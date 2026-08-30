"""
ClashEval stage 1: document activations, null controls and separation metrics.

Llama-3.1-8B-Instruct on the drugs+news headline subset (the two non-Wikipedia domains).

Stages (each cached, resumable with --force):
  0. Data prep (CPU): load data/clasheval_gpt4.pqt, dedup, apply exclusions, assert the
     expected row counts, select the drugs+news headline subset.
  1. Chunking (CPU): sentence-aligned, non-overlapping <=512-token chunks (blingfire split +
     greedy token packing). The chunker is applied to context_original and context_mod
     independently and never sees which side it is chunking, so it cannot leak the label.
  2. GPU (one HookedTransformer load for both):
       (a) last-token resid_post at every layer for every headline chunk, mean-aggregated per
           document -> H[n_layers, n_pairs, 2(orig/mod), d_model], cached to disk.
       (b) the same for nq_swap train (seed_42), used to build the shuffled-label control.
  3. Null controls (CPU): random unit direction and shuffled-factuality-label direction,
     5 seeds each -> results/clasheval_hidden/<model>/controls/.
  4. Metrics (CPU): paired AUROC = P(delta>0) + 0.5 P(delta=0), delta = h(orig).v - h(mod).v,
     averaged over all layers (no layer selection) and over the 5 identification seeds, with a
     95% bootstrap CI over pairs. Run for nq_swap, conflictqa and the two null controls.

Reads data/clasheval_gpt4.pqt, results/direction_identification/ and
data/normalized_dataset/nq_swap/seed_42/train.jsonl; writes results/clasheval_hidden/.

Usage:
    python src/experiments/clasheval_pipeline.py
    ... --force-hidden        # recompute GPU stage even if cache exists
    ... --skip-gpu            # metrics only, from cache (fails loudly if no cache)
    ... --batch-size 8 --max-chunk-tokens 512
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from utils import DATA_DIR, NORMALIZED_DIR, RESULTS_DIR  # noqa: E402

DATA_PQT = DATA_DIR / "clasheval_gpt4.pqt"
OUT_ROOT = RESULTS_DIR / "clasheval_hidden"

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HEADLINE_DOMAINS = ["drugs", "news"]
EXPECTED_TOTAL = 9855
EXPECTED_HEADLINE = 4755          # checked in stage 0; a mismatch is reported, not forced
DIRECTION_SOURCES = ["nq_swap", "conflictqa"]
SEEDS = [7, 42, 67, 89, 90]
PROCEDURE, POSITION = "context_only", "last_pos"
MAX_CHUNK_TOKENS = 512
BATCH_SIZE = 8
N_BOOT = 1000


def safe_model_id(m: str) -> str:
    return m.replace("/", "__")


# ============================================================================ stage 0: data prep

def load_eval_pairs() -> pd.DataFrame:
    """Dedup + exclusions. Returns the evaluation set over all 6 domains."""
    df = pd.read_parquet(DATA_PQT)
    raw_n = len(df)
    d = df.drop_duplicates(subset=["question", "context_original", "context_mod"]).reset_index(drop=True)
    dedup_n = len(d)

    mod_type = pd.to_numeric(d["mod_type"], errors="coerce")
    assert mod_type.isna().sum() == 0, "mod_type failed to parse as numeric for some rows"
    baseline = mod_type == 0
    n_baseline = int(baseline.sum())
    d = d[~baseline].reset_index(drop=True)

    degenerate = d["context_mod"] == d["context_original"]
    n_degenerate = int(degenerate.sum())
    d = d[~degenerate].reset_index(drop=True)

    print(f"[dataprep] raw={raw_n} dedup={dedup_n} excl_baseline(mod_type==0)={n_baseline} "
          f"excl_degenerate(context_mod==context_original)={n_degenerate} -> usable={len(d)}")
    if len(d) != EXPECTED_TOTAL:
        print(f"[dataprep][WARNING] expected {EXPECTED_TOTAL} evaluation pairs, got {len(d)}.")
    else:
        print(f"[dataprep] matches the expected {EXPECTED_TOTAL} exactly.")

    ans_mod = pd.to_numeric(d["answer_mod"], errors="coerce")
    n_undefined_sev = int((ans_mod == 0).sum())
    print(f"[dataprep] rows with undefined severity (answer_mod==0): {n_undefined_sev} -- "
          f"kept: the AUROC metric does not use severity.")

    d["pair_id"] = np.arange(len(d))
    print("[dataprep] per-domain counts (post-exclusion):")
    print(d["dataset"].value_counts().to_string())
    return d


def select_headline(d: pd.DataFrame, domains: list[str]) -> pd.DataFrame:
    sub = d[d["dataset"].isin(domains)].reset_index(drop=True)
    print(f"[dataprep] headline subset {domains}: {len(sub)} pairs")
    if domains == HEADLINE_DOMAINS and len(sub) != EXPECTED_HEADLINE:
        print(f"[dataprep][DISCREPANCY] expected {EXPECTED_HEADLINE} pairs for drugs+news, "
              f"measured {len(sub)} (delta {EXPECTED_HEADLINE - len(sub)}). Reported, not forced.")
    return sub


# ============================================================================ stage 1: chunking

def sentence_chunks(text: str, tok, max_tokens: int) -> list[str]:
    """Sentence-aligned, non-overlapping, <=max_tokens chunks. Deterministic, label-independent:
    takes only `text` and never knows whether it is chunking an original or a modified context."""
    from blingfire import text_to_sentences_and_offsets

    text = text or ""
    if not text.strip():
        return []
    _, offsets = text_to_sentences_and_offsets(text)
    sentences = [text[a:b] for a, b in offsets] if offsets else [text]

    chunks: list[str] = []
    cur: list[str] = []
    cur_n = 0
    for s in sentences:
        n = len(tok.encode(s, add_special_tokens=False))
        if cur and cur_n + n > max_tokens:
            chunks.append(" ".join(cur))
            cur, cur_n = [], 0
        cur.append(s)
        cur_n += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def identical_fraction(orig_chunks: list[str], mod_chunks: list[str]) -> float:
    """Fraction of the larger chunk-list that finds a byte-identical match in the other list
    (via difflib matching blocks over the chunk sequence, order-respecting). 1.0 = every chunk
    of the longer list is reproduced verbatim in the shorter one."""
    denom = max(len(orig_chunks), len(mod_chunks))
    if denom == 0:
        return 1.0
    sm = difflib.SequenceMatcher(None, orig_chunks, mod_chunks, autojunk=False)
    matched = sum(block.size for block in sm.get_matching_blocks())
    return matched / denom


def chunk_headline(df: pd.DataFrame, tok, max_tokens: int) -> tuple[list[list[str]], list[list[str]], np.ndarray]:
    orig_chunks_all, mod_chunks_all, fracs = [], [], []
    for _, row in df.iterrows():
        oc = sentence_chunks(row["context_original"], tok, max_tokens)
        mc = sentence_chunks(row["context_mod"], tok, max_tokens)
        orig_chunks_all.append(oc)
        mod_chunks_all.append(mc)
        fracs.append(identical_fraction(oc, mc))
    return orig_chunks_all, mod_chunks_all, np.array(fracs)


# ============================================================================ stage 2: GPU

def all_layer_last_token(model, texts: list[str], batch_size: int, max_tokens: int) -> torch.Tensor:
    """Last-token resid_post at EVERY layer, one forward pass per batch (hook grabs only
    position -1). Returns [n_layers, n_texts, d_model] bf16. Copied from
    src/experiments/raguard_layer_sweep.py::compute_all_layers (same pattern, reused verbatim
    for consistency with the RAGuard pipeline)."""
    import transformer_lens.utils as tl_utils
    from tqdm import tqdm

    device = model.cfg.device
    n_layers, d_model = model.cfg.n_layers, model.cfg.d_model
    hook_names = [tl_utils.get_act_name("resid_post", L) for L in range(n_layers)]

    tok = model.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    enc = [tok(t, return_tensors="pt", add_special_tokens=True).input_ids[0][:max_tokens] for t in texts]
    n_trunc = sum(1 for e in enc if e.shape[0] == max_tokens)
    print(f"    tokenized {len(texts)} texts | max_tokens={max_tokens} | "
          f"{n_trunc} at the cap ({n_trunc / max(1, len(texts)):.1%})")
    order = sorted(range(len(texts)), key=lambda i: enc[i].shape[0])
    out = torch.zeros(n_layers, len(texts), d_model, dtype=torch.bfloat16)

    grabbed: dict[str, torch.Tensor] = {}

    def make_hook(name):
        def hook(act, hook):
            grabbed[name] = act[:, -1, :].detach().float().cpu()
            return act
        return hook

    fwd_hooks = [(n, make_hook(n)) for n in hook_names]

    for s in tqdm(range(0, len(order), batch_size), desc="  resid_post all-layers"):
        idxs = order[s: s + batch_size]
        L = max(enc[i].shape[0] for i in idxs)
        batch = torch.full((len(idxs), L), tok.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros((len(idxs), L), dtype=torch.long, device=device)
        for r, i in enumerate(idxs):
            t = enc[i]
            batch[r, L - t.shape[0]:] = t.to(device)
            mask[r, L - t.shape[0]:] = 1
        grabbed.clear()
        with torch.no_grad():
            model.run_with_hooks(batch, fwd_hooks=fwd_hooks, attention_mask=mask,
                                 prepend_bos=False, return_type=None)
        for li, name in enumerate(hook_names):
            resid = grabbed[name]
            for r, i in enumerate(idxs):
                out[li, i] = resid[r].bfloat16()

    if not torch.isfinite(out).all():
        raise ValueError("Non-finite hidden states")
    return out


def aggregate_chunks_to_docs(H_chunks: torch.Tensor, group_id: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Mean over chunks within each (pair, side) group. H_chunks: [n_layers, n_chunks, d_model].
    Returns [n_layers, n_groups, d_model] float32 (cast to bf16 by the caller before saving)."""
    n_layers, n_chunks, d_model = H_chunks.shape
    Hf = H_chunks.float()
    counts = torch.bincount(group_id, minlength=n_groups).float().clamp(min=1)
    sums = torch.zeros(n_layers, n_groups, d_model, dtype=torch.float32)
    sums.index_add_(1, group_id, Hf)
    return sums / counts.view(1, -1, 1)


def load_nq_swap_train(seed: int = 42) -> tuple[list[str], list[str]]:
    """factual_context / non_factual_evidence texts, matching direction_identification.py's
    context_only text extraction exactly (READ-ONLY)."""
    path = NORMALIZED_DIR / "nq_swap" / f"seed_{seed}" / "train.jsonl"
    rows = [json.loads(l) for l in path.open() if l.strip()]
    pos = [r["factual_context"] for r in rows if r.get("factual_context")]
    neg = [r["non_factual_evidence"] for r in rows if r.get("non_factual_evidence")]
    return pos, neg


def run_gpu_stage(model_name: str, headline_df: pd.DataFrame, orig_chunks: list[list[str]],
                  mod_chunks: list[list[str]], batch_size: int, max_tokens: int,
                  hidden_dir: Path, force: bool) -> dict:
    """Everything that needs the model loaded, in one process/one GPU hold.
    Writes headline_doc_repr.pt (+meta.json) and nq_swap_train_acts.pt (+meta.json)."""
    doc_repr_path = hidden_dir / "headline_doc_repr.pt"
    doc_meta_path = hidden_dir / "headline_doc_repr_meta.json"
    train_acts_path = hidden_dir / "nq_swap_train_acts.pt"
    train_meta_path = hidden_dir / "nq_swap_train_acts_meta.json"

    need_doc = force or not (doc_repr_path.exists() and doc_meta_path.exists())
    need_train = force or not (train_acts_path.exists() and train_meta_path.exists())
    timings = {}
    if not need_doc and not need_train:
        print("[gpu] both caches present, skipping model load entirely.")
        return timings

    import transformer_lens.utils as tl_utils
    from transformer_lens import HookedTransformer

    device = tl_utils.get_device()
    print(f"[gpu] loading {model_name} on {device} (bf16) ...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained_no_processing(model_name, device=device, dtype="bfloat16")
    model.eval()
    timings["model_load_s"] = time.time() - t0
    print(f"[gpu] loaded in {timings['model_load_s']:.1f}s | n_layers={model.cfg.n_layers} d_model={model.cfg.d_model}")

    if need_doc:
        t0 = time.time()
        flat_texts: list[str] = []
        group_id: list[int] = []          # pair_idx * 2 + side
        n_chunks_orig, n_chunks_mod = [], []
        for pi in range(len(headline_df)):
            oc, mc = orig_chunks[pi], mod_chunks[pi]
            n_chunks_orig.append(len(oc))
            n_chunks_mod.append(len(mc))
            for c in oc:
                flat_texts.append(c)
                group_id.append(pi * 2 + 0)
            for c in mc:
                flat_texts.append(c)
                group_id.append(pi * 2 + 1)
        print(f"[gpu] headline chunks: {len(flat_texts)} flat chunks over {len(headline_df)} pairs "
              f"({len(flat_texts) / len(headline_df):.2f} chunks/doc-pair-side on average)")
        H_chunks = all_layer_last_token(model, flat_texts, batch_size, max_tokens)
        group_id_t = torch.tensor(group_id, dtype=torch.long)
        n_groups = len(headline_df) * 2
        H_doc = aggregate_chunks_to_docs(H_chunks, group_id_t, n_groups)  # [n_layers, n_groups, d_model] f32
        H_doc = H_doc.view(model.cfg.n_layers, len(headline_df), 2, model.cfg.d_model).bfloat16()
        del H_chunks
        hidden_dir.mkdir(parents=True, exist_ok=True)
        torch.save(H_doc, doc_repr_path)
        meta = {
            "model": model_name, "n_layers": model.cfg.n_layers, "d_model": model.cfg.d_model,
            "n_pairs": len(headline_df), "pair_ids": headline_df["pair_id"].tolist(),
            "domains": headline_df["dataset"].tolist(),
            "n_chunks_orig": n_chunks_orig, "n_chunks_mod": n_chunks_mod,
            "max_chunk_tokens": max_tokens, "batch_size": batch_size,
            "n_flat_chunks": len(flat_texts),
        }
        doc_meta_path.write_text(json.dumps(meta))
        timings["headline_hidden_s"] = time.time() - t0
        print(f"[gpu] wrote {doc_repr_path} {tuple(H_doc.shape)} "
              f"({doc_repr_path.stat().st_size / 1e6:.0f} MB) in {timings['headline_hidden_s']:.1f}s")
        del H_doc

    if need_train:
        t0 = time.time()
        pos, neg = load_nq_swap_train(seed=42)
        print(f"[gpu] nq_swap train (seed 42): {len(pos)} pos, {len(neg)} neg")
        H_pos = all_layer_last_token(model, pos, batch_size, max_tokens=1024)  # full doc, no chunking (matches direction_identification.py)
        H_neg = all_layer_last_token(model, neg, batch_size, max_tokens=1024)
        torch.save({"pos": H_pos, "neg": H_neg}, train_acts_path)
        train_meta_path.write_text(json.dumps({
            "model": model_name, "n_layers": model.cfg.n_layers, "d_model": model.cfg.d_model,
            "n_pos": len(pos), "n_neg": len(neg), "source": "nq_swap seed_42 train, context_only texts",
        }))
        timings["nq_swap_train_hidden_s"] = time.time() - t0
        print(f"[gpu] wrote {train_acts_path} in {timings['nq_swap_train_hidden_s']:.1f}s")

    del model
    torch.cuda.empty_cache()
    return timings


# ============================================================================ stage 3: null controls

def build_random_unit_controls(d_model: int, n_layers: int, seeds: list[int], out_dir: Path, force: bool) -> None:
    for seed in seeds:
        for L in range(n_layers):
            p = out_dir / "random_unit" / f"seed_{seed}" / f"layer_{L}" / "direction.pt"
            if p.exists() and not force:
                continue
            rng = np.random.default_rng(seed * 100_000 + L)
            v = rng.standard_normal(d_model).astype(np.float32)
            v = v / (np.linalg.norm(v) + 1e-8)
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.from_numpy(v), p)
            (p.parent / "meta.json").write_text(json.dumps(
                {"method": "random_unit", "seed": seed, "layer": L, "d_model": d_model}))
    print(f"[controls] random_unit written for {len(seeds)} seeds x {n_layers} layers")


def build_shuffled_label_controls(hidden_dir: Path, seeds: list[int], out_dir: Path, force: bool) -> None:
    """Diff-in-means with the pos/neg identity permuted, on the same nq_swap train activations
    used for the real direction (READ from our own GPU-stage cache, not from results/)."""
    train_acts_path = hidden_dir / "nq_swap_train_acts.pt"
    if not train_acts_path.exists():
        raise FileNotFoundError(f"{train_acts_path} missing; run the GPU stage first.")
    acts = torch.load(train_acts_path, map_location="cpu")
    H_pos, H_neg = acts["pos"].float(), acts["neg"].float()   # [n_layers, n, d_model] each
    n_layers = H_pos.shape[0]
    n_pos, n_neg = H_pos.shape[1], H_neg.shape[1]
    for seed in seeds:
        rng = np.random.default_rng(seed + 999_000)
        pool = torch.cat([H_pos, H_neg], dim=1)                # [n_layers, n_pos+n_neg, d_model]
        n_total = n_pos + n_neg
        perm_pos_idx = rng.choice(n_total, size=n_pos, replace=False)
        mask = np.zeros(n_total, dtype=bool)
        mask[perm_pos_idx] = True
        for L in range(n_layers):
            p = out_dir / "shuffled_label" / f"seed_{seed}" / f"layer_{L}" / "direction.pt"
            if p.exists() and not force:
                continue
            shuf_pos = pool[L, mask]
            shuf_neg = pool[L, ~mask]
            direction = shuf_pos.mean(dim=0) - shuf_neg.mean(dim=0)
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(direction, p)
            (p.parent / "meta.json").write_text(json.dumps({
                "method": "shuffled_label_diff_in_means", "seed": seed, "layer": L,
                "n_pos": n_pos, "n_neg": n_neg, "source": "nq_swap seed_42 train (own GPU-stage cache)",
            }))
    print(f"[controls] shuffled_label written for {len(seeds)} seeds x {n_layers} layers")


# ============================================================================ stage 4: metrics

def load_real_direction(model_name: str, dataset: str, seed: int, layer: int) -> torch.Tensor:
    p = (RESULTS_DIR / "direction_identification" / safe_model_id(model_name) / dataset
         / f"seed_{seed}" / PROCEDURE / f"layer_{layer}" / POSITION / "direction.pt")
    return torch.load(p, map_location="cpu").float()


def load_control_direction(hidden_dir: Path, method: str, seed: int, layer: int) -> torch.Tensor:
    p = hidden_dir / "controls" / method / f"seed_{seed}" / f"layer_{layer}" / "direction.pt"
    return torch.load(p, map_location="cpu").float()


def paired_correct_matrix(H_doc: torch.Tensor, direction_getter, seeds: list[int], n_layers: int) -> np.ndarray:
    """C[seed, layer, pair] in {0, 0.5, 1}: 1 if delta=h(orig).v - h(mod).v > 0, 0 if <0, 0.5 if ==0."""
    n_pairs = H_doc.shape[1]
    Hf = H_doc.float()
    C = np.empty((len(seeds), n_layers, n_pairs), dtype=np.float32)
    for si, seed in enumerate(seeds):
        for L in range(n_layers):
            v = direction_getter(seed, L)
            h_orig = (Hf[L, :, 0, :] @ v).numpy()
            h_mod = (Hf[L, :, 1, :] @ v).numpy()
            delta = h_orig - h_mod
            C[si, L] = np.where(delta > 0, 1.0, np.where(delta < 0, 0.0, 0.5))
    return C


def summarize_auroc(C: np.ndarray, n_boot: int, boot_seed: int) -> dict:
    n_seeds, n_layers, n_pairs = C.shape
    point = float(C.mean())
    per_layer_seed_mean = C.mean(axis=(0, 2))          # [n_layers], averaged over seeds+pairs -- for the curve
    flat = C.reshape(n_seeds * n_layers, n_pairs)
    rng = np.random.default_rng(boot_seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n_pairs, n_pairs)
        draws[b] = flat[:, idx].mean()
    return {
        "n_seeds": n_seeds, "n_layers": n_layers, "n_pairs": n_pairs,
        "auroc_all_layer_mean": point,
        "ci_lo": float(np.percentile(draws, 2.5)), "ci_hi": float(np.percentile(draws, 97.5)),
        "per_layer_curve": per_layer_seed_mean.tolist(),
    }


# ============================================================================ main

def main() -> None:
    ap = argparse.ArgumentParser(description="ClashEval pipeline (Llama-3.1-8B, drugs+news).")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--domains", nargs="+", default=HEADLINE_DOMAINS)
    ap.add_argument("--direction-sources", nargs="+", default=DIRECTION_SOURCES)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--max-chunk-tokens", type=int, default=MAX_CHUNK_TOKENS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="Recompute everything (hidden states + controls).")
    ap.add_argument("--force-hidden", action="store_true")
    ap.add_argument("--force-controls", action="store_true")
    ap.add_argument("--skip-gpu", action="store_true", help="Metrics only, from existing cache.")
    args = ap.parse_args()

    t_start = time.time()
    ms = safe_model_id(args.model)
    hidden_dir = OUT_ROOT / ms
    hidden_dir.mkdir(parents=True, exist_ok=True)

    # ---- stage 0: data prep
    full = load_eval_pairs()
    headline = select_headline(full, args.domains)

    # ---- stage 1: chunking + build check (c)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.time()
    orig_chunks, mod_chunks, fracs = chunk_headline(headline, tok, args.max_chunk_tokens)
    t_chunk = time.time() - t0
    chunk_check = {
        "n_pairs": len(headline), "mean_identical_fraction": float(fracs.mean()),
        "median_identical_fraction": float(np.median(fracs)),
        "p10": float(np.percentile(fracs, 10)), "p25": float(np.percentile(fracs, 25)),
        "frac_pairs_above_0.5": float((fracs > 0.5).mean()),
        "mean_n_chunks_orig": float(np.mean([len(c) for c in orig_chunks])),
        "mean_n_chunks_mod": float(np.mean([len(c) for c in mod_chunks])),
        "chunk_time_s": t_chunk,
    }
    (hidden_dir / "chunk_identity_check.json").write_text(json.dumps(chunk_check, indent=2))
    print(f"[chunk-check] mean identical-chunk fraction = {chunk_check['mean_identical_fraction']:.3f} "
          f"(median {chunk_check['median_identical_fraction']:.3f}) over {len(headline)} pairs, "
          f"{t_chunk:.1f}s")

    # ---- stage 2: GPU (skippable if caches already present)
    timings = {}
    if not args.skip_gpu:
        timings = run_gpu_stage(args.model, headline, orig_chunks, mod_chunks, args.batch_size,
                                args.max_chunk_tokens, hidden_dir, args.force or args.force_hidden)
    doc_repr_path = hidden_dir / "headline_doc_repr.pt"
    if not doc_repr_path.exists():
        raise FileNotFoundError(f"{doc_repr_path} missing and --skip-gpu was set; cannot compute metrics.")
    H_doc = torch.load(doc_repr_path, map_location="cpu")
    n_layers, n_pairs_cached, _, d_model = H_doc.shape
    print(f"[metrics] loaded H_doc {tuple(H_doc.shape)}")
    assert n_pairs_cached == len(headline), "cached doc repr does not match the current headline selection"

    # ---- stage 3: null controls
    build_random_unit_controls(d_model, n_layers, args.seeds, hidden_dir / "controls",
                               args.force or args.force_controls)
    build_shuffled_label_controls(hidden_dir, args.seeds, hidden_dir / "controls",
                                  args.force or args.force_controls)

    # ---- stage 4: metrics
    results = {}
    for ds in args.direction_sources:
        getter = lambda seed, L, ds=ds: load_real_direction(args.model, ds, seed, L)
        C = paired_correct_matrix(H_doc, getter, args.seeds, n_layers)
        results[ds] = summarize_auroc(C, args.n_boot, args.boot_seed)
        r = results[ds]
        print(f"[RESULT] {ds}: all-layer-mean paired AUROC = {r['auroc_all_layer_mean']:.4f} "
              f"(95% CI [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]) over {r['n_pairs']} pairs, "
              f"{r['n_seeds']} seeds x {r['n_layers']} layers")

    for method in ["random_unit", "shuffled_label"]:
        getter = lambda seed, L, method=method: load_control_direction(hidden_dir, method, seed, L)
        C = paired_correct_matrix(H_doc, getter, args.seeds, n_layers)
        results[method] = summarize_auroc(C, args.n_boot, args.boot_seed)
        r = results[method]
        gate = "PASS" if r["ci_lo"] <= 0.5 <= r["ci_hi"] else "FAIL"
        print(f"[BUILD CHECK {gate}] {method}: {r['auroc_all_layer_mean']:.4f} "
              f"(95% CI [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]) -- must contain 0.500")

    out = {
        "model": args.model, "domains": args.domains, "n_pairs_total_9855_check": len(full),
        "n_pairs_headline": len(headline), "chunk_check": chunk_check,
        "timings_s": timings, "wall_clock_total_s": time.time() - t_start,
        "results": results,
    }
    out_path = hidden_dir / "round1_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[write] {out_path}")
    print(f"[done] total wall clock: {out['wall_clock_total_s']:.1f}s")

    # Disk usage of what this run produced, for the cost-estimate report.
    total_bytes = sum(p.stat().st_size for p in hidden_dir.rglob("*") if p.is_file())
    print(f"[disk] {hidden_dir}: {total_bytes / 1e6:.1f} MB total")


if __name__ == "__main__":
    main()
