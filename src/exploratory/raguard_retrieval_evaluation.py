"""
RAGuard step E - does the factuality direction rerank misleading documents down?

Unlike the claim-verdict diagnostic (raguard_direction_evaluation.py), RAGuard's document
labels ARE factuality-relevant: a `misleading` document is one that pushes a reader toward
the WRONG verdict, so its content is misleading regardless of the claim's polarity, and a
`supporting` document pushes toward the right one. `unrelated` documents are topically
retrievable but non-evidential - RAGuard's hard negatives.

Same fusion as retrieval_evaluation.py, query = the claim text:

    score(d, q) = (1 - alpha) * zscore(s_cos) + alpha * zscore(s_proj)

but ranked WITHIN each query's own annotated document pool, not over a global corpus:

  * out-of-pool documents carry no label with respect to a given query - they are not
    `unrelated` in RAGuard's sense, and merging the two would destroy the distinction the
    experiment is about;
  * s_proj = H @ v has no query term, so on a global corpus alpha=1.0 returns the identical
    top-k for every query. Per-pool ranking keeps that column meaningful.

So this measures RERANKING, with an exact random baseline (the pool composition itself).

Query selection: only claims with >=1 supporting AND >=1 misleading document (350 of 2648).
The other 2298 have no headroom - 64% of all claims have no supporting document and 70% no
misleading one, so their label mix at top-k cannot move with alpha. Selected pools hold 3324
document rows (2929 unique texts): 960 supporting / 793 misleading / 1571 unrelated, over
166 verdict=True and 184 verdict=False claims. Every metric is also broken down by verdict.

Both z-scores are taken WITHIN the pool (retrieval_evaluation.py z-scores s_proj globally):
the candidate set is the pool, so that is what makes the two scores commensurate before
mixing. Direction normalization is therefore irrelevant here - zscore(c * s) == zscore(s).

Document text is `Title + "\n\n" + Full Text`: 28% of RAGuard's rows have a placeholder
Full Text (`[Link Post]`, `[deleted]`) yet carry real labels, and Title is always populated.
Dropping them would bias the pools; using Title too leaves 32 degenerate rows in 16331.
Rows are keyed by Document ID (unique); text is deduplicated only for the forward pass.

Documents are truncated to the leading --max-tokens tokens (default 1024). RAGuard's length
tail reaches ~8.5k tokens and attention is O(L^2), which OOMs a 24GB card at any batch size;
1024 leaves 97% of the selected documents intact (512 would leave 92%). Note the two scorers
see different windows either way: all-MiniLM-L6-v2 caps at 256 wordpieces internally, so the
SBERT side is always truncated harder than the projection side.

Layer: one per model, the modal best layer over the 7 combos in
results/raguard/figures/summary_mixed.json (gemma L2, Llama-8B L15, Llama-1B L4, Qwen L5 -
which also happen to be the 3-way mixture's best layer for all four). Override with --layer.

Reads:  data/raguard/{claims.csv,documents.csv}
        results/mixed_directions/<model>/<combo>/seed_42/context_only/layer_<L>/last_pos/direction.pt
        results/raguard/figures/summary_mixed.json
Writes: results/raguard_retrieval/docs.jsonl                              (selected pools)
        results/raguard_retrieval/sbert_embeddings.pt                     (model-independent)
        results/raguard_retrieval/<model>/layer_<L>/hidden_states_max<T>.pt
        results/raguard_retrieval/<model>/layer_<L>/<combo>/results.jsonl (one row per query/alpha/k)
        results/raguard_retrieval/<model>/layer_<L>/summary.jsonl
        results/raguard_retrieval/summary.jsonl                           (all models)

Needs a GPU for the hidden states (2929 documents, one forward pass per model); everything
after that is dot products, so the 7 combos are free.

Usage:
    python src/exploratory/raguard_retrieval_evaluation.py
    python src/exploratory/raguard_retrieval_evaluation.py --models meta-llama/Llama-3.2-1B-Instruct
    python src/exploratory/raguard_retrieval_evaluation.py --layer 15 --force-recompute
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
from utils import REPO_ROOT, RESULTS_DIR, logger, safe_model_id, setup_logging, write_jsonl

import transformer_lens.utils as tl_utils
from transformer_lens import HookedTransformer
from sentence_transformers import SentenceTransformer

MODELS = ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-1B-Instruct",
          "google/gemma-3-4b-it", "Qwen/Qwen2-7B-Instruct"]
COMBOS = ["conflictqa", "nq_swap", "longfact",
          "conflictqa+nq_swap", "conflictqa+longfact", "nq_swap+longfact",
          "conflictqa+nq_swap+longfact"]
LABELS = ["supporting", "misleading", "unrelated"]
SEED = 42                      # mixtures exist at seed 42 only
PROCEDURE = "context_only"
POSITION = "last_pos"
ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]
KS = [1, 2, 3, 5]

RAGUARD_DIR = REPO_ROOT / "data" / "raguard"
OUT_ROOT = RESULTS_DIR / "raguard_retrieval"
SUMMARY_MIXED = RESULTS_DIR / "raguard" / "figures" / "summary_mixed.json"
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 4
MAX_TOKENS = 1024              # see the truncation note in the module docstring

EXPECTED_QUERIES = 350         # claims with >=1 supporting and >=1 misleading document


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-8)


def best_layer(model_safe: str) -> int:
    """Modal best layer over the 7 combos in the claim-separation summary."""
    if not SUMMARY_MIXED.exists():
        raise FileNotFoundError(f"{SUMMARY_MIXED} not found. Run plot_raguard_claim_separation.py "
                                f"--direction-source mixed, or pass --layer.")
    summary = json.loads(SUMMARY_MIXED.read_text())
    if model_safe not in summary:
        raise KeyError(f"{model_safe} not in {SUMMARY_MIXED}; pass --layer explicitly.")
    layers = Counter(v["layer"] for v in summary[model_safe].values())
    return layers.most_common(1)[0][0]


def load_pools() -> tuple[List[Dict], List[str]]:
    """Selected queries with their document pools, plus the deduplicated document texts.

    Returns (queries, texts) where each query is
        {"claim_id", "claim", "verdict", "docs": [{"doc_id", "label", "text_idx"}, ...]}
    and `texts[text_idx]` is the document text. Rows are kept separate even when two rows in
    one pool share a text (33 cases, 4 with disagreeing labels): the label lives on the row.
    """
    csv.field_size_limit(10 ** 9)
    with (RAGUARD_DIR / "claims.csv").open(newline="") as f:
        claims = {int(r["Claim ID"]): r for r in csv.DictReader(f)}
    with (RAGUARD_DIR / "documents.csv").open(newline="") as f:
        doc_rows = list(csv.DictReader(f))

    pools: Dict[int, List[Dict]] = defaultdict(list)
    for r in doc_rows:
        pools[int(r["Claim ID"])].append(r)

    text_idx: Dict[str, int] = {}
    queries = []
    for claim_id in sorted(pools):
        rows = pools[claim_id]
        labels = Counter(r["Document Label"].strip() for r in rows)
        if not (labels["supporting"] and labels["misleading"]):
            continue
        docs = []
        for r in rows:
            text = f"{r['Title']}\n\n{r['Full Text']}".strip()
            docs.append({"doc_id": int(r["Document ID"]),
                         "label": r["Document Label"].strip(),
                         "text_idx": text_idx.setdefault(text, len(text_idx))})
        c = claims[claim_id]
        queries.append({"claim_id": claim_id,
                        "claim": c["Claim"].strip().strip('"').strip(),
                        "verdict": c["Verdict"].strip() == "True",
                        "docs": docs})

    texts = [t for t, _ in sorted(text_idx.items(), key=lambda kv: kv[1])]
    assert len(queries) == EXPECTED_QUERIES, f"Expected {EXPECTED_QUERIES} queries, got {len(queries)}"
    n_rows = sum(len(q["docs"]) for q in queries)
    mix = Counter(d["label"] for q in queries for d in q["docs"])
    logger.info(f"{len(queries)} queries ({sum(q['verdict'] for q in queries)} True / "
                f"{sum(not q['verdict'] for q in queries)} False) | {n_rows} document rows "
                f"({len(texts)} unique texts) | {dict(mix)}")
    return queries, texts


def compute_hidden_states(model_name: str, texts: List[str], layer: int, batch_size: int,
                          max_tokens: int) -> torch.Tensor:
    """Last-token resid_post at `layer` for every document text, truncated to `max_tokens`."""
    device = tl_utils.get_device()
    model = HookedTransformer.from_pretrained_no_processing(model_name, device=device, dtype="bfloat16")
    model.eval()
    hook_point = tl_utils.get_act_name("resid_post", layer)

    tok = model.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Keep the leading max_tokens (BOS + title + lead): attention is O(L^2) and RAGuard's
    # tail runs to ~8.5k tokens, which OOMs a 24GB card at any batch size.
    enc = [tok(t, return_tensors="pt", add_special_tokens=True).input_ids[0][:max_tokens] for t in texts]
    n_trunc = sum(1 for t, e in zip(texts, enc) if e.shape[0] == max_tokens)
    logger.info(f"Tokenized {len(texts)} documents | max_tokens={max_tokens} | "
                f"{n_trunc} at the cap ({n_trunc / len(texts):.1%})")
    order = sorted(range(len(texts)), key=lambda i: enc[i].shape[0])
    hidden = torch.zeros(len(texts), model.cfg.d_model)

    for s in tqdm(range(0, len(order), batch_size), desc="LLM hidden states"):
        idxs = order[s: s + batch_size]
        L = max(enc[i].shape[0] for i in idxs)
        batch = torch.full((len(idxs), L), tok.pad_token_id, dtype=torch.long, device=device)
        # Explicit mask, same reasoning as raguard_claim_activations.py: without it the real
        # tokens attend to the pads and a document's activation depends on its batch mates.
        mask = torch.zeros((len(idxs), L), dtype=torch.long, device=device)
        for r, i in enumerate(idxs):          # left-pad so index -1 is the true last token
            t = enc[i]
            batch[r, L - t.shape[0]:] = t.to(device)
            mask[r, L - t.shape[0]:] = 1
        with torch.no_grad():
            _, cache = model.run_with_cache(batch, names_filter=hook_point,
                                            attention_mask=mask, prepend_bos=False)
        resid = cache[hook_point][:, -1, :].detach().float().cpu()
        for r, i in enumerate(idxs):
            hidden[i] = resid[r]
        del cache

    del model
    torch.cuda.empty_cache()
    if not torch.isfinite(hidden).all():
        raise ValueError(f"Non-finite hidden states for {model_name} layer {layer}")
    return hidden


def evaluate_combo(queries: List[Dict], hidden: torch.Tensor, direction: torch.Tensor,
                   doc_cos: np.ndarray, doc_len: np.ndarray,
                   alphas: List[float], ks: List[int]) -> List[Dict]:
    """One row per (query, alpha, k): label mix and mean length of the top-k."""
    s_proj_all = (hidden @ direction).numpy()

    records = []
    for q in queries:
        pool = q["docs"]
        idx = np.array([d["text_idx"] for d in pool])
        lab = np.array([d["label"] for d in pool])
        n = len(pool)

        s_cos = zscore(doc_cos[q["claim_id"]])
        s_proj = zscore(s_proj_all[idx])
        lens = doc_len[idx]
        pool_frac = {L: float((lab == L).mean()) for L in LABELS}

        for alpha in alphas:
            scores = (1 - alpha) * s_cos + alpha * s_proj
            order = np.argsort(-scores)
            for k in ks:
                if k > n:
                    continue
                top = order[:k]
                top_lab = lab[top]
                records.append({
                    "claim_id": q["claim_id"], "verdict": q["verdict"],
                    "alpha": alpha, "k": k, "pool_size": n,
                    **{f"n_{L}": int((top_lab == L).sum()) for L in LABELS},
                    **{f"frac_{L}": float((top_lab == L).mean()) for L in LABELS},
                    **{f"pool_frac_{L}": pool_frac[L] for L in LABELS},
                    "mean_len_topk": float(lens[top].mean()),
                    "mean_len_pool": float(lens.mean()),
                    "topk_doc_ids": [pool[i]["doc_id"] for i in top],
                })
    return records


def summarize(records: List[Dict], model: str, layer: int, combo: str,
              alphas: List[float], ks: List[int]) -> List[Dict]:
    """Macro-average over queries, overall and split by claim verdict."""
    rows = []
    for alpha in alphas:
        for k in ks:
            sel = [r for r in records if r["alpha"] == alpha and r["k"] == k]
            if not sel:
                continue
            for split, subset in (("all", sel),
                                  ("verdict_true", [r for r in sel if r["verdict"]]),
                                  ("verdict_false", [r for r in sel if not r["verdict"]])):
                if not subset:
                    continue
                rows.append({
                    "model": model, "layer": layer, "combo": combo,
                    "alpha": alpha, "k": k, "split": split, "n_queries": len(subset),
                    **{f"frac_{L}": float(np.mean([r[f"frac_{L}"] for r in subset])) for L in LABELS},
                    # Random-ranking baseline: the pool composition itself.
                    **{f"random_{L}": float(np.mean([r[f"pool_frac_{L}"] for r in subset])) for L in LABELS},
                    "mean_len_topk": float(np.mean([r["mean_len_topk"] for r in subset])),
                    "mean_len_pool": float(np.mean([r["mean_len_pool"] for r in subset])),
                })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="RAGuard pool reranking by the factuality direction.")
    ap.add_argument("--models", nargs="+", default=MODELS, help="HuggingFace model ids.")
    ap.add_argument("--combos", nargs="+", default=COMBOS, help="Mixed-direction combos to score.")
    ap.add_argument("--layer", type=int, default=None,
                    help="Layer for every model (default: modal best layer per model from "
                         "results/raguard/figures/summary_mixed.json).")
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    ap.add_argument("--ks", type=int, nargs="+", default=KS)
    ap.add_argument("--sbert-model", default=SBERT_MODEL)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                    help="Truncate documents to this many leading tokens before the forward pass.")
    ap.add_argument("--force-recompute", action="store_true",
                    help="Recompute hidden states / embeddings even if cached.")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    setup_logging("raguard_retrieval_evaluation", OUT_ROOT)
    queries, texts = load_pools()

    write_jsonl(OUT_ROOT / "docs.jsonl", [{"text_idx": i, "text": t} for i, t in enumerate(texts)])
    doc_len = np.array([len(t.split()) for t in texts], dtype=float)

    # SBERT scores are model-independent: encode once, cache the document embeddings, and
    # precompute the per-query cosine vector over that query's pool.
    sbert_path = OUT_ROOT / "sbert_embeddings.pt"
    sbert_enc = SentenceTransformer(args.sbert_model)
    if sbert_path.exists() and not args.force_recompute:
        sbert_emb = torch.load(sbert_path, map_location="cpu")
        assert sbert_emb.shape[0] == len(texts), "Cached SBERT embeddings do not match the corpus"
        logger.info(f"Loaded cached SBERT embeddings {tuple(sbert_emb.shape)}")
    else:
        logger.info("Encoding documents with SBERT ...")
        sbert_emb = torch.tensor(sbert_enc.encode(texts, batch_size=64, show_progress_bar=True,
                                                  convert_to_numpy=True), dtype=torch.float32)
        torch.save(sbert_emb, sbert_path)
    doc_norm = sbert_emb / sbert_emb.norm(dim=1, keepdim=True)

    logger.info("Encoding claims with SBERT ...")
    q_emb = torch.tensor(sbert_enc.encode([q["claim"] for q in queries], batch_size=64,
                                          show_progress_bar=True, convert_to_numpy=True),
                         dtype=torch.float32)
    q_norm = q_emb / q_emb.norm(dim=1, keepdim=True)
    doc_cos = {q["claim_id"]: (doc_norm[[d["text_idx"] for d in q["docs"]]] @ q_norm[i]).numpy()
               for i, q in enumerate(queries)}

    all_summary = []
    for model_name in args.models:
        model_safe = safe_model_id(model_name)
        layer = args.layer if args.layer is not None else best_layer(model_safe)
        layer_dir = OUT_ROOT / model_safe / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"##### {model_name} | layer {layer} #####")

        # The token cap changes the activations, so it is part of the cache key.
        hidden_path = layer_dir / f"hidden_states_max{args.max_tokens}.pt"
        if hidden_path.exists() and not args.force_recompute:
            hidden = torch.load(hidden_path, map_location="cpu")
            assert hidden.shape[0] == len(texts), f"Cached hidden states in {layer_dir} do not match the corpus"
            logger.info(f"Loaded cached hidden states {tuple(hidden.shape)}")
        else:
            hidden = compute_hidden_states(model_name, texts, layer, args.batch_size, args.max_tokens)
            torch.save(hidden, hidden_path)

        model_summary = []
        for combo in args.combos:
            dpath = (RESULTS_DIR / "mixed_directions" / model_safe / combo / f"seed_{SEED}"
                     / PROCEDURE / f"layer_{layer}" / POSITION / "direction.pt")
            if not dpath.exists():
                logger.warning(f"Skip (no direction): {dpath}")
                continue
            direction = torch.load(dpath, map_location="cpu").float()
            records = evaluate_combo(queries, hidden, direction, doc_cos, doc_len,
                                     args.alphas, args.ks)
            write_jsonl(layer_dir / combo / "results.jsonl", records)
            model_summary += summarize(records, model_safe, layer, combo, args.alphas, args.ks)

        write_jsonl(layer_dir / "summary.jsonl", model_summary)
        all_summary += model_summary

        # Compact readout: the 3-way mixture at every alpha, k=3, all queries.
        for row in model_summary:
            if row["combo"] == "conflictqa+nq_swap+longfact" and row["k"] == 3 and row["split"] == "all":
                logger.info(f"  alpha={row['alpha']:.1f} k=3 | sup {row['frac_supporting']:.3f} "
                            f"mis {row['frac_misleading']:.3f} unrel {row['frac_unrelated']:.3f} "
                            f"| random sup {row['random_supporting']:.3f} mis {row['random_misleading']:.3f} "
                            f"| len {row['mean_len_topk']:.0f} (pool {row['mean_len_pool']:.0f})")

    write_jsonl(OUT_ROOT / "summary.jsonl", all_summary)
    logger.info(f"Wrote {len(all_summary)} summary rows -> {OUT_ROOT / 'summary.jsonl'}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
