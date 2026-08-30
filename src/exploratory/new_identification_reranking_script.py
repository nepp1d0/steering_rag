"""
Regression harness for the attention-mask fix: direction identification + retrieval
reranking end to end, for ONE model / layer / seed, so it can be run in minutes.

Background. `collect_side_acts` used to left-pad batches without passing an attention
mask. TransformerLens only derives one when tokenizer.padding_side == "left", which is
false by default for Llama and Qwen, so real tokens attended to pad tokens and a chunk's
activation depended on which other chunks shared its batch. This script re-runs the whole
pipeline with the fixed extractor and diffs every plot metric against the results already
on disk, which were produced by the buggy path.

Three stages in one process:

  1. Identify  re-extract directions for all three direction datasets at both positions,
               calling the REAL `collect_side_acts` / `resolve_side_spans` from
               direction_identification.py (this is a regression test of that code path,
               not a reimplementation).
  2. Rerank    score(d, q) = (1 - alpha) * z(cos_SBERT(q, d)) + alpha * z(h(d) . v),
               identical to retrieval_evaluation.compute_evaluation.

               The cached llm_hidden_states.pt / sbert_embeddings.pt from the existing
               retrieval_evaluation run are REUSED (after verifying they match the current
               corpus). The document side was never affected by the bug -- retrieval's
               compute_llm_hidden_states does set padding_side="left" -- so only the
               direction changes. Reusing them keeps the comparison exact (identical
               document representations) and skips ~4.3k forward passes.
               Use --recompute-tensors to recompute them anyway.
  3. Compare   every metric behind the pipeline's plots, OLD vs NEW vs delta.

Metrics (definitions taken from plot_retrieval_evaluation.score_layer and
plot_figure3.build_data, so the numbers are directly comparable to the paper figures):
    gold_recall@k, nonfactual_rate@k          per (alpha, k)
    mean_gold_rank, mean_nf_rank              per alpha
    gold_rank_gain    = mean_gold_rank(0) - mean_gold_rank(alpha)   (>0: gold climbed)
    nf_rank_change    = mean_nf_rank(0)   - mean_nf_rank(alpha)     (<0: non-factual sank)
    rank_separation_gain = gold_rank_gain - nf_rank_change
    gold_lift = gold_rate(a) - gold_rate(0);  nf_drop = nf_rate(0) - nf_rate(a)
    layer_score = gold_lift + nf_drop                              (score_layer quantity)

NOTHING under results/direction_identification/ or results/retrieval_evaluation/ is
written: those stay intact as the comparison baseline. All output goes to
results/new_identification_reranking/.

Outputs:
    results/new_identification_reranking/
        directions/<direction_dataset>/<position>/{direction.pt,meta.json}
        <eval>/<direction>/<position>/results.jsonl
        direction_comparison.json      cos(old, new) + norms per direction
        metrics.csv                    OLD / NEW / delta for every metric
        summary.txt                    compact table at SUMMARY_ALPHA

Usage:
    python src/exploratory/new_identification_reranking_script.py
    python src/exploratory/new_identification_reranking_script.py --layer 18
    python src/exploratory/new_identification_reranking_script.py --skip-identify   # rerank only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
from utils import (
    RESULTS_DIR,
    diff_in_means,
    load_normalized,
    logger,
    safe_model_id,
    setup_logging,
    write_jsonl,
)
from direction_identification import (
    collect_side_acts,
    compute_conflictqa_qa_spans,
    resolve_side_spans,
)
from retrieval_evaluation import (
    cache_matches_corpus,
    compute_evaluation,
    compute_llm_hidden_states,
    corpus_of,
)
from plot_retrieval_evaluation import compute_seed_metrics

import transformer_lens.utils as tl_utils
from transformer_lens import HookedTransformer
from sentence_transformers import SentenceTransformer

# --- Fixed single-cell scope (all overridable by flags). ---
MODEL = "Qwen/Qwen2-7B-Instruct"
LAYER = 26                       # Qwen's best layer for conflictqa->conflictqa/last_pos
SEED = 42
DIRECTION_DATASETS = ["nq_swap", "conflictqa", "longfact"]
# longfact rows have empty question/factual_context (one-sided data), so it cannot be an
# eval corpus -- there is no query or gold document to rank. Direction-only, matching
# retrieval_evaluation.DATASETS.
EVAL_DATASETS = ["nq_swap", "conflictqa"]
POSITIONS = ["last_pos", "entity_pos"]
PROCEDURE = "context_only"
NORMALIZE_PATH = "unnormalized"  # directions are used unnormalized, as in the pipeline
ALPHAS = [0.0, 0.3, 0.5, 1.0]
KS = [2, 5, 10]
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
IDENT_BATCH_SIZE = 8             # matches direction_identification default
RERANK_BATCH_SIZE = 4            # matches retrieval_evaluation default
SUMMARY_ALPHA = 0.5

OUT_ROOT = RESULTS_DIR / "new_identification_reranking"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def new_direction_path(direction_ds: str, position: str) -> Path:
    return OUT_ROOT / "directions" / direction_ds / position / "direction.pt"


def old_direction_path(model: str, direction_ds: str, seed: int, layer: int, position: str) -> Path:
    return (RESULTS_DIR / "direction_identification" / safe_model_id(model) / direction_ds
            / f"seed_{seed}" / PROCEDURE / f"layer_{layer}" / position / "direction.pt")


def old_layer_dir(model: str, eval_ds: str, direction_ds: str, seed: int, layer: int) -> Path:
    return (RESULTS_DIR / "retrieval_evaluation" / safe_model_id(model) / eval_ds / direction_ds
            / NORMALIZE_PATH / f"seed_{seed}" / PROCEDURE / f"layer_{layer}")


def new_results_path(eval_ds: str, direction_ds: str, position: str) -> Path:
    return OUT_ROOT / eval_ds / direction_ds / position / "results.jsonl"


# ---------------------------------------------------------------------------
# Stage 1 - direction identification (fixed extractor)
# ---------------------------------------------------------------------------

def identify_directions(model_name: str, layer: int, seed: int, batch_size: int) -> None:
    """Re-extract every (direction dataset, position) direction with the fixed extractor."""
    # conflictqa's entity locator is a separate 125M QA model; run it before the LLM loads.
    span_data: Dict[str, Dict[str, list]] = {}
    for ds in DIRECTION_DATASETS:
        samples = load_normalized(ds, seed)["train"]
        qa_spans = compute_conflictqa_qa_spans(samples) if ds == "conflictqa" else None
        span_data[ds] = {side: resolve_side_spans(ds, samples, side, qa_spans) for side in ("pos", "neg")}
        for side in ("pos", "neg"):
            items = span_data[ds][side]
            n_res = sum(1 for _, sp in items if sp is not None)
            logger.info(f"{ds} {side}: {len(items)} texts | entity spans resolved: {n_res}")

    device = tl_utils.get_device()
    logger.info(f"##### Loading {model_name} on {device} #####")
    model = HookedTransformer.from_pretrained_no_processing(model_name, device=device, dtype="bfloat16")
    hook_point = tl_utils.get_act_name("resid_post", layer)

    for ds in DIRECTION_DATASETS:
        logger.info(f"=== directions: {ds} | layer {layer} ===")
        pos_last, pos_ent = collect_side_acts(model, hook_point, layer, span_data[ds]["pos"],
                                              batch_size=batch_size, desc=f"{ds} pos acts")
        neg_last, neg_ent = collect_side_acts(model, hook_point, layer, span_data[ds]["neg"],
                                              batch_size=batch_size, desc=f"{ds} neg acts")
        stacks = {"last_pos": (pos_last, neg_last), "entity_pos": (pos_ent, neg_ent)}
        for position in POSITIONS:
            p_stack, n_stack = stacks[position]
            if p_stack is None or n_stack is None:
                logger.warning(f"Skipping {ds}/{position}: no activations collected.")
                continue
            direction = diff_in_means(p_stack, n_stack, normalize=False)
            out = new_direction_path(ds, position)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(direction, out)
            (out.parent / "meta.json").write_text(json.dumps({
                "method": "diff_in_means", "model": model_name, "dataset": ds, "layer": layer,
                "seed": seed, "procedure": PROCEDURE, "position": position,
                "n_pos": int(p_stack.shape[0]), "n_neg": int(n_stack.shape[0]),
                "d_model": int(direction.shape[0]),
                "norm_pre_normalize": float((p_stack.mean(0) - n_stack.mean(0)).norm().item()),
                "attention_mask_fix": True,
            }, indent=2))
            logger.info(f"Saved {ds}/{position} -> {out}  (||v||={direction.norm():.4f})")

    del model
    torch.cuda.empty_cache()


def compare_directions(model_name: str, layer: int, seed: int) -> Dict:
    """cos(old, new) and both norms, per (direction dataset, position)."""
    out: Dict[str, Dict] = {}
    for ds in DIRECTION_DATASETS:
        for position in POSITIONS:
            new_p, old_p = new_direction_path(ds, position), old_direction_path(model_name, ds, seed, layer, position)
            if not new_p.exists():
                continue
            new_v = torch.load(new_p, map_location="cpu").float()
            entry = {"norm_new": float(new_v.norm())}
            if old_p.exists():
                old_v = torch.load(old_p, map_location="cpu").float()
                entry["norm_old"] = float(old_v.norm())
                entry["cosine_old_new"] = float(
                    torch.nn.functional.cosine_similarity(old_v, new_v, dim=0))
            out[f"{ds}/{position}"] = entry
            logger.info(f"direction {ds}/{position}: {entry}")
    return out


# ---------------------------------------------------------------------------
# Stage 2 - reranking
# ---------------------------------------------------------------------------

def load_or_compute_tensors(model_name: str, eval_ds: str, direction_ds: str, seed: int,
                            layer: int, all_docs: List[str], sbert_enc: SentenceTransformer,
                            recompute: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """SBERT embeddings + LLM hidden states for the eval corpus.

    Reused from the existing retrieval_evaluation cache when it matches the corpus: the
    document side never depended on the direction, and compute_llm_hidden_states already
    sets padding_side="left", so those tensors are unaffected by the bug.
    """
    layer_dir = old_layer_dir(model_name, eval_ds, direction_ds, seed, layer)
    if not recompute and cache_matches_corpus(layer_dir, all_docs):
        logger.info(f"Reusing cached tensors from {layer_dir}")
        return (torch.load(layer_dir / "sbert_embeddings.pt", map_location="cpu"),
                torch.load(layer_dir / "llm_hidden_states.pt", map_location="cpu"))

    logger.info(f"Computing tensors for {eval_ds} ({len(all_docs)} docs) ...")
    emb = sbert_enc.encode(all_docs, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    sbert_emb = torch.tensor(emb, dtype=torch.float32)

    device = tl_utils.get_device()
    model = HookedTransformer.from_pretrained_no_processing(model_name, device=device, dtype="bfloat16")
    model.eval()
    hook_point = tl_utils.get_act_name("resid_post", layer)
    llm_hidden = compute_llm_hidden_states(model, all_docs, hook_point, RERANK_BATCH_SIZE)
    del model
    torch.cuda.empty_cache()

    cache_dir = OUT_ROOT / "_tensors" / eval_ds / f"layer_{layer}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(sbert_emb, cache_dir / "sbert_embeddings.pt")
    torch.save(llm_hidden, cache_dir / "llm_hidden_states.pt")
    return sbert_emb, llm_hidden


def rerank_all(model_name: str, layer: int, seed: int, recompute_tensors: bool) -> None:
    sbert_enc = SentenceTransformer(SBERT_MODEL)
    for eval_ds in EVAL_DATASETS:
        samples = load_normalized(eval_ds, seed)["test"]
        all_docs = corpus_of(samples)
        doc_idx = {d: i for i, d in enumerate(all_docs)}
        logger.info(f"=== eval {eval_ds}: {len(samples)} samples | {len(all_docs)} unique docs ===")

        # Tensors depend on (eval corpus, layer) only; the direction dataset in the cache
        # path is irrelevant to their content, so the first available cache is reused.
        sbert_emb, llm_hidden = None, None
        for direction_ds in DIRECTION_DATASETS:
            if sbert_emb is None:
                sbert_emb, llm_hidden = load_or_compute_tensors(
                    model_name, eval_ds, direction_ds, seed, layer, all_docs, sbert_enc,
                    recompute_tensors)

            for position in POSITIONS:
                dir_path = new_direction_path(direction_ds, position)
                if not dir_path.exists():
                    logger.warning(f"Skip {eval_ds}/{direction_ds}/{position}: no new direction.")
                    continue
                out_dir = new_results_path(eval_ds, direction_ds, position).parent
                out_dir.mkdir(parents=True, exist_ok=True)
                direction = torch.load(dir_path, map_location="cpu").float()
                logger.info(f"--- rerank {eval_ds} | dir={direction_ds} | {position} ---")
                compute_evaluation(llm_hidden, direction, sbert_emb, samples, all_docs, doc_idx,
                                   sbert_enc, ALPHAS, KS, out_dir)


# ---------------------------------------------------------------------------
# Stage 3 - metrics + old/new comparison
# ---------------------------------------------------------------------------

def derived_metrics(m: Dict, alpha: float, k: int) -> Dict[str, float]:
    """The quantities the pipeline's plots are built from, for one (alpha, k)."""
    gold_rate, nf_rate = m["gold_rate"], m["nf_rate"]
    out = {
        "gold_recall_at_k": gold_rate[(alpha, k)],
        "nonfactual_rate_at_k": nf_rate[(alpha, k)],
        "gold_lift": gold_rate[(alpha, k)] - gold_rate[(0.0, k)],
        "nf_drop": nf_rate[(0.0, k)] - nf_rate[(alpha, k)],
    }
    out["layer_score"] = out["gold_lift"] + out["nf_drop"]
    if m["has_rank"]:
        gr, nr = m["mean_gold_rank"], m["mean_nf_rank"]
        out["mean_gold_rank"] = gr[alpha]
        out["mean_nf_rank"] = nr[alpha]
        # Sign conventions follow plot_figure3.build_data.
        out["gold_rank_gain"] = gr[0.0] - gr[alpha]      # >0: gold climbed
        out["nf_rank_change"] = nr[0.0] - nr[alpha]      # <0: non-factual sank
        out["rank_separation_gain"] = out["gold_rank_gain"] - out["nf_rank_change"]
    return out


METRIC_NAMES = ["gold_recall_at_k", "nonfactual_rate_at_k", "gold_lift", "nf_drop",
                "layer_score", "mean_gold_rank", "mean_nf_rank", "gold_rank_gain",
                "nf_rank_change", "rank_separation_gain"]


def build_comparison(model_name: str, layer: int, seed: int) -> List[Dict]:
    rows: List[Dict] = []
    for eval_ds in EVAL_DATASETS:
        for direction_ds in DIRECTION_DATASETS:
            for position in POSITIONS:
                new_p = new_results_path(eval_ds, direction_ds, position)
                if not new_p.exists():
                    continue
                new_m = compute_seed_metrics(new_p)
                old_p = old_layer_dir(model_name, eval_ds, direction_ds, seed, layer) / position / "results.jsonl"
                old_m = compute_seed_metrics(old_p) if old_p.exists() else None
                if old_m is None:
                    logger.warning(f"No OLD baseline for {eval_ds}/{direction_ds}/{position} "
                                   f"(never evaluated); reporting NEW only.")
                for alpha in new_m["alphas"]:
                    for k in new_m["ks"]:
                        nd = derived_metrics(new_m, alpha, k)
                        od = derived_metrics(old_m, alpha, k) if old_m else {}
                        row = {"eval": eval_ds, "direction": direction_ds, "position": position,
                               "alpha": alpha, "k": k, "has_old": old_m is not None}
                        for name in METRIC_NAMES:
                            row[f"new_{name}"] = nd.get(name)
                            row[f"old_{name}"] = od.get(name)
                            row[f"delta_{name}"] = (
                                nd[name] - od[name] if name in nd and name in od else None)
                        rows.append(row)
    return rows


def write_outputs(rows: List[Dict], dir_cmp: Dict, model_name: str, layer: int, seed: int) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "direction_comparison.json").write_text(json.dumps(
        {"model": model_name, "layer": layer, "seed": seed, "directions": dir_cmp}, indent=2))

    csv_path = OUT_ROOT / "metrics.csv"
    fields = (["eval", "direction", "position", "alpha", "k", "has_old"]
              + [f"{p}_{n}" for n in METRIC_NAMES for p in ("old", "new", "delta")])
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    logger.info(f"Wrote {csv_path} ({len(rows)} rows)")

    # Compact glance table: one line per configuration at SUMMARY_ALPHA, largest change first.
    lines = [f"model={model_name}  layer={layer}  seed={seed}  alpha={SUMMARY_ALPHA}",
             "(rank metrics are k-independent; gold_recall/nf_rate shown at k=%d)" % KS[0], ""]
    lines.append("direction cosine old-vs-new:")
    for name, e in dir_cmp.items():
        c = e.get("cosine_old_new")
        lines.append(f"   {name:<24} cos={'n/a' if c is None else f'{c:+.4f}'}"
                     f"  ||v||old={e.get('norm_old', float('nan')):.4f}  ||v||new={e['norm_new']:.4f}")
    lines += ["", f"{'eval':<11}{'direction':<11}{'position':<11}"
                  f"{'sep_gain old':>13}{'new':>10}{'delta':>10}"
                  f"{'  |  score old':>15}{'new':>9}{'delta':>9}"]

    sel = [r for r in rows if r["alpha"] == SUMMARY_ALPHA and r["k"] == KS[0]]
    sel.sort(key=lambda r: -abs(r.get("delta_rank_separation_gain") or 0.0))
    for r in sel:
        fmt = lambda v, w=10, p=3: (" " * (w - 3) + "n/a") if v is None else f"{v:>{w}.{p}f}"
        lines.append(
            f"{r['eval']:<11}{r['direction']:<11}{r['position']:<11}"
            f"{fmt(r['old_rank_separation_gain'], 13)}{fmt(r['new_rank_separation_gain'])}"
            f"{fmt(r['delta_rank_separation_gain'])}"
            f"{fmt(r['old_layer_score'], 15)}{fmt(r['new_layer_score'], 9)}"
            f"{fmt(r['delta_layer_score'], 9)}")

    text = "\n".join(lines)
    (OUT_ROOT / "summary.txt").write_text(text + "\n")
    print("\n" + text + "\n")
    logger.info(f"Wrote {OUT_ROOT / 'summary.txt'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-cell regression harness for the attention-mask fix.")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--batch-size", type=int, default=IDENT_BATCH_SIZE)
    ap.add_argument("--skip-identify", action="store_true",
                    help="Reuse directions already under results/new_identification_reranking/directions/.")
    ap.add_argument("--skip-rerank", action="store_true", help="Only (re)compute the comparison tables.")
    ap.add_argument("--recompute-tensors", action="store_true",
                    help="Recompute document hidden states instead of reusing the retrieval_evaluation cache.")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    setup_logging("new_identification_reranking", OUT_ROOT)
    logger.info(f"model={args.model} | layer={args.layer} | seed={args.seed} | "
                f"directions={DIRECTION_DATASETS} | evals={EVAL_DATASETS} | positions={POSITIONS}")

    if not args.skip_identify:
        identify_directions(args.model, args.layer, args.seed, args.batch_size)
    else:
        logger.info("Skipping identification (--skip-identify).")

    dir_cmp = compare_directions(args.model, args.layer, args.seed)

    if not args.skip_rerank:
        rerank_all(args.model, args.layer, args.seed, args.recompute_tensors)
    else:
        logger.info("Skipping reranking (--skip-rerank).")

    rows = build_comparison(args.model, args.layer, args.seed)
    write_outputs(rows, dir_cmp, args.model, args.layer, args.seed)
    logger.info("Done.")


if __name__ == "__main__":
    main()
