"""
ClashEval pool-ranking: re-rank each question's own pool of perturbed variants.

Paired AUROC (clasheval_pipeline.py) is a separation diagnostic. The method as actually
deployed is re-ranking a retrieval pool. ClashEval already gives us that pool for free:
each question has ONE context_original and ~10 context_mod perturbed variants (drugs+news:
479 questions, 9.9 variants/question on average, verified below).

Per question, pool = {context_original} u {context_mod_i for every perturbed row of that
question}. Score every pool member as the paper does:

    score(d, q) = (1 - alpha) * z_pool(cos_SBERT(q, d)) + alpha * z_pool(h(d) . v)

z_pool = z-scored WITHIN the question's own pool (not the global corpus - the pool has ~11
members, so this is a local re-ranking task, matching how the method is actually deployed).
h(d).v is computed per layer from the ALREADY-CACHED headline_doc_repr.pt (no new GPU pass);
because A3 forbids layer selection on ClashEval, h(d).v is not one layer's projection but the
all-layer mean of the per-layer z-scored projection (z-score each layer within the pool, then
average the 32 z-scores) - the natural pool-ranking analogue of the primary AUROC metric.
Metrics (precision@1 of the uncorrupted original, MRR) are computed per seed then averaged
over the 5 identification seeds, then aggregated over the 479 questions (the natural cluster/
independent unit for this task) with a percentile bootstrap CI (resample questions).

Reuses src/experiments/clasheval_pipeline.py's data-prep (guarantees identical pair ordering
to the cached H_doc) and direction-loading helpers. No new LLM forward pass.

Usage:
    python src/experiments/clasheval_pool_ranking.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clasheval_pipeline import (       # noqa: E402
    DIRECTION_SOURCES, HEADLINE_DOMAINS, MODEL, OUT_ROOT, SEEDS, load_control_direction,
    load_eval_pairs, load_real_direction, safe_model_id, select_headline,
)
from utils import RESULTS_DIR          # noqa: E402

OUT_DIR = RESULTS_DIR / "clasheval_pool_ranking"
ALPHAS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
ALL_SOURCES = DIRECTION_SOURCES + ["random_unit", "shuffled_label"]
N_BOOT = 1000
BOOT_SEED = 0


def build_pools(headline_df) -> list[dict]:
    """One entry per question: pair-axis indices into H_doc, aligned as
    [orig_row_idx] + [row_idx for each perturbed variant], and the raw texts for SBERT."""
    pools = []
    for q, g in headline_df.groupby("question", sort=False):
        rows = g.index.tolist()                      # positional indices into headline_df / H_doc pair axis
        orig_text = g["context_original"].iloc[0]
        assert (g["context_original"] == orig_text).all(), "context_original differs within a question"
        pools.append({
            "question": q,
            "domain": g["dataset"].iloc[0],
            "orig_row": rows[0],                      # any row's side=0 is the same doc; use the first
            "mod_rows": rows,                          # each row's side=1 is one distinct variant
            "orig_text": orig_text,
            "mod_texts": g["context_mod"].tolist(),
        })
    return pools


def zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    if std < 1e-12:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ms = safe_model_id(MODEL)
    hidden_dir = OUT_ROOT / ms

    full = load_eval_pairs()
    headline = select_headline(full, HEADLINE_DOMAINS)
    pools = build_pools(headline)
    n_q = len(pools)
    print(f"[pool] {n_q} question pools, mean size = {np.mean([1 + len(p['mod_rows']) for p in pools]):.2f}")
    by_domain = {}
    for p in pools:
        by_domain[p["domain"]] = by_domain.get(p["domain"], 0) + 1
    print(f"[pool] per domain: {by_domain}")

    H_doc = torch.load(hidden_dir / "headline_doc_repr.pt", map_location="cpu").float()   # [n_layers, n_pairs, 2, d_model]
    n_layers = H_doc.shape[0]

    # ---- SBERT: encode every distinct document once (orig texts are shared across a question's
    # rows, mod texts are one per row) plus every question text.
    from sentence_transformers import SentenceTransformer
    print("[sbert] loading all-MiniLM-L6-v2 ...")
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    q_texts = [p["question"] for p in pools]
    q_emb = sbert.encode(q_texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=True)

    # Pool doc embeddings, computed per pool directly (small: ~11 docs x 479 pools ~ 5.2k texts total).
    pool_doc_texts, pool_doc_owner = [], []      # owner: (pool_idx, is_orig, mod_position_or_-1)
    for pi, p in enumerate(pools):
        pool_doc_texts.append(p["orig_text"])
        pool_doc_owner.append((pi, True, -1))
        for j, t in enumerate(p["mod_texts"]):
            pool_doc_texts.append(t)
            pool_doc_owner.append((pi, False, j))
    print(f"[sbert] encoding {len(pool_doc_texts)} pool documents ...")
    doc_emb = sbert.encode(pool_doc_texts, batch_size=64, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=True)
    del sbert

    # cos_sbert per pool member, grouped back by pool.
    cos_by_pool: list[np.ndarray] = [np.empty(1 + len(p["mod_rows"])) for p in pools]
    for gi, (pi, is_orig, j) in enumerate(pool_doc_owner):
        idx = 0 if is_orig else 1 + j
        cos_by_pool[pi][idx] = float(doc_emb[gi] @ q_emb[pi])

    # Diagnostic: how often is the SBERT term fully tied within a pool (std < 1e-12, i.e. the
    # 256-token truncation made every pool member's embedded text byte-identical)?
    n_degenerate_sbert = sum(1 for c in cos_by_pool if c.std() < 1e-12)
    print(f"[sbert] {n_degenerate_sbert}/{n_q} pools ({n_degenerate_sbert / n_q:.1%}) have a fully "
          f"tied SBERT cosine within the pool (256-token truncation cuts off before the edit).")

    # h(d).v per layer per pool member, from the cached doc-level activations directly (no SBERT
    # needed here). h_layer[pool][member] built once per direction/seed below.

    def align_scores_for_seed(source: str, seed: int) -> list[np.ndarray]:
        """Per pool, all-layer-mean of the per-layer z-scored projection h(d).v, for `source`/`seed`."""
        out = []
        # v: [n_layers, d_model]
        if source in ("nq_swap", "conflictqa"):
            v = torch.stack([load_real_direction(MODEL, source, seed, L) for L in range(n_layers)])
        else:
            v = torch.stack([load_control_direction(hidden_dir, source, seed, L) for L in range(n_layers)])
        for p in pools:
            rows = torch.tensor([p["orig_row"]] + p["mod_rows"], dtype=torch.long)
            sides = torch.tensor([0] + [1] * len(p["mod_rows"]), dtype=torch.long)
            Hm = H_doc[:, rows, sides, :]               # paired advanced index -> [n_layers, n_members, d_model]
            proj = torch.einsum("lmd,ld->lm", Hm, v).numpy()   # [n_layers, n_members]
            z = np.stack([zscore(proj[L]) for L in range(n_layers)])   # [n_layers, n_members]
            out.append(z.mean(axis=0))                 # [n_members], all-layer mean
        return out

    # SBERT's max_seq_length is 256 tokens; ClashEval documents average ~900 words (>1000
    # tokens) and the edit sits at a median offset well beyond that, so many pools
    # have BYTE-IDENTICAL truncated SBERT input for the original and every modified variant
    # -> exact-tied cosines. np.argsort is stable and the original is always built at pool
    # position 0, so naive argsort silently resolves every tie in the original's favor,
    # spuriously inflating P@1. Break ties with a random tiebreak sub-machine-precision jitter
    # (a fixed seed per pool so the run is reproducible) instead of array position.
    tie_rng = np.random.default_rng(12345)
    jitter_by_pool = [tie_rng.uniform(-1e-9, 1e-9, size=1 + len(p["mod_rows"])) for p in pools]

    results = {}
    for source in ALL_SOURCES:
        print(f"[score] {source} ...")
        seed_metrics = {alpha: {"p1": [], "mrr": []} for alpha in ALPHAS}   # per seed, per-question arrays
        for seed in SEEDS:
            align = align_scores_for_seed(source, seed)
            for alpha in ALPHAS:
                p1_arr = np.empty(n_q)
                mrr_arr = np.empty(n_q)
                for pi in range(n_q):
                    zc = zscore(cos_by_pool[pi])
                    zv = align[pi]
                    score = (1 - alpha) * zc + alpha * zv + jitter_by_pool[pi]
                    order = np.argsort(-score)          # descending; index 0 is the original
                    rank = int(np.where(order == 0)[0][0]) + 1
                    p1_arr[pi] = 1.0 if rank == 1 else 0.0
                    mrr_arr[pi] = 1.0 / rank
                seed_metrics[alpha]["p1"].append(p1_arr)
                seed_metrics[alpha]["mrr"].append(mrr_arr)

        source_out = {}
        rng = np.random.default_rng(BOOT_SEED)
        for alpha in ALPHAS:
            p1_seedmean = np.mean(seed_metrics[alpha]["p1"], axis=0)      # [n_q], averaged over 5 seeds
            mrr_seedmean = np.mean(seed_metrics[alpha]["mrr"], axis=0)
            p1_point, mrr_point = float(p1_seedmean.mean()), float(mrr_seedmean.mean())
            p1_draws = np.empty(N_BOOT)
            mrr_draws = np.empty(N_BOOT)
            for b in range(N_BOOT):
                idx = rng.integers(0, n_q, n_q)
                p1_draws[b] = p1_seedmean[idx].mean()
                mrr_draws[b] = mrr_seedmean[idx].mean()
            source_out[str(alpha)] = {
                "precision_at_1": p1_point,
                "p1_ci_lo": float(np.percentile(p1_draws, 2.5)), "p1_ci_hi": float(np.percentile(p1_draws, 97.5)),
                "mrr": mrr_point,
                "mrr_ci_lo": float(np.percentile(mrr_draws, 2.5)), "mrr_ci_hi": float(np.percentile(mrr_draws, 97.5)),
            }
            print(f"  alpha={alpha:.1f} | P@1={p1_point:.4f} [{source_out[str(alpha)]['p1_ci_lo']:.4f},"
                  f"{source_out[str(alpha)]['p1_ci_hi']:.4f}] | MRR={mrr_point:.4f} "
                  f"[{source_out[str(alpha)]['mrr_ci_lo']:.4f},{source_out[str(alpha)]['mrr_ci_hi']:.4f}]")
        results[source] = source_out

    out = {
        "model": MODEL, "domains": HEADLINE_DOMAINS, "n_questions": n_q,
        "n_questions_by_domain": by_domain, "mean_pool_size": float(np.mean([1 + len(p["mod_rows"]) for p in pools])),
        "n_pools_sbert_degenerate": n_degenerate_sbert,
        "frac_pools_sbert_degenerate": n_degenerate_sbert / n_q,
        "alphas": ALPHAS, "n_boot": N_BOOT, "results": results,
    }
    out_path = OUT_DIR / "pool_ranking_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
