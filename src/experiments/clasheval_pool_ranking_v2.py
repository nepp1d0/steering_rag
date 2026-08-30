"""
ClashEval stage 2: re-rank a frozen 12-document pool per question.

Pool composition (same domain only):
    1 target                  = the uncorrupted document for this question
    3 corrupted variants of the target                    (on-topic, corrupted)
    4 uncorrupted distractors = other questions' targets  (off-topic, uncorrupted)
    4 corrupted distractors   = other questions' variants (off-topic, corrupted)

Analytical reference lines that follow from the composition alone: random = 1/12 = 0.083,
relevance-only ceiling = 1/4 = 0.25, factuality-only ceiling = 1/5 = 0.20.

Pools are drawn ONCE and reused verbatim for every model (construction is text-only, so it is
model-independent). frozen_pools.json is checked against a recorded sha256 before use; a
mismatch stops the run.

Scoring, per pool: (1 - alpha) * z_pool(cos_SBERT(q, d)) + alpha * z_pool(h(d).v), with the
projection z-scored per layer and then averaged over all layers (no layer selection). Direction
sources are nq_swap, conflictqa and longfact, plus two nulls (shuffled_label, random_unit) at
100 independent draws each -- enough draws for the 2.5/97.5 percentiles to be real order
statistics.

Metrics per source and alpha: P@1 and MRR of the target, aggregated over questions with a
question-clustered bootstrap (N_BOOT = 2000). The resample matrix is shared across every
source, alpha and model, so all comparisons are paired; the same shared indices give the
paired delta P@1(alpha=0.5) - P@1(alpha=0) and its Wilcoxon signed-rank p-value.

Build check: alpha=0 carries no model or direction term (pure SBERT), so P@1(alpha=0) must be
identical for every source and model; a discrepancy means the pools were not shared correctly.

Also records, for nq_swap only, the same metric computed from each single (seed, layer) cell
instead of the all-layer mean: the 10th/90th percentile across that grid is an across-layer
spread, not sampling error. DEFF compares the naive pool-instance bootstrap variance against
the question-clustered one.

Results go to a model-scoped pool_ranking_v2_results__<safe_model_id>.json; frozen_pools.json
stays shared. The `records` domain is dropped (191 questions over 9 documents make the pools
ill-posed); years/names/locations have no cached activations and are not run.

Usage:
    python src/experiments/clasheval_pool_ranking_v2.py \
        --model meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clasheval_pipeline import (       # noqa: E402
    HEADLINE_DOMAINS, MODEL, OUT_ROOT, SEEDS, load_control_direction,
    load_eval_pairs, load_real_direction, safe_model_id, select_headline,
)
from clasheval_pool_ranking import zscore   # noqa: E402
from utils import RESULTS_DIR               # noqa: E402

OUT_DIR = RESULTS_DIR / "clasheval_pool_ranking_v2"
ALPHAS = [round(0.1 * i, 1) for i in range(11)]
PRIMARY_ALPHA = 0.5
DIRECTION_SOURCES = ["nq_swap", "conflictqa", "longfact"]
NULL_SOURCES = ["shuffled_label", "random_unit"]
ALL_SOURCES = DIRECTION_SOURCES + NULL_SOURCES
DRAW_SEEDS = [101, 102, 103, 104, 105]           # pool-instance draws (text-only, shared across models)
POOL_SIZE = 12
N_BOOT = 2000
BOOT_SEED = 0
REF_LINES = {"random": 1 / 12, "relevance_only_ceiling": 1 / 4, "factuality_only_ceiling": 1 / 5, "perfect": 1.0}
ALPHA0_P1_REFERENCE = 0.213836      # build check: alpha=0 is pure SBERT, so it cannot vary
ALPHA0_MRR_REFERENCE = 0.494223

# Both nulls use 100 draws so the 2.5/97.5 percentiles of the band are genuine order statistics:
# the min of B draws estimates the 1/(B+1) quantile, so 20 draws would put the nominal 2.5th
# percentile at the 4.8th. random_unit vectors are generated on the fly (independent Gaussian
# unit vectors per layer, seeded by draw_id) rather than read from the 5-seed disk cache.
SHUFFLED_LABEL_N_DRAWS = 100
SHUFFLED_LABEL_DRAW_IDS = list(range(1, SHUFFLED_LABEL_N_DRAWS + 1))
RANDOM_UNIT_N_DRAWS = 100
RANDOM_UNIT_DRAW_IDS = list(range(1, RANDOM_UNIT_N_DRAWS + 1))


def build_eligible_questions(headline_df) -> dict[str, list[dict]]:
    by_domain: dict[str, list[dict]] = {d: [] for d in HEADLINE_DOMAINS}
    for (dom, q), g in headline_df.groupby(["dataset", "question"], sort=False):
        rows = g.index.tolist()
        if len(rows) < 3:
            continue
        by_domain[dom].append({"question": q, "target_row": rows[0], "variant_rows": rows})
    return by_domain


def freeze_pools(by_domain: dict[str, list[dict]], draw_seeds: list[int]) -> dict:
    frozen: dict[int, dict[str, list[dict]]] = {}
    domain_offset = {d: i for i, d in enumerate(sorted(by_domain))}
    for ds in draw_seeds:
        frozen[ds] = {}
        for dom, qs in by_domain.items():
            rng = np.random.default_rng(ds * 31 + domain_offset[dom])
            n = len(qs)
            pools = []
            for qi, q in enumerate(qs):
                variant_rows = q["variant_rows"]
                corrupted_variants = rng.choice(variant_rows, size=3, replace=False).tolist()
                other_idx = [j for j in range(n) if j != qi]
                distractor_q_idx = rng.choice(other_idx, size=4, replace=False)
                uncorrupted_distractors = [qs[j]["target_row"] for j in distractor_q_idx]
                candidate_rows = [r for j in other_idx for r in qs[j]["variant_rows"]]
                corrupted_distractors = rng.choice(candidate_rows, size=4, replace=False).tolist()
                pool_rows = [q["target_row"]] + corrupted_variants + uncorrupted_distractors + corrupted_distractors
                pool_sides = [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
                assert len(pool_rows) == POOL_SIZE
                pools.append({"question": q["question"], "rows": pool_rows, "sides": pool_sides})
            frozen[ds][dom] = pools
    return frozen


def shuffled_label_direction_stack(H_pos: torch.Tensor, H_neg: torch.Tensor, draw_id: int, n_layers: int) -> torch.Tensor:
    rng = np.random.default_rng(draw_id + 5_000_000)
    n_pos, n_neg = H_pos.shape[1], H_neg.shape[1]
    n_total = n_pos + n_neg
    pool = torch.cat([H_pos, H_neg], dim=1)
    perm_pos_idx = rng.choice(n_total, size=n_pos, replace=False)
    mask = np.zeros(n_total, dtype=bool)
    mask[perm_pos_idx] = True
    vs = []
    for L in range(n_layers):
        v = pool[L, mask].mean(dim=0) - pool[L, ~mask].mean(dim=0)
        vs.append(v)
    return torch.stack(vs)


def random_unit_direction_stack(draw_id: int, d_model: int, n_layers: int) -> torch.Tensor:
    """20 independent draws of an isotropic random unit direction, one per layer. Distinct rng
    namespace from both the 5-seed disk-cached controls (seed*100_000+L) and the shuffled_label
    20-draw generator (draw_id+5_000_000), so none of the three collide."""
    vs = []
    for L in range(n_layers):
        rng = np.random.default_rng(draw_id * 1_000_003 + L + 8_000_000)
        v = rng.standard_normal(d_model).astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-8)
        vs.append(torch.from_numpy(v))
    return torch.stack(vs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    model = args.model

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ms = safe_model_id(model)
    hidden_dir = OUT_ROOT / ms

    full = load_eval_pairs()
    headline = select_headline(full, HEADLINE_DOMAINS)
    by_domain = build_eligible_questions(headline)
    n_excluded = {d: headline[headline["dataset"] == d]["question"].nunique() - len(by_domain[d])
                 for d in HEADLINE_DOMAINS}
    print(f"[eligibility] {[(d, len(by_domain[d])) for d in HEADLINE_DOMAINS]} eligible questions "
          f"(excluded for <3 own variants: {n_excluded})")

    # ---- frozen pools: shared across all models, reused verbatim, hash-checked
    # requirement that a redraw would silently invalidate every cross-model comparison).
    frozen_path = OUT_DIR / "frozen_pools.json"
    frozen_hash_path = OUT_DIR / "frozen_pools.sha256"
    if frozen_path.exists():
        raw_bytes = frozen_path.read_bytes()
        print(f"[pools] loaded frozen pools from {frozen_path} (shared across all models)")
    else:
        frozen = freeze_pools(by_domain, DRAW_SEEDS)
        raw_bytes = json.dumps(frozen).encode()
        frozen_path.write_bytes(raw_bytes)
        print(f"[pools] froze {sum(len(v) for dv in frozen.values() for v in dv.values())} "
              f"pool-instances -> {frozen_path}")

    file_hash = hashlib.sha256(raw_bytes).hexdigest()
    if frozen_hash_path.exists():
        expected_hash = frozen_hash_path.read_text().strip()
        if file_hash != expected_hash:
            raise RuntimeError(
                f"[BUILD CHECK FAIL] frozen_pools.json hash mismatch: expected {expected_hash} "
                f"(reference), got {file_hash}. Pools were NOT reused verbatim for "
                f"model={model} - STOPPING.")
        print(f"[BUILD CHECK PASS] frozen_pools.json sha256 matches the reference "
              f"({file_hash}) - pools are identical across models.")
    else:
        frozen_hash_path.write_text(file_hash)
        print(f"[pools] recorded reference hash {file_hash} (first run - reference for every "
              f"subsequent model).")

    frozen_raw = json.loads(raw_bytes)
    frozen = {int(ds): v for ds, v in frozen_raw.items()}

    H_doc = torch.load(hidden_dir / "headline_doc_repr.pt", map_location="cpu").float()
    n_layers = H_doc.shape[0]

    acts = torch.load(hidden_dir / "nq_swap_train_acts.pt", map_location="cpu")
    H_pos, H_neg = acts["pos"].float(), acts["neg"].float()

    all_pools = []
    for ds in DRAW_SEEDS:
        for dom in HEADLINE_DOMAINS:
            for p in frozen[ds][dom]:
                all_pools.append({"question": p["question"], "domain": dom, "draw_seed": ds,
                                  "rows": p["rows"], "sides": p["sides"]})
    n_pools = len(all_pools)
    print(f"[pools] {n_pools} total pool-instances "
          f"({len(DRAW_SEEDS)} draws x {sum(len(v) for v in by_domain.values())} questions)")

    from sentence_transformers import SentenceTransformer
    print("[sbert] loading all-MiniLM-L6-v2 ...")
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    unique_questions = sorted({p["question"] for p in all_pools})
    q_emb_map = dict(zip(unique_questions, sbert.encode(
        unique_questions, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=True)))

    orig_text = headline["context_original"].tolist()
    mod_text = headline["context_mod"].tolist()
    doc_key_set = sorted({(r, s) for p in all_pools for r, s in zip(p["rows"], p["sides"])})
    doc_texts = [orig_text[r] if s == 0 else mod_text[r] for r, s in doc_key_set]
    print(f"[sbert] encoding {len(doc_texts)} unique pool documents ...")
    doc_embs = sbert.encode(doc_texts, batch_size=64, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=True)
    doc_emb_map = {k: e for k, e in zip(doc_key_set, doc_embs)}
    del sbert

    n_degenerate = 0
    cos_by_pool = []
    for p in all_pools:
        d = np.stack([doc_emb_map[(r, s)] for r, s in zip(p["rows"], p["sides"])])
        c = d @ q_emb_map[p["question"]]
        cos_by_pool.append(c)
        if c.std() < 1e-12:
            n_degenerate += 1
    print(f"[sbert] {n_degenerate}/{n_pools} pool-instances ({n_degenerate / n_pools:.1%}) "
          f"have a fully tied SBERT cosine (256-token truncation).")

    tie_rng = np.random.default_rng(54321)
    jitter_by_pool = [tie_rng.uniform(-1e-9, 1e-9, size=POOL_SIZE) for _ in all_pools]

    # vectorized matrices, independent of source/alpha
    zc_matrix = np.stack([zscore(c) for c in cos_by_pool])              # [n_pools, 12]
    jitter_matrix = np.stack(jitter_by_pool)                            # [n_pools, 12]

    def align_scores_for_draw(source: str, draw_id: int, keep_full: bool = False):
        """Returns list of [12] all-layer-mean z per pool; if keep_full, also returns the
        [n_pools, n_layers, 12] full (non-collapsed) per-layer z matrix."""
        if source in ("nq_swap", "conflictqa", "longfact"):
            v = torch.stack([load_real_direction(model, source, draw_id, L) for L in range(n_layers)])
        elif source == "random_unit":
            v = random_unit_direction_stack(draw_id, H_doc.shape[-1], n_layers)
        elif source == "shuffled_label":
            v = shuffled_label_direction_stack(H_pos, H_neg, draw_id, n_layers)
        else:
            raise ValueError(source)
        out_mean = []
        out_full = np.empty((n_pools, n_layers, POOL_SIZE)) if keep_full else None
        for pi, p in enumerate(all_pools):
            rows = torch.tensor(p["rows"], dtype=torch.long)
            sides = torch.tensor(p["sides"], dtype=torch.long)
            Hm = H_doc[:, rows, sides, :]                       # [n_layers, 12, d_model]
            proj = torch.einsum("lmd,ld->lm", Hm, v).numpy()    # [n_layers, 12]
            z = np.stack([zscore(proj[L]) for L in range(n_layers)])   # [n_layers, 12]
            out_mean.append(z.mean(axis=0))
            if keep_full:
                out_full[pi] = z
        return out_mean, out_full

    SOURCE_DRAWS = {"nq_swap": SEEDS, "conflictqa": SEEDS, "longfact": SEEDS,
                    "random_unit": RANDOM_UNIT_DRAW_IDS, "shuffled_label": SHUFFLED_LABEL_DRAW_IDS}

    question_to_pool_idx: dict[str, list[int]] = {}
    for i, p in enumerate(all_pools):
        question_to_pool_idx.setdefault(p["question"], []).append(i)
    questions = list(question_to_pool_idx.keys())
    n_clusters = len(questions)
    q_idx_arr = [np.array(question_to_pool_idx[q]) for q in questions]

    # ---- ONE shared set of cluster-resample indices AND one naive (unclustered) resample-index
    # set, reused for EVERY source and EVERY alpha, so every comparison is paired.
    rng = np.random.default_rng(BOOT_SEED)
    boot_idx = rng.integers(0, n_clusters, size=(N_BOOT, n_clusters))       # [N_BOOT, n_clusters]
    naive_boot_idx = rng.integers(0, n_pools, size=(N_BOOT, n_pools))       # naive DEFF denominator

    results = {}
    all_p1_by_q: dict[str, dict[float, np.ndarray]] = {s: {} for s in ALL_SOURCES}
    all_mrr_by_q: dict[str, dict[float, np.ndarray]] = {s: {} for s in ALL_SOURCES}
    nq_swap_layer_z = []       # list over SEEDS of [n_pools, n_layers, 12] (kept only for nq_swap)

    # Draw-level P@1-by-question arrays, kept for the null sources ONLY at the primary alpha, so
    # the draw-sampling-vs-question-sampling variance decomposition can
    # be computed without re-running anything: one array per draw, not yet averaged over draws.
    null_per_draw_p1_by_q: dict[str, list[np.ndarray]] = {}

    for source in ALL_SOURCES:
        draws = SOURCE_DRAWS[source]
        keep_full = source == "nq_swap"
        print(f"[score] {source} ({len(draws)} draws) ...")
        per_alpha_p1 = {a: [] for a in ALPHAS}
        per_alpha_mrr = {a: [] for a in ALPHAS}
        per_pool_p1_flat = {a: None for a in ALPHAS}      # for naive/DEFF bootstrap (flat over pool instances)
        is_null = source in NULL_SOURCES
        if is_null:
            null_per_draw_p1_by_q[source] = []
        for draw_id in draws:
            align_mean, align_full = align_scores_for_draw(source, draw_id, keep_full=keep_full)
            if keep_full:
                nq_swap_layer_z.append(align_full)
            za_matrix = np.stack(align_mean)     # [n_pools, 12]
            for a in ALPHAS:
                score = (1 - a) * zc_matrix + a * za_matrix + jitter_matrix    # [n_pools, 12]
                argmax = score.argmax(axis=1)
                p1_arr = (argmax == 0).astype(float)
                rank = 1 + (score > score[:, 0:1]).sum(axis=1)
                mrr_arr = 1.0 / rank
                per_alpha_p1[a].append(p1_arr)
                per_alpha_mrr[a].append(mrr_arr)
                if is_null and a == PRIMARY_ALPHA:
                    p1_by_q_this_draw = np.array([p1_arr[question_to_pool_idx[q]].mean() for q in questions])
                    null_per_draw_p1_by_q[source].append(p1_by_q_this_draw)

        source_out = {}
        for a in ALPHAS:
            p1_drawmean = np.mean(per_alpha_p1[a], axis=0)      # [n_pools], mean over draws
            mrr_drawmean = np.mean(per_alpha_mrr[a], axis=0)
            p1_by_q = np.array([p1_drawmean[question_to_pool_idx[q]].mean() for q in questions])
            mrr_by_q = np.array([mrr_drawmean[question_to_pool_idx[q]].mean() for q in questions])
            all_p1_by_q[source][a] = p1_by_q
            all_mrr_by_q[source][a] = mrr_by_q
            p1_point, mrr_point = float(p1_by_q.mean()), float(mrr_by_q.mean())

            p1_boot = p1_by_q[boot_idx].mean(axis=1)     # [N_BOOT], shared resample indices
            mrr_boot = mrr_by_q[boot_idx].mean(axis=1)
            var_cluster = float(p1_boot.var(ddof=1))

            p1_naive_boot = p1_drawmean[naive_boot_idx].mean(axis=1)   # naive (ignores clustering)
            var_naive = float(p1_naive_boot.var(ddof=1))
            deff = var_cluster / var_naive if var_naive > 0 else float("nan")

            source_out[str(a)] = {
                "precision_at_1": p1_point,
                "p1_ci_lo": float(np.percentile(p1_boot, 2.5)), "p1_ci_hi": float(np.percentile(p1_boot, 97.5)),
                "p1_halfwidth": float((np.percentile(p1_boot, 97.5) - np.percentile(p1_boot, 2.5)) / 2),
                "mrr": mrr_point,
                "mrr_ci_lo": float(np.percentile(mrr_boot, 2.5)), "mrr_ci_hi": float(np.percentile(mrr_boot, 97.5)),
                "deff": deff, "var_cluster": var_cluster, "var_naive": var_naive,
            }
        results[source] = source_out
        p1_curve = " ".join(f"{a:.1f}:{source_out[str(a)]['precision_at_1']:.3f}" for a in ALPHAS)
        print(f"  P@1 by alpha: {p1_curve} | DEFF@0.5={source_out[str(PRIMARY_ALPHA)]['deff']:.2f}")

    # ---- build check: alpha=0 has no model/direction term -- must match the analytic reference
    # exactly.
    alpha0_checks = {}
    for source in ALL_SOURCES:
        p1_0 = results[source]["0.0"]["precision_at_1"]
        mrr_0 = results[source]["0.0"]["mrr"]
        ok_p1 = abs(p1_0 - ALPHA0_P1_REFERENCE) < 1e-6
        ok_mrr = abs(mrr_0 - ALPHA0_MRR_REFERENCE) < 1e-6
        alpha0_checks[source] = {"p1": p1_0, "mrr": mrr_0, "p1_matches_reference": ok_p1, "mrr_matches_reference": ok_mrr}
        gate = "PASS" if (ok_p1 and ok_mrr) else "FAIL"
        print(f"[BUILD CHECK {gate}] alpha=0 {source}: P@1={p1_0:.6f} (ref {ALPHA0_P1_REFERENCE}), "
              f"MRR={mrr_0:.6f} (ref {ALPHA0_MRR_REFERENCE})")
        if not (ok_p1 and ok_mrr):
            print(f"[BUILD CHECK FAIL] {source}: alpha=0 does not match the model-independent "
                  f"reference -- pools were not shared correctly for model={model}. STOPPING.")
            raise RuntimeError(f"alpha=0 build check failed for source={source}, model={model}")

    # ---- variance decomposition: for each null source, what fraction
    # of the TOTAL variance of its alpha=0.5 P@1 point estimate comes from draw sampling (which
    # B draws we happened to use) versus question sampling (the cluster bootstrap, already
    # computed above as var_cluster, holding the observed draws fixed)? Draw-sampling variance is
    # estimated by bootstrap-resampling the B observed draws themselves (2000 replicates),
    # holding the 477 questions fixed, and measuring how much the question-averaged point
    # estimate moves. PhD's pre-committed thresholds: <10% -> the 5-draw run is retro-validated,
    # nothing else needs re-examining; >30% -> every null-referenced statement needs revisiting.
    variance_decomposition = {}
    DRAW_BOOT = 2000
    drng = np.random.default_rng(BOOT_SEED + 999)
    for source in NULL_SOURCES:
        D = np.stack(null_per_draw_p1_by_q[source])          # [B, n_clusters]
        B = D.shape[0]
        draw_boot_idx = drng.integers(0, B, size=(DRAW_BOOT, B))       # resample DRAWS, not questions
        # for each draw-resample: average the resampled draws -> [n_clusters] -> mean over clusters
        draw_boot_point = D[draw_boot_idx].mean(axis=1).mean(axis=1)   # [DRAW_BOOT]
        var_draw = float(draw_boot_point.var(ddof=1))
        var_question = results[source][str(PRIMARY_ALPHA)]["var_cluster"]
        total = var_draw + var_question
        frac_draw = var_draw / total if total > 0 else float("nan")
        variance_decomposition[source] = {
            "n_draws": int(B), "var_draw": var_draw, "var_question": var_question,
            "frac_variance_from_draws": frac_draw,
            "verdict": ("retro-validated (<10%)" if frac_draw < 0.10 else
                       "re-examine all null-referenced statements (>30%)" if frac_draw > 0.30 else
                       "intermediate (10-30%), reported as such"),
        }
        print(f"[variance decomp] {source}: var_draw={var_draw:.3e} var_question={var_question:.3e} "
              f"frac_from_draws={frac_draw:.3f} ({variance_decomposition[source]['verdict']})")

    # ---- column-5 content: paired delta P@1(alpha=0.5) - P@1(alpha=0), per source, with a
    # paired cluster-bootstrap CI (shared boot_idx) and a Wilcoxon signed-rank p over questions.
    delta_vs_alpha0 = {}
    for source in ALL_SOURCES:
        for metric, table in (("p1", all_p1_by_q), ("mrr", all_mrr_by_q)):
            p_hi = table[source][PRIMARY_ALPHA]
            p_lo = table[source][0.0]
            point_delta = float(p_hi.mean() - p_lo.mean())
            d_hi = p_hi[boot_idx].mean(axis=1)
            d_lo = p_lo[boot_idx].mean(axis=1)
            delta_boot = d_hi - d_lo
            ci = (float(np.percentile(delta_boot, 2.5)), float(np.percentile(delta_boot, 97.5)))
            per_q_delta = p_hi - p_lo
            try:
                if np.allclose(per_q_delta, 0):
                    wp = 1.0
                else:
                    wp = float(wilcoxon(per_q_delta, zero_method="zsplit").pvalue)
            except Exception as e:
                wp = float("nan")
            delta_vs_alpha0[f"{source}__{metric}"] = {
                "point_delta": point_delta, "ci_lo": ci[0], "ci_hi": ci[1],
                "wilcoxon_p": wp, "excludes_0": not (ci[0] <= 0 <= ci[1]),
            }
    print(f"[column5] delta P@1(alpha=0.5)-P@1(alpha=0) per source: " +
          ", ".join(f"{s}={delta_vs_alpha0[f'{s}__p1']['point_delta']:+.4f}"
                    f"[{delta_vs_alpha0[f'{s}__p1']['ci_lo']:+.4f},{delta_vs_alpha0[f'{s}__p1']['ci_hi']:+.4f}]"
                    f"(p={delta_vs_alpha0[f'{s}__p1']['wilcoxon_p']:.4f})" for s in ALL_SOURCES))

    # ---- supplementary cross-source paired deltas at the primary alpha (kept from the prior
    # revision; useful in addition to the note-05-mandated column-5 content above).
    # Every direction/null source's paired delta against random_unit is reported (see the
    # instruction): no interpretive claim about any series is licensed without this comparison,
    # since rank alone ("longfact is second-best") is uninterpretable when the isotropic null
    # itself is not at the bottom of the ranking.
    pair_list = [("nq_swap", "conflictqa"), ("nq_swap", "random_unit"),
                 ("nq_swap", "shuffled_label"), ("conflictqa", "random_unit"),
                 ("nq_swap", "longfact"), ("longfact", "random_unit"),
                 ("shuffled_label", "random_unit"), ("conflictqa", "longfact"),
                 ("conflictqa", "shuffled_label"), ("longfact", "shuffled_label")]
    paired_cross_source = {}
    for a_name, b_name in pair_list:
        for metric, table in (("p1", all_p1_by_q), ("mrr", all_mrr_by_q)):
            pA, pB = table[a_name][PRIMARY_ALPHA], table[b_name][PRIMARY_ALPHA]
            point_delta = float(pA.mean() - pB.mean())
            dA, dB = pA[boot_idx].mean(axis=1), pB[boot_idx].mean(axis=1)
            delta_boot = dA - dB
            ci = (float(np.percentile(delta_boot, 2.5)), float(np.percentile(delta_boot, 97.5)))
            frac_le0, frac_ge0 = float((delta_boot <= 0).mean()), float((delta_boot >= 0).mean())
            p_value = min(1.0, 2 * min(frac_le0, frac_ge0))
            key = f"{a_name}_minus_{b_name}__{metric}"
            paired_cross_source[key] = {"point_delta": point_delta, "ci_lo": ci[0], "ci_hi": ci[1],
                                        "p_value_two_sided_bootstrap": p_value, "excludes_0": not (ci[0] <= 0 <= ci[1])}
            if metric == "p1":
                print(f"[paired delta, P@1, alpha={PRIMARY_ALPHA}] {a_name} - {b_name} = "
                      f"{point_delta:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] p={p_value:.4f} "
                      f"excl_0={paired_cross_source[key]['excludes_0']}")

    # ---- light band: nq_swap only, per-(seed,layer) cell metric (single-layer projection, not
    # the all-layer mean), 10th/90th percentile across the (seed x layer) grid, per alpha.
    layer_band = {}
    if nq_swap_layer_z:
        Z = np.stack(nq_swap_layer_z)     # [n_seeds, n_pools, n_layers, 12]
        n_seeds_nq = Z.shape[0]
        for a in ALPHAS:
            cells = np.empty((n_seeds_nq, n_layers))
            for si in range(n_seeds_nq):
                for L in range(n_layers):
                    za = Z[si, :, L, :]
                    score = (1 - a) * zc_matrix + a * za + jitter_matrix
                    argmax = score.argmax(axis=1)
                    p1_flat = (argmax == 0).astype(float)
                    p1_by_q_cell = np.array([p1_flat[question_to_pool_idx[q]].mean() for q in questions])
                    cells[si, L] = p1_by_q_cell.mean()
            layer_band[str(a)] = {
                "p10": float(np.percentile(cells, 10)), "p90": float(np.percentile(cells, 90)),
                "n_cells": int(cells.size),
            }
        print(f"[layer-band] nq_swap across-layer/seed spread (n_cells={n_seeds_nq * n_layers}): "
              f"alpha=0.5 -> [{layer_band['0.5']['p10']:.4f}, {layer_band['0.5']['p90']:.4f}]")

    headline_check = {
        s: {"p1_at_0.5": results[s][str(PRIMARY_ALPHA)]["precision_at_1"],
            "p1_ci_lo_at_0.5": results[s][str(PRIMARY_ALPHA)]["p1_ci_lo"],
            "p1_ci_hi_at_0.5": results[s][str(PRIMARY_ALPHA)]["p1_ci_hi"],
            "exceeds_relevance_ceiling_0.25": results[s][str(PRIMARY_ALPHA)]["precision_at_1"] > REF_LINES["relevance_only_ceiling"],
            "ci_excludes_0.25": results[s][str(PRIMARY_ALPHA)]["p1_ci_lo"] > REF_LINES["relevance_only_ceiling"]}
        for s in ALL_SOURCES
    }
    print(f"[headline] P@1 at alpha={PRIMARY_ALPHA} vs relevance-only ceiling (0.25): {headline_check}")

    out = {
        "model": model, "domains": HEADLINE_DOMAINS, "pool_size": POOL_SIZE, "n_layers": n_layers,
        "n_eligible_questions_by_domain": {d: len(by_domain[d]) for d in HEADLINE_DOMAINS},
        "n_excluded_lt3_variants": n_excluded, "draw_seeds": DRAW_SEEDS,
        "n_pool_instances": n_pools, "n_sbert_degenerate_pools": n_degenerate,
        "frac_sbert_degenerate": n_degenerate / n_pools,
        "frozen_pools_sha256": file_hash,
        "reference_lines": REF_LINES, "primary_alpha": PRIMARY_ALPHA, "headline_check": headline_check,
        "alpha0_build_check": alpha0_checks,
        "alphas": ALPHAS, "n_boot": N_BOOT, "results": results,
        "delta_vs_alpha0_primary": delta_vs_alpha0,
        "variance_decomposition": variance_decomposition,
        "paired_cross_source_at_primary_alpha": paired_cross_source,
        "nq_swap_layer_seed_band": layer_band,
        "shuffled_label_n_draws": SHUFFLED_LABEL_N_DRAWS,
        "random_unit_n_draws": RANDOM_UNIT_N_DRAWS,
        "direction_sources": DIRECTION_SOURCES, "null_sources": NULL_SOURCES,
        "note_years_names_locations": "NOT RUN - no cached H_doc for those domains; would require "
                                      "new GPU extraction.",
        "note_arm_b": "NOT RUN - no direction.pt for this configuration exists on disk.",
    }
    out_path = OUT_DIR / f"pool_ranking_v2_results__{ms}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
