"""
RAGuard step C - can the factuality direction separate true from false claims?

For every (model, direction dataset, seed, layer) we project the RAGuard claim activations
onto the existing diff-in-means direction and measure how well the projection ranks
verdict=True above verdict=False:

    Hc = H - mean(H)          # centering is mandatory: the residual stream has a large
    s  = Hc @ v               # common component (||mu|| ~ 15x ||v||) that buries the signal
                              # (same decision as direction_analysis_heatmap.py)

Metrics per configuration:
    auroc                 ranking AUROC of s vs verdict (below 0.5 = direction points the
                          other way; kept signed so the orientation is visible)
    auroc_abs             max(auroc, 1 - auroc)
    balanced_acc_best     best balanced accuracy over all thresholds AND both orientations
    balanced_acc_zero     balanced accuracy of the sign of s (threshold 0, as-is orientation)
    spearman_truthiness   rank correlation of s with the 6-level truthiness (pants-on-fire=0
                          .. true=5). A genuine truthfulness direction should be monotone
                          across the six levels, not merely binary-separating.

Three baselines make those numbers interpretable:
    probe ceiling   per (model, layer): out-of-fold AUROC of a 5-fold stratified-CV logistic
                    regression fit on the same centered activations. Upper bound on what is
                    linearly decodable - separates "the model does not represent claim truth"
                    from "it does, but our direction misses it".
    random null     per (model, layer): AUROC of N_RANDOM random unit vectors -> 2.5/97.5
                    percentile band. At n=2648 this should sit at 0.5 +- ~0.02.
    lexical floor   model-independent: TF-IDF + logistic regression on the claim text alone
                    (plus a claim-length-only control). Any AUROC below this floor proves
                    nothing about the representation.

`--direction-source` selects which directions are scored:
    single  results/direction_identification/<model>/<dataset>/...   (5 seeds)
    mixed   results/mixed_directions/<model>/<combo>/...             (7 combos, seed 42 only)
The two write to separate result files and share the same baselines, since the probe
ceiling and the random null depend only on (model, layer).

Reads:  data/raguard/claims.jsonl, results/raguard/<model>/hidden_states*/layer_<L>.pt,
        results/<source>/<model>/<name>/seed_<S>/context_only/layer_<L>/last_pos/direction.pt
Writes: results/raguard/<model>/claim_separation[_mixed].jsonl  (one row per name/seed/layer)
        results/raguard/<model>/baselines.jsonl                 (one row per layer)
        results/raguard/lexical_baseline.json

CPU only - everything reads precomputed tensors.

Usage:
    python src/exploratory/raguard_direction_evaluation.py
    python src/exploratory/raguard_direction_evaluation.py --direction-source mixed
    python src/exploratory/raguard_direction_evaluation.py --models meta-llama__Llama-3.2-1B-Instruct
    python src/exploratory/raguard_direction_evaluation.py --skip-probe      # fast pass
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
from utils import REPO_ROOT, RESULTS_DIR, logger, setup_logging, write_jsonl

MODELS = ["meta-llama__Llama-3.1-8B-Instruct", "meta-llama__Llama-3.2-1B-Instruct",
          "google__gemma-3-4b-it", "Qwen__Qwen2-7B-Instruct"]
DIRECTION_DATASETS = ["nq_swap", "conflictqa", "longfact"]
# Mixture combos from mixed_direction_identification.py. Note its singles are NOT the
# same vectors as the DIRECTION_DATASETS ones: every dataset is subsampled to an equal
# number of texts per side, so singles and mixtures stay comparable to each other.
COMBOS = ["conflictqa", "nq_swap", "longfact",
          "conflictqa+nq_swap", "conflictqa+longfact", "nq_swap+longfact",
          "conflictqa+nq_swap+longfact"]
# source -> (results subdirectory, direction names, output filename suffix)
DIRECTION_SOURCES = {
    "single": ("direction_identification", DIRECTION_DATASETS, ""),
    "mixed": ("mixed_directions", COMBOS, "_mixed"),
}
PROCEDURE = "context_only"
POSITION = "last_pos"          # RAGuard claims have no annotated entity span
CLAIMS_PATH = REPO_ROOT / "data" / "raguard" / "claims.jsonl"
OUT_ROOT = RESULTS_DIR / "raguard"

N_RANDOM = 200                 # random unit vectors for the null band
RANDOM_SEED = 0
PROBE_C = 1.0                  # matches probe_direction_identification.py
PROBE_MAX_ITER = 2000
N_FOLDS = 5


def auroc(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Ranking AUROC via the Mann-Whitney statistic. `scores` is [n] or [n, k].

    Equivalent to sklearn's roc_auc_score (ties handled by average ranks) but vectorized
    over the columns, which matters for the N_RANDOM null band.
    """
    s = scores[:, None] if scores.ndim == 1 else scores
    ranks = rankdata(s, axis=0)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    pos_rank_sum = ranks[labels].sum(axis=0)
    out = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return out[0] if scores.ndim == 1 else out


def balanced_acc_curve(scores: np.ndarray, labels: np.ndarray) -> float:
    """Best balanced accuracy over every threshold and both orientations."""
    order = np.argsort(-scores)
    y = labels[order]
    n_pos, n_neg = int(labels.sum()), len(labels) - int(labels.sum())
    tp = np.concatenate([[0], np.cumsum(y)])          # predict-positive on the top-i scores
    fp = np.concatenate([[0], np.cumsum(~y)])
    ba = 0.5 * (tp / n_pos + (n_neg - fp) / n_neg)
    return float(max(ba.max(), (1.0 - ba).max()))


def balanced_acc_at_zero(scores: np.ndarray, labels: np.ndarray) -> float:
    pred = scores > 0
    tpr = float((pred & labels).sum() / labels.sum())
    tnr = float((~pred & ~labels).sum() / (~labels).sum())
    return 0.5 * (tpr + tnr)


def load_claims() -> List[Dict]:
    if not CLAIMS_PATH.exists():
        raise FileNotFoundError(f"{CLAIMS_PATH} not found. Run raguard_normalization.py first.")
    with CLAIMS_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def hidden_dir(model: str, variant: str) -> Path:
    suffix = "" if variant == "raw" else f"_{variant}"
    return OUT_ROOT / model / f"hidden_states{suffix}"


def direction_path(model: str, dd: str, seed: int, layer: int,
                   subdir: str = "direction_identification") -> Path:
    return (RESULTS_DIR / subdir / model / dd / f"seed_{seed}"
            / PROCEDURE / f"layer_{layer}" / POSITION / "direction.pt")


def discover_direction_seeds(subdir: str, model: str, dd: str) -> List[int]:
    """Seeds available for one (source, model, direction name); mixtures only have seed 42."""
    base = RESULTS_DIR / subdir / model / dd
    return sorted(int(d.name.split("_")[1]) for d in base.glob("seed_*") if d.is_dir())


def lexical_baseline(claims: List[Dict]) -> Dict:
    """TF-IDF and length-only floors on the claim text (model-independent)."""
    X = [c["claim"] for c in claims]
    y = np.array([c["verdict"] for c in claims])
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    tfidf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2),
                          LogisticRegression(max_iter=PROBE_MAX_ITER, class_weight="balanced"))
    tfidf_scores = cross_val_predict(tfidf, X, y, cv=cv, method="decision_function")

    lengths = np.array([[len(x)] for x in X], dtype=float)
    len_scores = cross_val_predict(LogisticRegression(max_iter=PROBE_MAX_ITER, class_weight="balanced"),
                                   lengths, y, cv=cv, method="decision_function")

    out = {"tfidf_auroc": float(auroc(tfidf_scores, y)),
           "length_auroc": float(auroc(len_scores, y)),
           "n_claims": len(claims),
           "n_true": int(y.sum())}
    logger.info(f"Lexical floor: TF-IDF AUROC {out['tfidf_auroc']:.3f} | "
                f"length-only AUROC {out['length_auroc']:.3f}")
    return out


def probe_auroc(Hc: np.ndarray, y: np.ndarray) -> float:
    """Out-of-fold AUROC of a CV logistic probe on the centered activations."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    clf = LogisticRegression(C=PROBE_C, max_iter=PROBE_MAX_ITER, class_weight="balanced")
    scores = cross_val_predict(clf, Hc, y, cv=cv, method="decision_function")
    return float(auroc(scores, y))


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGuard claim separation by the factuality direction.")
    parser.add_argument("--models", nargs="+", default=MODELS, help="Filesystem-safe model ids.")
    parser.add_argument("--variant", default="raw", choices=["raw", "statement"],
                        help="Which activation variant to score (must exist from step B).")
    parser.add_argument("--direction-source", default="single", choices=list(DIRECTION_SOURCES),
                        help="'single' = direction_identification, 'mixed' = mixed_direction_identification combos.")
    parser.add_argument("--skip-probe", action="store_true",
                        help="Skip the supervised probe ceiling (the slow part).")
    parser.add_argument("--recompute-baselines", action="store_true",
                        help="Recompute probe/random baselines instead of reusing baselines.jsonl.")
    args = parser.parse_args()

    subdir, direction_names, suffix = DIRECTION_SOURCES[args.direction_source]

    setup_logging("raguard_direction_evaluation", OUT_ROOT)
    claims = load_claims()
    y = np.array([c["verdict"] for c in claims])
    truthiness = np.array([c["truthiness"] for c in claims])
    logger.info(f"{len(claims)} claims | {int(y.sum())} True / {int((~y).sum())} False")

    lex_path = OUT_ROOT / "lexical_baseline.json"
    if not lex_path.exists():
        lex_path.parent.mkdir(parents=True, exist_ok=True)
        lex_path.write_text(json.dumps(lexical_baseline(claims), indent=2))
    else:
        logger.info(f"Lexical floor cached: {json.loads(lex_path.read_text())}")

    rng = np.random.default_rng(RANDOM_SEED)

    for model in args.models:
        hdir = hidden_dir(model, args.variant)
        meta_path = hdir / "meta.json"
        if not meta_path.exists():
            logger.warning(f"No activations for {model} at {hdir}. Run raguard_claim_activations.py first.")
            continue
        meta = json.loads(meta_path.read_text())
        assert meta["claim_ids"] == [c["claim_id"] for c in claims], \
            f"Activation row order for {model} does not match claims.jsonl"

        # Probe / random baselines depend only on (model, layer), not on the direction
        # source, so a second source reuses the first run's baselines.jsonl.
        base_path = OUT_ROOT / model / "baselines.jsonl"
        cached_base = {}
        if base_path.exists() and not args.recompute_baselines:
            with base_path.open() as f:
                cached_base = {b["layer"]: b for b in (json.loads(l) for l in f if l.strip())}
            logger.info(f"Reusing cached baselines for {model} ({len(cached_base)} layers).")

        seeds_by_name = {dd: discover_direction_seeds(subdir, model, dd) for dd in direction_names}
        logger.info(f"{model} | source={args.direction_source} | "
                    + " ".join(f"{dd}:seeds{seeds_by_name[dd]}" for dd in direction_names))

        rows, base_rows = [], []
        for layer in range(meta["n_layers"]):
            H = torch.load(hdir / f"layer_{layer}.pt", map_location="cpu").numpy()
            Hc = H - H.mean(axis=0, keepdims=True)

            # --- our directions ---
            for dd in direction_names:
                for seed in seeds_by_name[dd]:
                    dpath = direction_path(model, dd, seed, layer, subdir)
                    if not dpath.exists():
                        continue
                    v = torch.load(dpath, map_location="cpu").float().numpy()
                    s = Hc @ v
                    a = float(auroc(s, y))
                    rows.append({
                        "model": model, "direction_dataset": dd, "seed": seed, "layer": layer,
                        "auroc": a, "auroc_abs": max(a, 1.0 - a),
                        "balanced_acc_best": balanced_acc_curve(s, y),
                        "balanced_acc_zero": balanced_acc_at_zero(s, y),
                        "spearman_truthiness": float(spearmanr(s, truthiness).statistic),
                    })

            # --- random-direction null band + probe ceiling ---
            if layer in cached_base:
                base_rows.append(cached_base[layer])
            else:
                R = rng.standard_normal((H.shape[1], N_RANDOM))
                R /= np.linalg.norm(R, axis=0, keepdims=True)
                rand_aurocs = auroc(Hc @ R, y)
                base = {
                    "model": model, "layer": layer,
                    "random_p2.5": float(np.percentile(rand_aurocs, 2.5)),
                    "random_p50": float(np.percentile(rand_aurocs, 50)),
                    "random_p97.5": float(np.percentile(rand_aurocs, 97.5)),
                    "random_abs_p97.5": float(np.percentile(np.maximum(rand_aurocs, 1 - rand_aurocs), 97.5)),
                }
                base["probe_auroc"] = None if args.skip_probe else probe_auroc(Hc, y)
                base_rows.append(base)
                logger.info(f"{model} L{layer}: probe={base['probe_auroc']} "
                            f"null=[{base['random_p2.5']:.3f}, {base['random_p97.5']:.3f}]")

        write_jsonl(OUT_ROOT / model / f"claim_separation{suffix}.jsonl", rows)
        if not cached_base:
            write_jsonl(base_path, base_rows)

        # Compact summary: best layer per direction name, averaged over seeds.
        for dd in direction_names:
            by_layer: Dict[int, List[float]] = {}
            for r in rows:
                if r["direction_dataset"] == dd:
                    by_layer.setdefault(r["layer"], []).append(r["auroc_abs"])
            if not by_layer:
                continue
            best = max(by_layer, key=lambda L: float(np.mean(by_layer[L])))
            logger.info(f"{model} | {dd}: best layer {best} | "
                        f"AUROC_abs {np.mean(by_layer[best]):.3f} +/- {np.std(by_layer[best]):.3f}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
