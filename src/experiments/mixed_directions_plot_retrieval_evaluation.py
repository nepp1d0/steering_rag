"""
Plot the mixed-direction retrieval evaluation and select the best layer per group.

Clone of plot_retrieval_evaluation.py pointed at the mixed-direction results. A "group"
is (model, eval_dataset, combo, normalize, procedure); the direction dataset of the
original is a dataset *combo* here.

Mixed directions exist for seed 42 only, so there is a single seed per group: the
per-seed aggregation plots of the original are dropped (nothing to average), and
top_layers_<procedure>.json is scored from that one seed.

Outputs:
    results/mixed_directions_retrieval_evaluation/**/retrieval_plot.png
    results/mixed_directions_retrieval_evaluation/<model>/<eval>/<combo>/<norm>/top_layers_<proc>.json
    results/mixed_directions_top_retrieval_evaluation/   (best layer per group, copied)

Usage:
    python -m src.experiments.mixed_directions_plot_retrieval_evaluation
"""

from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
EVAL_DIR = RESULTS_DIR / "mixed_directions_retrieval_evaluation"
TOP_DIR = RESULTS_DIR / "mixed_directions_top_retrieval_evaluation"

SEED_RE = re.compile(r"seed_(\d+)")
LAYER_RE = re.compile(r"layer_(\d+)")


def plot_results(results_path: Path, out_path: Path) -> None:
    # {(alpha, k): [gold_hits, nonfactual_hits, total]}
    buckets: dict[tuple, list] = defaultdict(lambda: [0, 0, 0])
    rank_records: list[dict] = []
    with results_path.open() as f:
        for line in f:
            r = json.loads(line)
            key = (r["alpha"], r["k"])
            buckets[key][0] += int(r["gold_in_topk"])
            buckets[key][1] += int(r["nonfactual_in_topk"])
            buckets[key][2] += 1
            if "gold_rank" in r:
                rank_records.append(r)

    alphas = sorted({k[0] for k in buckets})
    ks = sorted({k[1] for k in buckets})

    has_rank = bool(rank_records)
    fig, axes = plt.subplots(1, 3 if has_rank else 2, figsize=(15 if has_rank else 10, 4), sharey=False)
    ax1, ax2 = axes[0], axes[1]

    for alpha in alphas:
        gold_rates = [buckets[(alpha, k)][0] / buckets[(alpha, k)][2] for k in ks]
        nf_rates = [buckets[(alpha, k)][1] / buckets[(alpha, k)][2] for k in ks]
        label = f"α={alpha:.1f}" + (" (baseline)" if alpha == 0.0 else "")
        ls = "--" if alpha == 0.0 else "-"
        ax1.plot(ks, gold_rates, marker="o", linestyle=ls, label=label)
        ax2.plot(ks, nf_rates, marker="o", linestyle=ls, label=label)

    title = "/".join(results_path.relative_to(EVAL_DIR).parts[:-1])
    for ax, ylabel in [(ax1, "gold recall@k"), (ax2, "non-factual rate@k")]:
        ax.set_xlabel("k")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ks)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.set_title(title, fontsize=7)

    if has_rank:
        ax3 = axes[2]
        first_k = ks[0]
        mean_gold, mean_nf = [], []
        for alpha in alphas:
            rows = [r for r in rank_records if r["alpha"] == alpha and r["k"] == first_k]
            mean_gold.append(sum(r["gold_rank"] for r in rows) / len(rows))
            mean_nf.append(sum(r["nonfactual_rank"] for r in rows) / len(rows))
        ax3.plot(alphas, mean_gold, marker="o", label="gold (lower=better)")
        ax3.plot(alphas, mean_nf, marker="o", label="non-factual (higher=better)")
        ax3.set_yscale("log")
        ax3.set_xlabel("α")
        ax3.set_ylabel("mean rank (log scale)")
        ax3.legend(fontsize=8)
        ax3.set_title(title, fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def compute_seed_metrics(results_path: Path) -> dict:
    """Per-seed metrics: recall/rate per (alpha, k) and mean ranks per alpha."""
    buckets: dict[tuple, list] = defaultdict(lambda: [0, 0, 0])  # (alpha,k)->[gold,nf,total]
    rank_rows: list[dict] = []
    with results_path.open() as f:
        for line in f:
            r = json.loads(line)
            key = (r["alpha"], r["k"])
            buckets[key][0] += int(r["gold_in_topk"])
            buckets[key][1] += int(r["nonfactual_in_topk"])
            buckets[key][2] += 1
            if "gold_rank" in r:
                rank_rows.append(r)

    alphas = sorted({k[0] for k in buckets})
    ks = sorted({k[1] for k in buckets})
    gold_rate = {key: b[0] / b[2] for key, b in buckets.items()}
    nf_rate = {key: b[1] / b[2] for key, b in buckets.items()}

    mean_gold_rank, mean_nf_rank = {}, {}
    if rank_rows:
        first_k = ks[0]  # rank is independent of k; pick one to avoid double-counting
        for alpha in alphas:
            rows = [r for r in rank_rows if r["alpha"] == alpha and r["k"] == first_k]
            mean_gold_rank[alpha] = sum(r["gold_rank"] for r in rows) / len(rows)
            mean_nf_rank[alpha] = sum(r["nonfactual_rank"] for r in rows) / len(rows)

    return {"alphas": alphas, "ks": ks, "gold_rate": gold_rate, "nf_rate": nf_rate,
            "mean_gold_rank": mean_gold_rank, "mean_nf_rank": mean_nf_rank,
            "has_rank": bool(rank_rows)}


def score_layer(metrics: list[dict]) -> tuple[float, float | None]:
    """Best steering configuration for a layer: for each alpha>0 score it against the
    alpha=0 baseline (meaned across seeds and k), then keep the best alpha.

    Returns (score, best_alpha). The score sums two top-k recall terms, each in [-1, 1]:
      gold_lift  = gold_rate(a) - gold_rate(0)   (panel 1, higher better)
      nf_drop    = nf_rate(0)   - nf_rate(a)     (panel 2, higher better)
    A global rank-separation term was deliberately dropped: it is query-independent at
    alpha=1 (pure projection -> same ranking for every question) and rewarded that
    degenerate config despite near-zero top-k recall, which tanks end-to-end accuracy.
    """
    alphas = metrics[0]["alphas"]
    ks = metrics[0]["ks"]
    steer = [a for a in alphas if a > 0]

    best_score, best_alpha = float("-inf"), None
    for a in steer:
        gold_lift = np.mean([m["gold_rate"][(a, k)] - m["gold_rate"][(0.0, k)]
                             for m in metrics for k in ks])
        nf_drop = np.mean([m["nf_rate"][(0.0, k)] - m["nf_rate"][(a, k)]
                           for m in metrics for k in ks])
        s = float(gold_lift + nf_drop)
        if s > best_score:
            best_score, best_alpha = s, a
    return best_score, best_alpha


def layer_group_key(results_path: Path) -> tuple:
    """(model, eval, combo, normalize, procedure) — identity ignoring both seed and layer."""
    parts = results_path.relative_to(EVAL_DIR).parts
    seed_idx = next(i for i, p in enumerate(parts) if SEED_RE.fullmatch(p))
    return parts[:seed_idx] + parts[seed_idx + 1:-2]  # drop seed, and trailing layer + filename


def write_top_layers(results_files: list[Path]) -> None:
    """Score every layer (best alpha>0 vs baseline) and dump a top_layers_<procedure>.json
    next to the seed_* folders for each group."""
    groups: dict[tuple, dict[int, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for r in results_files:
        layer = int(LAYER_RE.fullmatch(r.parts[-2]).group(1))
        groups[layer_group_key(r)][layer].append(r)

    for gkey, layers in sorted(groups.items()):
        scored = []
        for layer, paths in layers.items():
            metrics = [compute_seed_metrics(p) for p in paths]
            score, best_alpha = score_layer(metrics)
            scored.append({"layer": layer, "best_alpha": best_alpha,
                           "score": round(score, 4), "n_seeds": len(paths)})
        scored.sort(key=lambda d: -d["score"])
        *prefix, procedure = gkey
        out_path = EVAL_DIR / Path(*prefix) / f"top_layers_{procedure}.json"
        out_path.write_text(json.dumps(
            {"group": "/".join(gkey), "top5": scored[:5], "ranking": scored}, indent=2))
        print(f"Wrote {out_path}  (best: layer {scored[0]['layer']} "
              f"@ alpha={scored[0]['best_alpha']}, score={scored[0]['score']})")


def copy_top_results() -> None:
    """Copy the best-layer seed dirs for each group into mixed_directions_top_retrieval_evaluation/.

    Reads the top_layers_*.json written by write_top_layers, so call it afterwards.
    """
    json_files = sorted(EVAL_DIR.rglob("top_layers_*.json"))
    print(f"Found {len(json_files)} top_layers JSON files for copying.")

    for json_path in json_files:
        data = json.loads(json_path.read_text())
        best = data["ranking"][0]
        if best["best_alpha"] is None:
            print(f"Skip (no valid alpha): {json_path}")
            continue
        best_layer = best["layer"]
        procedure = json_path.stem.replace("top_layers_", "")

        normalize_dir = json_path.parent  # .../model/eval/combo/normalize/
        for seed_dir in sorted(normalize_dir.glob("seed_*")):
            src = seed_dir / procedure / f"layer_{best_layer}"
            if not src.exists():
                print(f"Warning: dir not found: {src}")
                continue
            dst = TOP_DIR / src.relative_to(EVAL_DIR)
            if dst.exists():
                print(f"Skip (exists): {dst}")
                continue
            shutil.copytree(src, dst)
            print(f"Copied -> {dst}")

        dst_json = TOP_DIR / json_path.relative_to(EVAL_DIR)
        if not dst_json.exists():
            dst_json.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(json_path, dst_json)
            print(f"Copied json -> {dst_json}")


def main() -> None:
    # Only results under a seed_*/ path; the cache/ dirs hold no results.jsonl.
    results_files = sorted(p for p in EVAL_DIR.rglob("results.jsonl")
                           if any(SEED_RE.fullmatch(part) for part in p.parts))
    print(f"Found {len(results_files)} results.jsonl files.")

    for r in results_files:
        out = r.parent / "retrieval_plot.png"
        plot_results(r, out)
        print(f"Wrote {out}")

    # Per-group layer ranking (best alpha per layer) -> top_layers_*.json.
    write_top_layers(results_files)

    # Copy best-layer dirs to mixed_directions_top_retrieval_evaluation/ for version control.
    copy_top_results()


if __name__ == "__main__":
    main()
