"""
Plot retrieval evaluation results.

Auto-discovers all results.jsonl under results/retrieval_evaluation/ and saves
a two-panel line plot (gold recall@k and non-factual rate@k vs k, one line per α)
next to each results file as retrieval_plot.png.

Expected layout (position-aware, see migrate_results_to_position_layout.py):
    .../<normalize>/seed_<S>/<procedure>/layer_<L>/<position>/results.jsonl

Usage:
    python src/experiments/plot_retrieval_evaluation.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from utils import RESULTS_DIR


def plot_results(results_path: Path, out_path: Path) -> None:
    # {(alpha, k): [gold_hits, nonfactual_hits, total]}
    buckets: dict[tuple, list] = defaultdict(lambda: [0, 0, 0])
    # {(alpha, sample_idx): (gold_rank, nf_rank)} — populated from smallest k to deduplicate
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
        nf_rates   = [buckets[(alpha, k)][1] / buckets[(alpha, k)][2] for k in ks]
        label = f"α={alpha:.1f}" + (" (baseline)" if alpha == 0.0 else "")
        ls = "--" if alpha == 0.0 else "-"
        ax1.plot(ks, gold_rates, marker="o", linestyle=ls, label=label)
        ax2.plot(ks, nf_rates,   marker="o", linestyle=ls, label=label)

    title = "/".join(results_path.relative_to(RESULTS_DIR / "retrieval_evaluation").parts[:-1])
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
        ax3.plot(alphas, mean_nf,   marker="o", label="non-factual (higher=better)")
        ax3.set_yscale("log")
        ax3.set_xlabel("α")
        ax3.set_ylabel("mean rank (log scale)")
        ax3.legend(fontsize=8)
        ax3.set_title(title, fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


SEED_RE = re.compile(r"seed_(\d+)")
LAYER_RE = re.compile(r"layer_(\d+)")
POSITIONS = ("last_pos", "entity_pos")


def find_seed(results_path: Path) -> int | None:
    """Return the seed number if a `seed_<N>` component is in the path, else None."""
    for part in results_path.parts:
        m = SEED_RE.fullmatch(part)
        if m:
            return int(m.group(1))
    return None


def group_key_of(results_path: Path) -> tuple:
    """Path identity ignoring the seed: (model, eval, dir, normalize, procedure, layer)."""
    parts = results_path.relative_to(RESULTS_DIR / "retrieval_evaluation").parts
    seed_idx = next(i for i, p in enumerate(parts) if SEED_RE.fullmatch(p))
    return parts[:seed_idx] + parts[seed_idx + 1:-1]


def aggregated_dir_of(results_path: Path) -> Path:
    """The `aggregated_plots` dir living next to the `seed_*` folders."""
    seed_dir = results_path
    while not SEED_RE.fullmatch(seed_dir.name):
        seed_dir = seed_dir.parent
    return seed_dir.parent / "aggregated_plots"


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
    """(model, eval, dir, normalize, procedure, position) — identity ignoring seed and layer."""
    parts = results_path.relative_to(RESULTS_DIR / "retrieval_evaluation").parts
    seed_idx = next(i for i, p in enumerate(parts) if SEED_RE.fullmatch(p))
    # after the seed: (procedure, layer_<L>, position, results.jsonl)
    return parts[:seed_idx] + parts[seed_idx + 1:-3] + (parts[-2],)


def write_top_layers(results_files: list[Path]) -> None:
    """Score every layer (best alpha>0 vs baseline, meaned across seeds) and dump a
    top_layers_<procedure>_<position>.json next to the seed_* folders for each group."""
    groups: dict[tuple, dict[int, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for r in results_files:
        if find_seed(r) is None or r.parts[-2] not in POSITIONS:
            continue
        layer = int(LAYER_RE.fullmatch(r.parts[-3]).group(1))
        groups[layer_group_key(r)][layer].append(r)

    for gkey, layers in sorted(groups.items()):
        scored = []
        for layer, paths in layers.items():
            metrics = [compute_seed_metrics(p) for p in paths]
            score, best_alpha = score_layer(metrics)
            scored.append({"layer": layer, "best_alpha": best_alpha,
                           "score": round(score, 4), "n_seeds": len(paths)})
        scored.sort(key=lambda d: -d["score"])
        *prefix, procedure, position = gkey
        out_path = RESULTS_DIR / "retrieval_evaluation" / Path(*prefix) / f"top_layers_{procedure}_{position}.json"
        out_path.write_text(json.dumps(
            {"group": "/".join(gkey), "top5": scored[:5], "ranking": scored}, indent=2))
        print(f"Wrote {out_path}  (best: layer {scored[0]['layer']} "
              f"@ alpha={scored[0]['best_alpha']}, score={scored[0]['score']})")


def copy_top_results() -> None:
    """Copy the best-layer seed dirs for each (group, position) into top_retrieval_evaluation/.

    Reads existing top_layers_*.json files (written by write_top_layers), so call
    this after write_top_layers has run. For each json only its own position subdir is
    copied (positions may pick different best layers), plus the layer-level shared
    files (tensors + docs.jsonl) that the heatmap / figure scripts read.
    """
    TOP_DIR = RESULTS_DIR / "top_retrieval_evaluation"
    json_files = sorted((RESULTS_DIR / "retrieval_evaluation").rglob("top_layers_*.json"))
    print(f"Found {len(json_files)} top_layers JSON files for copying.")

    for json_path in json_files:
        stem = json_path.stem.replace("top_layers_", "")
        position = next((p for p in POSITIONS if stem.endswith(p)), None)
        if position is None:
            print(f"Skip (no position in name): {json_path}")
            continue
        procedure = stem[: -len(position) - 1]

        data = json.loads(json_path.read_text())
        best = data["ranking"][0]
        if best["best_alpha"] is None:
            print(f"Skip (no valid alpha): {json_path}")
            continue
        best_layer = best["layer"]

        normalize_dir = json_path.parent  # .../model/eval/direction/normalize/
        for seed_dir in sorted(normalize_dir.glob("seed_*")):
            src = seed_dir / procedure / f"layer_{best_layer}"
            if not (src / position / "results.jsonl").exists():
                print(f"Warning: results not found: {src / position}")
                continue
            dst = TOP_DIR / src.relative_to(RESULTS_DIR / "retrieval_evaluation")
            if (dst / position / "results.jsonl").exists():
                print(f"Skip (exists): {dst / position}")
                continue
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():  # layer-level shared files (tensors, docs.jsonl)
                if f.is_file() and not (dst / f.name).exists():
                    shutil.copy2(f, dst / f.name)
            shutil.copytree(src / position, dst / position, dirs_exist_ok=True)
            print(f"Copied -> {dst / position}")

        dst_json = TOP_DIR / json_path.relative_to(RESULTS_DIR / "retrieval_evaluation")
        dst_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(json_path, dst_json)


def _agg(values: list[float]) -> tuple[float, float]:
    """Mean and sample std across seeds (std=0 when fewer than 2 seeds)."""
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def plot_aggregated(seed_paths: list[Path], seeds: list[int], out_path: Path, title: str) -> None:
    """Mean ± std across seeds, same panels as plot_results but with error bars."""
    metrics = [compute_seed_metrics(p) for p in seed_paths]
    alphas = metrics[0]["alphas"]
    ks = metrics[0]["ks"]
    has_rank = all(m["has_rank"] for m in metrics)

    n_panels = 3 if has_rank else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    ax1, ax2 = axes[0], axes[1]

    dodge = 0.15  # horizontal nudge per alpha so error-bar caps don't collide at the same k
    for j, alpha in enumerate(alphas):
        off = (j - (len(alphas) - 1) / 2) * dodge
        gold_means, gold_stds, nf_means, nf_stds = [], [], [], []
        for k in ks:
            gm, gs = _agg([m["gold_rate"][(alpha, k)] for m in metrics])
            nm, ns = _agg([m["nf_rate"][(alpha, k)] for m in metrics])
            gold_means.append(gm); gold_stds.append(gs)
            nf_means.append(nm); nf_stds.append(ns)
        label = f"α={alpha:.1f}" + (" (baseline)" if alpha == 0.0 else "")
        ls = "--" if alpha == 0.0 else "-"
        xs = [k + off for k in ks]
        ax1.errorbar(xs, gold_means, yerr=gold_stds, marker="o", markersize=3, linestyle=ls, capsize=3, label=label)
        ax2.errorbar(xs, nf_means, yerr=nf_stds, marker="o", markersize=3, linestyle=ls, capsize=3, label=label)

    for ax, ylabel in [(ax1, "gold recall@k"), (ax2, "non-factual rate@k")]:
        ax.set_xlabel("k"); ax.set_ylabel(ylabel)
        ax.set_xticks(ks); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8); ax.set_title(title, fontsize=7)

    if has_rank:
        ax3 = axes[2]
        for name, key in [("gold (lower=better)", "mean_gold_rank"),
                          ("non-factual (higher=better)", "mean_nf_rank")]:
            means, stds = [], []
            for alpha in alphas:
                m_, s_ = _agg([mm[key][alpha] for mm in metrics])
                means.append(m_); stds.append(s_)
            means, stds = np.array(means), np.array(stds)
            lower = np.minimum(stds, means * 0.999)  # keep lower whisker > 0 on log scale
            ax3.errorbar(alphas, means, yerr=[lower, stds], marker="o", markersize=3, capsize=3, label=name)
        ax3.set_yscale("log"); ax3.set_xlabel("α")
        ax3.set_ylabel("mean rank (log scale)")
        ax3.legend(fontsize=8); ax3.set_title(title, fontsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    results_files = sorted((RESULTS_DIR / "retrieval_evaluation").rglob("results.jsonl"))
    print(f"Found {len(results_files)} results.jsonl files.")

    # Per-file plots (unchanged) — includes legacy results without a seed in the path.
    for r in results_files:
        out = r.parent / "retrieval_plot.png"
        plot_results(r, out)
        print(f"Wrote {out}")

    # Aggregation over seeds — only results whose path contains a `seed_<N>` component.
    groups: dict[tuple, list] = defaultdict(list)
    for r in results_files:
        if find_seed(r) is not None:
            groups[group_key_of(r)].append(r)

    print(f"Aggregating {len(groups)} layer groups over seeds.")
    for key, paths in sorted(groups.items()):
        paths = sorted(paths)
        seeds = sorted(find_seed(p) for p in paths)
        procedure, layer, position = key[-3], key[-2], key[-1]
        out = aggregated_dir_of(paths[0]) / f"{procedure}_{layer}_{position}.png"
        title = "/".join(key) + f"  ({len(seeds)} seeds: {', '.join(map(str, seeds))})"
        plot_aggregated(paths, seeds, out, title)
        print(f"Wrote {out}")

    # Per-group layer ranking (best alpha per layer, across seeds) -> top_layers_*.json.
    write_top_layers(results_files)

    # Copy best-layer dirs to top_retrieval_evaluation/ for version control.
    copy_top_results()


if __name__ == "__main__":
    main()
