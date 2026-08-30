"""
RAGuard step D - plots for the claim-separation diagnostic.

Reads results/raguard/<model>/{claim_separation.jsonl,baselines.jsonl} and
results/raguard/lexical_baseline.json. Produces:

  auroc_by_layer_<model>[_mixed].pdf   AUROC vs layer, one line per direction source (+-1 std
                               over the direction seeds), overlaid with the supervised probe
                               ceiling, the random-direction null band (2.5-97.5 pct), the
                               0.5 chance line and the TF-IDF lexical floor.
  auroc_summary[_mixed].pdf    models x direction sources heatmap, each cell at its own best
                               layer (chosen by |AUROC - 0.5|), annotated with signed AUROC.

`--direction-source mixed` plots the mixture combos from mixed_direction_identification.py
instead of the single-dataset directions. Mixtures exist at seed 42 only, so their +-1 std
ribbon is zero-width - the spread across seeds is not available for them.

AUROC is plotted SIGNED, not absolute: the direction has a definite orientation (factual is
positive), so a consistent AUROC < 0.5 would itself be a finding and must not be hidden.

Writes to results/raguard/figures/. CPU only.

Usage:
    python src/exploratory/plot_raguard_claim_separation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
from utils import RESULTS_DIR, logger, setup_logging

MODELS = ["google__gemma-3-4b-it", "meta-llama__Llama-3.1-8B-Instruct",
          "meta-llama__Llama-3.2-1B-Instruct", "Qwen__Qwen2-7B-Instruct"]
OUT_ROOT = RESULTS_DIR / "raguard"
FIG_DIR = OUT_ROOT / "figures"

# Singles keep the colours used for the single-source figures; mixtures are dashed so a
# combo is visually distinguishable from the datasets it is built from.
DD_COLORS = {
    "nq_swap": "#2a78d6", "conflictqa": "#e34948", "longfact": "#3f9a6d",
    "conflictqa+nq_swap": "#8e5fa8", "conflictqa+longfact": "#c47f2e",
    "nq_swap+longfact": "#2f8f9d", "conflictqa+nq_swap+longfact": "#111111",
}
DD_STYLES = {name: ("--" if "+" in name else "-") for name in DD_COLORS}
DIRECTION_SOURCES = {
    "single": (["nq_swap", "conflictqa", "longfact"], ""),
    "mixed": (["conflictqa", "nq_swap", "longfact",
               "conflictqa+nq_swap", "conflictqa+longfact", "nq_swap+longfact",
               "conflictqa+nq_swap+longfact"], "_mixed"),
}
DIVERGING_CMAP = LinearSegmentedColormap.from_list("blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"])


def read_jsonl(path: Path) -> List[Dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def per_layer_stats(rows: List[Dict], dd: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(layers, mean AUROC over seeds, std) for one direction dataset."""
    by_layer = defaultdict(list)
    for r in rows:
        if r["direction_dataset"] == dd:
            by_layer[r["layer"]].append(r["auroc"])
    layers = np.array(sorted(by_layer))
    mean = np.array([np.mean(by_layer[L]) for L in layers])
    std = np.array([np.std(by_layer[L]) for L in layers])
    return layers, mean, std


def plot_model(model: str, rows: List[Dict], base: List[Dict], tfidf: float,
               direction_names: List[str], suffix: str) -> None:
    base_layers = np.array([b["layer"] for b in base])
    lo = np.array([b["random_p2.5"] for b in base])
    hi = np.array([b["random_p97.5"] for b in base])
    probe = [b["probe_auroc"] for b in base]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(base_layers, lo, hi, color="#b0b0b0", alpha=0.35, lw=0,
                    label="random directions (95% band)")
    ax.axhline(0.5, color="#666666", lw=1, ls="--", label="chance")
    ax.axhline(tfidf, color="#9467bd", lw=1.2, ls=":", label=f"TF-IDF floor ({tfidf:.3f})")
    if all(p is not None for p in probe):
        ax.plot(base_layers, probe, color="#111111", lw=1.6, ls="-.", label="supervised probe ceiling")

    for dd in direction_names:
        layers, mean, std = per_layer_stats(rows, dd)
        if not len(layers):
            continue
        ax.plot(layers, mean, color=DD_COLORS[dd], lw=1.8, ls=DD_STYLES[dd],
                marker="o", ms=3, label=f"direction: {dd}")
        ax.fill_between(layers, mean - std, mean + std, color=DD_COLORS[dd], alpha=0.18, lw=0)

    ax.set_xlabel("layer")
    ax.set_ylabel("AUROC (true vs false claim)")
    ax.set_title(f"RAGuard claim separation - {model}" + (" (mixed directions)" if suffix else ""))
    ax.legend(fontsize=7, loc="best", framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    out = FIG_DIR / f"auroc_by_layer_{model}{suffix}.pdf"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info(f"Wrote {out}")


def plot_summary(summary: Dict[str, Dict[str, tuple[int, float]]],
                 direction_names: List[str], suffix: str) -> None:
    models = [m for m in MODELS if m in summary]
    grid = np.full((len(models), len(direction_names)), np.nan)
    for i, m in enumerate(models):
        for j, dd in enumerate(direction_names):
            if dd in summary[m]:
                grid[i, j] = summary[m][dd][1]

    fig, ax = plt.subplots(figsize=(1.6 * len(direction_names) + 3.5, 0.7 * len(models) + 2))
    span = max(0.02, float(np.nanmax(np.abs(grid - 0.5))))
    im = ax.imshow(grid, cmap=DIVERGING_CMAP, norm=TwoSlopeNorm(vmin=0.5 - span, vcenter=0.5, vmax=0.5 + span))
    ax.set_xticks(range(len(direction_names)), direction_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)), models, fontsize=8)
    ax.set_xlabel("direction source")
    ax.set_title("Best-layer AUROC, true vs false RAGuard claims")
    for i, m in enumerate(models):
        for j, dd in enumerate(direction_names):
            if dd in summary[m]:
                layer, val = summary[m][dd]
                ax.text(j, i, f"{val:.3f}\nL{layer}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="AUROC")
    fig.tight_layout()
    out = FIG_DIR / f"auroc_summary{suffix}.pdf"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the RAGuard claim-separation diagnostic.")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--direction-source", default="single", choices=list(DIRECTION_SOURCES),
                        help="'single' = direction_identification, 'mixed' = mixture combos.")
    args = parser.parse_args()

    direction_names, suffix = DIRECTION_SOURCES[args.direction_source]
    setup_logging("plot_raguard_claim_separation", OUT_ROOT)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tfidf = json.loads((OUT_ROOT / "lexical_baseline.json").read_text())["tfidf_auroc"]

    summary: Dict[str, Dict[str, tuple[int, float]]] = {}
    for model in args.models:
        res_path = OUT_ROOT / model / f"claim_separation{suffix}.jsonl"
        if not res_path.exists():
            logger.warning(f"No results for {model}; skipping.")
            continue
        rows = read_jsonl(res_path)
        base = read_jsonl(OUT_ROOT / model / "baselines.jsonl")
        plot_model(model, rows, base, tfidf, direction_names, suffix)

        summary[model] = {}
        for dd in direction_names:
            layers, mean, _ = per_layer_stats(rows, dd)
            if not len(layers):
                continue
            # Best layer = strongest separation in either orientation.
            best = int(np.argmax(np.abs(mean - 0.5)))
            summary[model][dd] = (int(layers[best]), float(mean[best]))

    if summary:
        plot_summary(summary, direction_names, suffix)
        (FIG_DIR / f"summary{suffix}.json").write_text(json.dumps(
            {m: {dd: {"layer": v[0], "auroc": v[1]} for dd, v in d.items()} for m, d in summary.items()},
            indent=2))
    logger.info("Done.")


if __name__ == "__main__":
    main()
