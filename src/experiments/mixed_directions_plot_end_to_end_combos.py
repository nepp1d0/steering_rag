"""
Mixed-direction end-to-end figure: delta accuracy per combo, grouped by model.

Companion to mixed_directions_plot_combos.py: same panels as that figure's bottom row
(one panel per eval dataset, bars grouped by model, x axis = identification combo), but
the bar height is the end-to-end answer-accuracy change instead of the rank separation
gain:

    delta = accuracy(alpha = that cell's selected alpha) - accuracy(alpha = 0)

alpha=0 is similarity-only retrieval and carries no direction term, so it is the same
baseline for all 7 combos of a given (model, eval); the baselines are asserted identical
as a build check.

Mixed directions exist for seed 42 only: one seed per cell, so no error bars (matching
figure_combos.pdf). Each combo runs at its own selected alpha and layer, both chosen on
the retrieval task, so bars differ in more than the combo — the per-cell values are
tabulated in the sidecar.

Reads results/mixed_directions_end_to_end_evaluation/ (2.5 GB of jsonl), so the per-cell
accuracies are cached next to the figure; --refresh recomputes them.

Usage:
    python src/experiments/mixed_directions_plot_end_to_end_combos.py
    python src/experiments/mixed_directions_plot_end_to_end_combos.py --refresh
    python src/experiments/mixed_directions_plot_end_to_end_combos.py --ks 2 5 10
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.append(str(Path(__file__).resolve().parent))
from mixed_directions_plot_combos import (
    COMBOS,
    COMBO_LABELS,
    DATASETS,
    DATASET_LABELS,
    MODELS_BY_SIZE,
    MODEL_COLORS,
    RC,
    _style,
)

from utils import RESULTS_DIR

EVAL_ROOT = RESULTS_DIR / "mixed_directions_end_to_end_evaluation"
FIG_DIR = RESULTS_DIR / "figures" / "mixed"
OUT_PATH = FIG_DIR / "figure_combos_end_to_end.pdf"
CACHE_PATH = FIG_DIR / "figure_combos_end_to_end_data.json"

DEFAULT_KS = [2]          # one row per k; the bottom row of figure_combos.pdf is a single row


def summarize_cell(results_path: Path) -> dict:
    """Accuracy per (alpha, k) for one cell, plus its layer and selected alpha."""
    cfg = json.loads((results_path.parent / "config.json").read_text())
    hits: dict[str, int] = {}
    total: dict[str, int] = {}
    with results_path.open() as f:
        for line in f:
            r = json.loads(line)
            key = f"{r['alpha']}|{r['k']}"
            hits[key] = hits.get(key, 0) + int(r["is_correct"])
            total[key] = total.get(key, 0) + 1
    return {"layer": cfg["layer"], "alpha": cfg["alphas"][1],
            "selection_source": cfg.get("selection_source"),
            "n": max(total.values()) if total else 0,
            "acc": {k: hits[k] / total[k] for k in hits}}


def build_cache() -> dict:
    cells = {}
    for model, _, _ in MODELS_BY_SIZE:
        for eval_ds in DATASETS:
            for combo in COMBOS:
                cell_dir = EVAL_ROOT / model / eval_ds / combo
                paths = sorted(cell_dir.rglob("results.jsonl"))
                if not paths:
                    print(f"Missing: {model} / {eval_ds} / {combo}")
                    continue
                if len(paths) > 1:
                    print(f"Warning: {len(paths)} layers for {model}/{eval_ds}/{combo}, using {paths[0]}")
                cells[f"{model}|{eval_ds}|{combo}"] = summarize_cell(paths[0])
                print(f"Read {model} / {eval_ds} / {combo}")
    return {"generated": datetime.now().isoformat(timespec="seconds"),
            "source": str(EVAL_ROOT.relative_to(RESULTS_DIR.parent)), "cells": cells}


def load_cache(refresh: bool) -> dict:
    if not refresh and CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    data = build_cache()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(data, indent=2))
    print(f"Wrote {CACHE_PATH}")
    return data


def delta_of(cell: dict, k: int) -> float | None:
    """accuracy(selected alpha) - accuracy(0) at this k."""
    base = cell["acc"].get(f"0.0|{k}")
    fused = cell["acc"].get(f"{cell['alpha']}|{k}")
    if base is None or fused is None:
        return None
    return fused - base


def check_baselines(cells: dict, ks: list[int]) -> None:
    """The alpha=0 baseline carries no direction term, so it must be identical across
    the 7 combos of a given (model, eval). A mismatch means the cells disagree on the
    corpus or the retrieval, and the deltas would not be comparable."""
    for model, _, _ in MODELS_BY_SIZE:
        for eval_ds in DATASETS:
            for k in ks:
                vals = {round(c["acc"][f"0.0|{k}"], 6)
                        for key, c in cells.items()
                        if key.startswith(f"{model}|{eval_ds}|") and f"0.0|{k}" in c["acc"]}
                if len(vals) > 1:
                    raise ValueError(f"alpha=0 baseline differs across combos for "
                                     f"{model}/{eval_ds} at k={k}: {sorted(vals)}")


def out_path_for(ks: list[int]) -> Path:
    """Multi-k variants get their own filename so they don't overwrite the default figure."""
    if ks == DEFAULT_KS:
        return OUT_PATH
    return OUT_PATH.with_name(f"{OUT_PATH.stem}_k{'-'.join(str(k) for k in ks)}.pdf")


def make_figure(cells: dict, ks: list[int]) -> None:
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(len(ks), len(DATASETS),
                             figsize=(7.2, 2.9 * len(ks)), squeeze=False)

    x = np.arange(len(COMBOS))
    width = 0.2
    for row, k in enumerate(ks):
        row_axes = [axes[row][i] for i in range(len(DATASETS))]
        for ax, eval_ds in zip(row_axes, DATASETS):
            for i, (model, label, _) in enumerate(MODELS_BY_SIZE):
                offs = (i - (len(MODELS_BY_SIZE) - 1) / 2) * width
                for j, combo in enumerate(COMBOS):
                    cell = cells.get(f"{model}|{eval_ds}|{combo}")
                    if cell is None:
                        continue
                    d = delta_of(cell, k)
                    if d is None:
                        continue
                    ax.bar(j + offs, d, width=width, color=MODEL_COLORS[model],
                           edgecolor="white", linewidth=0.4, zorder=2)
            ax.axhline(0, color="#90A4AE", linewidth=0.8, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels([COMBO_LABELS[c] for c in COMBOS], rotation=30,
                               ha="right", fontsize=6.5)
            ax.set_title(f"{DATASET_LABELS[eval_ds]} — k={k}", fontsize=9, pad=4)
            _style(ax)
            if row == len(ks) - 1:
                ax.set_xlabel("identification dataset(s)")
        row_axes[0].set_ylabel(f"$\\Delta$ accuracy vs $\\alpha=0$\n(k={k})")

        # Shared y-limits within the row so the two eval datasets are directly comparable.
        lo = min(ax.get_ylim()[0] for ax in row_axes)
        hi = max(ax.get_ylim()[1] for ax in row_axes)
        for ax in row_axes:
            ax.set_ylim(lo, hi)

    handles = [Patch(facecolor=MODEL_COLORS[m], edgecolor="white", label=label)
               for m, label, _ in MODELS_BY_SIZE]
    fig.legend(handles=handles, ncol=4, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=1.6)

    fig.tight_layout(pad=0.8)
    out = out_path_for(ks)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def write_sidecar(cells: dict, ks: list[int]) -> None:
    lines = [
        f"# {out_path_for(ks).name} — end-to-end accuracy change per identification combo",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}  ",
        "Script: `src/experiments/mixed_directions_plot_end_to_end_combos.py`  ",
        "Companion to `figure_combos.pdf` (same panels as its bottom row, same palette).",
        "",
        "## What is plotted",
        "",
        "Bar height = `accuracy(selected alpha) - accuracy(alpha=0)` on the eval dataset's own",
        "test split, one panel per eval dataset, bars grouped by model, x axis = the",
        "identification dataset combo the direction was fit on. Positive = the factuality",
        "direction improved the generated answer over similarity-only retrieval.",
        "",
        "Accuracy is the alias metric of `end_to_end_evaluation.py`: the generated answer is",
        "correct if it contains any ground-truth alias of length >= 4 characters. Greedy",
        "decoding, max 64 new tokens, top-k documents in the prompt.",
        "",
        "alpha=0 carries no direction term, so it is one shared baseline per (model, eval);",
        "the script asserts it is identical across all 7 combos before plotting.",
        "",
        "## Data",
        "",
        "Source: `results/mixed_directions_end_to_end_evaluation/` written by",
        "`src/experiments/mixed_directions_end_to_end_evaluation.py`. Per-cell accuracies are",
        "cached in `figure_combos_end_to_end_data.json` next to this file.",
        "",
        "Pipeline: `dataset_normalization.py` -> `mixed_direction_identification.py` ->",
        "`mixed_directions_retrieval_evaluation.py` -> `mixed_directions_plot_retrieval_evaluation.py`",
        "(selects layer + alpha per cell, writes `top_layers_context_only.json`) ->",
        "`mixed_directions_end_to_end_evaluation.py` -> this script.",
        "",
        f"- models: {', '.join(lb for _, lb, _ in MODELS_BY_SIZE)}",
        f"- eval datasets: {', '.join(DATASET_LABELS[d] for d in DATASETS)} "
        "(ConflictQA 1835 test samples / 3608 docs, NQ-Swap 342 / 681)",
        f"- combos: {', '.join(COMBO_LABELS[c] for c in COMBOS)}",
        f"- k: {', '.join(str(k) for k in ks)}",
        "- seed: 42 only (mixed directions exist for this seed alone)",
        "- procedure: `context_only`, position: `last_pos`, directions unnormalized",
        "- scoring: `(1-alpha)*z(sbert_cos) + alpha*z(projection)`, z-scored over the corpus",
        "",
        "## Selection protocol",
        "",
        "Each cell uses the single best layer and alpha chosen by",
        "`mixed_directions_plot_retrieval_evaluation.py` on the *retrieval* task (top-k recall",
        "lift + non-factual drop vs the alpha=0 baseline), not tuned on end-to-end accuracy.",
        "Per-cell layer and alpha:",
        "",
        "| model | eval | combo | layer | alpha |",
        "|---|---|---|---|---|",
    ]
    for model, label, _ in MODELS_BY_SIZE:
        for eval_ds in DATASETS:
            for combo in COMBOS:
                c = cells.get(f"{model}|{eval_ds}|{combo}")
                if c:
                    lines.append(f"| {label} | {DATASET_LABELS[eval_ds]} | "
                                 f"{COMBO_LABELS[combo]} | {c['layer']} | {c['alpha']} |")
    lines += [
        "",
        "## Reading and limitations",
        "",
        "- **One seed, no error bars.** Every bar is a point estimate. Binomial se is ~0.011 on",
        "  ConflictQA and ~0.025 on NQ-Swap, so differences below ~0.03 (ConflictQA) or ~0.07",
        "  (NQ-Swap) between neighbouring bars are not resolvable. Paired bootstrap CIs over the",
        "  shared question set would be cheap to add and are the right fix before publication.",
        "- **Bars differ in more than the combo.** Each cell carries its own selected layer and",
        "  alpha (table above); this is deliberately not marked on the bars, so it belongs in the",
        "  caption. Most cells are alpha=0.5. In particular",
        "  Llama-3.2-1B / ConflictQA / LF was selected at alpha=1.0, the degenerate",
        "  query-independent configuration (pure projection ranks every question identically),",
        "  which is why that bar collapses. Qwen / ConflictQA / NQS was selected at layer 0,",
        "  essentially the embedding layer. Both are selection artifacts of a single-seed",
        "  retrieval score, not properties of the identification corpus.",
        "- **The sign flips with k and with the eval dataset.** Gains concentrate at k=2 on",
        "  ConflictQA and mostly vanish or reverse by k=10; on NQ-Swap most cells are negative.",
        "  Plot all three k (`--ks 2 5 10`) before drawing any conclusion from the k=2 panel.",
        "",
    ]
    md_path = out_path_for(ks).with_suffix(".md")
    md_path.write_text("\n".join(lines))
    print(f"Wrote {md_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="Recompute the per-cell accuracy cache from the jsonl results.")
    ap.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS,
                    help="One row of panels per k (default: 2).")
    args = ap.parse_args()

    data = load_cache(args.refresh)
    cells = data["cells"]
    print(f"Loaded {len(cells)} cells.")
    check_baselines(cells, args.ks)
    make_figure(cells, args.ks)
    write_sidecar(cells, args.ks)


if __name__ == "__main__":
    main()
