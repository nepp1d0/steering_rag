"""
Paper-ready mixed-direction re-ranking figure (single 2x2 PDF), fully rank-based.

Sibling of plot_figure3.py, but the varying dimension is the *combo* of identification
datasets ("conflictqa", "conflictqa+nq_swap", ...) instead of the model.

Top row:    mean rank vs alpha — gold (left) and non-factual (right), one line per combo,
            for the fixed TOP_ROW_MODEL x TOP_ROW_DATASET cell.
Bottom row: rank separation gain at SCATTER_ALPHA per combo, grouped bars by model —
            ConflictQA (left) and NQ-Swap (right).

Mixed directions exist for seed 42 only, so there is a single seed per cell: no error
bands or caps are drawn (their spread would be identically zero). Each point also comes
from that combo's own best layer, selected on the same seed it is reported on, so small
between-combo gaps are within selection noise; read the consistent patterns.

Usage:
    python src/experiments/mixed_directions_plot_combos.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.append(str(Path(__file__).resolve().parent))
from plot_retrieval_evaluation import (  # noqa: E402
    RESULTS_DIR,
    compute_seed_metrics,
)

TOP_DIR = RESULTS_DIR / "mixed_directions_top_retrieval_evaluation"

# ── Config ───────────────────────────────────────────────────────────────────
NORMALIZE = "unnormalized"
PROCEDURE = "context_only"
TOP_ROW_MODEL = "meta-llama__Llama-3.1-8B-Instruct"   # cell shown in the top row
TOP_ROW_DATASET = "conflictqa"
SCATTER_ALPHA = 0.3              # fixed alpha for the bottom-row separation gain
DROP_ALPHA_ONE = False           # set True to hide the degenerate alpha=1.0 point in the top row

OUT_PATH = RESULTS_DIR / "figures" / "mixed" / "figure_combos.pdf"

MODELS_BY_SIZE = [
    ("meta-llama__Llama-3.2-1B-Instruct", "Llama-3.2-1B", 1.2),
    ("google__gemma-3-4b-it",             "Gemma-3-4B",   4.3),
    ("Qwen__Qwen2-7B-Instruct",           "Qwen2-7B",     7.6),
    ("meta-llama__Llama-3.1-8B-Instruct", "Llama-3.1-8B", 8.0),
]
MODEL_COLORS = {   # same palette as plot_figure3.py so the two figures read as siblings
    "meta-llama__Llama-3.2-1B-Instruct": "#FFB300",  # Amber 600
    "google__gemma-3-4b-it":             "#00897B",  # Teal 600
    "Qwen__Qwen2-7B-Instruct":           "#7E57C2",  # Deep Purple 400
    "meta-llama__Llama-3.1-8B-Instruct": "#1E88E5",  # Blue 600
}

# Ordered singles -> pairs -> triple; the bottom-row x axis follows this order.
COMBOS = [
    "conflictqa",
    "nq_swap",
    "longfact",
    "conflictqa+nq_swap",
    "conflictqa+longfact",
    "nq_swap+longfact",
    "conflictqa+nq_swap+longfact",
]
COMBO_LABELS = {
    "conflictqa": "CQA",
    "nq_swap": "NQS",
    "longfact": "LF",
    "conflictqa+nq_swap": "CQA+NQS",
    "conflictqa+longfact": "CQA+LF",
    "nq_swap+longfact": "NQS+LF",
    "conflictqa+nq_swap+longfact": "CQA+NQS+LF",
}
COMBO_COLORS = {
    "conflictqa":                  "#E53935",  # Red 600
    "nq_swap":                     "#1E88E5",  # Blue 600
    "longfact":                    "#43A047",  # Green 600
    "conflictqa+nq_swap":          "#8E24AA",  # Purple 600
    "conflictqa+longfact":         "#FB8C00",  # Orange 600
    "nq_swap+longfact":            "#00ACC1",  # Cyan 600
    "conflictqa+nq_swap+longfact": "#37474F",  # Blue Grey 800
}
# Singles dashed, mixes solid — the single-vs-mixed contrast is the point of the figure.
COMBO_LINESTYLES = {c: ("--" if "+" not in c else "-") for c in COMBOS}
COMBO_LINEWIDTHS = {c: (1.3 if "+" not in c else 1.8) for c in COMBOS}

DATASETS = ["conflictqa", "nq_swap"]
DATASET_LABELS = {"conflictqa": "ConflictQA", "nq_swap": "NQ-Swap"}

RC = {
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


# ── Data discovery / aggregation ──────────────────────────────────────────────
def parse_parts(path: Path):
    """Mixed layout: model/eval/combo/normalize/seed_N/procedure/layer_M/results.jsonl

    One component shorter than the main pipeline's — mixed_directions_retrieval_evaluation.py
    predates the position-as-leaf layout, so there is no <position> level.
    """
    p = path.relative_to(TOP_DIR).parts
    if len(p) < 8:
        return None
    return {"model": p[0], "eval": p[1], "combo": p[2], "normalize": p[3],
            "seed": p[4], "procedure": p[5], "layer": p[6]}


def collect_groups() -> dict[tuple, list[Path]]:
    groups: dict[tuple, list[Path]] = defaultdict(list)
    for f in sorted(TOP_DIR.rglob("results.jsonl")):
        meta = parse_parts(f)
        if meta is None:
            continue
        if meta["normalize"] != NORMALIZE or meta["procedure"] != PROCEDURE:
            continue
        groups[(meta["model"], meta["eval"], meta["combo"])].append(f)
    return groups


def build_data(groups) -> dict[tuple, dict]:
    """One seed per cell, so metrics stay plain floats — no mean/std across seeds."""
    data: dict[tuple, dict] = {}
    for key, paths in groups.items():
        m = compute_seed_metrics(sorted(paths)[0])
        if not m["alphas"] or not m["has_rank"]:
            continue
        alphas = m["alphas"]
        plot_alphas = [a for a in alphas if not (DROP_ALPHA_ONE and a == 1.0)]
        entry = {
            "alphas": plot_alphas,
            "gold_rank": [m["mean_gold_rank"][a] for a in plot_alphas],
            "nf_rank": [m["mean_nf_rank"][a] for a in plot_alphas],
        }

        # Separation gain at SCATTER_ALPHA.  delta = baseline_rank - rank(alpha)
        #   gold gain:  positive -> gold climbed toward the top (good)
        #   nf change:  negative -> non-factual sank down the list (good)
        # separation = gold gain + magnitude of the nf drop.
        sa = SCATTER_ALPHA if SCATTER_ALPHA in alphas else min(alphas, key=lambda a: abs(a - SCATTER_ALPHA))
        if 0.0 in m["mean_gold_rank"] and sa in m["mean_gold_rank"]:
            dg = m["mean_gold_rank"][0.0] - m["mean_gold_rank"][sa]
            dn = m["mean_nf_rank"][0.0] - m["mean_nf_rank"][sa]
            entry.update({"sep": dg - dn, "scatter_alpha": sa})
        data[key] = entry
    return data


# ── Styling ───────────────────────────────────────────────────────────────────
def _style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#BDBDBD")
    ax.spines["bottom"].set_color("#BDBDBD")
    ax.tick_params(length=0, labelsize=8)
    ax.grid(axis=grid_axis, color="#ECEFF1", linewidth=0.7)
    ax.set_axisbelow(True)


def make_figure(data) -> None:
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))

    # ── Top row: mean rank vs alpha, one line per combo (fixed model x eval) ──
    ax_g, ax_n = axes[0][0], axes[0][1]
    for combo in COMBOS:
        d = data.get((TOP_ROW_MODEL, TOP_ROW_DATASET, combo))
        if not d:
            continue
        a = np.array(d["alphas"])
        for ax, key in [(ax_g, "gold_rank"), (ax_n, "nf_rank")]:
            ax.plot(a, np.array(d[key]), marker="o", markersize=3.5,
                    linewidth=COMBO_LINEWIDTHS[combo], linestyle=COMBO_LINESTYLES[combo],
                    color=COMBO_COLORS[combo], markeredgecolor="white", markeredgewidth=0.4)
    ax_g.set_yscale("log"); ax_n.set_yscale("log")
    model_label = next(lb for m, lb, _ in MODELS_BY_SIZE if m == TOP_ROW_MODEL)
    cell = f"{model_label} — {DATASET_LABELS[TOP_ROW_DATASET]}"
    ax_g.set_title(f"Gold document — {cell}", fontsize=9, pad=4)
    ax_n.set_title(f"Non-factual document — {cell}", fontsize=9, pad=4)
    ax_g.set_ylabel("mean rank (log)\nlower = better")
    ax_n.set_ylabel("mean rank (log)\nhigher = better")
    ax_g.set_xlabel(r"$\alpha$"); ax_n.set_xlabel(r"$\alpha$")
    _style(ax_g); _style(ax_n)

    combo_handles = [Line2D([0], [0], color=COMBO_COLORS[c], linestyle=COMBO_LINESTYLES[c],
                            linewidth=COMBO_LINEWIDTHS[c], label=COMBO_LABELS[c])
                     for c in COMBOS]
    ax_g.legend(handles=combo_handles, fontsize=6, frameon=False, loc="upper left", ncol=2)

    # ── Bottom row: separation gain at SCATTER_ALPHA, bars per combo grouped by model ──
    x = np.arange(len(COMBOS))
    width = 0.2
    bar_axes = [axes[1][0], axes[1][1]]
    for ax, d_ds in zip(bar_axes, DATASETS):
        for i, (model, label, _) in enumerate(MODELS_BY_SIZE):
            offs = (i - (len(MODELS_BY_SIZE) - 1) / 2) * width
            xs, ys = [], []
            for j, combo in enumerate(COMBOS):
                d = data.get((model, d_ds, combo))
                if not d or "sep" not in d:
                    continue
                xs.append(j + offs); ys.append(d["sep"])
            ax.bar(xs, ys, width=width, color=MODEL_COLORS[model],
                   edgecolor="white", linewidth=0.4, zorder=2)
        ax.axhline(0, color="#90A4AE", linewidth=0.8, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([COMBO_LABELS[c] for c in COMBOS], rotation=30,
                           ha="right", fontsize=6.5)
        ax.set_title(f"{DATASET_LABELS[d_ds]}", fontsize=9, pad=4)
        ax.set_xlabel("identification dataset(s)")
        _style(ax)
    bar_axes[0].set_ylabel(f"rank separation gain\n(positions, $\\alpha={SCATTER_ALPHA:g}$)")

    # Shared y-limits so the two eval datasets are directly comparable.
    lo = min(ax.get_ylim()[0] for ax in bar_axes)
    hi = max(ax.get_ylim()[1] for ax in bar_axes)
    for ax in bar_axes:
        ax.set_ylim(lo, hi)

    model_handles = [Patch(facecolor=MODEL_COLORS[m], edgecolor="white", label=label)
                     for m, label, _ in MODELS_BY_SIZE]
    fig.legend(handles=model_handles, ncol=4, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=1.6)

    fig.tight_layout(pad=0.8)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


def main() -> None:
    groups = collect_groups()
    print(f"Found {len(groups)} (model, dataset, combo) groups under {TOP_DIR}.")
    data = build_data(groups)
    make_figure(data)


if __name__ == "__main__":
    main()
