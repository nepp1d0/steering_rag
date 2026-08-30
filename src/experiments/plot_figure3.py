"""
Paper-ready re-ranking figure (single 2x2 PDF), fully rank-based.

Top row:    mean rank vs alpha — gold (left) and non-factual (right), all models, TOP_ROW_DATASET.
Bottom-left: rank deltas at alpha=SCATTER_ALPHA — gold rank gain (x) vs non-factual rank change (y).
Bottom-right: rank separation gain at SCATTER_ALPHA vs model size.

Usage:
    python src/experiments/plot_reranking_figure.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.append(str(Path(__file__).resolve().parent))
from plot_retrieval_evaluation import (  # noqa: E402
    RESULTS_DIR,
    compute_seed_metrics,
    _agg,
)

TOP_DIR = RESULTS_DIR / "top_retrieval_evaluation"

# ── Config ───────────────────────────────────────────────────────────────────
NORMALIZE = "unnormalized"
PROCEDURE = "context_only"
# Which direction (identification) dataset the figure is built from:
#   "same"    -> in-domain: direction == eval (reproduces the original figure)
#   <dataset> -> use that direction (e.g. "longfact") tested on the eval datasets
# The chosen value also becomes an extra output level: figures/<DIRECTION_DATASET>/...
DIRECTION_DATASET = "nq_swap"
# Direction position: "last_pos" keeps the original output paths; other positions
# ("entity_pos") add an extra output level: .../<POSITION>/figure_3_reranking.pdf
POSITION = "last_pos"
TOP_ROW_DATASET = "nq_swap"   # which dataset's mean-rank curves go in the top row
SCATTER_ALPHA = 0.3              # fixed alpha for the bottom-row rank deltas
DROP_ALPHA_ONE = False           # set True to hide the degenerate alpha=1.0 point in the top row

_FIG_DIR = RESULTS_DIR / "figures" / DIRECTION_DATASET
if POSITION != "last_pos":
    _FIG_DIR = _FIG_DIR / POSITION
OUT_PATH = _FIG_DIR / "figure_3_reranking.pdf"

MODELS_BY_SIZE = [
    ("meta-llama__Llama-3.2-1B-Instruct", "Llama-3.2-1B", 1.2),
    ("google__gemma-3-4b-it",             "Gemma-3-4B",   4.3),
    ("Qwen__Qwen2-7B-Instruct",           "Qwen2-7B",     7.6),
    ("meta-llama__Llama-3.1-8B-Instruct", "Llama-3.1-8B", 8.0),
]
COLORS = {
    "meta-llama__Llama-3.2-1B-Instruct": "#FFB300",  # Amber 600
    "google__gemma-3-4b-it":             "#00897B",  # Teal 600
    "Qwen__Qwen2-7B-Instruct":           "#7E57C2",  # Deep Purple 400
    "meta-llama__Llama-3.1-8B-Instruct": "#1E88E5",  # Blue 600
}

# Eval datasets shown in the bottom panels (longfact is never an eval, only a direction).
DATASETS = ["conflictqa", "nq_swap"]
DATASET_LABELS = {"conflictqa": "ConflictQA", "nq_swap": "NQ-Swap", "longfact": "LongFact"}
MARKERS = {"conflictqa": "o", "nq_swap": "^", "longfact": "s"}
LINESTYLES = {"conflictqa": "-", "nq_swap": "--", "longfact": "-"}

RC = {
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


# ── Data discovery / aggregation ──────────────────────────────────────────────
def parse_parts(path: Path):
    """Assumed layout: model/eval/direction/normalize/seed_N/procedure/layer_M/position/results.jsonl"""
    p = path.relative_to(TOP_DIR).parts
    if len(p) < 9:
        return None
    return {"model": p[0], "eval": p[1], "direction": p[2], "normalize": p[3],
            "seed": p[4], "procedure": p[5], "layer": p[6], "position": p[7]}


def collect_groups() -> dict[tuple, list[Path]]:
    groups: dict[tuple, list[Path]] = defaultdict(list)
    for f in sorted(TOP_DIR.rglob("results.jsonl")):
        meta = parse_parts(f)
        if meta is None:
            continue
        if meta["normalize"] != NORMALIZE or meta["procedure"] != PROCEDURE or meta["position"] != POSITION:
            continue
        if DIRECTION_DATASET == "same":
            if meta["eval"] != meta["direction"]:   # in-domain only
                continue
        elif meta["direction"] != DIRECTION_DATASET:
            continue
        groups[(meta["model"], meta["eval"])].append(f)
    return groups


def build_data(groups) -> dict[tuple, dict]:
    data: dict[tuple, dict] = {}
    for (model, ds), paths in groups.items():
        metrics = [compute_seed_metrics(p) for p in sorted(paths)]
        if not metrics or not metrics[0]["alphas"] or not metrics[0]["has_rank"]:
            continue
        alphas = metrics[0]["alphas"]
        plot_alphas = [a for a in alphas if not (DROP_ALPHA_ONE and a == 1.0)]
        entry = {"alphas": plot_alphas}

        # Top-row mean-rank curves (rank is independent of k).
        gr_m, gr_s, nr_m, nr_s = [], [], [], []
        for a in plot_alphas:
            gm, gs = _agg([m["mean_gold_rank"][a] for m in metrics])
            nm, ns = _agg([m["mean_nf_rank"][a] for m in metrics])
            gr_m.append(gm); gr_s.append(gs); nr_m.append(nm); nr_s.append(ns)
        entry.update(gold_rank_m=gr_m, gold_rank_s=gr_s, nf_rank_m=nr_m, nf_rank_s=nr_s)

        # Bottom-row rank deltas at SCATTER_ALPHA.  delta = baseline_rank - rank(alpha)
        #   gold:        positive  -> climbed toward the top (good)
        #   non-factual: negative  -> sank down the list (good)
        sa = SCATTER_ALPHA if SCATTER_ALPHA in alphas else min(alphas, key=lambda a: abs(a - SCATTER_ALPHA))
        if all(0.0 in m["mean_gold_rank"] and sa in m["mean_gold_rank"] for m in metrics):
            dg = [m["mean_gold_rank"][0.0] - m["mean_gold_rank"][sa] for m in metrics]   # gold gain
            dn = [m["mean_nf_rank"][0.0] - m["mean_nf_rank"][sa] for m in metrics]        # nf change (neg)
            sep = [g - n for g, n in zip(dg, dn)]   # separation gain = gold gain + nf drop magnitude
            entry.update({
                "gx_m": _agg(dg)[0], "gx_s": _agg(dg)[1],
                "ny_m": _agg(dn)[0], "ny_s": _agg(dn)[1],
                "sep_m": _agg(sep)[0], "sep_s": _agg(sep)[1],
                "scatter_alpha": sa,
            })
        data[(model, ds)] = entry
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

    # ── Top row: mean rank vs alpha for TOP_ROW_DATASET ──
    ds = TOP_ROW_DATASET
    ax_g, ax_n = axes[0][0], axes[0][1]
    for model, label, _ in MODELS_BY_SIZE:
        d = data.get((model, ds))
        if not d or "gold_rank_m" not in d:
            continue
        a = np.array(d["alphas"])
        for ax, km, ks_ in [(ax_g, "gold_rank_m", "gold_rank_s"),
                            (ax_n, "nf_rank_m", "nf_rank_s")]:
            m_ = np.array(d[km]); s_ = np.array(d[ks_])
            ax.fill_between(a, np.clip(m_ - s_, 1.0, None), m_ + s_,
                            color=COLORS[model], alpha=0.12, linewidth=0)
            ax.plot(a, m_, marker="o", markersize=3.5, linewidth=1.8, color=COLORS[model],
                    markeredgecolor="white", markeredgewidth=0.4)
    ax_g.set_yscale("log"); ax_n.set_yscale("log")
    ax_g.set_title(f"Gold document \u2014 {DATASET_LABELS[ds]}", fontsize=9, pad=4)
    ax_n.set_title(f"Non-factual document \u2014 {DATASET_LABELS[ds]}", fontsize=9, pad=4)
    ax_g.set_ylabel("mean rank (log)\nlower = better")
    ax_n.set_ylabel("mean rank (log)\nhigher = better")
    ax_g.set_xlabel(r"$\alpha$"); ax_n.set_xlabel(r"$\alpha$")
    _style(ax_g); _style(ax_n)

    # ── Bottom-left: rank deltas at alpha=SCATTER_ALPHA (both datasets) ──
    axB = axes[1][0]
    for d_ds in DATASETS:
        for model, label, _ in MODELS_BY_SIZE:
            d = data.get((model, d_ds))
            if not d or "gx_m" not in d:
                continue
            axB.errorbar(d["gx_m"], d["ny_m"], xerr=d["gx_s"], yerr=d["ny_s"],
                         marker=MARKERS[d_ds], markersize=8, color=COLORS[model],
                         ecolor=COLORS[model], elinewidth=0.7, capsize=2,
                         markeredgecolor="white", markeredgewidth=0.5, linestyle="none", zorder=3)
    axB.axhline(0, color="#CFD8DC", linewidth=0.8)
    axB.axvline(0, color="#CFD8DC", linewidth=0.8)
    axB.set_xlabel("gold rank gain (positions, $+$ = climbed)")
    axB.set_ylabel("non-factual rank change\n(positions, $-$ = sank)")
    axB.set_title(f"Rank change at $\\alpha={SCATTER_ALPHA:g}$", fontsize=9, pad=4)
    _style(axB, grid_axis="both")
    ds_handles = [Line2D([0], [0], marker=MARKERS[d], color="#607D8B", linestyle="none",
                         markersize=7, label=DATASET_LABELS[d]) for d in DATASETS]
    axB.legend(handles=ds_handles, fontsize=7, frameon=False, loc="upper left")

    # ── Bottom-right: rank separation gain at SCATTER_ALPHA vs model size ──
    axC = axes[1][1]
    sizes = [s for _, _, s in MODELS_BY_SIZE]
    for d_ds in DATASETS:
        xs, ys, yerr, cols = [], [], [], []
        for model, label, size in MODELS_BY_SIZE:
            d = data.get((model, d_ds))
            if not d or "sep_m" not in d:
                continue
            xs.append(size); ys.append(d["sep_m"]); yerr.append(d["sep_s"]); cols.append(COLORS[model])
        if not xs:
            continue
        axC.plot(xs, ys, linestyle=LINESTYLES[d_ds], color="#B0BEC5", linewidth=1.3, zorder=1)
        axC.errorbar(xs, ys, yerr=yerr, fmt="none", ecolor="#B0BEC5", elinewidth=0.7, capsize=2, zorder=2)
        for x, y, c in zip(xs, ys, cols):
            axC.plot(x, y, marker=MARKERS[d_ds], markersize=8, color=c,
                     markeredgecolor="white", markeredgewidth=0.5, linestyle="none", zorder=3)
    axC.set_xscale("log")
    axC.set_xticks(sizes)
    axC.set_xticklabels([f"{s:g}B" for s in sizes])
    axC.minorticks_off()
    axC.set_xlabel("model size (params, log)")
    axC.set_ylabel(f"rank separation gain\n(positions, $\\alpha={SCATTER_ALPHA:g}$)")
    axC.set_title("Effect vs scale", fontsize=9, pad=4)
    _style(axC)
    c_handles = [Line2D([0], [0], marker=MARKERS[d], color="#B0BEC5", linestyle=LINESTYLES[d],
                        markersize=7, label=DATASET_LABELS[d]) for d in DATASETS]
    axC.legend(handles=c_handles, fontsize=7, frameon=False, loc="upper left")

    # Shared model legend on top
    model_handles = [Line2D([0], [0], color=COLORS[m], marker="o", linewidth=2,
                            markeredgecolor="white", markeredgewidth=0.4, label=label)
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
    print(f"Found {len(groups)} (model, dataset) groups under {TOP_DIR}.")
    data = build_data(groups)
    make_figure(data)


if __name__ == "__main__":
    main()