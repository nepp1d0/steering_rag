"""
Figure 4: end-to-end answer accuracy vs k.

One line plot per dataset slice, one colour per model (ordered by parameter count).
Solid = the retrieved pool re-ranked by the factuality score at its operating alpha,
dashed = baseline (alpha=0, similarity only). Mean +/- std across seeds.

One operating alpha per (model, eval, direction, normalize) was evaluated, so there is
exactly one re-ranked curve per model. The k axis is linear, at the true k positions, so
the uneven 5 -> 10 gap is not evenly spaced.

Reuses the results layout written by the end-to-end evaluation:
  end_to_end_evaluation/<model>/<eval>/<direction>/<normalize>/.../seed_<n>/layer_<m>/results.jsonl
each row: {"alpha": float, "k": int, "is_correct": bool, ...}

Outputs (under results/end_to_end_evaluation/figures/):
  figure_4_<eval>.pdf        -- one single-column line plot per dataset
  figure_4_end_to_end.pdf    -- full-width 1xN panel, all datasets side by side (main paper)

Set RESULTS_DIR in the environment to point the script at a results tree elsewhere.

Usage:
    python src/experiments/plot_figure_4.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from utils import RESULTS_DIR

E2E_DIR = RESULTS_DIR / "end_to_end_evaluation"
FIG_DIR = E2E_DIR / "figures"

SEED_RE = re.compile(r"seed_(\d+)")
LAYER_RE = re.compile(r"layer_(\d+)")

# --- which slice the main figure shows -------------------------------------
# In-domain means the steering direction was fit on the same dataset we evaluate
# on (direction == eval). Change NORMALIZE to "normalized" for the other variant;
# push the off-diagonal / other-normalize versions to the appendix with the same
# helpers below.
NORMALIZE = "unnormalized"
DATASETS = [("conflictqa", "ConflictQA"), ("nq_swap", "NQ-Swap")]

# --- house style ------------------------------------------------------------
# Models ordered by parameter count so the capability trend reads left-to-right.
# Keys are the on-disk directory names (note the "__" separator).
MODELS = [
    "meta-llama__Llama-3.2-1B-Instruct",
    "google__gemma-3-4b-it",
    "Qwen__Qwen2-7B-Instruct",
    "meta-llama__Llama-3.1-8B-Instruct",
]
MODEL_LABELS = {
    "meta-llama__Llama-3.2-1B-Instruct": "Llama-3.2-1B",
    "google__gemma-3-4b-it":             "Gemma-3-4B",
    "Qwen__Qwen2-7B-Instruct":           "Qwen2-7B",
    "meta-llama__Llama-3.1-8B-Instruct": "Llama-3.1-8B",
}
# Okabe-Ito colourblind-safe palette, same order as MODELS above.
COLORS = ["#E69F00", "#009E73", "#CC79A7", "#0072B2"]
# Redundant non-colour cue (never rely on colour alone), same order.
MARKERS = ["^", "s", "D", "o"]

_RCPARAMS = {
    "pdf.fonttype": 42,          # embed TrueType so the PDF has real, editable fonts
    "ps.fonttype": 42,
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
}


def file_accuracies(path: Path) -> tuple[float, list[int], dict[int, float], dict[int, float]]:
    """(steered_alpha, ks, {k: baseline_acc}, {k: steered_acc}) for one results.jsonl.

    Baseline is alpha==0; the steered curve uses the (single) positive alpha present.
    """
    buckets: dict[tuple, list] = defaultdict(lambda: [0, 0])  # (alpha,k) -> [hits, total]
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            b = buckets[(r["alpha"], r["k"])]
            b[0] += int(r["is_correct"])
            b[1] += 1
    ks = sorted({k for _, k in buckets})
    steer_alpha = next(a for a, _ in sorted(buckets) if a > 0)
    base = {k: buckets[(0.0, k)][0] / buckets[(0.0, k)][1] for k in ks}
    steer = {k: buckets[(steer_alpha, k)][0] / buckets[(steer_alpha, k)][1] for k in ks}
    return steer_alpha, ks, base, steer


def aggregate_seeds(seed_map: dict[int, tuple]) -> dict:
    """Mean/std across seeds for one (model, slice). seed_map: seed -> file_accuracies tuple."""
    ks = sorted(next(iter(seed_map.values()))[1])
    alpha = next(iter(seed_map.values()))[0]

    def mean_std(idx: int) -> tuple[list[float], list[float]]:
        means, stds = [], []
        for k in ks:
            vals = [seed_map[s][idx][k] for s in seed_map]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        return means, stds

    base_mean, base_std = mean_std(2)
    steer_mean, steer_std = mean_std(3)
    return {
        "ks": ks, "alpha": alpha,
        "base_mean": base_mean, "base_std": base_std,
        "steer_mean": steer_mean, "steer_std": steer_std,
    }


def _ylim(per_model: dict) -> tuple[float, float, float]:
    """(bottom, top, step) with a 0 floor and a little headroom above the data."""
    ymax = 0.0
    for cur in per_model.values():
        for m, s in ((cur["base_mean"], cur["base_std"]), (cur["steer_mean"], cur["steer_std"])):
            ymax = max(ymax, max(a + b for a, b in zip(m, s)))
    step = 0.1 if ymax > 0.25 else 0.05
    top = min(1.0, np.ceil(ymax * 1.12 / step) * step)
    return 0.0, top, step


def _draw_lines(ax, per_model: dict, ks_all: list[int], ylim: tuple, show_ylabel: bool) -> None:
    bottom, top, step = ylim
    for i, model in enumerate(MODELS):
        if model not in per_model:
            continue
        cur = per_model[model]
        ks = cur["ks"]
        # Baseline: dashed, no marker, drawn underneath.
        ax.errorbar(ks, cur["base_mean"], yerr=cur["base_std"], color=COLORS[i],
                    linestyle="--", linewidth=1.3, marker="", capsize=2,
                    elinewidth=0.7, alpha=0.85, zorder=3)
        # Steered: solid, model marker, on top.
        ax.errorbar(ks, cur["steer_mean"], yerr=cur["steer_std"], color=COLORS[i],
                    linestyle="-", linewidth=1.8, marker=MARKERS[i], markersize=4.5,
                    capsize=2, elinewidth=0.7, zorder=4)

    ax.set_xticks(ks_all)
    ax.set_xticklabels([str(k) for k in ks_all])
    margin = (ks_all[-1] - ks_all[0]) * 0.06
    ax.set_xlim(ks_all[0] - margin, ks_all[-1] + margin)
    ax.set_xlabel(r"$k$ (retrieved docs)")

    ax.set_ylim(bottom, top)
    ax.set_yticks(np.arange(bottom, top + 1e-9, step))
    if show_ylabel:
        ax.set_ylabel("End-to-end accuracy", labelpad=6)
    else:
        ax.set_yticklabels([])

    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.tick_params(axis="both", which="both", length=0)
    ax.yaxis.grid(True, linewidth=0.5, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)


def _model_handles(per_model: dict, show_alpha: bool) -> list:
    handles = []
    for i, model in enumerate(MODELS):
        if model not in per_model:
            continue
        lbl = MODEL_LABELS[model]
        if show_alpha:
            lbl += rf" ($\alpha$={per_model[model]['alpha']:g})"
        handles.append(mlines.Line2D([], [], color=COLORS[i], marker=MARKERS[i],
                                      markersize=4.5, linewidth=1.8, label=lbl))
    return handles


def _style_handles() -> list:
    return [
        mlines.Line2D([], [], color="0.35", linestyle="-", linewidth=1.8,
                      label="factuality re-ranking"),
        mlines.Line2D([], [], color="0.35", linestyle="--", linewidth=1.3,
                      label=r"baseline (similarity, $\alpha$=0)"),
    ]


def plot_single(per_model: dict, out_path: Path) -> None:
    """Single-column line plot for one dataset."""
    plt.rcParams.update(_RCPARAMS)
    ks_all = sorted({k for cur in per_model.values() for k in cur["ks"]})
    ylim = _ylim(per_model)

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    _draw_lines(ax, per_model, ks_all, ylim, show_ylabel=True)

    # Model legend above; condition legend tucked inside the (free) upper-left.
    mleg = ax.legend(handles=_model_handles(per_model, show_alpha=True),
                     ncol=2, frameon=False, loc="lower center",
                     bbox_to_anchor=(0.5, 1.0), handlelength=1.4,
                     handletextpad=0.4, columnspacing=1.2)
    ax.add_artist(mleg)
    ax.legend(handles=_style_handles(), frameon=False, loc="upper left",
              handlelength=1.8, fontsize=7)

    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_panel(slices: list[tuple[str, dict]], out_path: Path) -> None:
    """Full-width 1xN panel, one dataset per column, shared y-axis and legend.

    `slices` is a list of (column_label, per_model) in display order.
    """
    plt.rcParams.update(_RCPARAMS)
    n = len(slices)
    ks_all = sorted({k for _, pm in slices for cur in pm.values() for k in cur["ks"]})

    # Shared y-axis across panels so datasets are directly comparable.
    merged = {m: c for _, pm in slices for m, c in pm.items()}
    ylim = _ylim(merged)

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    legend_source = {}
    for j, (col_label, per_model) in enumerate(slices):
        _draw_lines(axes[j], per_model, ks_all, ylim, show_ylabel=(j == 0))
        axes[j].set_title(col_label, fontsize=9, pad=4)
        legend_source.update(per_model)

    # One shared model legend above the whole figure (no per-model alpha here,
    # since alpha can differ across datasets -- list those in the appendix table).
    ordered = {m: legend_source[m] for m in MODELS if m in legend_source}
    fig.legend(handles=_model_handles(ordered, show_alpha=False),
               ncol=len(ordered), frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), handlelength=1.4,
               handletextpad=0.4, columnspacing=1.4)
    axes[0].legend(handles=_style_handles(), frameon=False, loc="upper left",
                   handlelength=1.8, fontsize=7)

    fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.99))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def collect() -> dict:
    """gkey (model, eval, direction, normalize, ...) -> layer -> seed -> file_accuracies."""
    files = sorted(E2E_DIR.rglob("results.jsonl"))
    print(f"Found {len(files)} results.jsonl files.")
    groups: dict[tuple, dict[int, dict[int, tuple]]] = defaultdict(lambda: defaultdict(dict))
    for f in files:
        parts = f.relative_to(E2E_DIR).parts
        seed_idx = next(i for i, p in enumerate(parts) if SEED_RE.fullmatch(p))
        gkey = parts[:seed_idx]
        seed = int(SEED_RE.fullmatch(parts[seed_idx]).group(1))
        layer = int(LAYER_RE.fullmatch(parts[-2]).group(1))
        groups[gkey][layer][seed] = file_accuracies(f)
    return groups


def slice_per_model(groups: dict, eval_ds: str) -> dict:
    """In-domain (direction == eval), chosen NORMALIZE: model -> aggregated curves.

    Assumes gkey = (model, eval, direction, normalize, ...). One layer expected per
    group; if more are present, the lowest-numbered layer is used and a note printed.
    """
    per_model: dict[str, dict] = {}
    for gkey, layer_seed in groups.items():
        if len(gkey) < 4:
            continue
        model, ev, direction, normalize = gkey[0], gkey[1], gkey[2], gkey[3]
        if ev != eval_ds or direction != eval_ds or normalize != NORMALIZE:
            continue
        if model not in MODEL_LABELS:
            continue
        if len(layer_seed) > 1:
            print(f"  note: {'/'.join(gkey)} has {len(layer_seed)} layers "
                  f"{sorted(layer_seed)}; using {min(layer_seed)}")
        layer = min(layer_seed)
        per_model[model] = aggregate_seeds(layer_seed[layer])
        per_model[model]["layer"] = layer
    return per_model


def print_deltas(eval_ds: str, per_model: dict) -> None:
    """Sanity-check the scaling narrative: steered - baseline per model and k."""
    print(f"\n  [{eval_ds}] steered - baseline (delta) by model, in size order:")
    for model in MODELS:
        if model not in per_model:
            continue
        cur = per_model[model]
        deltas = [s - b for s, b in zip(cur["steer_mean"], cur["base_mean"])]
        cells = "  ".join(f"k={k}: {d:+.3f}" for k, d in zip(cur["ks"], deltas))
        print(f"    {MODEL_LABELS[model]:<14} (L{cur['layer']}, "
              f"a={cur['alpha']:g})   {cells}")


def main() -> None:
    groups = collect()

    panel_slices = []
    for ds_key, ds_label in DATASETS:
        per_model = slice_per_model(groups, ds_key)
        if not per_model:
            print(f"No in-domain/{NORMALIZE} data for {ds_key}; skipping.")
            continue
        print_deltas(ds_key, per_model)
        plot_single(per_model, FIG_DIR / f"figure_4_{ds_key}.pdf")
        panel_slices.append((ds_label, per_model))

    if panel_slices:
        plot_panel(panel_slices, FIG_DIR / "figure_4_end_to_end.pdf")


if __name__ == "__main__":
    main()