"""
Figure 6 - end-to-end generalization: re-ranking ClashEval pools with a factuality direction
identified on a different corpus.

One row, four panels (models ordered by capacity). Each panel plots Accuracy@1 of the target
document inside its frozen 12-document pool against the fusion weight alpha, for three
identification datasets. alpha=0 is pure SBERT relevance and carries no model or direction term,
so all curves meet at the same point in every panel.

Two analytic reference lines, both properties of the pool composition alone (12 documents, of
which 4 are on-topic and 5 are uncorrupted):
    Random         = 1/12 = 0.083
    Relevance only = 1/4  = 0.250   <- a perfect relevance ranker cannot exceed this

Reads (read-only) the per-model result JSONs written by
agents_work/code/clasheval_pool_ranking_v2.py. Plotting only: no GPU, no recomputation.

Usage:
    python src/exploratory/plot_figure_6_end_to_end_generalization.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
# Source data still lives under agents_work/; both paths are single constants so they can be
# repointed in one line once the data and the rest of the code move.
DATA_DIR = REPO_ROOT / "results" / "clasheval_pool_ranking_v2"
OUT_PATH = REPO_ROOT / "agents_work" / "figures" / "figure_6_end_to_end_generalization.pdf"

# ── Config ───────────────────────────────────────────────────────────────────
MODELS_BY_SIZE = [
    ("meta-llama__Llama-3.2-1B-Instruct", "Llama-3.2-1B"),
    ("google__gemma-3-4b-it",             "Gemma-3-4B"),
    ("Qwen__Qwen2-7B-Instruct",           "Qwen2-7B"),
    ("meta-llama__Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
]
DATASETS = ["nq_swap", "conflictqa", "longfact"]
DATASET_LABELS = {"nq_swap": "NQ-Swap", "conflictqa": "ConflictQA", "longfact": "LongFact"}
COLORS = {
    "nq_swap":    "#1E88E5",   # Blue 600
    "conflictqa": "#D55E00",   # vermillion, matches figure 3b's non-factual hue
    "longfact":   "#00897B",   # Teal 600
}

RANDOM_BASELINE = 1 / 12
RELEVANCE_ONLY = 1 / 4
RANDOM_C, RELEVANCE_C = "#B0BEC5", "#607D8B"

XLIM = (-0.03, 1.03)
YLIM = (0.0, 0.45)          # data span incl. CIs is 0.011-0.417
XTICKS = [0.0, 0.5, 1.0]

RC = {
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


# ── Data ─────────────────────────────────────────────────────────────────────
def load_results() -> dict[str, dict]:
    """model key -> the pool-ranking result JSON for that model."""
    data = {}
    for key, label in MODELS_BY_SIZE:
        path = DATA_DIR / f"pool_ranking_v2_results__{key}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing results for {label}: {path}")
        data[key] = json.loads(path.read_text())
    return data


def curve(model_data: dict, dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """alphas, Accuracy@1, CI low, CI high for one (model, identification dataset).

    "precision_at_1" in the stored JSON is the fraction of pools whose top-1 document is the
    target. There is exactly one correct document per pool, so precision@1, recall@1 and
    accuracy@1 are the same quantity; the figure uses the unambiguous name.
    """
    alphas = model_data["alphas"]
    table = model_data["results"][dataset]
    acc = np.array([table[str(a)]["precision_at_1"] for a in alphas])
    lo = np.array([table[str(a)]["p1_ci_lo"] for a in alphas])
    hi = np.array([table[str(a)]["p1_ci_hi"] for a in alphas])
    return np.array(alphas), acc, lo, hi


# ── Styling ──────────────────────────────────────────────────────────────────
def _style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#BDBDBD")
    ax.spines["bottom"].set_color("#BDBDBD")
    ax.tick_params(length=0, labelsize=8)
    ax.grid(axis=grid_axis, color="#ECEFF1", linewidth=0.7)
    ax.set_axisbelow(True)


# ── Figure ───────────────────────────────────────────────────────────────────
def make_figure(data: dict[str, dict]) -> None:
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2), sharey=True)

    for ax, (key, label) in zip(axes, MODELS_BY_SIZE):
        ax.axhline(RANDOM_BASELINE, color=RANDOM_C, linestyle=":", linewidth=1.0, zorder=1)
        ax.axhline(RELEVANCE_ONLY, color=RELEVANCE_C, linestyle="--", linewidth=1.0, zorder=1)

        for ds in DATASETS:
            a, acc, lo, hi = curve(data[key], ds)
            ax.fill_between(a, lo, hi, color=COLORS[ds], alpha=0.12, linewidth=0, zorder=2)
            ax.plot(a, acc, marker="o", markersize=3.5, linewidth=1.8, color=COLORS[ds],
                    markeredgecolor="white", markeredgewidth=0.4, zorder=3)

        ax.set_title(label, fontsize=9, pad=4)
        ax.set_xlabel(r"$\alpha$")
        ax.set_xlim(*XLIM)
        ax.set_xticks(XTICKS)
        _style(ax)

    axes[0].set_ylim(*YLIM)
    axes[0].set_ylabel("Accuracy@1")

    handles = [Line2D([0], [0], color=COLORS[ds], marker="o", linewidth=1.8, markersize=3.5,
                      markeredgecolor="white", markeredgewidth=0.4, label=DATASET_LABELS[ds])
               for ds in DATASETS]
    handles += [
        Line2D([0], [0], color=RANDOM_C, linestyle=":", linewidth=1.0, label="Random"),
        Line2D([0], [0], color=RELEVANCE_C, linestyle="--", linewidth=1.0, label="Relevance only"),
    ]
    fig.legend(handles=handles, ncol=5, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=1.6)

    fig.tight_layout(pad=0.8)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


def main() -> None:
    data = load_results()
    print(f"Loaded {len(data)} models from {DATA_DIR}.")
    make_figure(data)


if __name__ == "__main__":
    main()
