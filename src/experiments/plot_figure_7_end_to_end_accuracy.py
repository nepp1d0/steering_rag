"""
Figure 7 - end-to-end answer accuracy on ClashEval, the companion to figure 6.

Figure 6 shows the fused score retrieves the uncorrupted document more often. This shows what
that buys downstream: the top-k documents are handed to the same model, which answers the
question, and the answer is checked against ClashEval's numeric ground truth.

Two rows (k=1, k=2) x four columns (models, ordered by capacity). Each panel plots end-to-end
accuracy against the fusion weight alpha, with one line per identification dataset -- the same
three series, in the same colours, as figure 6.

alpha=0 is similarity-only retrieval and carries no direction term, so all three series meet
there; that shared value is the baseline and is drawn as a single gray dashed line per panel
(figure 4's baseline convention, on figure 6's axis). Above the dashed line = better than plain
relevance. The shared alpha=0 value is asserted, as a build check, to be identical across the
three directions.

Bands are 95% bootstrap intervals over the 477 questions, 2,000 resamples, with the resample
indices SHARED across every direction, alpha, k and model, so all comparisons are paired.

Reads the JSONL written by src/experiments/clasheval_end_to_end_generation.py. Plotting only.

Usage:
    python src/experiments/plot_figure_7_end_to_end_accuracy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.append(str(Path(__file__).resolve().parent))
from utils import RESULTS_DIR  # noqa: E402

GEN_DIR = RESULTS_DIR / "clasheval_end_to_end"
FIG_DIR = RESULTS_DIR / "figures" / "clasheval"
OUT_PATH = FIG_DIR / "figure_7_end_to_end_accuracy.pdf"

# ── Config ───────────────────────────────────────────────────────────────────
MODELS_BY_SIZE = [
    ("meta-llama__Llama-3.2-1B-Instruct", "Llama-3.2-1B"),
    ("google__gemma-3-4b-it",             "Gemma-3-4B"),
    ("Qwen__Qwen2-7B-Instruct",           "Qwen2-7B"),
    ("meta-llama__Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
]
DATASETS = ["nq_swap", "conflictqa", "longfact"]
DATASET_LABELS = {"nq_swap": "NQ-Swap", "conflictqa": "ConflictQA", "longfact": "LongFact"}
COLORS = {                                  # identical to figure 6
    "nq_swap":    "#1E88E5",
    "conflictqa": "#D55E00",
    "longfact":   "#00897B",
}
KS = [1, 2]
BASELINE_C = "#607D8B"

PRIMARY_ALPHA = 0.5          # pre-committed, inherited from the RAGuard setting
N_BOOT = 2000
BOOT_SEED = 0
XLIM = (-0.03, 1.03)
YLIM = (0.0, 0.42)
XTICKS = [0.0, 0.5, 1.0]

RC = {
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


# ── Data ─────────────────────────────────────────────────────────────────────
def load_model(key: str) -> tuple[list[float], dict[tuple[str, int, float], np.ndarray]]:
    """alphas, and {(direction, k, alpha): per-question 0/1 correct}, shared question order."""
    path = GEN_DIR / f"end_to_end__{key}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing generations: {path}")
    rows = [json.loads(l) for l in path.open() if l.strip()]

    have = {r.get("direction") for r in rows}
    missing = set(DATASETS) - have
    if missing:
        raise ValueError(f"{path.name} has directions {sorted(have)}; missing {sorted(missing)}. "
                         f"Re-run clasheval_end_to_end_generation.py for this model.")

    questions = sorted({r["question"] for r in rows})
    q_index = {q: i for i, q in enumerate(questions)}
    alphas = sorted({r["alpha"] for r in rows})

    out: dict[tuple[str, int, float], np.ndarray] = {}
    for ds in DATASETS:
        for k in KS:
            for a in alphas:
                out[(ds, k, a)] = np.full(len(questions), np.nan)
    for r in rows:
        out[(r["direction"], r["k"], r["alpha"])][q_index[r["question"]]] = float(r["label"] == "correct")
    for key_, arr in out.items():
        assert not np.isnan(arr).any(), f"missing cells for {key_} in {key}"

    # build check: alpha=0 has no direction term, so the three series must be identical there
    for k in KS:
        base = out[(DATASETS[0], k, alphas[0])]
        for ds in DATASETS[1:]:
            assert np.array_equal(base, out[(ds, k, alphas[0])]), \
                f"alpha=0 differs between {DATASETS[0]} and {ds} at k={k} in {key}"
    return alphas, out


def band(values: np.ndarray, boot_idx: np.ndarray) -> tuple[float, float, float]:
    draws = values[boot_idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


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
def make_figure() -> None:
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.2), sharey=True, sharex=True)

    rng = np.random.default_rng(BOOT_SEED)
    boot_idx = None

    for ci, (key, label) in enumerate(MODELS_BY_SIZE):
        alphas, cells = load_model(key)
        if boot_idx is None:      # one shared resample matrix for every panel, direction, k, alpha
            n_q = len(cells[(DATASETS[0], KS[0], alphas[0])])
            boot_idx = rng.integers(0, n_q, size=(N_BOOT, n_q))

        for ri, k in enumerate(KS):
            ax = axes[ri][ci]
            # baseline: shared alpha=0 value (asserted identical across the three directions)
            ax.axhline(cells[(DATASETS[0], k, alphas[0])].mean(), color=BASELINE_C,
                       linestyle="--", linewidth=0.9, alpha=0.7, zorder=1)

            for ds in DATASETS:
                pts = [band(cells[(ds, k, a)], boot_idx) for a in alphas]
                y = np.array([p[0] for p in pts])
                lo = np.array([p[1] for p in pts])
                hi = np.array([p[2] for p in pts])
                ax.fill_between(alphas, lo, hi, color=COLORS[ds], alpha=0.12, linewidth=0, zorder=2)
                ax.plot(alphas, y, marker="o", markersize=3.5, linewidth=1.8, color=COLORS[ds],
                        markeredgecolor="white", markeredgewidth=0.4, zorder=3)

            if ri == 0:
                ax.set_title(label, fontsize=9, pad=4)
            if ri == len(KS) - 1:
                ax.set_xlabel(r"$\alpha$")
            if ci == 0:
                ax.set_ylabel(f"End-to-end accuracy\n$k = {k}$")
            ax.set_xlim(*XLIM)
            ax.set_xticks(XTICKS)
            _style(ax)

    axes[0][0].set_ylim(*YLIM)

    handles = [Line2D([0], [0], color=COLORS[ds], marker="o", linewidth=1.8, markersize=3.5,
                      markeredgecolor="white", markeredgewidth=0.4, label=DATASET_LABELS[ds])
               for ds in DATASETS]
    handles.append(Line2D([0], [0], color=BASELINE_C, linestyle="--", linewidth=0.9,
                          label=r"baseline (similarity only, $\alpha$=0)"))
    fig.legend(handles=handles, ncol=4, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=1.6)

    fig.tight_layout(pad=0.8)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


def print_deltas() -> None:
    """Paired delta vs alpha=0, with a paired bootstrap CI, at two alphas:

      - PRIMARY_ALPHA=0.5, pre-committed before any ClashEval result existed. This is the
        honest headline: no alpha was chosen on this data.
      - each series' best alpha, which IS selected on the evaluation data and is therefore
        optimistically biased. Printed as a diagnostic, never as the claim.
    """
    rng = np.random.default_rng(BOOT_SEED)
    boot_idx = None
    for key, label in MODELS_BY_SIZE:
        alphas, cells = load_model(key)
        if boot_idx is None:
            n_q = len(cells[(DATASETS[0], KS[0], alphas[0])])
            boot_idx = rng.integers(0, n_q, size=(N_BOOT, n_q))
        for k in KS:
            for ds in DATASETS:
                base = cells[(ds, k, alphas[0])]
                means = {a: cells[(ds, k, a)].mean() for a in alphas}
                best = max(alphas, key=lambda x: means[x])
                for tag, a in (("pre-committed a=0.5", PRIMARY_ALPHA),
                               (f"best a={best:.1f} (selected on data)", best)):
                    d = cells[(ds, k, a)] - base
                    draws = d[boot_idx].mean(axis=1)
                    lo, hi = np.percentile(draws, 2.5), np.percentile(draws, 97.5)
                    # resolvable in EITHER direction; a CI wholly below 0 is a real decrease
                    verdict = "resolvable" if (lo > 0 or hi < 0) else "CI includes 0"
                    print(f"  {label:<14} k={k} {DATASET_LABELS[ds]:<11} {tag:<34} "
                          f"{means[alphas[0]]:.3f} -> {means[a]:.3f}  "
                          f"delta={d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]  {verdict}")


def main() -> None:
    print_deltas()
    make_figure()


if __name__ == "__main__":
    main()
