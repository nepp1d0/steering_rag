"""
Figure 7, bar variant - end-to-end answer accuracy on ClashEval, laid out by direction x model.

Same data, same metric and the same build checks as plot_figure_7_end_to_end_accuracy.py
(which it imports from and does NOT overwrite): only the rendering differs.

    rows    = identification dataset (NQ-Swap / ConflictQA / LongFact)
    columns = model, ordered by capacity
    bars    = one per fusion weight alpha, grouped in pairs for k=1 (full colour) and
              k=2 (lightened), with 95% bootstrap CIs

alpha=0 is similarity-only retrieval and carries no direction term: it is the same value in
all three rows, so its bars are drawn in the neutral baseline colour and repeated as two
dashed reference lines (k=1 dashed, k=2 dotted). Bars above their line beat plain relevance.
The shared alpha=0 value is asserted identical across the three directions by load_model().

Bands are 95% bootstrap intervals over the 477 questions, 2,000 resamples, with the resample
indices SHARED across every direction, alpha, k and model, so all comparisons are paired.

Also writes a single-alpha reduction of the same bars (make_simple_figure): the sweep collapsed
to ALPHA_FIXED, so x carries the model and the panels carry the direction. Same k=1/k=2 pairing,
same CIs, same alpha=0 reference. That alpha is chosen post hoc -- see its sidecar.

Plotting only: reads the JSONL written by src/experiments/clasheval_end_to_end_generation.py.

Usage:
    python src/experiments/plot_figure_7_end_to_end_accuracy_bars.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.append(str(Path(__file__).resolve().parent))
from plot_figure_7_end_to_end_accuracy import (  # noqa: E402
    BASELINE_C,
    BOOT_SEED,
    COLORS,
    DATASETS,
    DATASET_LABELS,
    FIG_DIR,
    KS,
    MODELS_BY_SIZE,
    N_BOOT,
    RC,
    YLIM,
    _style,
    band,
    load_model,
)

OUT_PATH = FIG_DIR / "figure_7_end_to_end_accuracy_bars.pdf"

K_LIGHTEN = 0.45         # k=2 bars: fraction of the row colour kept when blending toward white
GROUP_WIDTH = 0.86       # total width of the k=1/k=2 pair at each alpha
FIGSIZE = (9.6, 6.0)

# Single-alpha variant: same bars, sweep collapsed to one operating point.
ALPHA_FIXED = 0.3
SIMPLE_OUT_PATH = FIG_DIR / "figure_7_end_to_end_accuracy_alpha03.pdf"
SIMPLE_FIGSIZE = (7.6, 2.8)


def lighten(color: str, keep: float) -> tuple:
    """Blend toward white; `keep`=1 is the original colour, 0 is white."""
    return tuple(1.0 - (1.0 - c) * keep for c in to_rgb(color))


def make_figure() -> None:
    plt.rcParams.update(RC)
    nrow, ncol = len(DATASETS), len(MODELS_BY_SIZE)
    fig, axes = plt.subplots(nrow, ncol, figsize=FIGSIZE, sharey=True, sharex=True)

    rng = np.random.default_rng(BOOT_SEED)
    boot_idx = None
    width = GROUP_WIDTH / len(KS)

    for ci, (key, label) in enumerate(MODELS_BY_SIZE):
        alphas, cells = load_model(key)
        if boot_idx is None:   # one shared resample matrix for every panel, direction, k, alpha
            n_q = len(cells[(DATASETS[0], KS[0], alphas[0])])
            boot_idx = rng.integers(0, n_q, size=(N_BOOT, n_q))
        x = np.arange(len(alphas))

        for ri, ds in enumerate(DATASETS):
            ax = axes[ri][ci]

            # One baseline per k: the shared alpha=0 value, which carries no direction term.
            for k, ls in zip(KS, ("--", ":")):
                ax.axhline(cells[(ds, k, alphas[0])].mean(), color=BASELINE_C,
                           linestyle=ls, linewidth=0.9, alpha=0.7, zorder=1)

            for ki, k in enumerate(KS):
                offs = (ki - (len(KS) - 1) / 2) * width
                pts = [band(cells[(ds, k, a)], boot_idx) for a in alphas]
                y = np.array([p[0] for p in pts])
                lo = np.array([p[1] for p in pts])
                hi = np.array([p[2] for p in pts])
                keep = 1.0 if ki == 0 else K_LIGHTEN
                cols = [lighten(BASELINE_C if a == 0.0 else COLORS[ds], keep) for a in alphas]
                ax.bar(x + offs, y, width=width * 0.9, color=cols,
                       edgecolor="white", linewidth=0.3, zorder=2)
                ax.errorbar(x + offs, y, yerr=[y - lo, hi - y], fmt="none",
                            ecolor="#455A64", elinewidth=0.5, capsize=0.8, zorder=3)

            if ri == 0:
                ax.set_title(label, fontsize=9, pad=4)
            if ri == nrow - 1:
                ax.set_xlabel(r"$\alpha$")
            if ci == 0:
                ax.set_ylabel(f"{DATASET_LABELS[ds]}\nend-to-end accuracy")
            ax.set_xticks(x[::2])
            ax.set_xticklabels([f"{a:g}" for a in alphas[::2]], fontsize=7)
            _style(ax)

    axes[0][0].set_ylim(*YLIM)

    handles = [Patch(facecolor=COLORS[ds], edgecolor="white", label=DATASET_LABELS[ds])
               for ds in DATASETS]
    handles.append(Patch(facecolor=BASELINE_C, edgecolor="white",
                         label=r"$\alpha$=0 (similarity only)"))
    handles += [Patch(facecolor=lighten("#455A64", 1.0), edgecolor="white", label=f"$k$={KS[0]}"),
                Patch(facecolor=lighten("#455A64", K_LIGHTEN), edgecolor="white",
                      label=f"$k$={KS[1]} (lighter)")]
    fig.legend(handles=handles, ncol=6, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=1.6)

    fig.tight_layout(pad=0.8)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


def make_simple_figure() -> None:
    """Same bars as make_figure(), with the alpha sweep collapsed to ALPHA_FIXED.

    With one alpha there is nothing left to put on x, so x carries the model and the panels
    carry the direction: the 3x4 grid above becomes 1x3. The grammar is unchanged -- k=1 in
    full colour, k=2 lightened, paired 95% CIs, alpha=0 as a dashed (k=1) / dotted (k=2)
    reference. The reference is drawn as one segment per model group rather than an axhline,
    because the alpha=0 value differs by model (it is shared only across directions).
    """
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, len(DATASETS), figsize=SIMPLE_FIGSIZE, sharey=True)

    loaded = []
    for key, label in MODELS_BY_SIZE:
        alphas, cells = load_model(key)
        a_fix = min(alphas, key=lambda a: abs(a - ALPHA_FIXED))
        assert np.isclose(a_fix, ALPHA_FIXED), \
            f"alpha={ALPHA_FIXED} is not on the grid for {key}; nearest is {a_fix}"
        loaded.append((label, cells, alphas[0], a_fix))

    rng = np.random.default_rng(BOOT_SEED)   # same seed//N_BOOT as make_figure: paired with it
    n_q = len(loaded[0][1][(DATASETS[0], KS[0], loaded[0][2])])
    boot_idx = rng.integers(0, n_q, size=(N_BOOT, n_q))

    x = np.arange(len(loaded))
    width = GROUP_WIDTH / len(KS)

    for di, ds in enumerate(DATASETS):
        ax = axes[di]
        for ki, (k, ls) in enumerate(zip(KS, ("--", ":"))):
            offs = (ki - (len(KS) - 1) / 2) * width
            pts = [band(cells[(ds, k, a_fix)], boot_idx) for _, cells, _, a_fix in loaded]
            y = np.array([p[0] for p in pts])
            lo = np.array([p[1] for p in pts])
            hi = np.array([p[2] for p in pts])
            ax.bar(x + offs, y, width=width * 0.9,
                   color=lighten(COLORS[ds], 1.0 if ki == 0 else K_LIGHTEN),
                   edgecolor="white", linewidth=0.3, zorder=2)
            ax.errorbar(x + offs, y, yerr=[y - lo, hi - y], fmt="none",
                        ecolor="#455A64", elinewidth=0.5, capsize=0.8, zorder=3)
            base = np.array([cells[(ds, k, a0)].mean() for _, cells, a0, _ in loaded])
            ax.hlines(base, x - GROUP_WIDTH / 2, x + GROUP_WIDTH / 2, color=BASELINE_C,
                      linestyle=ls, linewidth=0.9, alpha=0.7, zorder=4)

        ax.set_title(DATASET_LABELS[ds], fontsize=9, pad=4)
        if di == 0:
            ax.set_ylabel("End-to-end accuracy\n" + rf"$\alpha$ = {ALPHA_FIXED:g}")
        ax.set_xticks(x)
        ax.set_xticklabels([lb for lb, _, _, _ in loaded], fontsize=6.5, rotation=25, ha="right")
        _style(ax)

    axes[0].set_ylim(*YLIM)

    handles = [
        Patch(facecolor=lighten("#455A64", 1.0), edgecolor="white", label=f"$k$={KS[0]}"),
        Patch(facecolor=lighten("#455A64", K_LIGHTEN), edgecolor="white",
              label=f"$k$={KS[1]} (lighter)"),
        Line2D([0], [0], color=BASELINE_C, linestyle="--", linewidth=0.9,
               label=rf"$\alpha$=0, $k$={KS[0]}"),
        Line2D([0], [0], color=BASELINE_C, linestyle=":", linewidth=0.9,
               label=rf"$\alpha$=0, $k$={KS[1]}"),
    ]
    fig.legend(handles=handles, ncol=4, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=1.6)

    fig.tight_layout(pad=0.8)
    SIMPLE_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SIMPLE_OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {SIMPLE_OUT_PATH}")


def write_sidecar() -> None:
    lines = [
        f"# {OUT_PATH.name} — end-to-end accuracy on ClashEval, by direction x model",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}  ",
        "Script: `src/experiments/plot_figure_7_end_to_end_accuracy_bars.py`  ",
        "Bar re-rendering of `figure_7_end_to_end_accuracy.pdf`; that figure is not overwritten.",
        "",
        "## What is plotted",
        "",
        "Rows = identification dataset, columns = model (ordered by capacity), one bar per fusion",
        "weight alpha, grouped in pairs: k=1 in full colour, k=2 lightened. Bar height is the",
        "fraction of the 477 ClashEval questions answered correctly when the top-k documents of the",
        "fused ranking are handed to the same model.",
        "",
        "    score(doc, q) = (1 - alpha) * z(sbert_cos) + alpha * z(projection onto v_fact)",
        "",
        "alpha=0 is similarity-only retrieval and carries no direction term, so it is one shared",
        "value across the three rows: drawn in the neutral baseline colour and repeated as two",
        "dashed reference lines (k=1 dashed, k=2 dotted). `load_model()` asserts that the alpha=0",
        "per-question outcomes are identical across the three directions, as a build check.",
        "",
        "Error bars are 95% bootstrap intervals over the 477 questions, 2,000 resamples, with the",
        "resample indices shared across every direction, alpha, k and model, so all comparisons are",
        "paired.",
        "",
        "## Data",
        "",
        "Source: `results/clasheval_end_to_end/end_to_end__<model>.jsonl`, written by",
        "`src/experiments/clasheval_end_to_end_generation.py`. Plotting only — no regeneration,",
        "no GPU. Data loading, the alpha=0 build check, the bootstrap and the palette are imported",
        "from `plot_figure_7_end_to_end_accuracy.py` rather than reimplemented.",
        "",
        f"- models: {', '.join(lb for _, lb in MODELS_BY_SIZE)}",
        f"- identification datasets: {', '.join(DATASET_LABELS[d] for d in DATASETS)}",
        "- alphas: 0.0 to 1.0 in steps of 0.1 (11 bars per panel per k)",
        f"- k: {', '.join(str(k) for k in KS)}",
        "- questions: 477, frozen 12-document pool per question",
        "",
        "## Selection protocol",
        "",
        "None on this figure: every alpha is shown, so nothing is selected on the evaluation data.",
        "The pre-committed headline alpha is 0.5 (inherited from the RAGuard setting, fixed before",
        "any ClashEval result existed). `plot_figure_7_end_to_end_accuracy.py --` prints the paired",
        "deltas at both alpha=0.5 and each series' best alpha, the latter being optimistically",
        "biased and reported only as a diagnostic.",
        "",
        "## Reading",
        "",
        "Bars above their own dashed/dotted baseline beat similarity-only retrieval. The shape to",
        "read is the interior peak: for Gemma-3-4B, Qwen2-7B and Llama-3.1-8B accuracy rises over",
        "alpha ~0.1-0.4 and then collapses toward alpha=1, where the ranking becomes",
        "query-independent (pure projection scores every question identically). Llama-3.2-1B is",
        "flat-to-declining everywhere, consistent with its direction carrying little usable signal.",
        "The three rows are near-identical in shape, which is the generalization claim: the peak",
        "does not depend on which corpus the direction was identified on.",
        "",
        "Overlapping CIs at neighbouring alphas mean the exact location of the peak is not",
        "resolvable; the presence of the peak, and the collapse at high alpha, are.",
        "",
    ]
    OUT_PATH.with_suffix(".md").write_text("\n".join(lines))
    print(f"Wrote {OUT_PATH.with_suffix('.md')}")


def write_simple_sidecar() -> None:
    lines = [
        f"# {SIMPLE_OUT_PATH.name} — end-to-end accuracy on ClashEval at a single alpha",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}  ",
        "Script: `src/experiments/plot_figure_7_end_to_end_accuracy_bars.py`  ",
        f"Single-alpha reduction of `figure_7_end_to_end_accuracy_bars.pdf`; neither that figure",
        "nor `figure_7_end_to_end_accuracy.pdf` is overwritten.",
        "",
        "## What is plotted",
        "",
        f"The same bars as the full-sweep figure, restricted to alpha = {ALPHA_FIXED:g}. With one",
        "alpha there is nothing left for the x-axis, so x carries the model (ordered by capacity)",
        "and the three panels carry the identification dataset. Within each model group: k=1 in",
        "full colour, k=2 lightened, the same pairing as the full-sweep figure. Bar height is the",
        "fraction of the 477 ClashEval questions answered correctly when the top-k documents of",
        "the fused ranking are handed to the same model.",
        "",
        "    score(doc, q) = (1 - alpha) * z(sbert_cos) + alpha * z(projection onto v_fact)",
        "",
        "alpha=0 is similarity-only retrieval and carries no direction term. It is drawn as a",
        "reference segment over each model group — dashed for k=1, dotted for k=2 — rather than as",
        "an axhline, because the alpha=0 value differs by model; it is shared only across the three",
        "directions, and `load_model()` asserts that sharing as a build check.",
        "",
        "Error bars are 95% bootstrap intervals over the 477 questions, 2,000 resamples, drawn from",
        "the same BOOT_SEED and N_BOOT as the companion figures, so the intervals are paired with",
        "them and across every direction, k and model here.",
        "",
        "## Data",
        "",
        "Source: `results/clasheval_end_to_end/end_to_end__<model>.jsonl`, written by",
        "`src/experiments/clasheval_end_to_end_generation.py`. Plotting only — no regeneration, no",
        "GPU. Data loading, the alpha=0 build check, the bootstrap and the palette are imported from",
        "`plot_figure_7_end_to_end_accuracy.py` rather than reimplemented.",
        "",
        f"- models: {', '.join(lb for _, lb in MODELS_BY_SIZE)}",
        f"- identification datasets: {', '.join(DATASET_LABELS[d] for d in DATASETS)}",
        f"- alpha: {ALPHA_FIXED:g} only, plus alpha=0 as the reference line",
        f"- k: {', '.join(str(k) for k in KS)}",
        "- questions: 477, frozen 12-document pool per question",
        "",
        "## Selection protocol",
        "",
        f"**alpha = {ALPHA_FIXED:g} is fixed post hoc and is NOT the pre-committed value.** The",
        "pre-committed headline alpha is 0.5 (`PRIMARY_ALPHA` in",
        "`plot_figure_7_end_to_end_accuracy.py`), inherited from the RAGuard setting and fixed",
        f"before any ClashEval result existed. {ALPHA_FIXED:g} sits inside the peak region (~0.1-0.4)",
        "visible in the full sweep, so it is chosen with the evaluation data in view and any effect",
        "size read off this figure is optimistically biased. This figure is a presentational",
        "reduction chosen post hoc: quote it alongside the full sweep",
        "(`figure_7_end_to_end_accuracy.pdf`), which shows every alpha and selects nothing, and use",
        "`python src/experiments/plot_figure_7_end_to_end_accuracy.py` for the paired deltas at the",
        "pre-committed alpha=0.5.",
        "",
        "## Reading",
        "",
        "Bars above their own dashed (k=1) / dotted (k=2) segment beat similarity-only retrieval at",
        "that model. The comparison the figure is for is across models at a single operating point:",
        "whether the gain over the alpha=0 reference holds as capacity grows, and whether the three",
        "panels agree. Near-identical panels are the generalization claim — the gain does not depend",
        "on which corpus the direction was identified on.",
        "",
        "Overlapping CIs between two bars mean their difference is not resolvable at 477 questions.",
        "",
    ]
    SIMPLE_OUT_PATH.with_suffix(".md").write_text("\n".join(lines))
    print(f"Wrote {SIMPLE_OUT_PATH.with_suffix('.md')}")


def main() -> None:
    make_figure()
    write_sidecar()
    make_simple_figure()
    write_simple_sidecar()


if __name__ == "__main__":
    main()
