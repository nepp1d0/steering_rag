"""
RAGuard step F - plots for the pool-reranking evaluation.

Reads results/raguard_retrieval/summary.jsonl (written by raguard_retrieval_evaluation.py)
and produces:

  alpha_sweep_<model>.pdf   4x3 grid, rows = k, cols = supporting/misleading/unrelated.
                            x = alpha, one line per mixed-direction combo, dashed horizontal
                            line = that label's random-ranking baseline (the pool composition).
                            All three labels are shown together on purpose: misleading alone
                            falls with alpha, but so does supporting, and the pair only means
                            something against the random line.
  net_contrast.pdf          one panel per model, y = frac_supporting - frac_misleading vs
                            alpha at k=K_SUMMARY. The single-number "does the direction beat
                            pure SBERT" plot; dashed line = the same contrast under random
                            ranking.
  summary_heatmap.pdf       models x combos, cell = net contrast at alpha=A_SUMMARY minus net
                            contrast at alpha=0, i.e. gain over pure SBERT. Diverging colormap
                            centered at 0: positive = the direction helped.
  controls.pdf              (a) verdict split - frac_misleading vs alpha for verdict_true vs
                            verdict_false on the 3-way mixture, since RAGuard's misleading
                            documents attach asymmetrically to true and false claims;
                            (b) length control - mean_len_topk vs alpha against the pool mean,
                            because supporting/misleading documents are ~30% longer than
                            unrelated ones and a projection that merely sorts by length would
                            show up here.

Means over the 350 queries, no error bars (summary.jsonl stores means only).

Writes to results/raguard_retrieval/figures/. CPU only.

Usage:
    python src/exploratory/plot_raguard_retrieval.py
    python src/exploratory/plot_raguard_retrieval.py --k 1 --alpha 0.7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
from utils import RESULTS_DIR, logger, setup_logging

MODELS = ["google__gemma-3-4b-it", "meta-llama__Llama-3.1-8B-Instruct",
          "meta-llama__Llama-3.2-1B-Instruct", "Qwen__Qwen2-7B-Instruct"]
COMBOS = ["conflictqa", "nq_swap", "longfact",
          "conflictqa+nq_swap", "conflictqa+longfact", "nq_swap+longfact",
          "conflictqa+nq_swap+longfact"]
LABELS = ["supporting", "misleading", "unrelated"]
THREE_WAY = "conflictqa+nq_swap+longfact"

# Same colours / dashed-for-mixture convention as plot_raguard_claim_separation.py.
DD_COLORS = {
    "nq_swap": "#2a78d6", "conflictqa": "#e34948", "longfact": "#3f9a6d",
    "conflictqa+nq_swap": "#8e5fa8", "conflictqa+longfact": "#c47f2e",
    "nq_swap+longfact": "#2f8f9d", "conflictqa+nq_swap+longfact": "#111111",
}
DD_STYLES = {name: ("--" if "+" in name else "-") for name in DD_COLORS}
MODEL_COLORS = dict(zip(MODELS, ["#3f9a6d", "#2a78d6", "#c47f2e", "#8e5fa8"]))
DIVERGING_CMAP = LinearSegmentedColormap.from_list("blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"])

OUT_ROOT = RESULTS_DIR / "raguard_retrieval"
FIG_DIR = OUT_ROOT / "figures"
K_SUMMARY = 3
A_SUMMARY = 0.5


def read_jsonl(path: Path) -> List[Dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def pick(rows: List[Dict], **kw) -> List[Dict]:
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def curve(rows: List[Dict], field: str, **kw) -> tuple[np.ndarray, np.ndarray]:
    """(alphas, field) sorted by alpha for the rows matching kw."""
    sel = sorted(pick(rows, **kw), key=lambda r: r["alpha"])
    return np.array([r["alpha"] for r in sel]), np.array([r[field] for r in sel])


def baseline(rows: List[Dict], field: str, **kw) -> float:
    """Random-ranking baseline: pool composition, constant across alpha/combo/k."""
    sel = pick(rows, **kw)
    return float(np.mean([r[field] for r in sel]))


def plot_alpha_sweep(rows: List[Dict], model: str, ks: List[int]) -> None:
    fig, axes = plt.subplots(len(ks), len(LABELS), figsize=(4.2 * len(LABELS), 2.9 * len(ks)),
                             sharex=True, sharey="col")
    axes = np.atleast_2d(axes)
    for i, k in enumerate(ks):
        for j, label in enumerate(LABELS):
            ax = axes[i, j]
            rnd = baseline(rows, f"random_{label}", model=model, k=k, split="all")
            ax.axhline(rnd, color="#666666", lw=1.2, ls="--",
                       label="random ranking" if (i == 0 and j == 0) else None)
            for combo in COMBOS:
                a, y = curve(rows, f"frac_{label}", model=model, combo=combo, k=k, split="all")
                if not len(a):
                    continue
                ax.plot(a, y, color=DD_COLORS[combo], ls=DD_STYLES[combo], lw=1.8,
                        marker="o", ms=3.5, label=combo if (i == 0 and j == 0) else None)
            ax.grid(alpha=0.25, lw=0.5)
            if i == 0:
                ax.set_title(label, fontsize=11)
            if j == 0:
                ax.set_ylabel(f"k={k}\nfraction in top-k", fontsize=9)
            if i == len(ks) - 1:
                ax.set_xlabel("alpha (0 = SBERT only, 1 = projection only)", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=4, loc="lower center", framealpha=0.9)
    fig.suptitle(f"RAGuard pool reranking - {model}", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    out = FIG_DIR / f"alpha_sweep_{model}.pdf"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info(f"Wrote {out}")


def plot_net_contrast(rows: List[Dict], models: List[str], k: int) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models):
        rnd = (baseline(rows, "random_supporting", model=model, k=k, split="all")
               - baseline(rows, "random_misleading", model=model, k=k, split="all"))
        ax.axhline(rnd, color="#666666", lw=1.2, ls="--", label="random ranking")
        for combo in COMBOS:
            a, sup = curve(rows, "frac_supporting", model=model, combo=combo, k=k, split="all")
            _, mis = curve(rows, "frac_misleading", model=model, combo=combo, k=k, split="all")
            if not len(a):
                continue
            ax.plot(a, sup - mis, color=DD_COLORS[combo], ls=DD_STYLES[combo], lw=1.8,
                    marker="o", ms=3.5, label=combo)
        ax.set_title(model, fontsize=9)
        ax.set_xlabel("alpha")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel(f"supporting - misleading @k={k}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, ncol=4, loc="lower center", framealpha=0.9)
    fig.suptitle(f"Net factuality gain of the retrieved set (k={k})", fontsize=12)
    fig.tight_layout(rect=(0, 0.14, 1, 0.96))
    out = FIG_DIR / "net_contrast.pdf"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info(f"Wrote {out}")


def net_at(rows: List[Dict], model: str, combo: str, k: int, alpha: float) -> float | None:
    sel = pick(rows, model=model, combo=combo, k=k, alpha=alpha, split="all")
    if not sel:
        return None
    return sel[0]["frac_supporting"] - sel[0]["frac_misleading"]


def plot_summary_heatmap(rows: List[Dict], models: List[str], k: int, alpha: float) -> Dict:
    """Gain over pure SBERT: net contrast at `alpha` minus net contrast at alpha=0."""
    grid = np.full((len(models), len(COMBOS)), np.nan)
    for i, model in enumerate(models):
        for j, combo in enumerate(COMBOS):
            hi, lo = net_at(rows, model, combo, k, alpha), net_at(rows, model, combo, k, 0.0)
            if hi is not None and lo is not None:
                grid[i, j] = hi - lo

    span = max(1e-3, float(np.nanmax(np.abs(grid))))
    fig, ax = plt.subplots(figsize=(1.5 * len(COMBOS) + 3.5, 0.7 * len(models) + 2.4))
    im = ax.imshow(grid, cmap=DIVERGING_CMAP,
                   norm=TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span))
    ax.set_xticks(range(len(COMBOS)), COMBOS, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)), models, fontsize=8)
    ax.set_xlabel("direction combo")
    ax.set_title(f"Gain over pure SBERT: net contrast at alpha={alpha} minus alpha=0 (k={k})\n"
                 "positive = the direction improved the retrieved set", fontsize=10)
    for i in range(len(models)):
        for j in range(len(COMBOS)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:+.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="delta net contrast")
    fig.tight_layout()
    out = FIG_DIR / "summary_heatmap.pdf"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info(f"Wrote {out}")
    return {m: {c: (None if np.isnan(grid[i, j]) else float(grid[i, j]))
                for j, c in enumerate(COMBOS)} for i, m in enumerate(models)}


def plot_controls(rows: List[Dict], models: List[str], k: int) -> None:
    fig, (ax_v, ax_l) = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) verdict split on the 3-way mixture: solid = true claims, dotted = false claims.
    for model in models:
        for split, ls in (("verdict_true", "-"), ("verdict_false", ":")):
            a, y = curve(rows, "frac_misleading", model=model, combo=THREE_WAY, k=k, split=split)
            if not len(a):
                continue
            ax_v.plot(a, y, color=MODEL_COLORS[model], ls=ls, lw=1.7, marker="o", ms=3.5,
                      label=f"{model.split('__')[-1]} ({split.replace('verdict_', '')})")
    ax_v.set_xlabel("alpha")
    ax_v.set_ylabel(f"frac misleading @k={k}")
    ax_v.set_title(f"Verdict split ({THREE_WAY})", fontsize=10)
    ax_v.legend(fontsize=6.5, ncol=2, framealpha=0.9)
    ax_v.grid(alpha=0.25, lw=0.5)

    # (b) length control: does the projection just sort by document length?
    for model in models:
        a, y = curve(rows, "mean_len_topk", model=model, combo=THREE_WAY, k=k, split="all")
        if not len(a):
            continue
        ax_l.plot(a, y, color=MODEL_COLORS[model], lw=1.7, marker="o", ms=3.5,
                  label=model.split("__")[-1])
    pool_len = baseline(rows, "mean_len_pool", combo=THREE_WAY, k=k, split="all")
    ax_l.axhline(pool_len, color="#666666", lw=1.2, ls="--", label=f"pool mean ({pool_len:.0f})")
    ax_l.set_xlabel("alpha")
    ax_l.set_ylabel(f"mean document length in top-{k} (words)")
    ax_l.set_title(f"Length control ({THREE_WAY})", fontsize=10)
    ax_l.legend(fontsize=7, framealpha=0.9)
    ax_l.grid(alpha=0.25, lw=0.5)

    fig.tight_layout()
    out = FIG_DIR / "controls.pdf"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info(f"Wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot the RAGuard pool-reranking evaluation.")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--k", type=int, default=K_SUMMARY, help="k for the summary figures.")
    ap.add_argument("--alpha", type=float, default=A_SUMMARY, help="alpha for the heatmap.")
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging("plot_raguard_retrieval", OUT_ROOT)

    summary_path = OUT_ROOT / "summary.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found. Run raguard_retrieval_evaluation.py first.")
    rows = read_jsonl(summary_path)
    models = [m for m in args.models if any(r["model"] == m for r in rows)]
    ks = sorted({r["k"] for r in rows})
    logger.info(f"{len(rows)} summary rows | models={models} | ks={ks} "
                f"| alphas={sorted({r['alpha'] for r in rows})}")

    for model in models:
        plot_alpha_sweep(rows, model, ks)
    plot_net_contrast(rows, models, args.k)
    gains = plot_summary_heatmap(rows, models, args.k, args.alpha)
    plot_controls(rows, models, args.k)

    (FIG_DIR / "summary.json").write_text(json.dumps(
        {"k": args.k, "alpha": args.alpha, "gain_over_sbert": gains}, indent=2))

    # Readout: is any (model, combo) better than pure SBERT at this alpha?
    best = max(((g, m, c) for m, d in gains.items() for c, g in d.items() if g is not None),
               default=None)
    if best is not None:
        logger.info(f"Best gain over SBERT at alpha={args.alpha}, k={args.k}: "
                    f"{best[0]:+.4f} ({best[1]} / {best[2]})")
    logger.info("Done.")


if __name__ == "__main__":
    main()
