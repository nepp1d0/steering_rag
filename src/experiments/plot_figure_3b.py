"""
Figure 3b — LLM-as-judge baseline vs. our internal factuality direction.

ConflictQA only, each model judges itself. Single 2x2 PDF:
  (top-left)  A · head-to-head gold rank gain at alpha=SCATTER_ALPHA (internal vs judge), per model
  (top-right) B · per-query verdict on gold rank: internal wins / ties / loses vs judge
  (bottom-left)  C · internal factuality score: factual vs non-factual documents (REP_MODEL)
  (bottom-right) D · LLM-judge score: factual vs non-factual documents (REP_MODEL)

Reads:
  internal -> results/top_retrieval_evaluation/<model>/<ds>/<ds>/<norm>/seed_*/<proc>/layer_*/
  judge    -> results/llms_scoring_evaluation/<model>/<ds>/seed_*/
  direction-> results/direction_identification/<model>/<ds>/seed_*/<proc>/layer_*/last_pos/direction.pt

Usage:
    python -m src.experiments.plot_figure_3b
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_retrieval_evaluation import RESULTS_DIR, compute_seed_metrics, _agg

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import load_normalized  # noqa: E402

# Optional deps.
try:
    from scipy.stats import gaussian_kde
    HAVE_KDE = True
except Exception:
    HAVE_KDE = False
try:
    from sklearn.metrics import roc_auc_score
    HAVE_AUROC = True
except Exception:
    HAVE_AUROC = False

# ── Config ───────────────────────────────────────────────────────────────────
TOP_DIR = RESULTS_DIR / "top_retrieval_evaluation"
JUDGE_DIR = RESULTS_DIR / "llms_scoring_evaluation"
DIRECTION_DIR = RESULTS_DIR / "direction_identification"
OUT_PATH = RESULTS_DIR / "figures" / "figure_3b_judge_comparison.pdf"

DATASET = "conflictqa"     # this comparison is ConflictQA-only
NORMALIZE = "unnormalized"
PROCEDURE = "context_only"
POSITION = "last_pos"
SCATTER_ALPHA = 0.5
REP_MODEL = "meta-llama__Llama-3.1-8B-Instruct"   # model shown in the density panels

MODELS_BY_SIZE = [
    ("meta-llama__Llama-3.2-1B-Instruct", "Llama-3.2-1B", 1.2),
    ("google__gemma-3-4b-it",             "Gemma-3-4B",   4.3),
    ("Qwen__Qwen2-7B-Instruct",           "Qwen2-7B",     7.6),
    ("meta-llama__Llama-3.1-8B-Instruct", "Llama-3.1-8B", 8.0),
]
COLORS = {
    "meta-llama__Llama-3.2-1B-Instruct": "#FFB300",
    "google__gemma-3-4b-it":             "#00897B",
    "Qwen__Qwen2-7B-Instruct":           "#7E57C2",
    "meta-llama__Llama-3.1-8B-Instruct": "#1E88E5",
}
GOLD_C, NF_C = "#0072B2", "#D55E00"        # factual / non-factual in the density panels
WIN_C, TIE_C, LOSS_C = "#1E88E5", "#CFD8DC", "#D55E00"

RC = {"font.family": "sans-serif", "font.size": 8, "axes.linewidth": 0.6,
      "pdf.fonttype": 42, "ps.fonttype": 42}


# ── small IO helpers ──────────────────────────────────────────────────────────
def _read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _aeq(a, b):
    return abs(float(a) - float(b)) < 1e-6


def internal_seed_paths(model: str) -> list[Path]:
    base = TOP_DIR / model / DATASET / DATASET / NORMALIZE
    return sorted(base.glob(f"seed_*/{PROCEDURE}/layer_*/results.jsonl"))


def judge_seed_paths(model: str) -> list[Path]:
    return sorted((JUDGE_DIR / model / DATASET).glob("seed_*/results.jsonl"))


def _seed_of(path: Path) -> int | None:
    for p in path.parts:
        if p.startswith("seed_"):
            try:
                return int(p.split("_")[1])
            except ValueError:
                return None
    return None


def per_sample_gold_rank(path: Path, alpha: float) -> dict[int, int]:
    """sample_idx -> gold_rank at the given alpha (rank is k-independent; take first)."""
    out: dict[int, int] = {}
    for r in _read_jsonl(path):
        if not _aeq(r["alpha"], alpha):
            continue
        si = r["sample_idx"]
        if si not in out:
            out[si] = r["gold_rank"]
    return out


# ── Panel A & B inputs ──────────────────────────────────────────────────────
def gold_gain(seed_paths: list[Path]) -> tuple[float, float]:
    """Mean ± std (over seeds) of gold rank gain = rank(0) - rank(SCATTER_ALPHA)."""
    metrics = [compute_seed_metrics(p) for p in seed_paths]
    gains = []
    for m in metrics:
        mr = m["mean_gold_rank"]
        if 0.0 in mr and SCATTER_ALPHA in mr:
            gains.append(mr[0.0] - mr[SCATTER_ALPHA])
    if not gains:
        return float("nan"), 0.0
    return _agg(gains)


def per_query_verdict(model: str) -> tuple[float, float, float]:
    """% queries where the internal method ranks gold higher / equal / lower than the judge."""
    int_by_seed = {_seed_of(p): p for p in internal_seed_paths(model)}
    jud_by_seed = {_seed_of(p): p for p in judge_seed_paths(model)}
    common = sorted(set(int_by_seed) & set(jud_by_seed) - {None})
    win = tie = loss = 0
    for seed in common:
        gi = per_sample_gold_rank(int_by_seed[seed], SCATTER_ALPHA)
        gj = per_sample_gold_rank(jud_by_seed[seed], SCATTER_ALPHA)
        for si in set(gi) & set(gj):
            if gi[si] < gj[si]:      # lower rank = better position = internal wins
                win += 1
            elif gi[si] == gj[si]:
                tie += 1
            else:
                loss += 1
    tot = win + tie + loss
    if tot == 0:
        return 0.0, 0.0, 0.0
    return 100 * win / tot, 100 * tie / tot, 100 * loss / tot


# ── Panels C & D inputs (separability of the raw scores) ────────────────────
def _doc_labels(seed: int, docs_path: Path) -> tuple[np.ndarray, list[str]]:
    """Return (label array, texts) where label is +1 factual, -1 non-factual, 0 other."""
    rows = sorted(_read_jsonl(docs_path), key=lambda r: r[0])   # (doc_id, text)
    texts = [r[1] for r in rows]
    samples = load_normalized(DATASET, seed)["test"]
    gold = {s["factual_context"] for s in samples}
    nf = {s["non_factual_evidence"] for s in samples}
    labels = np.array([1 if t in gold else (-1 if t in nf else 0) for t in texts])
    return labels, texts


def internal_scores(model: str) -> tuple[np.ndarray, np.ndarray]:
    """Pooled standardized internal scores for (factual, non-factual) docs across seeds."""
    gold_all, nf_all = [], []
    for path in internal_seed_paths(model):
        seed = _seed_of(path)
        layer = next((p for p in path.parts if p.startswith("layer_")), None)
        if seed is None or layer is None:
            continue
        hidden_p = path.parent / "llm_hidden_states.pt"
        dir_p = (DIRECTION_DIR / model / DATASET / f"seed_{seed}" / PROCEDURE / layer / POSITION / "direction.pt")
        docs_p = path.parent / "docs.jsonl"
        if not (hidden_p.exists() and dir_p.exists() and docs_p.exists()):
            print(f"  [internal] missing files for seed {seed}, skipping")
            continue
        hidden = torch.load(hidden_p, map_location="cpu").float()
        direction = torch.load(dir_p, map_location="cpu").float()
        proj = (hidden @ direction).numpy()
        proj = (proj - proj.mean()) / (proj.std() + 1e-8)   # standardize per seed
        labels, _ = _doc_labels(seed, docs_p)
        gold_all.append(proj[labels == 1]); nf_all.append(proj[labels == -1])
    return (np.concatenate(gold_all) if gold_all else np.array([]),
            np.concatenate(nf_all) if nf_all else np.array([]))


def judge_scores(model: str) -> tuple[np.ndarray, np.ndarray]:
    """Pooled standardized LLM-judge scores for (factual, non-factual) docs across seeds."""
    gold_all, nf_all = [], []
    for sp in judge_seed_paths(model):
        seed = _seed_of(sp)
        scores_p = sp.parent / "llm_scores.pt"
        docs_p = sp.parent / "docs.jsonl"
        if seed is None or not (scores_p.exists() and docs_p.exists()):
            print(f"  [judge] missing files for seed {seed}, skipping")
            continue
        s = torch.load(scores_p, map_location="cpu").float().numpy()
        s = (s - s.mean()) / (s.std() + 1e-8)
        labels, _ = _doc_labels(seed, docs_p)
        gold_all.append(s[labels == 1]); nf_all.append(s[labels == -1])
    return (np.concatenate(gold_all) if gold_all else np.array([]),
            np.concatenate(nf_all) if nf_all else np.array([]))


# ── Styling ───────────────────────────────────────────────────────────────────
def _style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#BDBDBD")
    ax.spines["bottom"].set_color("#BDBDBD")
    ax.tick_params(length=0, labelsize=8)
    if grid_axis != "none":
        ax.grid(axis=grid_axis, color="#ECEFF1", linewidth=0.7)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)


def _density(ax, gold, nf, title):
    if len(gold) == 0 and len(nf) == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color="#999")
        ax.set_title(title, fontsize=9, pad=4)
        _style(ax, grid_axis="none"); ax.set_yticks([]); return
    allv = np.concatenate([gold, nf])
    lo, hi = allv.min(), allv.max(); pad = 0.12 * (hi - lo + 1e-9)
    xs = np.linspace(lo - pad, hi + pad, 200)
    for vals, color, label in [(gold, GOLD_C, "factual"), (nf, NF_C, "non-factual")]:
        if len(vals) < 2:
            continue
        if HAVE_KDE and vals.std() > 0:
            ys = gaussian_kde(vals)(xs)
            ax.fill_between(xs, ys, color=color, alpha=0.22, linewidth=0)
            ax.plot(xs, ys, color=color, linewidth=1.5, label=label)
        else:
            ax.hist(vals, bins=30, density=True, color=color, alpha=0.4, label=label)
    if HAVE_AUROC and len(gold) and len(nf):
        y = np.r_[np.ones(len(gold)), np.zeros(len(nf))]
        try:
            auc = roc_auc_score(y, np.r_[gold, nf])
            ax.text(0.97, 0.95, f"AUROC {auc:.2f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7.5, color="#555")
        except Exception:
            pass
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel("standardized factuality score")
    ax.set_yticks([])
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    _style(ax, grid_axis="none")


# ── Figure ────────────────────────────────────────────────────────────────────
def make_figure() -> None:
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    labels = [lbl for _, lbl, _ in MODELS_BY_SIZE]
    ypos = np.arange(len(MODELS_BY_SIZE))[::-1]   # first model on top

    # A — head-to-head gold rank gain
    axA = axes[0][0]
    h = 0.36
    for i, (model, lbl, _) in enumerate(MODELS_BY_SIZE):
        y = ypos[i]
        gi_m, gi_s = gold_gain(internal_seed_paths(model))
        gj_m, gj_s = gold_gain(judge_seed_paths(model))
        c = COLORS[model]
        axA.barh(y + h / 2, gi_m, height=h, xerr=gi_s, color=c, edgecolor="white",
                 linewidth=0.5, error_kw=dict(elinewidth=0.7, capsize=2), zorder=3)
        axA.barh(y - h / 2, gj_m, height=h, xerr=gj_s, facecolor="none", edgecolor=c,
                 linewidth=1.3, hatch="////", error_kw=dict(elinewidth=0.7, capsize=2), zorder=3)
    axA.axvline(0, color="#CFD8DC", linewidth=0.8)
    axA.set_yticks(ypos); axA.set_yticklabels(labels)
    axA.set_xlabel(f"gold rank gain at $\\alpha={SCATTER_ALPHA:g}$ (positions, $+$ = better)")
    axA.set_title("Head-to-head: gold promotion", fontsize=9, pad=4)
    _style(axA, grid_axis="x")
    axA.legend(handles=[Patch(facecolor="#777", edgecolor="white", label="internal direction"),
                        Patch(facecolor="none", edgecolor="#777", hatch="////", label="LLM-as-judge")],
               fontsize=7, frameon=False, loc="lower right")

    # B — per-query verdict
    axB = axes[0][1]
    for i, (model, lbl, _) in enumerate(MODELS_BY_SIZE):
        y = ypos[i]
        w, t, l = per_query_verdict(model)
        axB.barh(y, w, color=WIN_C, edgecolor="white", linewidth=0.5, zorder=3)
        axB.barh(y, t, left=w, color=TIE_C, edgecolor="white", linewidth=0.5, zorder=3)
        axB.barh(y, l, left=w + t, color=LOSS_C, edgecolor="white", linewidth=0.5, zorder=3)
    axB.set_yticks(ypos); axB.set_yticklabels(labels)
    axB.set_xlim(0, 100)
    axB.set_xlabel("% of queries")
    axB.set_title("Per-query verdict: gold rank", fontsize=9, pad=4)
    _style(axB, grid_axis="x")
    axB.legend(handles=[Patch(facecolor=WIN_C, label="internal wins"),
                        Patch(facecolor=TIE_C, label="tie"),
                        Patch(facecolor=LOSS_C, label="judge wins")],
               fontsize=7, frameon=False, loc="lower right", ncol=1)

    # C / D — score separability for REP_MODEL
    rep_label = dict((m, l) for m, l, _ in MODELS_BY_SIZE).get(REP_MODEL, REP_MODEL)
    gI, nI = internal_scores(REP_MODEL)
    gJ, nJ = judge_scores(REP_MODEL)
    _density(axes[1][0], gI, nI, f"Internal score \u2014 {rep_label}")
    _density(axes[1][1], gJ, nJ, f"LLM-judge score \u2014 {rep_label}")

    fig.suptitle(f"ConflictQA \u2014 internal factuality direction vs. LLM-as-judge", fontsize=10, y=1.02)
    fig.tight_layout(pad=0.9)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


def main() -> None:
    make_figure()


if __name__ == "__main__":
    main()