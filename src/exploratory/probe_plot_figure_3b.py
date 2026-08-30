"""
Figure 3b for the LOGISTIC-REGRESSION PROBE directions - probe direction vs. LLM-as-judge.

Same layout as plot_figure_3b.py, with the "internal" side coming from the probe tree and
the judge baseline reused as-is (the LLM-as-judge is method-independent). Emits BOTH
positions (last_pos, entity_pos).

ConflictQA only, each model judges itself. Produces one 2x2 PDF per position:
  (top-left)     Panel A - gold rank gain at alpha=0.5: internal direction vs judge, per model
  (top-right)    Panel B - per-query verdict on gold rank: internal wins / ties / loses vs judge
  (bottom-left)  Panel C - internal direction score: factual vs non-factual documents (REP_MODEL)
  (bottom-right) Panel D - LLM-judge score: factual vs non-factual documents (REP_MODEL)

Reads:
  internal  -> results/probes/top_retrieval_evaluation/<model>/<ds>/<ds>/<norm>/seed_*/<proc>/layer_*/<position>/
  judge     -> results/llms_scoring_evaluation/<model>/<ds>/seed_*/           (method-independent baseline)
  direction -> results/probes/direction_identification/<model>/<ds>/seed_*/<proc>/layer_*/<position>/direction.pt

Outputs:
  results/probes/figures/[<DIRECTION_DATASET>/][<position>/]figure_3b_judge_comparison.pdf

Usage:
    python src/exploratory/probe_plot_figure_3b.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Put this script's own dir on sys.path so the bare import below resolves whether the
# script is run from src/experiments/ or as `python src/exploratory/probe_plot_figure_3b.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Two helpers reused from the probe retrieval plotting script:
#   compute_seed_metrics(path) -> dict with "mean_gold_rank": {alpha: mean rank}, ...
#   _agg(list_of_values)       -> (mean, std) across seeds
from probe_plot_retrieval_evaluation import RESULTS_DIR, compute_seed_metrics, _agg  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
from utils import load_normalized  # noqa: E402

# gaussian_kde gives the smooth density curves in panels C/D.
# If scipy is missing we fall back to plain histograms.
try:
    from scipy.stats import gaussian_kde
    HAVE_KDE = True
except Exception:
    HAVE_KDE = False

# ── Config ───────────────────────────────────────────────────────────────────
TOP_DIR = RESULTS_DIR / "probes" / "top_retrieval_evaluation"
JUDGE_DIR = RESULTS_DIR / "llms_scoring_evaluation"     # method-independent baseline (not under probes/)
DIRECTION_DIR = RESULTS_DIR / "probes" / "direction_identification"
DATASET = "conflictqa"     # this comparison is ConflictQA-only
# Which direction (identification) dataset the internal method uses:
#   "same"    -> in-domain: direction == eval (reproduces the original figure)
#   <dataset> -> use that direction (e.g. "longfact") evaluated on DATASET
DIRECTION_DATASET = "conflictqa"
DIRECTION_DS = DATASET if DIRECTION_DATASET == "same" else DIRECTION_DATASET
NORMALIZE = "unnormalized"
PROCEDURE = "context_only"
# Direction positions: both are emitted. "last_pos" keeps the base output path; other
# positions ("entity_pos") add an extra output level.
POSITIONS = ["last_pos", "entity_pos"]
SCATTER_ALPHA = 0.3        # mixing weight used for the head-to-head panels A and B
REP_MODEL = "meta-llama__Llama-3.1-8B-Instruct"   # model shown in the density panels C/D


def out_path_for(position: str) -> Path:
    fig_dir = RESULTS_DIR / "probes" / "figures"
    if DIRECTION_DATASET != "same":
        fig_dir = fig_dir / DIRECTION_DATASET
    if position != "last_pos":
        fig_dir = fig_dir / position
    return fig_dir / "figure_3b_judge_comparison.pdf"


# Models listed small -> large; each row is (folder name, short label, size in B params).
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
def read_jsonl(path: Path):
    """Yield one parsed object per non-empty line of a .jsonl file."""
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def approx_equal(a, b) -> bool:
    """True if two floats are equal up to a tiny tolerance (alphas are stored as floats)."""
    return abs(float(a) - float(b)) < 1e-6


def standardize(values: np.ndarray) -> np.ndarray:
    """Center to mean 0 and scale to std 1 (so internal and judge scores are comparable)."""
    return (values - values.mean()) / (values.std() + 1e-8)


def internal_seed_paths(model: str, position: str) -> list[Path]:
    """All per-seed results.jsonl files for the probe direction method."""
    base = TOP_DIR / model / DATASET / DIRECTION_DS / NORMALIZE
    return sorted(base.glob(f"seed_*/{PROCEDURE}/layer_*/{position}/results.jsonl"))


def judge_seed_paths(model: str) -> list[Path]:
    """All per-seed results.jsonl files for the LLM-as-judge baseline."""
    return sorted((JUDGE_DIR / model / DATASET).glob("seed_*/results.jsonl"))


def seed_of(path: Path):
    """Pull the integer seed out of a path like .../seed_42/...  (None if not found)."""
    for part in path.parts:
        if part.startswith("seed_"):
            try:
                return int(part.split("_")[1])
            except ValueError:
                return None
    return None


def per_sample_gold_rank(path: Path, alpha: float) -> dict:
    """Map sample_idx -> gold_rank at the given alpha.

    gold_rank does not depend on k, so for each sample we just keep the first row we see.
    """
    out = {}
    for r in read_jsonl(path):
        if not approx_equal(r["alpha"], alpha):
            continue
        si = r["sample_idx"]
        if si not in out:
            out[si] = r["gold_rank"]
    return out


# ── Panel A & B inputs ──────────────────────────────────────────────────────
def gold_gain(seed_paths: list[Path]) -> tuple[float, float]:
    """Mean +/- std (across seeds) of gold rank gain = rank(alpha=0) - rank(alpha=0.5).

    Positive = re-ranking moved the gold document to a better (lower) rank.
    """
    gains = []
    for path in seed_paths:
        mean_rank = compute_seed_metrics(path)["mean_gold_rank"]   # {alpha: mean gold rank}
        if 0.0 in mean_rank and SCATTER_ALPHA in mean_rank:
            gains.append(mean_rank[0.0] - mean_rank[SCATTER_ALPHA])
    if not gains:
        return float("nan"), 0.0
    return _agg(gains)


def per_query_verdict(model: str, position: str) -> tuple[float, float, float]:
    """For each query, does the internal method rank gold better than the judge?

    Returns the percentage of queries where internal wins / ties / loses.
    A lower rank means a better position, so internal wins when its rank is smaller.
    """
    internal_by_seed = {seed_of(p): p for p in internal_seed_paths(model, position)}
    judge_by_seed = {seed_of(p): p for p in judge_seed_paths(model)}

    # Only compare seeds that exist for both methods (and drop a missing/None seed).
    common_seeds = set(internal_by_seed) & set(judge_by_seed)
    common_seeds.discard(None)

    win = tie = loss = 0
    for seed in sorted(common_seeds):
        internal_rank = per_sample_gold_rank(internal_by_seed[seed], SCATTER_ALPHA)
        judge_rank = per_sample_gold_rank(judge_by_seed[seed], SCATTER_ALPHA)
        for si in set(internal_rank) & set(judge_rank):
            if internal_rank[si] < judge_rank[si]:
                win += 1
            elif internal_rank[si] == judge_rank[si]:
                tie += 1
            else:
                loss += 1

    total = win + tie + loss
    if total == 0:
        return 0.0, 0.0, 0.0
    return 100 * win / total, 100 * tie / total, 100 * loss / total


# ── Panels C & D inputs (separability of the raw scores) ────────────────────
def doc_labels(seed: int, docs_path: Path) -> np.ndarray:
    """Label each document: +1 if it is a factual context, -1 if non-factual, 0 otherwise.

    docs.jsonl stores one [doc_id, text] pair per line. We sort by doc_id so the order
    matches the rows of the saved score / hidden-state tensors.
    """
    rows = sorted(read_jsonl(docs_path), key=lambda pair: pair[0])   # [doc_id, text]
    texts = [text for _, text in rows]

    samples = load_normalized(DATASET, seed)["test"]
    factual_texts = {s["factual_context"] for s in samples}
    nonfactual_texts = {s["non_factual_evidence"] for s in samples}

    labels = []
    for text in texts:
        if text in factual_texts:
            labels.append(1)
        elif text in nonfactual_texts:
            labels.append(-1)
        else:
            labels.append(0)
    return np.array(labels)


def internal_scores(model: str, position: str) -> tuple[np.ndarray, np.ndarray]:
    """Standardized probe direction scores, pooled across seeds.

    Returns (scores for factual docs, scores for non-factual docs).
    Score = (document hidden state) dot (factuality direction).
    """
    factual_scores, nonfactual_scores = [], []
    for path in internal_seed_paths(model, position):
        seed = seed_of(path)
        layer = next((p for p in path.parts if p.startswith("layer_")), None)
        if seed is None or layer is None:
            continue

        # results.jsonl sits in the position subdir; shared tensors live one level up (layer dir).
        hidden_p = path.parent.parent / "llm_hidden_states.pt"
        direction_p = DIRECTION_DIR / model / DIRECTION_DS / f"seed_{seed}" / PROCEDURE / layer / position / "direction.pt"
        docs_p = path.parent.parent / "docs.jsonl"
        if not (hidden_p.exists() and direction_p.exists() and docs_p.exists()):
            print(f"  [internal] missing files for seed {seed}, skipping")
            continue

        hidden = torch.load(hidden_p, map_location="cpu").float()       # [n_docs, d_model]
        direction = torch.load(direction_p, map_location="cpu").float() # [d_model]
        proj = (hidden @ direction).numpy()                             # [n_docs]
        proj = standardize(proj)

        labels = doc_labels(seed, docs_p)
        factual_scores.append(proj[labels == 1])
        nonfactual_scores.append(proj[labels == -1])

    factual = np.concatenate(factual_scores) if factual_scores else np.array([])
    nonfactual = np.concatenate(nonfactual_scores) if nonfactual_scores else np.array([])
    return factual, nonfactual


def judge_scores(model: str) -> tuple[np.ndarray, np.ndarray]:
    """Standardized LLM-judge scores, pooled across seeds.

    Returns (scores for factual docs, scores for non-factual docs).
    Score = the 0-1 factuality value the model assigned to each document.
    """
    factual_scores, nonfactual_scores = [], []
    for path in judge_seed_paths(model):
        seed = seed_of(path)
        scores_p = path.parent / "llm_scores.pt"
        docs_p = path.parent / "docs.jsonl"
        if seed is None or not (scores_p.exists() and docs_p.exists()):
            print(f"  [judge] missing files for seed {seed}, skipping")
            continue

        scores = torch.load(scores_p, map_location="cpu").float().numpy()   # [n_docs]
        scores = standardize(scores)

        labels = doc_labels(seed, docs_p)
        factual_scores.append(scores[labels == 1])
        nonfactual_scores.append(scores[labels == -1])

    factual = np.concatenate(factual_scores) if factual_scores else np.array([])
    nonfactual = np.concatenate(nonfactual_scores) if nonfactual_scores else np.array([])
    return factual, nonfactual


# ── Styling ───────────────────────────────────────────────────────────────────
def style_axes(ax, grid_axis="y"):
    """Light, consistent axis styling: hide top/right spines, soft grid, no tick marks."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#BDBDBD")
    ax.spines["bottom"].set_color("#BDBDBD")
    ax.tick_params(length=0, labelsize=8)
    if grid_axis != "none":
        ax.grid(axis=grid_axis, color="#ECEFF1", linewidth=0.7)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)


def plot_density(ax, factual, nonfactual, title):
    """Draw overlaid factual vs non-factual score distributions in one panel."""
    if len(factual) == 0 and len(nonfactual) == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color="#999")
        ax.set_title(title, fontsize=9, pad=4)
        style_axes(ax, grid_axis="none")
        ax.set_yticks([])
        return

    # Shared x range covering both groups, with a little padding on each side.
    all_values = np.concatenate([factual, nonfactual])
    lo, hi = all_values.min(), all_values.max()
    pad = 0.12 * (hi - lo + 1e-9)
    xs = np.linspace(lo - pad, hi + pad, 200)

    for values, color, label in [(factual, GOLD_C, "factual"),
                                 (nonfactual, NF_C, "non-factual")]:
        if len(values) < 2:
            continue
        if HAVE_KDE and values.std() > 0:
            ys = gaussian_kde(values)(xs)
            ax.fill_between(xs, ys, color=color, alpha=0.22, linewidth=0)
            ax.plot(xs, ys, color=color, linewidth=1.5, label=label)
        else:
            ax.hist(values, bins=30, density=True, color=color, alpha=0.4, label=label)

    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel("standardized factuality score")
    # y is a probability density (each curve integrates to 1). The absolute height
    # is not independently meaningful, so we label the axis but hide the numeric ticks.
    ax.set_ylabel("density")
    ax.set_yticks([])
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    # vertical background grid at the score ticks (y-ticks are hidden, so a y-grid
    # would be invisible) — matches the grid style of panels A and B.
    style_axes(ax, grid_axis="x")


# ── Figure ────────────────────────────────────────────────────────────────────
def make_figure(position: str) -> None:
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    labels = [label for _, label, _ in MODELS_BY_SIZE]
    ypos = np.arange(len(MODELS_BY_SIZE))[::-1]   # reverse so the first model sits on top

    # ── Panel A — head-to-head gold rank gain ───────────────────────────────
    # Two bars per model: solid = internal direction, hatched outline = LLM-as-judge.
    axA = axes[0][0]
    bar_h = 0.36
    for i, (model, label, _) in enumerate(MODELS_BY_SIZE):
        y = ypos[i]
        internal_mean, internal_std = gold_gain(internal_seed_paths(model, position))
        judge_mean, judge_std = gold_gain(judge_seed_paths(model))
        color = COLORS[model]
        axA.barh(y + bar_h / 2, internal_mean, height=bar_h, xerr=internal_std,
                 color=color, edgecolor="white", linewidth=0.5,
                 error_kw=dict(elinewidth=0.7, capsize=2), zorder=3)
        axA.barh(y - bar_h / 2, judge_mean, height=bar_h, xerr=judge_std,
                 facecolor="none", edgecolor=color, linewidth=1.3, hatch="////",
                 error_kw=dict(elinewidth=0.7, capsize=2), zorder=3)
    axA.axvline(0, color="#CFD8DC", linewidth=0.8)
    axA.set_yticks(ypos)
    axA.set_yticklabels(labels)
    axA.set_xlabel(f"gold rank gain at $\\alpha={SCATTER_ALPHA:g}$ (positions, $+$ = better)")
    axA.set_title("Head-to-head: gold promotion", fontsize=9, pad=4)
    style_axes(axA, grid_axis="x")
    # Legend swatches are generic gray (not per-model), so draw two invisible
    # zero-width bars purely to create the two labelled entries.
    axA.barh(0, 0, color="#777", edgecolor="white", label="internal direction")
    axA.barh(0, 0, facecolor="none", edgecolor="#777", hatch="////", label="LLM-as-judge")
    axA.legend(fontsize=7, frameon=False, loc="lower left")

    # ── Panel B — per-query verdict ─────────────────────────────────────────
    # One stacked bar per model: win (blue) + tie (gray) + loss (orange) = 100%.
    axB = axes[0][1]
    for i, (model, label, _) in enumerate(MODELS_BY_SIZE):
        y = ypos[i]
        win, tie, loss = per_query_verdict(model, position)
        # Only label the first model's segments so the legend has exactly three entries.
        axB.barh(y, win, color=WIN_C, edgecolor="white", linewidth=0.5, zorder=3,
                 label="internal wins" if i == 0 else None)
        axB.barh(y, tie, left=win, color=TIE_C, edgecolor="white", linewidth=0.5, zorder=3,
                 label="tie" if i == 0 else None)
        axB.barh(y, loss, left=win + tie, color=LOSS_C, edgecolor="white", linewidth=0.5, zorder=3,
                 label="judge wins" if i == 0 else None)
    axB.set_yticks(ypos)
    axB.set_yticklabels(labels)
    axB.set_xlim(0, 100)
    axB.set_xlabel("% of queries")
    axB.set_title("Per-query verdict: gold rank", fontsize=9, pad=4)
    style_axes(axB, grid_axis="x")
    axB.legend(fontsize=7, frameon=False, loc="lower right", ncol=1)

    # ── Panels C & D — score separability for the representative model ───────
    rep_label = dict((m, l) for m, l, _ in MODELS_BY_SIZE).get(REP_MODEL, REP_MODEL)
    internal_factual, internal_nonfactual = internal_scores(REP_MODEL, position)
    judge_factual, judge_nonfactual = judge_scores(REP_MODEL)
    plot_density(axes[1][0], internal_factual, internal_nonfactual, f"Internal score — {rep_label}")
    plot_density(axes[1][1], judge_factual, judge_nonfactual, f"LLM-judge score — {rep_label}")

    direction_note = "" if DIRECTION_DATASET == "same" else f" ({DIRECTION_DS} direction)"
    position_note = "" if position == "last_pos" else f" [{position}]"
    fig.suptitle(f"ConflictQA — probe factuality direction{direction_note} vs. LLM-as-judge{position_note}",
                 fontsize=10, y=1.02)
    fig.tight_layout(pad=0.9)
    out_path = out_path_for(position)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    for position in POSITIONS:
        make_figure(position)


if __name__ == "__main__":
    main()
