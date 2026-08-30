"""
Top row of figure 3, rebuilt from the attention-mask-fix re-run.

Same panels as the top row of plot_figure3.py -- mean gold rank and mean non-factual rank
vs alpha, log scale, one line per model -- but:
  * reads results/retrieval_evaluation_maskfix/ instead of top_retrieval_evaluation/,
  * picks each model's layer by RANK SEPARATION at SEPARATION_ALPHA (plot_figure3 selects
    layers with score_layer, which deliberately drops the separation term),
  * no confidence bands: the re-run has a single seed, so a band would be zero-width.

Rank separation at alpha (same quantity as plot_figure3's bottom row):

    gold_gain = mean_gold_rank[0] - mean_gold_rank[alpha]   (> 0: gold climbed)
    nf_change = mean_nf_rank[0]   - mean_nf_rank[alpha]     (< 0: non-factual sank)
    separation = gold_gain - nf_change

Note on alpha: at alpha=1.0 the score is the projection alone, so the ranking is the same
for every question and the separation is large while top-k recall collapses. That is why
selection uses a fixed SEPARATION_ALPHA rather than the best alpha over all of them.

The config block mirrors plot_figure3.py, so with the same DIRECTION_DATASET / POSITION /
TOP_ROW_DATASET the output is directly comparable to figure_3_reranking.pdf.

Outputs:
    results/figures_maskfix/<direction>/[<position>/]figure_3_top_row_maskfix.pdf
    results/figures_maskfix/<direction>/[<position>/]best_layers.json

Usage:
    python src/exploratory/plot_figure3_maskfix.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
from plot_retrieval_evaluation import RESULTS_DIR, compute_seed_metrics

MASKFIX_DIR = RESULTS_DIR / "retrieval_evaluation_maskfix"

# ── Config (mirrors plot_figure3.py) ──────────────────────────────────────────
NORMALIZE = "unnormalized"
PROCEDURE = "context_only"
DIRECTION_DATASET = "longfact"    # which identification dataset the directions come from
POSITION = "entity_pos"             # "last_pos" or "entity_pos"
TOP_ROW_DATASET = "nq_swap"    # which eval dataset's curves are plotted
SEED = 42                         # the re-run has this seed only
# Alpha at which rank separation is measured when choosing each model's best layer.
SEPARATION_ALPHA = 0.3

MODELS_BY_SIZE = [
    ("meta-llama__Llama-3.2-1B-Instruct", "Llama-3.2-1B"),
    ("google__gemma-3-4b-it",             "Gemma-3-4B"),
    ("Qwen__Qwen2-7B-Instruct",           "Qwen2-7B"),
    ("meta-llama__Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
]
COLORS = {
    "meta-llama__Llama-3.2-1B-Instruct": "#FFB300",
    "google__gemma-3-4b-it":             "#00897B",
    "Qwen__Qwen2-7B-Instruct":           "#7E57C2",
    "meta-llama__Llama-3.1-8B-Instruct": "#1E88E5",
}
DATASET_LABELS = {"conflictqa": "ConflictQA", "nq_swap": "NQ-Swap", "longfact": "LongFact"}

RC = {
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

_FIG_DIR = RESULTS_DIR / "figures_maskfix" / DIRECTION_DATASET
if POSITION != "last_pos":
    _FIG_DIR = _FIG_DIR / POSITION
OUT_PATH = _FIG_DIR / "figure_3_top_row_maskfix.pdf"
LAYERS_PATH = _FIG_DIR / "best_layers.json"


def parse_parts(path: Path) -> dict | None:
    """Layout: model/eval/direction/normalize/seed_N/procedure/layer_M/position/results.jsonl"""
    p = path.relative_to(MASKFIX_DIR).parts
    if len(p) < 9:
        return None
    return {"model": p[0], "eval": p[1], "direction": p[2], "normalize": p[3],
            "seed": p[4], "procedure": p[5], "layer": int(p[6].split("_")[1]), "position": p[7]}


def rank_separation(metrics: dict, alpha: float) -> float | None:
    """gold rank gain minus non-factual rank change at `alpha`; higher is better."""
    g, n = metrics["mean_gold_rank"], metrics["mean_nf_rank"]
    if 0.0 not in g or alpha not in g:
        return None
    return (g[0.0] - g[alpha]) - (n[0.0] - n[alpha])


def best_layer_per_model() -> dict[str, dict]:
    """For the configured (eval, direction, position, seed): the highest-separation layer."""
    per_model: dict[str, list] = defaultdict(list)
    for f in sorted(MASKFIX_DIR.rglob("results.jsonl")):
        meta = parse_parts(f)
        if meta is None:
            continue
        if (meta["eval"] != TOP_ROW_DATASET or meta["direction"] != DIRECTION_DATASET
                or meta["position"] != POSITION or meta["normalize"] != NORMALIZE
                or meta["procedure"] != PROCEDURE or meta["seed"] != f"seed_{SEED}"):
            continue
        m = compute_seed_metrics(f)
        if not m["has_rank"]:
            continue
        sep = rank_separation(m, SEPARATION_ALPHA)
        if sep is None:
            continue
        per_model[meta["model"]].append({"layer": meta["layer"], "separation": sep, "metrics": m})

    best: dict[str, dict] = {}
    for model, rows in per_model.items():
        rows.sort(key=lambda r: -r["separation"])
        best[model] = {"layer": rows[0]["layer"], "separation": rows[0]["separation"],
                       "metrics": rows[0]["metrics"],
                       "ranking": [{"layer": r["layer"], "separation": round(r["separation"], 3)}
                                   for r in rows]}
    return best


def _style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#BDBDBD")
    ax.spines["bottom"].set_color("#BDBDBD")
    ax.tick_params(length=0, labelsize=8)
    ax.grid(axis="y", color="#ECEFF1", linewidth=0.7)
    ax.set_axisbelow(True)


def make_figure(best: dict[str, dict]) -> None:
    plt.rcParams.update(RC)
    fig, (ax_g, ax_n) = plt.subplots(1, 2, figsize=(7.2, 2.8))

    handles, labels = [], []
    for model, label in MODELS_BY_SIZE:
        d = best.get(model)
        if not d:
            continue
        m = d["metrics"]
        alphas = m["alphas"]
        for ax, key in [(ax_g, "mean_gold_rank"), (ax_n, "mean_nf_rank")]:
            line, = ax.plot(alphas, [m[key][a] for a in alphas], marker="o", markersize=3.5,
                            linewidth=1.8, color=COLORS[model],
                            markeredgecolor="white", markeredgewidth=0.4)
        handles.append(line)
        labels.append(f"{label} (L{d['layer']})")

    ax_g.set_yscale("log")
    ax_n.set_yscale("log")
    ax_g.set_title(f"Gold document — {DATASET_LABELS[TOP_ROW_DATASET]}", fontsize=9, pad=4)
    ax_n.set_title(f"Non-factual document — {DATASET_LABELS[TOP_ROW_DATASET]}", fontsize=9, pad=4)
    ax_g.set_ylabel("mean rank (log)\nlower = better")
    ax_n.set_ylabel("mean rank (log)\nhigher = better")
    for ax in (ax_g, ax_n):
        ax.set_xlabel(r"$\alpha$")
        _style(ax)

    fig.legend(handles, labels, ncol=4, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=1.6)
    fig.tight_layout(pad=0.8)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


def main() -> None:
    best = best_layer_per_model()
    print(f"Config: eval={TOP_ROW_DATASET} | direction={DIRECTION_DATASET} | "
          f"position={POSITION} | seed={SEED} | separation alpha={SEPARATION_ALPHA}")
    if not best:
        print(f"No results matched under {MASKFIX_DIR}.")
        return

    print(f"\n{'model':>34} {'best layer':>11} {'separation':>11}   layer ranking (layer:sep)")
    for model, label in MODELS_BY_SIZE:
        d = best.get(model)
        if not d:
            print(f"{model:>34} {'-':>11} {'-':>11}   (no results)")
            continue
        rank_str = "  ".join(f"L{r['layer']}:{r['separation']:.1f}" for r in d["ranking"])
        print(f"{model:>34} {d['layer']:>11} {d['separation']:>11.2f}   {rank_str}")

    make_figure(best)

    LAYERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAYERS_PATH.write_text(json.dumps({
        "eval": TOP_ROW_DATASET, "direction": DIRECTION_DATASET, "position": POSITION,
        "seed": SEED, "separation_alpha": SEPARATION_ALPHA,
        "best": {m: {"layer": d["layer"], "separation": d["separation"], "ranking": d["ranking"]}
                 for m, d in best.items()},
    }, indent=2))
    print(f"Wrote {LAYERS_PATH}")


if __name__ == "__main__":
    main()
