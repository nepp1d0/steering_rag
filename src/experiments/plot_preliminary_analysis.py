"""
Plot preliminary analysis results.

Per dataset (single figures):
  1. accuracy_all.pdf                 – all questions
  2. accuracy_no_context_correct.pdf  – no-context-correct subset

Combined panels (one PDF each):
  3. accuracy_grid.pdf        – 2x2: {all, subset} x {ConflictQA, NQ-Swap}
  4. figure_2_motivation.pdf  – 1x2: subset only, both datasets (main paper)

Usage:
    python -m src.experiments.plot_preliminary_analysis
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
PA_DIR = RESULTS_DIR / "preliminary_analysis"

# Ordered by parameter count so the capability trend reads left-to-right.
MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "google/gemma-3-4b-it",
    "Qwen/Qwen2-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]
MODEL_LABELS = {
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "meta-llama/Llama-3.2-1B-Instruct": "Llama-3.2-1B",
    "google/gemma-3-4b-it":             "Gemma-3-4B",
    "Qwen/Qwen2-7B-Instruct":           "Qwen2-7B",
}
# Okabe–Ito colourblind-safe palette, same order as MODELS above.
COLORS = ["#E69F00", "#009E73", "#CC79A7", "#0072B2"]

CONDITIONS = [
    "no_context",
    "factual_only",
    "non_factual_only",
    "both_factual_first",
    "both_non_factual_first",
]
CONDITION_LABELS = [
    "No Context",
    "Factual\nOnly",
    "Non-Factual\nOnly",
    "Both\n(Factual First)",
    "Both\n(NF First)",
]
DATASET_TITLES = {"nq_swap": "NQ-Swap", "conflictqa": "ConflictQA"}

_RCPARAMS = {
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,   # embed real (editable) text in the PDF, not outlines
    "ps.fonttype": 42,
}


def safe_model_id(hf_id: str) -> str:
    return hf_id.replace("/", "__")


def is_correct(record: dict) -> bool:
    gt = [str(a).strip() for a in record["ground_truth"][:3]]
    answer = record["generated_answer"].lower()
    return any(a.lower() in answer for a in gt if a)


def load_records(model: str, dataset: str) -> list[dict]:
    path = PA_DIR / safe_model_id(model) / dataset / "results.jsonl"
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_accuracy(records: list[dict], questions: set[str] | None = None) -> dict[str, float]:
    buckets: dict[str, list] = defaultdict(lambda: [0, 0])
    for r in records:
        if questions is not None and r["question"] not in questions:
            continue
        b = buckets[r["condition"]]
        b[0] += int(is_correct(r))
        b[1] += 1
    return {c: (buckets[c][0] / buckets[c][1] if buckets[c][1] > 0 else 0.0) for c in CONDITIONS}


def no_context_correct_questions(records: list[dict]) -> set[str]:
    return {r["question"] for r in records if r["condition"] == "no_context" and is_correct(r)}


def _draw_grouped(ax, acc_by_model: dict, annotate: bool = False, show_xticklabels: bool = True) -> None:
    """Draw one grouped-bar panel into the given axes (no legend/title here)."""
    n_cond = len(CONDITIONS)
    n_models = len(MODELS)
    bar_w = 0.18
    group_w = n_models * bar_w + 0.14
    x = np.arange(n_cond) * group_w

    for i, model in enumerate(MODELS):
        accs = [acc_by_model[model].get(c, 0.0) for c in CONDITIONS]
        offsets = x + (i - (n_models - 1) / 2) * bar_w
        bars = ax.bar(offsets, accs, width=bar_w * 0.9, label=MODEL_LABELS[model],
                      color=COLORS[i], edgecolor="none", zorder=3)
        if annotate:
            for bar, val in zip(bars, accs):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=6, color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels(CONDITION_LABELS if show_xticklabels else [])
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlim(-group_w * 0.55, x[-1] + group_w * 0.55)

    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.tick_params(axis="both", which="both", length=0)
    ax.yaxis.grid(True, linewidth=0.5, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)


def plot_grouped_bar(acc_by_model: dict, title: str, out_path: Path, annotate: bool = False) -> None:
    plt.rcParams.update(_RCPARAMS)

    # Sized for a full-text-width figure* (~7in). For a single \columnwidth
    # figure use roughly figsize=(3.5, 2.4) and set the legend to ncol=2.
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    _draw_grouped(ax, acc_by_model, annotate=annotate, show_xticklabels=True)
    ax.set_ylabel("Accuracy", labelpad=6)
    ax.legend(ncol=len(MODELS), frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), handlelength=1.0,
              handletextpad=0.4, columnspacing=1.4)
    # Title omitted on purpose: let the LaTeX \caption carry it.
    # ax.set_title(title, fontsize=9, pad=24)

    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_grid(
    acc: dict,
    out_path: Path,
    row_keys: list[str],
    row_labels: list[str],
    col_keys: list[str],
    col_labels: list[str],
    annotate: bool = False,
) -> None:
    """Matrix of grouped-bar panels in one PDF.

    `acc` is keyed by (row_key, col_key) -> {model: {condition: accuracy}}.
    Column headers carry the dataset name; row headers (set to "" to hide)
    carry the configuration. One shared legend sits above the whole figure.
    """
    plt.rcParams.update(_RCPARAMS)
    n_rows, n_cols = len(row_keys), len(col_keys)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.3 * n_rows),
                             sharey=True, squeeze=False)

    for r, rk in enumerate(row_keys):
        for c, ck in enumerate(col_keys):
            ax = axes[r][c]
            show_x = (r == n_rows - 1)  # only the bottom row needs condition labels
            _draw_grouped(ax, acc[(rk, ck)], annotate=annotate, show_xticklabels=show_x)
            if r == 0:
                ax.set_title(col_labels[c], fontsize=9, pad=6)
            if c == 0:
                ax.set_ylabel("Accuracy", labelpad=6)
        if row_labels[r]:
            axes[r][-1].annotate(row_labels[r], xy=(1.03, 0.5), xycoords="axes fraction",
                                 rotation=-90, va="center", ha="left",
                                 fontsize=8.5, color="#333333")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(MODELS), frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), handlelength=1.0,
               handletextpad=0.4, columnspacing=1.4)

    fig.tight_layout(pad=0.6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    datasets = ["nq_swap", "conflictqa"]
    grid_acc: dict = {}

    for dataset in datasets:
        all_records = {m: load_records(m, dataset) for m in MODELS}
        ds_label = DATASET_TITLES[dataset]

        # All questions
        acc_all = {m: compute_accuracy(recs) for m, recs in all_records.items()}
        grid_acc[("all", dataset)] = acc_all
        plot_grouped_bar(
            acc_all,
            title=f"Answer Accuracy by Context Condition — {ds_label}",
            out_path=PA_DIR / dataset / "accuracy_all.pdf",
        )

        # No-context-correct subset
        acc_nc = {}
        for model, recs in all_records.items():
            nc_qs = no_context_correct_questions(recs)
            acc_nc[model] = compute_accuracy(recs, questions=nc_qs)
        grid_acc[("nc", dataset)] = acc_nc
        plot_grouped_bar(
            acc_nc,
            title=f"Answer Accuracy (no-context-correct subset) — {ds_label}",
            out_path=PA_DIR / dataset / "accuracy_no_context_correct.pdf",
        )

    # Combined 2x2 panel (all four) — for the appendix / completeness.
    # Rename row labels here if you prefer e.g. "memory-only" over the subset wording.
    plot_grid(
        grid_acc,
        out_path=PA_DIR / "accuracy_grid.pdf",
        row_keys=["all", "nc"],
        row_labels=["All questions", "No-context-correct subset"],
        col_keys=["conflictqa", "nq_swap"],
        col_labels=["ConflictQA", "NQ-Swap"],
    )

    # Main-paper motivation figure: subset row only, both datasets side by side.
    plot_grid(
        grid_acc,
        out_path=PA_DIR / "figure_2_motivation.pdf",
        row_keys=["nc"],
        row_labels=[""],  # caption carries the conditioning; no in-panel row label
        col_keys=["conflictqa", "nq_swap"],
        col_labels=["ConflictQA", "NQ-Swap"],
    )


if __name__ == "__main__":
    main()