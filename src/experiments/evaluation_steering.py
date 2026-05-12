"""
Step 3 - Evaluate steering runs.

For every `runs.jsonl` produced by `steering.py` under
    results/steering/<model>/<eval_dataset>/<id_dataset>/<procedure>/layer_<L>/<position>/runs.jsonl
build a bar plot comparing baseline vs steered accuracy, split by document order
(factual-first vs non_factual-first), and save it under the mirrored path
    results/steering_evaluation/<...>/accuracy.png

Accuracy is computed by simple keyword matching: a generation is correct if at
least one ground-truth keyword appears (case-insensitive) in the generation.

Usage:
    python -m src.experiments.evaluation_steering
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def is_correct(generation: str, ground_truth: list) -> bool:
    """Return True if any ground-truth keyword appears in the generation (case-insensitive)."""
    g = generation.lower()
    return any(kw.lower() in g for kw in ground_truth)


def plot_runs(runs_path: Path, out_path: Path) -> None:
    """Read a runs.jsonl and save a 4-bar accuracy plot (2 doc-orders x 2 conditions)."""
    # buckets[order_key] = [n_baseline_correct, n_steered_correct, total]
    buckets = {"factual-non_factual": [0, 0, 0], "non_factual-factual": [0, 0, 0]}
    with runs_path.open() as f:
        for line in f:
            r = json.loads(line)
            key = "-".join(r["doc_order"])
            buckets[key][0] += int(is_correct(r["baseline_generation"], r["ground_truth"]))
            buckets[key][1] += int(is_correct(r["steered_generation"], r["ground_truth"]))
            buckets[key][2] += 1

    orders = list(buckets.keys())
    # Accuracy = correct / total (guard against empty buckets).
    base_acc = [buckets[o][0] / buckets[o][2] if buckets[o][2] else 0.0 for o in orders]
    steer_acc = [buckets[o][1] / buckets[o][2] if buckets[o][2] else 0.0 for o in orders]
    totals = [buckets[o][2] for o in orders]

    x = [0, 1]
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - w / 2 for i in x], base_acc, w, label="baseline", color="#4C72B0")
    ax.bar([i + w / 2 for i in x], steer_acc, w, label="steered", color="#DD8452")

    # Annotate each bar with its bucket sample count (same total for baseline/steered of an order).
    for i in x:
        ax.text(i - w / 2, base_acc[i], f"n={totals[i]}", ha="center", va="bottom", fontsize=9)
        ax.text(i + w / 2, steer_acc[i], f"n={totals[i]}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(orders)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_xlabel("document order")
    ax.set_title("/".join(runs_path.relative_to(RESULTS_DIR / "steering").parts[:-1]), fontsize=8)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    src_root = RESULTS_DIR / "steering"
    dst_root = RESULTS_DIR / "steering_evaluation"
    runs = sorted(src_root.rglob("runs.jsonl"))
    print(f"Found {len(runs)} runs.jsonl files.")
    for r in runs:
        out = dst_root / r.relative_to(src_root).parent / "accuracy.png"
        plot_runs(r, out)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
