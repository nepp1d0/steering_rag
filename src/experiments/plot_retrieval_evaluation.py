"""
Plot retrieval evaluation results.

Auto-discovers all results.jsonl under results/retrieval_evaluation/ and saves
a two-panel line plot (gold recall@k and non-factual rate@k vs k, one line per α)
next to each results file as retrieval_plot.png.

Usage:
    python -m src.experiments.plot_retrieval_evaluation
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def plot_results(results_path: Path, out_path: Path) -> None:
    # {(alpha, k): [gold_hits, nonfactual_hits, total]}
    buckets: dict[tuple, list] = defaultdict(lambda: [0, 0, 0])
    # {(alpha, sample_idx): (gold_rank, nf_rank)} — populated from smallest k to deduplicate
    rank_records: list[dict] = []
    with results_path.open() as f:
        for line in f:
            r = json.loads(line)
            key = (r["alpha"], r["k"])
            buckets[key][0] += int(r["gold_in_topk"])
            buckets[key][1] += int(r["nonfactual_in_topk"])
            buckets[key][2] += 1
            if "gold_rank" in r:
                rank_records.append(r)

    alphas = sorted({k[0] for k in buckets})
    ks = sorted({k[1] for k in buckets})

    has_rank = bool(rank_records)
    fig, axes = plt.subplots(1, 3 if has_rank else 2, figsize=(15 if has_rank else 10, 4), sharey=False)
    ax1, ax2 = axes[0], axes[1]

    for alpha in alphas:
        gold_rates = [buckets[(alpha, k)][0] / buckets[(alpha, k)][2] for k in ks]
        nf_rates   = [buckets[(alpha, k)][1] / buckets[(alpha, k)][2] for k in ks]
        label = f"α={alpha:.1f}" + (" (baseline)" if alpha == 0.0 else "")
        ls = "--" if alpha == 0.0 else "-"
        ax1.plot(ks, gold_rates, marker="o", linestyle=ls, label=label)
        ax2.plot(ks, nf_rates,   marker="o", linestyle=ls, label=label)

    title = "/".join(results_path.relative_to(RESULTS_DIR / "retrieval_evaluation").parts[:-1])
    for ax, ylabel in [(ax1, "gold recall@k"), (ax2, "non-factual rate@k")]:
        ax.set_xlabel("k")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ks)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.set_title(title, fontsize=7)

    if has_rank:
        ax3 = axes[2]
        first_k = ks[0]
        p_vals = []
        for alpha in alphas:
            rows = [r for r in rank_records if r["alpha"] == alpha and r["k"] == first_k]
            p_vals.append(sum(r["gold_rank"] < r["nonfactual_rank"] for r in rows) / len(rows))
        ax3.plot(alphas, p_vals, marker="o", color="steelblue")
        ax3.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="random")
        ax3.set_xlabel("α")
        ax3.set_ylabel("P(rank_gold < rank_nonfactual)")
        ax3.set_ylim(0, 1.05)
        ax3.legend(fontsize=8)
        ax3.set_title(title, fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    results_files = sorted((RESULTS_DIR / "retrieval_evaluation").rglob("results.jsonl"))
    print(f"Found {len(results_files)} results.jsonl files.")
    for r in results_files:
        out = r.parent / "retrieval_plot.png"
        plot_results(r, out)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
