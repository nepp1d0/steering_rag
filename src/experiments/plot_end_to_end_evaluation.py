"""
Plot end-to-end evaluation results.

One figure per (model, eval, direction): end-to-end answer accuracy vs k, with a
dashed baseline line (alpha=0, un-steered) and one solid line per top-5 layer
(steered at its best alpha). Mean +/- std across seeds. Layers are ordered/labelled
by their retrieval-score rank (from top_layers_<procedure>.json) so the plot shows
whether the layers our scoring ranked highest also help most end-to-end.

Usage:
    python -m src.experiments.plot_end_to_end_evaluation
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
E2E_DIR = RESULTS_DIR / "end_to_end_evaluation"
PROCEDURE = "context_only"
POSITION = "last_pos"
SEED_RE = re.compile(r"seed_(\d+)")
LAYER_RE = re.compile(r"layer_(\d+)")


def file_accuracies(path: Path) -> tuple[float, list[int], dict[int, float], dict[int, float]]:
    """(best_alpha, ks, {k: baseline_acc}, {k: steered_acc}) for one results.jsonl."""
    buckets: dict[tuple, list] = defaultdict(lambda: [0, 0])  # (alpha,k) -> [hits, total]
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            b = buckets[(r["alpha"], r["k"])]
            b[0] += int(r["is_correct"])
            b[1] += 1
    ks = sorted({k for _, k in buckets})
    best_alpha = next(a for a, _ in sorted(buckets) if a > 0)
    base = {k: buckets[(0.0, k)][0] / buckets[(0.0, k)][1] for k in ks}
    steer = {k: buckets[(best_alpha, k)][0] / buckets[(best_alpha, k)][1] for k in ks}
    return best_alpha, ks, base, steer


def layer_rank_order(gkey: tuple) -> list[int]:
    """Layers ordered by retrieval score (from the matching top_layers file); [] if absent."""
    path = RESULTS_DIR / "retrieval_evaluation" / Path(*gkey) / f"top_layers_{PROCEDURE}_{POSITION}.json"
    if not path.exists():
        return []
    return [e["layer"] for e in json.loads(path.read_text())["top5"]]


def plot_group(gkey: tuple, layer_seed: dict[int, dict[int, tuple]], out_path: Path) -> None:
    ks = sorted(next(iter(next(iter(layer_seed.values())).values()))[1])
    ranked = layer_rank_order(gkey)
    layers = [l for l in ranked if l in layer_seed] + sorted(set(layer_seed) - set(ranked))

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Baseline (alpha=0) is layer-independent: average it across seeds (and layers, identical).
    base_per_seed: dict[int, dict[int, float]] = {}
    for seed_map in layer_seed.values():
        for seed, (_, _, base, _) in seed_map.items():
            base_per_seed[seed] = base
    base_mean = [np.mean([base_per_seed[s][k] for s in base_per_seed]) for k in ks]
    base_std = [np.std([base_per_seed[s][k] for s in base_per_seed], ddof=1) if len(base_per_seed) > 1 else 0.0
                for k in ks]
    ax.errorbar(ks, base_mean, yerr=base_std, color="black", linestyle="--", marker="s",
                capsize=3, linewidth=2, label="baseline (α=0)", zorder=10)

    for rank, layer in enumerate(layers, start=1):
        seed_map = layer_seed[layer]
        best_alpha = next(iter(seed_map.values()))[0]
        means = [np.mean([seed_map[s][3][k] for s in seed_map]) for k in ks]
        stds = [np.std([seed_map[s][3][k] for s in seed_map], ddof=1) if len(seed_map) > 1 else 0.0 for k in ks]
        rank_tag = f"#{rank} " if ranked else ""
        ax.errorbar(ks, means, yerr=stds, marker="o", markersize=4, capsize=3,
                    label=f"{rank_tag}L{layer} (α={best_alpha:g})")

    ax.set_xlabel("k (retrieved docs)")
    ax.set_ylabel("end-to-end accuracy")
    ax.set_xticks(ks)
    ax.set_ylim(0, max(max(base_mean), 0.05) * 1.6)
    ax.set_title("/".join(gkey), fontsize=8)
    ax.legend(fontsize=7, title="top-5 layers (by retrieval score)" if ranked else None)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    files = sorted(E2E_DIR.rglob("results.jsonl"))
    print(f"Found {len(files)} results.jsonl files.")

    # gkey (model, eval, direction, normalize) -> layer -> seed -> accuracies
    groups: dict[tuple, dict[int, dict[int, tuple]]] = defaultdict(lambda: defaultdict(dict))
    for f in files:
        parts = f.relative_to(E2E_DIR).parts
        seed_idx = next(i for i, p in enumerate(parts) if SEED_RE.fullmatch(p))
        gkey = parts[:seed_idx]
        seed = int(SEED_RE.fullmatch(parts[seed_idx]).group(1))
        layer = int(LAYER_RE.fullmatch(parts[-2]).group(1))
        groups[gkey][layer][seed] = file_accuracies(f)

    for gkey, layer_seed in sorted(groups.items()):
        out_path = E2E_DIR / Path(*gkey) / "end_to_end_plot.png"
        plot_group(gkey, layer_seed, out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
