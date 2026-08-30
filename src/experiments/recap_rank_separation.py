"""
Recap metric over every retrieval-evaluation combination: rank separation gain.

For each cell (model x direction dataset x eval dataset x position), using the
best-layer results copied into results/top_retrieval_evaluation/:

    gold_rank_gain  = mean_gold_rank(alpha=0) - mean_gold_rank(alpha)   (+ = gold climbed)
    nf_rank_loss    = mean_nf_rank(alpha)     - mean_nf_rank(alpha=0)   (+ = non-factual sank)
    separation_gain = gold_rank_gain + nf_rank_loss

Everything is computed per seed first, then aggregated (mean, std) across seeds,
so the std matches the error bars in figure 3.

Outputs (default results/figures/recap/):
    rank_separation.json  all cells, all alphas > 0, plus coverage gaps
    rank_separation.csv   one row per cell at --alpha, ready to paste into a table
    rank_separation.md    provenance sidecar

Usage:
    python src/experiments/recap_rank_separation.py
    python src/experiments/recap_rank_separation.py --alpha 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from plot_retrieval_evaluation import RESULTS_DIR, _agg, compute_seed_metrics

NORMALIZE = "unnormalized"
PROCEDURE = "context_only"

# Order of the table rows: models by size, as in figure 3.
MODELS = [
    ("meta-llama__Llama-3.2-1B-Instruct", "Llama-3.2-1B"),
    ("google__gemma-3-4b-it", "Gemma-3-4B"),
    ("Qwen__Qwen2-7B-Instruct", "Qwen2-7B"),
    ("meta-llama__Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
]
DIRECTIONS = ["conflictqa", "nq_swap", "longfact"]
EVALS = ["conflictqa", "nq_swap"]
POSITIONS = ["last_pos", "entity_pos"]
EXPECTED_SEEDS = 5


def parse_parts(path: Path, top_dir: Path):
    """Layout: model/eval/direction/normalize/seed_N/procedure/layer_L/position/results.jsonl"""
    p = path.relative_to(top_dir).parts
    if len(p) < 9:
        return None
    return {"model": p[0], "eval": p[1], "direction": p[2], "normalize": p[3],
            "seed": int(p[4].split("_")[1]), "procedure": p[5],
            "layer": int(p[6].split("_")[1]), "position": p[7]}


def corpus_size_of(results_path: Path) -> int:
    """Number of documents in the layer-level docs.jsonl shared by all positions."""
    docs_path = results_path.parent.parent / "docs.jsonl"
    if not docs_path.exists():
        return 0
    with docs_path.open() as f:
        return sum(1 for line in f if line.strip())


def label_of(direction: str, eval_ds: str, position: str) -> str:
    return f"{direction} -> {eval_ds} ({position})"


def build_cell(model: str, model_label: str, direction: str, eval_ds: str,
               position: str, paths: list[Path], top_dir: Path) -> dict:
    # Numeric seed order, so per-seed lists line up with the reported `seeds`.
    paths = sorted(paths, key=lambda p: parse_parts(p, top_dir)["seed"])
    metas = [parse_parts(p, top_dir) for p in paths]
    metrics = [compute_seed_metrics(p) for p in paths]
    sizes = [corpus_size_of(p) for p in paths]
    seeds = [m["seed"] for m in metas]
    layers = sorted({m["layer"] for m in metas})

    alphas = metrics[0]["alphas"]
    by_alpha = {}
    for a in [x for x in alphas if x > 0]:
        gold_gain = [m["mean_gold_rank"][0.0] - m["mean_gold_rank"][a] for m in metrics]
        nf_loss = [m["mean_nf_rank"][a] - m["mean_nf_rank"][0.0] for m in metrics]
        sep = [g + n for g, n in zip(gold_gain, nf_loss)]
        # Corpora differ in size between eval datasets, so also report the gain as a
        # fraction of the corpus: a 100-position shift means different things in a
        # 652-doc and a 1500-doc corpus.
        sep_norm = [s / n for s, n in zip(sep, sizes) if n]

        gg_m, gg_s = _agg(gold_gain)
        nf_m, nf_s = _agg(nf_loss)
        sp_m, sp_s = _agg(sep)
        spn_m, spn_s = _agg(sep_norm) if sep_norm else (None, None)
        by_alpha[f"{a:g}"] = {
            "gold_rank_gain_mean": gg_m, "gold_rank_gain_std": gg_s,
            "nf_rank_loss_mean": nf_m, "nf_rank_loss_std": nf_s,
            "separation_gain_mean": sp_m, "separation_gain_std": sp_s,
            "separation_gain_normalized_mean": spn_m,
            "separation_gain_normalized_std": spn_s,
            "baseline_gold_rank": _agg([m["mean_gold_rank"][0.0] for m in metrics])[0],
            "baseline_nf_rank": _agg([m["mean_nf_rank"][0.0] for m in metrics])[0],
            "gold_rank": _agg([m["mean_gold_rank"][a] for m in metrics])[0],
            "nf_rank": _agg([m["mean_nf_rank"][a] for m in metrics])[0],
            "per_seed_separation_gain": sep,
        }

    return {
        "label": label_of(direction, eval_ds, position),
        "model": model,
        "model_label": model_label,
        "direction": direction,
        "eval": eval_ds,
        "position": position,
        "layer": layers[0] if len(layers) == 1 else layers,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "corpus_size_mean": float(np.mean(sizes)) if sizes else 0.0,
        "alphas": alphas,
        "by_alpha": by_alpha,
    }


def write_csv(cells: list[dict], alpha: float, out_path: Path) -> None:
    key = f"{alpha:g}"
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "combination", "direction", "eval", "position", "layer",
                    "n_seeds", "gold_rank_gain", "gold_rank_gain_std",
                    "nf_rank_loss", "nf_rank_loss_std",
                    "separation_gain", "separation_gain_std",
                    "separation_gain_normalized"])
        for c in cells:
            a = c["by_alpha"].get(key)
            if a is None:
                continue
            norm = a["separation_gain_normalized_mean"]
            w.writerow([
                c["model_label"], c["label"], c["direction"], c["eval"], c["position"],
                c["layer"], c["n_seeds"],
                f"{a['gold_rank_gain_mean']:.2f}", f"{a['gold_rank_gain_std']:.2f}",
                f"{a['nf_rank_loss_mean']:.2f}", f"{a['nf_rank_loss_std']:.2f}",
                f"{a['separation_gain_mean']:.2f}", f"{a['separation_gain_std']:.2f}",
                "" if norm is None else f"{norm:.4f}",
            ])


def write_sidecar(out_path: Path, top_dir: Path, alpha: float, cells: list[dict],
                  missing: list[str], incomplete: list[str]) -> None:
    lines = [
        "# Rank separation gain — recap over all retrieval-evaluation combinations",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}  ",
        f"Script: `src/experiments/recap_rank_separation.py`  ",
        f"Outputs: `rank_separation.json`, `rank_separation.csv` (alpha={alpha:g}), this file.",
        "",
        "## Metric",
        "",
        "Per seed, from `results.jsonl` (ranks are independent of k, so the smallest k is used):",
        "",
        "```",
        "gold_rank_gain  = mean_gold_rank(alpha=0) - mean_gold_rank(alpha)   # + = gold climbed",
        "nf_rank_loss    = mean_nf_rank(alpha)     - mean_nf_rank(alpha=0)   # + = non-factual sank",
        "separation_gain = gold_rank_gain + nf_rank_loss",
        "```",
        "",
        "Both terms are in rank positions and both are 'higher is better'. The score is",
        "computed per seed and then averaged; the reported std is the across-seed sample",
        "std (ddof=1) of the per-seed value, i.e. the same quantity as the error bars in",
        "figure 3. `separation_gain_normalized` divides the per-seed gain by that seed's",
        "corpus size before aggregating, because the ConflictQA and NQ-Swap corpora differ",
        "in size and raw rank positions are not comparable across eval datasets.",
        "",
        "## Data",
        "",
        f"Source: `{top_dir.relative_to(RESULTS_DIR.parent)}` — the best-layer results copied by",
        "`plot_retrieval_evaluation.copy_top_results()`, i.e. for every",
        "(model, eval, direction, position) group the single layer that maximises",
        "`score_layer` (best alpha>0 top-k recall lift + non-factual drop vs the alpha=0",
        "baseline, meaned over seeds and k). The fused score being evaluated is",
        "`(1-alpha)*zscore(sbert_cos) + alpha*zscore(projection)`.",
        "",
        f"- normalize: `{NORMALIZE}` (directions are not unit-normalised)",
        f"- procedure: `{PROCEDURE}`",
        f"- models: {', '.join(m[1] for m in MODELS)}",
        f"- direction datasets: {', '.join(DIRECTIONS)}",
        f"- eval datasets: {', '.join(EVALS)}",
        f"- positions: {', '.join(POSITIONS)}",
        f"- alphas reported: every alpha > 0 present in the results (CSV uses {alpha:g})",
        f"- seeds: up to {EXPECTED_SEEDS} split seeds per cell (7, 42, 67, 89, 90)",
        "- `FULL_*` model directories (older, pre-position layout) are excluded.",
        "",
        "A cell labelled `A -> B (pos)` means: direction identified on dataset A, applied to",
        "the retrieval corpus of eval dataset B, using the direction from identification",
        "position `pos`. Documents are always projected at their last token; the position",
        "only selects which identified direction is loaded.",
        "",
        "## Reading",
        "",
        "Positive `separation_gain` means the projection term pushes the gold document up",
        "and the non-factual document down relative to pure SBERT retrieval. Off-diagonal",
        "cells (direction dataset != eval dataset, and every `longfact ->` cell) are the",
        "generalization evidence: the direction was never fit on the eval corpus.",
        "",
        "## Coverage",
        "",
        f"{len(cells)} of {len(MODELS) * len(DIRECTIONS) * len(EVALS) * len(POSITIONS)} grid cells present.",
        "",
    ]
    if incomplete:
        lines += ["Cells with fewer than the expected seeds (std is unreliable or zero):", ""]
        lines += [f"- {s}" for s in incomplete] + [""]
    if missing:
        lines += ["Cells with no results on disk:", ""]
        lines += [f"- {s}" for s in missing] + [""]
    out_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.3,
                    help="Alpha used for the CSV table (the JSON always holds every alpha>0).")
    ap.add_argument("--source", default="top_retrieval_evaluation",
                    help="Folder under results/ to read (best-layer copies).")
    ap.add_argument("--out-dir", default=None,
                    help="Output folder (default: results/figures/recap).")
    args = ap.parse_args()

    top_dir = RESULTS_DIR / args.source
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR / "figures" / "recap"
    out_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple, list[Path]] = defaultdict(list)
    for f in sorted(top_dir.rglob("results.jsonl")):
        meta = parse_parts(f, top_dir)
        if meta is None or meta["model"].startswith("FULL_"):
            continue
        if meta["normalize"] != NORMALIZE or meta["procedure"] != PROCEDURE:
            continue
        groups[(meta["model"], meta["direction"], meta["eval"], meta["position"])].append(f)

    model_labels = dict(MODELS)
    cells, missing, incomplete = [], [], []
    for model, model_label in MODELS:
        for direction in DIRECTIONS:
            for eval_ds in EVALS:
                for position in POSITIONS:
                    paths = sorted(groups.get((model, direction, eval_ds, position), []))
                    name = f"{model_label} | {label_of(direction, eval_ds, position)}"
                    if not paths:
                        missing.append(name)
                        continue
                    cell = build_cell(model, model_label, direction, eval_ds,
                                      position, paths, top_dir)
                    if not cell["by_alpha"]:
                        missing.append(name + "  (no alpha>0 / no rank fields)")
                        continue
                    if cell["n_seeds"] < EXPECTED_SEEDS:
                        incomplete.append(f"{name}  ({cell['n_seeds']}/{EXPECTED_SEEDS} seeds)")
                    cells.append(cell)

    unknown = sorted({k[0] for k in groups} - set(model_labels))
    for m in unknown:
        print(f"Note: skipped unlisted model directory {m}")

    payload = {
        "meta": {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "source": str(top_dir.relative_to(RESULTS_DIR.parent)),
            "normalize": NORMALIZE,
            "procedure": PROCEDURE,
            "csv_alpha": args.alpha,
            "expected_seeds": EXPECTED_SEEDS,
            "metric_definition": {
                "gold_rank_gain": "mean_gold_rank(alpha=0) - mean_gold_rank(alpha), + = gold climbed",
                "nf_rank_loss": "mean_nf_rank(alpha) - mean_nf_rank(alpha=0), + = non-factual sank",
                "separation_gain": "gold_rank_gain + nf_rank_loss (rank positions, higher = better)",
                "separation_gain_normalized": "per-seed separation_gain / corpus size, then aggregated",
                "aggregation": "per seed first, then mean and sample std (ddof=1) across seeds",
                "label": "'<direction dataset> -> <eval dataset> (<identification position>)'",
            },
            "n_cells": len(cells),
            "n_grid": len(MODELS) * len(DIRECTIONS) * len(EVALS) * len(POSITIONS),
        },
        "cells": cells,
        "incomplete": incomplete,
        "missing": missing,
    }

    json_path = out_dir / "rank_separation.json"
    csv_path = out_dir / "rank_separation.csv"
    md_path = out_dir / "rank_separation.md"
    json_path.write_text(json.dumps(payload, indent=2))
    write_csv(cells, args.alpha, csv_path)
    write_sidecar(md_path, top_dir, args.alpha, cells, missing, incomplete)

    print(f"Wrote {json_path}  ({len(cells)} cells, {len(missing)} missing, "
          f"{len(incomplete)} incomplete)")
    print(f"Wrote {csv_path}  (alpha={args.alpha:g})")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
