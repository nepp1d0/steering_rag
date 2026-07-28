"""
Direction analysis heatmaps: average projection of eval-dataset document
embeddings onto the (raw) factuality direction.

For EACH model we produce TWO heatmaps (3 rows x 2 cols):
  - rows    = direction source dataset  ["nq_swap", "longfact", "conflictqa"]
  - columns = eval dataset              ["conflictqa", "nq_swap"]
  - Plot A "factual samples":     cell = mean over the eval FACTUAL docs of cos(doc_emb - mu, direction)
  - Plot B "non-factual samples": cell = mean over the eval NON-FACTUAL docs of cos(doc_emb - mu, direction)
    where mu is the eval corpus mean embedding (per model/eval/layer/seed).

Design decisions (locked):
  1. Centered cosine: the residual stream is dominated by a large "common mean"
     component shared by every document (||mu|| ~ 15x ||v|| for gemma). The
     diff-in-means direction has that component subtracted out, so a raw cosine(H_i, v)
     mostly measures the common component and buries the factual/non-factual signal
     (yielding a near-constant ~0.06 for every doc). Subtracting the corpus mean mu
     removes that baseline and exposes which side of the direction each doc falls on;
     L2-normalizing keeps cells in [-1, 1] and comparable across the different per-cell
     top layers. (Verified: on same-dataset conflictqa the diff-in-means direction is a
     92%-balanced-accuracy factual/non-factual classifier, and centered cosine recovers
     factual > 0 > non-factual as expected.)
  2. Per-cell top layer, read from top_layers_<procedure>_<position>.json under
     top_retrieval_evaluation/<model>/<eval>/<direction>/unnormalized/.
  3. Average over 5 seeds {7,42,67,89,90}, matched direction seed == eval seed; keep std.
  4. Average over UNIQUE docs (dedup), not per-sample occurrences.
  5. Two separate colorbars (factual / non-factual on different scales).

No GPU / no model loading — everything reads precomputed tensors from disk.

Usage:
    python -m src.experiments.direction_analysis_heatmap
    python -m src.experiments.direction_analysis_heatmap --models google__gemma-3-4b-it
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import RESULTS_DIR, load_normalized, logger, setup_logging

MODELS = [
    "google__gemma-3-4b-it",
    "meta-llama__Llama-3.1-8B-Instruct",
    "meta-llama__Llama-3.2-1B-Instruct",
    "Qwen__Qwen2-7B-Instruct",
]
EVAL_DATASETS = ["conflictqa", "nq_swap"]        # columns
DIRECTION_DATASETS = ["nq_swap", "longfact", "conflictqa"]  # rows
SEEDS = [7, 42, 67, 89, 90]
PROCEDURE = "context_only"
POSITIONS = ["last_pos", "entity_pos"]

# Diverging blue<->red with a neutral gray midpoint (dataviz skill diverging pair).
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"]
)


def discover_top_layer(model: str, eval_ds: str, direction_ds: str, position: str) -> int:
    """Top layer for a (model, eval, direction, position) group, from its top_layers json."""
    path = (RESULTS_DIR / "top_retrieval_evaluation" / model / eval_ds / direction_ds
            / "unnormalized" / f"top_layers_{PROCEDURE}_{position}.json")
    return int(json.loads(path.read_text())["ranking"][0]["layer"])


def compute_cell(model: str, eval_ds: str, direction_ds: str, layer: int, position: str) -> tuple[float, float, float, float]:
    """Compute (fac_mean, fac_std, nf_mean, nf_std) across seeds for one cell."""
    group = RESULTS_DIR / "top_retrieval_evaluation" / model / eval_ds / direction_ds / "unnormalized"
    fac_seed_vals, nf_seed_vals = [], []
    for s in SEEDS:
        cell_dir = group / f"seed_{s}" / PROCEDURE / f"layer_{layer}"
        # Doc embeddings at the top layer.
        H = torch.load(cell_dir / "llm_hidden_states.pt", map_location="cpu").float()  # [N, d_model]

        # Authoritative corpus for this tensor: docs.jsonl saved alongside it (row i = id i).
        pairs = [json.loads(line) for line in (cell_dir / "docs.jsonl").read_text().splitlines() if line.strip()]
        id_to_text = {i: t for i, t in pairs}
        canonical_docs = [id_to_text[i] for i in range(len(id_to_text))]
        assert len(canonical_docs) == H.shape[0], (
            f"docs.jsonl/hidden mismatch for {model}/{eval_ds}/{direction_ds} seed {s}: "
            f"{len(canonical_docs)} docs vs {H.shape[0]} hidden states"
        )

        samples = load_normalized(eval_ds, s)["test"]

        # Precomputed diff-in-means direction (loaded, not recomputed).
        v = torch.load(
            RESULTS_DIR / "direction_identification" / model / direction_ds
            / f"seed_{s}" / PROCEDURE / f"layer_{layer}" / position / "direction.pt",
            map_location="cpu",
        ).float()  # [d_model]

        # Centered cosine (see module docstring, decision 1): subtract the eval corpus
        # mean to remove the dominating common-mean residual component, then L2-normalize
        # so cells are in [-1, 1] and comparable across the different per-cell top layers.
        vn = v / (v.norm() + 1e-8)                    # [d_model]
        Hc = H - H.mean(dim=0, keepdim=True)          # center by eval corpus mean
        proj = (Hc @ vn) / (Hc.norm(dim=1) + 1e-8)    # [N] centered cosine similarities

        # Average over UNIQUE (dedup) docs, labeled by membership in the current split.
        # Canonical docs that no longer appear in it (dataset drift) drop out of the means.
        fac_docs = set(x["factual_context"] for x in samples)
        nf_docs = set(x["non_factual_evidence"] for x in samples)
        fac_idx = [i for i, d in enumerate(canonical_docs) if d in fac_docs]
        nf_idx = [i for i, d in enumerate(canonical_docs) if d in nf_docs]

        fac_seed_vals.append(proj[fac_idx].mean().item())
        nf_seed_vals.append(proj[nf_idx].mean().item())

    fac = np.array(fac_seed_vals, dtype=float)
    nf = np.array(nf_seed_vals, dtype=float)
    return (
        float(fac.mean()), float(fac.std(ddof=1)),
        float(nf.mean()), float(nf.std(ddof=1)),
    )


def render_heatmap(mean_mat, std_mat, layer_mat, title, out_stem: Path) -> None:
    """3 rows x 2 cols heatmap; annotate mean\n±std and the per-cell layer."""
    mean_mat = np.array(mean_mat, dtype=float)
    std_mat = np.array(std_mat, dtype=float)

    # Symmetric diverging norm centered at 0 so the gray midpoint means "zero projection".
    vabs = float(np.max(np.abs(mean_mat)))
    vabs = vabs if vabs > 0 else 1.0
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(mean_mat, cmap=DIVERGING_CMAP, norm=norm, aspect="auto")

    ax.set_xticks(range(len(EVAL_DATASETS)))
    ax.set_xticklabels([f"eval:\n{e}" for e in EVAL_DATASETS], fontsize=10)
    ax.set_yticks(range(len(DIRECTION_DATASETS)))
    ax.set_yticklabels([f"direction:\n{d}" for d in DIRECTION_DATASETS], fontsize=10)
    ax.set_xlabel("eval dataset", fontsize=10)
    ax.set_ylabel("direction source dataset", fontsize=10)
    ax.set_title(title, fontsize=11, pad=12)

    # Recessive gridlines between cells.
    ax.set_xticks(np.arange(-0.5, len(EVAL_DATASETS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DIRECTION_DATASETS), 1), minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=2)
    ax.tick_params(which="minor", length=0)

    for i in range(len(DIRECTION_DATASETS)):
        for j in range(len(EVAL_DATASETS)):
            m, s, L = mean_mat[i, j], std_mat[i, j], layer_mat[i][j]
            # White ink on the deep ends of the ramp, dark ink near the neutral midpoint.
            txt_color = "#ffffff" if abs(m) > 0.55 * vabs else "#0b0b0b"
            ax.text(j, i, f"{m:.2f}\n±{s:.2f}", ha="center", va="center",
                    color=txt_color, fontsize=11, fontweight="bold")
            ax.text(j, i + 0.34, f"L={L}", ha="center", va="center",
                    color=txt_color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("mean centered cosine(doc_emb - mu, direction)", fontsize=9)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_model(model: str, position: str) -> None:
    logger.info(f"=== Model: {model} | position: {position} ===")
    # last_pos keeps the original output paths; other positions get a subdir.
    out_dir = RESULTS_DIR / "direction_analysis_heatmap" / model
    if position != "last_pos":
        out_dir = out_dir / position
    out_dir.mkdir(parents=True, exist_ok=True)
    title_suffix = "" if position == "last_pos" else f" ({position})"

    n_rows, n_cols = len(DIRECTION_DATASETS), len(EVAL_DATASETS)
    fac_mean = [[0.0] * n_cols for _ in range(n_rows)]
    fac_std = [[0.0] * n_cols for _ in range(n_rows)]
    nf_mean = [[0.0] * n_cols for _ in range(n_rows)]
    nf_std = [[0.0] * n_cols for _ in range(n_rows)]
    layer_mat = [[0] * n_cols for _ in range(n_rows)]

    for i, direction_ds in enumerate(DIRECTION_DATASETS):
        for j, eval_ds in enumerate(EVAL_DATASETS):
            layer = discover_top_layer(model, eval_ds, direction_ds, position)
            layer_mat[i][j] = layer
            fm, fs, nm, ns = compute_cell(model, eval_ds, direction_ds, layer, position)
            fac_mean[i][j], fac_std[i][j] = fm, fs
            nf_mean[i][j], nf_std[i][j] = nm, ns
            logger.info(
                f"  dir={direction_ds:10s} eval={eval_ds:10s} L={layer:2d} | "
                f"factual={fm:+.3f}±{fs:.3f}  nonfactual={nm:+.3f}±{ns:.3f}"
            )

    render_heatmap(
        fac_mean, fac_std, layer_mat,
        f"{model}\nfactuality projection — factual samples{title_suffix}",
        out_dir / "factuality_projection_factual_samples",
    )
    render_heatmap(
        nf_mean, nf_std, layer_mat,
        f"{model}\nfactuality projection — non-factual samples{title_suffix}",
        out_dir / "factuality_projection_nonfactual_samples",
    )

    matrices = {
        "model": model,
        "position": position,
        "row_labels": [f"direction: {d}" for d in DIRECTION_DATASETS],
        "col_labels": [f"eval: {e}" for e in EVAL_DATASETS],
        "seeds": SEEDS,
        "layers": layer_mat,
        "factual_samples": {"mean": fac_mean, "std": fac_std},
        "nonfactual_samples": {"mean": nf_mean, "std": nf_std},
    }
    (out_dir / "matrices.json").write_text(json.dumps(matrices, indent=2))
    logger.info(f"  Wrote outputs -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()

    setup_logging("direction_analysis_heatmap", RESULTS_DIR / "direction_analysis_heatmap")
    logger.info(f"Models: {args.models}")

    for model in args.models:
        for position in POSITIONS:
            try:
                process_model(model, position)
            except FileNotFoundError as e:
                logger.warning(f"Skip {model}/{position}: missing file ({e})")

    logger.info("Done.")


if __name__ == "__main__":
    main()
