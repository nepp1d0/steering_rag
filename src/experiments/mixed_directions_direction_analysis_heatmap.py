"""
Direction analysis heatmaps for the MIXED-dataset directions: average projection of
eval-dataset document embeddings onto the (raw) mixed factuality direction.

For EACH model we produce TWO heatmaps (7 rows x 2 cols):
  - rows    = direction source combo   (singles, pairs, and the full mixture)
  - columns = eval dataset             ["conflictqa", "nq_swap"]
  - Plot A "factual samples":     cell = mean over the eval FACTUAL docs of cos(doc_emb - mu, direction)
  - Plot B "non-factual samples": cell = mean over the eval NON-FACTUAL docs of cos(doc_emb - mu, direction)
    where mu is the eval corpus mean embedding (per model/eval/layer).

Design decisions inherited from direction_analysis_heatmap.py:
  1. Centered cosine: the residual stream is dominated by a large "common mean"
     component shared by every document (||mu|| ~ 15x ||v|| for gemma). The
     diff-in-means direction has that component subtracted out, so a raw cosine(H_i, v)
     mostly measures the common component and buries the factual/non-factual signal.
     Subtracting the corpus mean mu removes that baseline; L2-normalizing keeps cells in
     [-1, 1] and comparable across the different per-cell top layers.
  2. Per-cell top layer, discovered as the single layer_* dir under
     mixed_directions_top_retrieval_evaluation/<model>/<eval>/<combo>/unnormalized/seed_42/context_only/.
  3. Average over UNIQUE docs (dedup), not per-sample occurrences.
  4. Two separate colorbars (factual / non-factual on different scales).

Differs from the original in one respect: mixed directions are computed for seed 42 only,
so there is no average/std over seeds — each cell is a single number.

No GPU / no model loading — everything reads precomputed tensors from disk. The hidden
states live in the shared per-(model, eval, layer) cache written by
mixed_directions_retrieval_evaluation.py, not inside the per-combo result dirs.

Usage:
    python -m src.experiments.mixed_directions_direction_analysis_heatmap
    python -m src.experiments.mixed_directions_direction_analysis_heatmap --models google__gemma-3-4b-it
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import REPO_ROOT, RESULTS_DIR, load_normalized, logger, setup_logging

MODELS = [
    "google__gemma-3-4b-it",
    "meta-llama__Llama-3.1-8B-Instruct",
    "meta-llama__Llama-3.2-1B-Instruct",
    "Qwen__Qwen2-7B-Instruct",
]
EVAL_DATASETS = ["conflictqa", "nq_swap"]     # columns
COMBOS = [                                    # rows
    "conflictqa",
    "nq_swap",
    "longfact",
    "conflictqa+nq_swap",
    "conflictqa+longfact",
    "nq_swap+longfact",
    "conflictqa+nq_swap+longfact",
]
SEED = 42
PROCEDURE = "context_only"
POSITION = "last_pos"

EVAL_DIR = RESULTS_DIR / "mixed_directions_retrieval_evaluation"
TOP_DIR = RESULTS_DIR / "mixed_directions_top_retrieval_evaluation"
DIRECTIONS_ROOT = RESULTS_DIR / "mixed_directions"
OUT_ROOT = RESULTS_DIR / "mixed_directions_direction_analysis_heatmap"

# Diverging blue<->red with a neutral gray midpoint (dataviz skill diverging pair).
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"]
)


def _load_working(dataset_id: str, seed: int) -> list[dict]:
    """Test split from the working-tree normalized dataset."""
    return load_normalized(dataset_id, seed)["test"]


def _load_committed(dataset_id: str, seed: int) -> list[dict]:
    """Test split from the git-committed (HEAD) normalized dataset."""
    rel = f"data/normalized_dataset/{dataset_id}/seed_{seed}/test.jsonl"
    raw = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=str(REPO_ROOT)).decode()
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def load_matching_samples(dataset_id: str, seed: int, canonical_docs: list[str]) -> list[dict]:
    """Return the test split whose reconstructed corpus matches `canonical_docs`.

    The cached `docs.jsonl` is the authoritative corpus for its hidden-states tensor, so we
    pick whichever dataset version (working tree or HEAD) reproduces that corpus exactly,
    guaranteeing the factual/non-factual doc indices line up with the stored tensor.
    """
    for loader in (_load_working, _load_committed):
        samples = loader(dataset_id, seed)
        all_docs = sorted(
            set(x["factual_context"] for x in samples)
            | set(x["non_factual_evidence"] for x in samples)
        )
        if all_docs == canonical_docs:
            return samples
    raise RuntimeError(
        f"Neither working-tree nor HEAD {dataset_id}/seed_{seed} reproduces the stored "
        f"corpus ({len(canonical_docs)} docs)."
    )


def discover_top_layer(model: str, eval_ds: str, combo: str) -> int:
    """Return the single top layer for a (model, eval, combo) group."""
    ctx = TOP_DIR / model / eval_ds / combo / "unnormalized" / f"seed_{SEED}" / PROCEDURE
    layer_dirs = sorted(ctx.glob("layer_*"))
    assert len(layer_dirs) == 1, f"Expected exactly one layer_* under {ctx}, found {layer_dirs}"
    return int(layer_dirs[0].name.split("_")[1])


def compute_cell(model: str, eval_ds: str, combo: str, layer: int) -> tuple[float, float]:
    """Compute (fac_mean, nf_mean) for one cell (single seed, so no std)."""
    # Doc embeddings at the top layer, from the shared per-(model, eval, layer) cache.
    layer_cache = EVAL_DIR / model / eval_ds / "cache" / f"layer_{layer}"
    H = torch.load(layer_cache / "llm_hidden_states.pt", map_location="cpu").float()  # [N, d_model]

    # Authoritative corpus for this tensor: docs.jsonl saved alongside it (row i = id i).
    pairs = [json.loads(line) for line in (layer_cache / "docs.jsonl").read_text().splitlines() if line.strip()]
    id_to_text = {i: t for i, t in pairs}
    canonical_docs = [id_to_text[i] for i in range(len(id_to_text))]
    assert len(canonical_docs) == H.shape[0], (
        f"docs.jsonl/hidden mismatch for {model}/{eval_ds} layer {layer}: "
        f"{len(canonical_docs)} docs vs {H.shape[0]} hidden states"
    )

    samples = load_matching_samples(eval_ds, SEED, canonical_docs)
    all_docs = canonical_docs
    doc_idx = {d: i for i, d in enumerate(all_docs)}

    # Precomputed mixed diff-in-means direction (loaded, not recomputed).
    v = torch.load(
        DIRECTIONS_ROOT / model / combo / f"seed_{SEED}" / PROCEDURE
        / f"layer_{layer}" / POSITION / "direction.pt",
        map_location="cpu",
    ).float()  # [d_model]

    # Centered cosine (see module docstring, decision 1): subtract the eval corpus mean to
    # remove the dominating common-mean residual component, then L2-normalize so cells are
    # in [-1, 1] and comparable across the different per-cell top layers.
    vn = v / (v.norm() + 1e-8)                    # [d_model]
    Hc = H - H.mean(dim=0, keepdim=True)          # center by eval corpus mean
    proj = (Hc @ vn) / (Hc.norm(dim=1) + 1e-8)    # [N] centered cosine similarities

    # Average over UNIQUE (dedup) docs, not per-sample occurrences.
    fac_docs = sorted(set(x["factual_context"] for x in samples))
    nf_docs = sorted(set(x["non_factual_evidence"] for x in samples))
    fac_idx = [doc_idx[d] for d in fac_docs]
    nf_idx = [doc_idx[d] for d in nf_docs]

    return float(proj[fac_idx].mean().item()), float(proj[nf_idx].mean().item())


def render_heatmap(mean_mat, layer_mat, title, out_stem: Path) -> None:
    """7 rows x 2 cols heatmap; annotate the mean and the per-cell layer."""
    mean_mat = np.array(mean_mat, dtype=float)

    # Symmetric diverging norm centered at 0 so the gray midpoint means "zero projection".
    vabs = float(np.max(np.abs(mean_mat)))
    vabs = vabs if vabs > 0 else 1.0
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)

    fig, ax = plt.subplots(figsize=(6.4, 8.4))
    im = ax.imshow(mean_mat, cmap=DIVERGING_CMAP, norm=norm, aspect="auto")

    ax.set_xticks(range(len(EVAL_DATASETS)))
    ax.set_xticklabels([f"eval:\n{e}" for e in EVAL_DATASETS], fontsize=10)
    ax.set_yticks(range(len(COMBOS)))
    ax.set_yticklabels(COMBOS, fontsize=9)
    ax.set_xlabel("eval dataset", fontsize=10)
    ax.set_ylabel("direction source combo", fontsize=10)
    ax.set_title(title, fontsize=11, pad=12)

    # Recessive gridlines between cells.
    ax.set_xticks(np.arange(-0.5, len(EVAL_DATASETS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(COMBOS), 1), minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=2)
    ax.tick_params(which="minor", length=0)

    for i in range(len(COMBOS)):
        for j in range(len(EVAL_DATASETS)):
            m, L = mean_mat[i, j], layer_mat[i][j]
            # White ink on the deep ends of the ramp, dark ink near the neutral midpoint.
            txt_color = "#ffffff" if abs(m) > 0.55 * vabs else "#0b0b0b"
            ax.text(j, i, f"{m:.2f}", ha="center", va="center",
                    color=txt_color, fontsize=11, fontweight="bold")
            ax.text(j, i + 0.28, f"L={L}", ha="center", va="center",
                    color=txt_color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("mean centered cosine(doc_emb - mu, direction)", fontsize=9)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_model(model: str) -> None:
    logger.info(f"=== Model: {model} ===")
    out_dir = OUT_ROOT / model
    out_dir.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = len(COMBOS), len(EVAL_DATASETS)
    fac_mean = [[0.0] * n_cols for _ in range(n_rows)]
    nf_mean = [[0.0] * n_cols for _ in range(n_rows)]
    layer_mat = [[0] * n_cols for _ in range(n_rows)]

    for i, combo in enumerate(COMBOS):
        for j, eval_ds in enumerate(EVAL_DATASETS):
            layer = discover_top_layer(model, eval_ds, combo)
            layer_mat[i][j] = layer
            fm, nm = compute_cell(model, eval_ds, combo, layer)
            fac_mean[i][j], nf_mean[i][j] = fm, nm
            logger.info(
                f"  combo={combo:28s} eval={eval_ds:10s} L={layer:2d} | "
                f"factual={fm:+.3f}  nonfactual={nm:+.3f}"
            )

    render_heatmap(
        fac_mean, layer_mat,
        f"{model}\nfactuality projection — factual samples (mixed directions)",
        out_dir / "factuality_projection_factual_samples",
    )
    render_heatmap(
        nf_mean, layer_mat,
        f"{model}\nfactuality projection — non-factual samples (mixed directions)",
        out_dir / "factuality_projection_nonfactual_samples",
    )

    matrices = {
        "model": model,
        "row_labels": [f"direction: {c}" for c in COMBOS],
        "col_labels": [f"eval: {e}" for e in EVAL_DATASETS],
        "seed": SEED,
        "layers": layer_mat,
        "factual_samples": {"mean": fac_mean},
        "nonfactual_samples": {"mean": nf_mean},
    }
    (out_dir / "matrices.json").write_text(json.dumps(matrices, indent=2))
    logger.info(f"  Wrote outputs -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    setup_logging("mixed_directions_direction_analysis_heatmap", OUT_ROOT)
    logger.info(f"Models: {args.models}")

    for model in args.models:
        if not (TOP_DIR / model).is_dir():
            logger.warning(f"No mixed top-retrieval results for {model}, skipping.")
            continue
        process_model(model)

    logger.info("Done.")


if __name__ == "__main__":
    main()
