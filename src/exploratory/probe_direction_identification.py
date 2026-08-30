"""
Step 1c - Direction identification via logistic-regression probes on the residual stream.

Same activation collection as `direction_identification.py` ("context_only" procedure
at last_pos / entity_pos), but the direction is the weight vector of an L2-regularized
logistic regression trained to separate pos from neg activations (mean-centered, no
bias term, balanced class weights). Downstream scoring is unchanged: documents are
still ranked by `hidden @ direction`, and the z-score in retrieval_evaluation absorbs
the probe's arbitrary scale and missing bias.

Outputs:
    results/probes/direction_identification/<model>/<dataset>/seed_<S>/context_only/layer_<L>/<position>/{direction.pt,meta.json}

Usage:
    python src/exploratory/probe_direction_identification.py --automated
    python src/exploratory/probe_direction_identification.py --automated --force-recompute
    python src/exploratory/probe_direction_identification.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --dataset nq_swap \
        --layers 10,15,20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
from utils import (
    RESULTS_DIR,
    load_normalized,
    logger,
    safe_model_id,
    setup_logging,
)
from direction_identification import (
    CONTEXT_ONLY_POSITIONS,
    MODELS,
    collect_side_acts,
    compute_conflictqa_qa_spans,
    discover_seeds,
    parse_layers,
    resolve_side_spans,
)

import transformer_lens.utils as tl_utils
from transformer_lens import HookedTransformer


PROBE_C = 1.0
PROBE_MAX_ITER = 2000


def fit_probe_direction(pos_stack: torch.Tensor, neg_stack: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """L2 logistic regression on mean-centered activations (pos=1, neg=0).

    Returns (weight vector [d_model], train accuracy). class_weight="balanced"
    because entity_pos drops different numbers of samples per side.
    """
    X = torch.cat([pos_stack, neg_stack]).float().numpy()
    y = np.concatenate([np.ones(pos_stack.shape[0]), np.zeros(neg_stack.shape[0])])
    X = X - X.mean(axis=0)
    clf = LogisticRegression(penalty="l2", C=PROBE_C, fit_intercept=False,
                             class_weight="balanced", max_iter=PROBE_MAX_ITER)
    clf.fit(X, y)
    direction = torch.from_numpy(clf.coef_[0]).float()
    return direction, float(clf.score(X, y))


def save_probe_direction(out_dir: Path, direction: torch.Tensor, train_accuracy: float,
                         meta_extra: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(direction, out_dir / "direction.pt")
    meta = {
        "method": "logistic_regression",
        "C": PROBE_C,
        "fit_intercept": False,
        "class_weight": "balanced",
        "centered": True,
        "max_iter": PROBE_MAX_ITER,
        "train_accuracy": train_accuracy,
        "d_model": int(direction.shape[0]),
        "norm": float(direction.norm().item()),
        **meta_extra,
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved probe direction -> {out_dir / 'direction.pt'} (train_acc={train_accuracy:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Logistic-regression probe direction identification (context_only).")
    parser.add_argument("--model", default=None, help="HuggingFace model id (e.g. meta-llama/Llama-3.1-8B-Instruct).")
    parser.add_argument("--dataset", default=None, choices=list(CONTEXT_ONLY_POSITIONS.keys()),
                        help="Normalized dataset id.")
    parser.add_argument("--automated", action="store_true",
                        help="Run all models x datasets (all seeds and layers; both positions are always computed).")
    parser.add_argument("--layers", type=parse_layers, default=None,
                        help="Comma-separated list of layers. If omitted, all model layers are used.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Split seed. If omitted, runs for all seeds found in the normalized dataset.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of samples per forward pass.")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Recompute directions even if they already exist on disk.")
    args = parser.parse_args()

    if args.automated:
        models, datasets = MODELS, list(CONTEXT_ONLY_POSITIONS)
    else:
        if not args.model or not args.dataset:
            parser.error("--model and --dataset are required without --automated")
        models, datasets = [args.model], [args.dataset]

    setup_logging("probe_direction_identification", RESULTS_DIR / "probes" / "direction_identification")
    logger.info(f"models={models} | datasets={datasets} | layers={args.layers} | seed={args.seed}")

    device = tl_utils.get_device()
    for model_name in models:
        logger.info(f"##### Model: {model_name} (loading on {device}) #####")
        model = HookedTransformer.from_pretrained_no_processing(model_name, device=device, dtype="bfloat16")
        layers = args.layers if args.layers is not None else list(range(model.cfg.n_layers))

        for dataset in datasets:
            ctx_positions = CONTEXT_ONLY_POSITIONS[dataset]
            seeds = [args.seed] if args.seed is not None else discover_seeds(dataset)
            if not seeds:
                logger.warning(f"No seed directories found for '{dataset}'. Run dataset_normalization.py first.")
                continue

            for seed in seeds:
                out_root = (RESULTS_DIR / "probes" / "direction_identification"
                            / safe_model_id(model_name) / dataset / f"seed_{seed}")
                todo = [L for L in layers if args.force_recompute or not all(
                    (out_root / "context_only" / f"layer_{L}" / pos / "direction.pt").exists()
                    for pos in ctx_positions)]
                if not todo:
                    logger.info(f"Skip {model_name} | {dataset} | seed {seed} (all layers computed).")
                    continue
                logger.info(f"=== {model_name} | {dataset} | seed {seed} | layers to do: {todo} ===")

                samples = load_normalized(dataset, seed)["train"]
                logger.info(f"Loaded {len(samples)} train samples.")
                qa_spans = compute_conflictqa_qa_spans(samples) if dataset == "conflictqa" else None
                side_data = {side: resolve_side_spans(dataset, samples, side, qa_spans) for side in ("pos", "neg")}
                for side in ("pos", "neg"):
                    n_texts = len(side_data[side])
                    n_resolved = sum(1 for _, sp in side_data[side] if sp is not None)
                    logger.info(f"Seed {seed} {side}: {n_texts} texts | entity spans resolved: "
                                f"{n_resolved} ({n_texts - n_resolved} dropped from entity_pos)")

                for layer in todo:
                    hook_point = tl_utils.get_act_name("resid_post", layer)
                    logger.info(f"=== Layer {layer} ({hook_point}) ===")

                    logger.info("-> procedure: context_only")
                    pos_last, pos_ent = collect_side_acts(model, hook_point, layer, side_data["pos"],
                                                          batch_size=args.batch_size, desc="pos acts")
                    neg_last, neg_ent = collect_side_acts(model, hook_point, layer, side_data["neg"],
                                                          batch_size=args.batch_size, desc="neg acts")
                    stacks_by_pos = {"last_pos": (pos_last, neg_last), "entity_pos": (pos_ent, neg_ent)}
                    for pos_name in ctx_positions:
                        p_stack, n_stack = stacks_by_pos[pos_name]
                        if p_stack is None or n_stack is None:
                            logger.warning(f"Skipping {pos_name}: no activations collected.")
                            continue
                        direction, train_acc = fit_probe_direction(p_stack, n_stack)
                        save_probe_direction(
                            out_root / "context_only" / f"layer_{layer}" / pos_name,
                            direction, train_acc,
                            {"model": model_name, "dataset": dataset, "layer": layer, "seed": seed,
                             "procedure": "context_only", "position": pos_name,
                             "n_pos": int(p_stack.shape[0]), "n_neg": int(n_stack.shape[0])},
                        )

        del model
        torch.cuda.empty_cache()

    logger.info("Done.")


if __name__ == "__main__":
    main()
