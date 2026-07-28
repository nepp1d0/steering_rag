"""
Step 1d - Direction identification via logistic-regression probes on mixtures of datasets.

Same procedure as `probe_direction_identification.py`, but the train splits of several
datasets are pooled before fitting the probe (mirrors `mixed_direction_identification.py`:
same combos, same per-side balancing, activations extracted once per dataset per layer
and reused across combos).

Outputs:
    results/probes/mixed_directions/<model>/<combo>/seed_<S>/context_only/layer_<L>/<position>/{direction.pt,meta.json}

where <combo> is e.g. "conflictqa", "conflictqa+nq_swap" or "conflictqa+nq_swap+longfact".

Usage:
    python -m src.experiments.mixed_probe_direction_identification --automated
    python -m src.experiments.mixed_probe_direction_identification --automated --force-recompute
    python -m src.experiments.mixed_probe_direction_identification \
        --model google/gemma-3-4b-it \
        --layers 10,15
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import (
    RESULTS_DIR,
    logger,
    safe_model_id,
    setup_logging,
)
from src.experiments.direction_identification import (
    MODELS,
    collect_side_acts,
    discover_seeds,
    parse_layers,
)
from src.experiments.mixed_direction_identification import (
    COMBOS,
    DATASETS,
    MIX_SEED,
    POSITIONS,
    combo_name,
    load_side_data,
)
from src.experiments.probe_direction_identification import (
    fit_probe_direction,
    save_probe_direction,
)

import transformer_lens.utils as tl_utils
from transformer_lens import HookedTransformer


def main() -> None:
    parser = argparse.ArgumentParser(description="Logistic-regression probe direction identification on dataset mixtures.")
    parser.add_argument("--model", default=None, help="HuggingFace model id (e.g. google/gemma-3-4b-it).")
    parser.add_argument("--automated", action="store_true",
                        help="Run all models (all seeds and layers; both positions are always computed).")
    parser.add_argument("--layers", type=parse_layers, default=None,
                        help="Comma-separated list of layers. If omitted, all model layers are used.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Split seed, applied to every dataset. If omitted, runs for all seeds.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of samples per forward pass.")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Recompute directions even if they already exist on disk.")
    args = parser.parse_args()

    if args.automated:
        models = MODELS
    else:
        if not args.model:
            parser.error("--model is required without --automated")
        models = [args.model]
    seeds = [args.seed] if args.seed is not None else discover_seeds("nq_swap")

    setup_logging("mixed_probe_direction_identification", RESULTS_DIR / "probes" / "mixed_directions")
    logger.info(f"models={models} | datasets={DATASETS} | layers={args.layers} | seeds={seeds}")

    device = tl_utils.get_device()
    for model_name in models:
        out_base = RESULTS_DIR / "probes" / "mixed_directions" / safe_model_id(model_name)
        out_base.mkdir(parents=True, exist_ok=True)
        logger.info(f"##### Model: {model_name} (loading on {device}) #####")
        model = HookedTransformer.from_pretrained_no_processing(model_name, device=device, dtype="bfloat16")
        layers = args.layers if args.layers is not None else list(range(model.cfg.n_layers))

        for seed in seeds:
            todo = [L for L in layers if args.force_recompute or not all(
                (out_base / combo_name(c) / f"seed_{seed}" / "context_only" / f"layer_{L}"
                 / pos / "direction.pt").exists() for c in COMBOS for pos in POSITIONS)]
            if not todo:
                logger.info(f"Skip {model_name} | seed {seed} (all layers computed).")
                continue
            logger.info(f"=== {model_name} | seed {seed} | layers to do: {todo} ===")

            # --- Load the train splits and resolve entity spans (model-independent). ---
            data: Dict[str, Dict[str, List[Tuple[str, Optional[Tuple[int, int]]]]]] = {}
            for ds in DATASETS:
                data[ds] = load_side_data(ds, seed)
                for side in ("pos", "neg"):
                    n_texts = len(data[ds][side])
                    n_resolved = sum(1 for _, sp in data[ds][side] if sp is not None)
                    logger.info(f"{ds} {side}: {n_texts} texts | entity spans resolved: "
                                f"{n_resolved} ({n_texts - n_resolved} dropped from entity_pos)")

            # --- Balance: subsample every dataset to the same number of texts per side. ---
            n = min(len(data[ds][side]) for ds in DATASETS for side in ("pos", "neg"))
            logger.info(f"Subsampling every dataset to {n} texts per side (mix_seed={MIX_SEED}).")
            rng = random.Random(MIX_SEED)
            for ds in DATASETS:
                for side in ("pos", "neg"):
                    data[ds][side] = rng.sample(data[ds][side], n)

            for layer in todo:
                hook_point = tl_utils.get_act_name("resid_post", layer)
                logger.info(f"=== Layer {layer} ({hook_point}) ===")

                # Extract once per dataset, reuse across every combo.
                acts: Dict[str, Dict[str, Dict[str, Optional[torch.Tensor]]]] = {}
                for ds in DATASETS:
                    logger.info(f"-> activations: {ds}")
                    acts[ds] = {}
                    for side in ("pos", "neg"):
                        last, ent = collect_side_acts(model, hook_point, layer, data[ds][side],
                                                      batch_size=args.batch_size, desc=f"{ds} {side} acts")
                        acts[ds][side] = {"last_pos": last, "entity_pos": ent}

                for combo in COMBOS:
                    name = combo_name(combo)
                    for pos_name in POSITIONS:
                        if any(acts[ds][side][pos_name] is None for ds in combo for side in ("pos", "neg")):
                            logger.warning(f"Skipping {name}/{pos_name}: missing activations.")
                            continue
                        pos_stack = torch.cat([acts[ds]["pos"][pos_name] for ds in combo])
                        neg_stack = torch.cat([acts[ds]["neg"][pos_name] for ds in combo])
                        direction, train_acc = fit_probe_direction(pos_stack, neg_stack)
                        save_probe_direction(
                            out_base / name / f"seed_{seed}" / "context_only" / f"layer_{layer}" / pos_name,
                            direction, train_acc,
                            {"model": model_name, "dataset": name, "datasets": combo, "layer": layer,
                             "seed": seed, "procedure": "context_only", "position": pos_name,
                             "n_pos": int(pos_stack.shape[0]), "n_neg": int(neg_stack.shape[0]),
                             "n_per_dataset": n, "mix_seed": MIX_SEED},
                        )

        del model
        torch.cuda.empty_cache()

    logger.info("Done.")


if __name__ == "__main__":
    main()
