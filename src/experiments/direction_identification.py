"""
Step 1 - Direction identification via diff-in-means on the residual stream.

Two procedures are run for every call:

  - "context_only": positive activations come from the factual context, negative ones
    from the non-factual evidence. Positions: `last_pos` for both datasets, plus
    `entity_pos` for nq_swap.

  - "ab_choice": each sample is turned into a single A/B-choice prompt where both
    chunks are shown and a label ("A" or "B") is appended. The pos prompt ends with
    the label of the factual chunk; the neg prompt ends with the wrong label. The
    A/B ordering is shuffled (seed 42) so neither side is always "A".
    Position: `choice_token` (last token, i.e. right after the appended label).

Outputs:
    results/direction_identification/<model>/<dataset>/<procedure>/layer_<L>/<position>/{direction.pt,meta.json}

Usage:
    python -m src.experiments.direction_identification \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --dataset nq_swap \
        --layers 10,15,20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import (
    NORMALIZED_DIR,
    RESULTS_DIR,
    diff_in_means,
    get_last_residual,
    get_residual_at_positions,
    load_normalized,
    logger,
    make_ab_choice_prompt,
    safe_model_id,
    setup_logging,
)

import transformer_lens.utils as tl_utils
from transformer_lens import HookedTransformer


# Per-dataset positions used by the "context_only" procedure.
CONTEXT_ONLY_POSITIONS: Dict[str, List[str]] = {
    "nq_swap":    ["last_pos", "entity_pos"],
    "conflictqa": ["last_pos"],
}

AB_CHOICE_SEED = 42


def parse_layers(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def discover_seeds(dataset: str) -> List[int]:
    dirs = sorted((NORMALIZED_DIR / dataset).glob("seed_*"), key=lambda d: int(d.name.split("_")[1]))
    return [int(d.name.split("_")[1]) for d in dirs if d.is_dir()]


def collect_context_only(
    model,
    hook_point: str,
    samples: List[Dict],
    want_entity_pos: bool,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Returns `{position: {"pos": [n,d], "neg": [n,d]}}` for the context-only procedure."""
    pos_last, neg_last = [], []
    pos_ent, neg_ent = [], []

    for s in tqdm(samples, desc="ctx-only acts"):
        ent_pos = s["factual_answer"][0] if (want_entity_pos and s.get("factual_answer")) else None
        ent_neg = s["non_factual_answer"][0] if (want_entity_pos and s.get("non_factual_answer")) else None

        a_pos = get_residual_at_positions(model, hook_point, s["factual_context"], ent_pos)
        a_neg = get_residual_at_positions(model, hook_point, s["non_factual_evidence"], ent_neg)

        pos_last.append(a_pos["last_pos"])
        neg_last.append(a_neg["last_pos"])

        if want_entity_pos and "entity_pos" in a_pos and "entity_pos" in a_neg:
            pos_ent.append(a_pos["entity_pos"])
            neg_ent.append(a_neg["entity_pos"])

    out = {"last_pos": {"pos": torch.stack(pos_last), "neg": torch.stack(neg_last)}}
    if want_entity_pos and pos_ent:
        out["entity_pos"] = {"pos": torch.stack(pos_ent), "neg": torch.stack(neg_ent)}
    return out


def collect_ab_choice(
    model,
    hook_point: str,
    samples: List[Dict],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Builds an A/B-choice prompt per sample with the factual chunk randomly placed at A or B.
    Pos = prompt ending with the correct label, Neg = prompt ending with the wrong label.
    Returns activations at the last (label) token.
    """
    rng = random.Random(AB_CHOICE_SEED)
    n = len(samples)
    factual_is_a = [True] * (n // 2) + [False] * (n - n // 2)
    rng.shuffle(factual_is_a)

    pos_acts, neg_acts = [], []
    for s, fact_a in tqdm(list(zip(samples, factual_is_a)), desc="ab-choice acts"):
        if fact_a:
            ctx_a, ctx_b = s["factual_context"], s["non_factual_evidence"]
            correct, wrong = "A", "B"
        else:
            ctx_a, ctx_b = s["non_factual_evidence"], s["factual_context"]
            correct, wrong = "B", "A"

        prompt_pos = make_ab_choice_prompt(ctx_a, ctx_b, correct)
        prompt_neg = make_ab_choice_prompt(ctx_a, ctx_b, wrong)

        pos_acts.append(get_last_residual(model, hook_point, prompt_pos))
        neg_acts.append(get_last_residual(model, hook_point, prompt_neg))

    return {"choice_token": {"pos": torch.stack(pos_acts), "neg": torch.stack(neg_acts)}}


def save_direction(
    out_dir: Path,
    direction: torch.Tensor,
    pos_stack: torch.Tensor,
    neg_stack: torch.Tensor,
    meta_extra: Dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(direction, out_dir / "direction.pt")
    meta = {
        "method": "diff_in_means",
        "n_samples": int(pos_stack.shape[0]),
        "d_model": int(direction.shape[0]),
        "norm_pre_normalize": float((pos_stack.mean(0) - neg_stack.mean(0)).norm().item()),
        **meta_extra,
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved direction -> {out_dir / 'direction.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diff-in-means direction identification (context_only + ab_choice).")
    parser.add_argument("--model", required=True, help="HuggingFace model id (e.g. meta-llama/Llama-3.1-8B-Instruct).")
    parser.add_argument("--dataset", required=True, choices=list(CONTEXT_ONLY_POSITIONS.keys()),
                        help="Normalized dataset id.")
    parser.add_argument("--layers", type=parse_layers, default=None,
                        help="Comma-separated list of layers. If omitted, all model layers are used.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Split seed. If omitted, runs for all seeds found in the normalized dataset.")
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else discover_seeds(args.dataset)
    if not seeds:
        logger.error(f"No seed directories found for '{args.dataset}'. Run dataset_normalization.py first.")
        return

    log_root = RESULTS_DIR / "direction_identification" / safe_model_id(args.model) / args.dataset
    log_root.mkdir(parents=True, exist_ok=True)
    setup_logging("direction_identification", log_root)
    logger.info(f"model={args.model} | dataset={args.dataset} | layers={args.layers} | seeds={seeds}")

    device = tl_utils.get_device()
    logger.info(f"Loading model on {device} ...")
    model = HookedTransformer.from_pretrained(args.model, device=device, dtype="bfloat16")
    layers = args.layers if args.layers is not None else list(range(model.cfg.n_layers))
    logger.info(f"Layers: {layers}")

    ctx_positions = CONTEXT_ONLY_POSITIONS[args.dataset]
    want_entity_pos = "entity_pos" in ctx_positions

    for seed in seeds:
        logger.info(f"=== Seed {seed} ===")
        samples = load_normalized(args.dataset, seed)["train"]
        logger.info(f"Loaded {len(samples)} train samples.")
        out_root = RESULTS_DIR / "direction_identification" / safe_model_id(args.model) / args.dataset / f"seed_{seed}"

        for layer in layers:
            if all((out_root / proc / f"layer_{layer}" / pos / "direction.pt").exists()
                   for proc, pos in [("context_only", "last_pos"), ("ab_choice", "choice_token")]):
                logger.info(f"Skip layer {layer} (already computed).")
                continue
            hook_point = tl_utils.get_act_name("resid_post", layer)
            logger.info(f"=== Layer {layer} ({hook_point}) ===")

            # Procedure 1: context_only
            logger.info("-> procedure: context_only")
            ctx_acts = collect_context_only(model, hook_point, samples, want_entity_pos=want_entity_pos)
            for pos_name in ctx_positions:
                if pos_name not in ctx_acts:
                    logger.warning(f"Skipping {pos_name}: no activations collected.")
                    continue
                stacks = ctx_acts[pos_name]
                direction = diff_in_means(stacks["pos"], stacks["neg"], normalize=False)
                save_direction(
                    out_root / "context_only" / f"layer_{layer}" / pos_name,
                    direction, stacks["pos"], stacks["neg"],
                    {"model": args.model, "dataset": args.dataset, "layer": layer, "seed": seed,
                     "procedure": "context_only", "position": pos_name},
                )

            # Procedure 2: ab_choice
            logger.info("-> procedure: ab_choice")
            ab_acts = collect_ab_choice(model, hook_point, samples)
            for pos_name, stacks in ab_acts.items():
                direction = diff_in_means(stacks["pos"], stacks["neg"], normalize=False)
                save_direction(
                    out_root / "ab_choice" / f"layer_{layer}" / pos_name,
                    direction, stacks["pos"], stacks["neg"],
                    {"model": args.model, "dataset": args.dataset, "layer": layer, "seed": seed,
                     "procedure": "ab_choice", "position": pos_name, "ab_seed": AB_CHOICE_SEED},
                )

    logger.info("Done.")


if __name__ == "__main__":
    main()
