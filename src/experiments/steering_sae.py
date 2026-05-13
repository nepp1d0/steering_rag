"""
Step 2 (SAE variant) - Steering experiment using SAE-derived directions.

Identical to steering.py in every detail except it only processes directions
produced by direction_identification_sae.py (procedures: sae_context_only,
sae_ab_choice).  Because the direction.pt format is identical, the hook
injection and batched generation are unchanged.

Outputs land under:
    results/steering/<model>/<eval_dataset>/<id_dataset>/<sae_procedure>/layer_<L>/<pos>/runs.jsonl

evaluation_steering.py discovers these automatically alongside diff-in-means runs.

Usage:
    python -m src.experiments.steering_sae \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --eval-dataset nq_swap \
        --alpha 20.0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import RESULTS_DIR, load_normalized, logger, safe_model_id, setup_logging
from src.experiments.steering import (
    build_rag_prompt,
    batched_generate,
    parse_direction_path,
    load_direction,
    SHUFFLE_SEED,
    MAX_NEW_TOKENS_DEFAULT,
    BATCH_SIZE_DEFAULT,
)

import transformer_lens.utils as tl_utils
from transformer_lens import HookedTransformer


def find_sae_directions(model_id: str) -> List[Path]:
    """Return all direction.pt files produced by direction_identification_sae.py."""
    root = RESULTS_DIR / "direction_identification" / safe_model_id(model_id)
    return sorted(
        p for p in root.rglob("direction.pt")
        if p.relative_to(root).parts[1].startswith("sae_")
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="SAE steering experiment over SAE-identified directions.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval-dataset", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--normalize", action="store_true",
                    help="L2-normalise directions before applying alpha.")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    args = ap.parse_args()

    out_root = RESULTS_DIR / "steering" / safe_model_id(args.model) / args.eval_dataset
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging("steering_sae", out_root)
    logger.info(f"model={args.model} | eval={args.eval_dataset} | alpha={args.alpha} | normalize={args.normalize}")

    samples = load_normalized(args.eval_dataset)
    logger.info(f"Loaded {len(samples)} samples for '{args.eval_dataset}'.")

    direction_files = find_sae_directions(args.model)
    if not direction_files:
        logger.error(
            f"No SAE directions found for model '{args.model}'. "
            "Run direction_identification_sae.py first."
        )
        return
    logger.info(f"Found {len(direction_files)} SAE direction(s).")

    device = tl_utils.get_device()
    model = HookedTransformer.from_pretrained(args.model, device=device, dtype="bfloat16")
    model.eval()

    rng = random.Random(SHUFFLE_SEED)
    prompts_and_orders = [build_rag_prompt(model, s, rng) for s in samples]
    prompts = [p for p, _ in prompts_and_orders]

    logger.info("Generating baseline (no steering) ...")
    baseline = batched_generate(model, prompts, args.max_new_tokens, args.batch_size, "baseline")

    for dir_path in direction_files:
        info = parse_direction_path(dir_path, args.model)
        out_dir = out_root / info["id_dataset"] / info["procedure"] / f"layer_{info['layer']}" / info["position"]
        out_dir.mkdir(parents=True, exist_ok=True)
        runs_path = out_dir / "runs.jsonl"
        if runs_path.exists():
            logger.info(f"Skip (already exists): {runs_path}")
            continue

        logger.info(f"Steering with {dir_path} ...")
        d = load_direction(dir_path, args.normalize, device, model.cfg.dtype)
        hook_point = tl_utils.get_act_name("resid_post", info["layer"])

        def steer_hook(resid, hook, _v=d, _a=args.alpha):
            return resid + _a * _v

        with model.hooks(fwd_hooks=[(hook_point, steer_hook)]):
            steered = batched_generate(
                model, prompts, args.max_new_tokens, args.batch_size,
                f"steer L{info['layer']}/{info['position']}",
            )

        with runs_path.open("w") as f:
            for i, (sample, (prompt, order), b, s) in enumerate(
                zip(samples, prompts_and_orders, baseline, steered)
            ):
                row = {
                    "id": i,
                    "question": sample["question"],
                    "factual_answer": sample["factual_answer"],
                    "non_factual_answer": sample["non_factual_answer"],
                    "doc_order": order,
                    "baseline_generation": b,
                    "steered_generation": s,
                    "alpha": args.alpha,
                    "normalized_direction": args.normalize,
                    "prompt": prompt,
                    "ground_truth": sample["ground_truth"],
                    **info,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {runs_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
