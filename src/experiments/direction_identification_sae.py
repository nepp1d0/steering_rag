"""
Step 1 (SAE variant) - Direction identification via SAE contrastive feature selection.

Mirrors direction_identification.py but replaces diff-in-means in residual space with:
  1. Collect residual activations identically (context_only / ab_choice procedures)
  2. Encode them through a pretrained SAE  ->  sparse feature activations
  3. Feature-level diff: mean(pos_feats) - mean(neg_feats)  for every feature
  4. Keep the top-K features with the largest positive score
  5. Direction = weighted sum of their SAE decoder columns  (weight = feature diff score)
     saved un-normalised (same convention as direction_identification.py)

Procedures saved:
  sae_context_only  (mirrors context_only)
  sae_ab_choice     (mirrors ab_choice)

Outputs:
    results/direction_identification/<model>/<dataset>/sae_context_only/layer_<L>/<pos>/{direction.pt,meta.json}
    results/direction_identification/<model>/<dataset>/sae_ab_choice/layer_<L>/choice_token/{direction.pt,meta.json}

Pre-trained SAE availability (as of 2025):
  - Llama-3.1-8B: use release "llama_scope_lxr_8x"  (base model; instruct fine-tune is fine in practice)
  - Gemma-3-4b-it: NO public pre-trained SAEs exist yet. Script will raise NotImplementedError.

Usage:
    python -m src.experiments.direction_identification_sae \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --dataset nq_swap \
        --layers 10,15,20 \
        [--top-k 10] \
        [--sae-release llama_scope_lxr_8x] \
        [--sae-id-template "blocks.{layer}.hook_resid_post"]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import RESULTS_DIR, load_normalized, logger, safe_model_id, setup_logging
from src.experiments.direction_identification import (
    CONTEXT_ONLY_POSITIONS,
    collect_context_only,
    collect_ab_choice,
)

import transformer_lens.utils as tl_utils
from transformer_lens import HookedTransformer
from sae_lens import SAE


# Default (release, sae_id_template) per model family keyword.
SAE_DEFAULTS: Dict[str, Tuple[str, str]] = {
    "llama": ("llama_scope_lxr_8x", "l{layer}r_8x"),
}


def load_sae(model_id: str, layer: int, sae_release: str | None, sae_id_tpl: str, device: str) -> SAE:
    if sae_release is None:
        for family, (release, tpl) in SAE_DEFAULTS.items():
            if family in model_id.lower():
                sae_release, sae_id_tpl = release, tpl
                break
        else:
            raise NotImplementedError(
                f"No default SAE release configured for '{model_id}'.\n"
                "Pass --sae-release and --sae-id-template explicitly.\n"
                "Tip: list available releases with:\n"
                "  python -c \"from sae_lens.pretrained_saes import get_pretrained_saes_directory; "
                "print(list(get_pretrained_saes_directory().keys()))\"\n"
                "Note: Gemma-3-4b-it has no public pre-trained SAEs as of 2025."
            )
    sae_id = sae_id_tpl.format(layer=layer)
    logger.info(f"Loading SAE: release={sae_release!r}  sae_id={sae_id!r}")
    # Load to CPU first: safetensors safe_open rejects bare "cuda" (needs "cuda:0" or "cpu").
    sae, _, _ = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device="cpu")
    sae = sae.to(device)
    sae.eval()
    return sae


def sae_direction(
    sae: SAE,
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    top_k: int,
) -> Tuple[torch.Tensor, float, List[int]]:
    """
    Build an SAE-based steering direction from contrastive residual activations.

    Feature selection follows Xin et al. (ACL 2025): frequency-based separation score
        sep(i) = freq(pos, i) - freq(neg, i)
    where freq(S, i) = proportion of samples in S where feature i fires (activation > 0).
    This is more robust than mean difference for TopK SAEs where activations are sparse.

    The direction is a weighted sum of the top-k decoder columns (weights = sep scores).

    Returns:
        direction      - un-normalised weighted sum of top-k decoder columns [d_model], float32, cpu
        norm_pre       - L2 norm of that raw direction (for meta.json)
        top_idx_list   - list of the selected SAE feature indices (for interpretability)
    """
    sae_dtype = next(sae.parameters()).dtype
    device = next(sae.parameters()).device

    with torch.no_grad():
        pos_feats = sae.encode(pos_acts.to(device=device, dtype=sae_dtype))  # [n, d_sae]
        neg_feats = sae.encode(neg_acts.to(device=device, dtype=sae_dtype))  # [n, d_sae]

    # Frequency-based separation score (Xin et al. 2025)
    pos_freq = (pos_feats > 0).float().mean(0)   # [d_sae]
    neg_freq = (neg_feats > 0).float().mean(0)
    sep_score = pos_freq - neg_freq              # [d_sae]

    top_idx = sep_score.topk(top_k).indices      # [top_k]
    weights = sep_score[top_idx]                 # [top_k]

    # Weighted sum of decoder columns -> direction in residual stream space
    direction = (weights.unsqueeze(1) * sae.W_dec[top_idx]).sum(0)  # [d_model]
    norm_pre = direction.norm().item()

    return direction.detach().cpu().float(), norm_pre, top_idx.cpu().tolist()


def save_direction(out_dir: Path, direction: torch.Tensor, meta: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(direction, out_dir / "direction.pt")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"Saved -> {out_dir / 'direction.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SAE direction identification (sae_context_only + sae_ab_choice).")
    parser.add_argument("--model", required=True, help="HuggingFace model id.")
    parser.add_argument("--dataset", required=True, choices=list(CONTEXT_ONLY_POSITIONS.keys()))
    parser.add_argument("--layers", required=True, help="Comma-separated layer numbers, e.g. '10,15,20'.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of SAE features to combine into direction.")
    parser.add_argument("--sae-release", default=None, help="sae_lens release name (auto-detected for Llama).")
    parser.add_argument("--sae-id-template", default="l{layer}r_8x",
                        help="SAE id template with {layer} placeholder.")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    device = tl_utils.get_device()

    out_root = RESULTS_DIR / "direction_identification" / safe_model_id(args.model) / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging("direction_identification_sae", out_root)
    logger.info(f"model={args.model} | dataset={args.dataset} | layers={layers} | top_k={args.top_k}")

    samples = load_normalized(args.dataset)["train"]
    logger.info(f"Loaded {len(samples)} train samples.")

    model = HookedTransformer.from_pretrained(args.model, device=device, dtype="bfloat16")
    model.eval()

    want_entity_pos = "entity_pos" in CONTEXT_ONLY_POSITIONS[args.dataset]
    ctx_positions = CONTEXT_ONLY_POSITIONS[args.dataset]

    for layer in layers:
        hook_point = tl_utils.get_act_name("resid_post", layer)
        logger.info(f"=== Layer {layer} ({hook_point}) ===")

        sae = load_sae(args.model, layer, args.sae_release, args.sae_id_template, device)

        # --- sae_context_only ---
        logger.info("-> procedure: sae_context_only")
        ctx_acts = collect_context_only(model, hook_point, samples, want_entity_pos=want_entity_pos)
        for pos_name in ctx_positions:
            if pos_name not in ctx_acts:
                logger.warning(f"Skipping {pos_name}: no activations collected.")
                continue
            stacks = ctx_acts[pos_name]
            direction, norm_pre, top_idx = sae_direction(sae, stacks["pos"], stacks["neg"], args.top_k)
            save_direction(
                out_root / "sae_context_only" / f"layer_{layer}" / pos_name,
                direction,
                {
                    "method": "sae",
                    "n_samples": int(stacks["pos"].shape[0]),
                    "d_model": int(direction.shape[0]),
                    "norm_pre_normalize": norm_pre,
                    "model": args.model,
                    "dataset": args.dataset,
                    "layer": layer,
                    "procedure": "sae_context_only",
                    "position": pos_name,
                    "top_k_features": args.top_k,
                    "top_feature_indices": top_idx,
                },
            )

        # --- sae_ab_choice ---
        logger.info("-> procedure: sae_ab_choice")
        ab_acts = collect_ab_choice(model, hook_point, samples)
        for pos_name, stacks in ab_acts.items():
            direction, norm_pre, top_idx = sae_direction(sae, stacks["pos"], stacks["neg"], args.top_k)
            save_direction(
                out_root / "sae_ab_choice" / f"layer_{layer}" / pos_name,
                direction,
                {
                    "method": "sae",
                    "n_samples": int(stacks["pos"].shape[0]),
                    "d_model": int(direction.shape[0]),
                    "norm_pre_normalize": norm_pre,
                    "model": args.model,
                    "dataset": args.dataset,
                    "layer": layer,
                    "procedure": "sae_ab_choice",
                    "position": pos_name,
                    "top_k_features": args.top_k,
                    "top_feature_indices": top_idx,
                },
            )

    logger.info("Done.")


if __name__ == "__main__":
    main()
