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
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import (
    NORMALIZED_DIR,
    RESULTS_DIR,
    diff_in_means,
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
    batch_size: int = 8,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Returns `{position: {"pos": [n,d], "neg": [n,d]}}` for the context-only procedure."""
    tok = model.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    pos_last, neg_last = [], []
    pos_ent, neg_ent = [], []

    def run_batch(texts: List[str], entities: List[Optional[str]]):
        # Tokenize individually with BOS to know each sequence's original length.
        enc = [tok(t, return_tensors="pt", add_special_tokens=True).input_ids[0] for t in texts]
        orig_lens = [e.shape[0] for e in enc]
        L = max(orig_lens)
        # Left-pad the batch.
        batch = torch.full((len(texts), L), tok.pad_token_id, dtype=torch.long, device=model.cfg.device)
        for r, e in enumerate(enc):
            batch[r, L - e.shape[0]:] = e.to(model.cfg.device)
        with torch.no_grad():
            _, cache = model.run_with_cache(batch, names_filter=hook_point, prepend_bos=False)
        resid = cache[hook_point]  # [B, L, d_model]
        last = resid[:, -1, :].detach().cpu()
        ent_acts: List[Optional[torch.Tensor]] = []
        for r, (text, entity, orig_len) in enumerate(zip(texts, entities, orig_lens)):
            act = None
            if entity:
                idx_in_text = text.find(entity)
                if idx_in_text >= 0:
                    prefix = text[:idx_in_text + len(entity)]
                    # entity_tok_idx: 0-based index into the BOS-prepended token sequence.
                    entity_tok_idx = tok(prefix, return_tensors="pt", add_special_tokens=True).input_ids.shape[1] - 1
                    # Map into left-padded batch position.
                    padded_idx = L - orig_len + entity_tok_idx
                    padded_idx = max(0, min(padded_idx, L - 1))
                    act = resid[r, padded_idx, :].detach().cpu()
            ent_acts.append(act)
        return last, ent_acts

    for i in tqdm(range(0, len(samples), batch_size), desc="ctx-only acts"):
        b = samples[i:i + batch_size]
        pos_entities = [s["factual_answer"][0] if (want_entity_pos and s.get("factual_answer")) else None for s in b]
        neg_entities = [s["non_factual_answer"][0] if (want_entity_pos and s.get("non_factual_answer")) else None for s in b]
        pos_last_b, pos_ent_b = run_batch([s["factual_context"] for s in b], pos_entities)
        neg_last_b, neg_ent_b = run_batch([s["non_factual_evidence"] for s in b], neg_entities)
        pos_last.extend(pos_last_b.unbind(0))
        neg_last.extend(neg_last_b.unbind(0))
        if want_entity_pos:
            for pe, ne in zip(pos_ent_b, neg_ent_b):
                if pe is not None and ne is not None:
                    pos_ent.append(pe)
                    neg_ent.append(ne)

    out = {"last_pos": {"pos": torch.stack(pos_last), "neg": torch.stack(neg_last)}}
    if want_entity_pos and pos_ent:
        out["entity_pos"] = {"pos": torch.stack(pos_ent), "neg": torch.stack(neg_ent)}
    return out


def collect_ab_choice(
    model,
    hook_point: str,
    samples: List[Dict],
    batch_size: int = 8,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Builds an A/B-choice prompt per sample with the factual chunk randomly placed at A or B.
    Pos = prompt ending with the correct label, Neg = prompt ending with the wrong label.
    Returns activations at the last (label) token.
    """
    tok = model.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    rng = random.Random(AB_CHOICE_SEED)
    n = len(samples)
    factual_is_a = [True] * (n // 2) + [False] * (n - n // 2)
    rng.shuffle(factual_is_a)

    pos_acts, neg_acts = [], []

    def run_batch(texts: List[str]) -> torch.Tensor:
        enc = [tok(t, return_tensors="pt", add_special_tokens=True).input_ids[0] for t in texts]
        orig_lens = [e.shape[0] for e in enc]
        L = max(orig_lens)
        batch = torch.full((len(texts), L), tok.pad_token_id, dtype=torch.long, device=model.cfg.device)
        for r, e in enumerate(enc):
            batch[r, L - e.shape[0]:] = e.to(model.cfg.device)
        with torch.no_grad():
            _, cache = model.run_with_cache(batch, names_filter=hook_point, prepend_bos=False)
        return cache[hook_point][:, -1, :].detach().cpu()

    zipped = list(zip(samples, factual_is_a))
    for i in tqdm(range(0, len(zipped), batch_size), desc="ab-choice acts"):
        b = zipped[i:i + batch_size]
        pos_texts, neg_texts = [], []
        for s, fact_a in b:
            if fact_a:
                ctx_a, ctx_b = s["factual_context"], s["non_factual_evidence"]
                correct, wrong = "A", "B"
            else:
                ctx_a, ctx_b = s["non_factual_evidence"], s["factual_context"]
                correct, wrong = "B", "A"
            pos_texts.append(make_ab_choice_prompt(ctx_a, ctx_b, correct))
            neg_texts.append(make_ab_choice_prompt(ctx_a, ctx_b, wrong))
        pos_acts.extend(run_batch(pos_texts).unbind(0))
        neg_acts.extend(run_batch(neg_texts).unbind(0))

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
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of samples per forward pass.")
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
    model = HookedTransformer.from_pretrained_no_processing(args.model, device=device, dtype="bfloat16")
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
                   for proc, pos in [("context_only", "last_pos")]):#, ("ab_choice", "choice_token")]):
                logger.info(f"Skip layer {layer} (already computed).")
                continue
            hook_point = tl_utils.get_act_name("resid_post", layer)
            logger.info(f"=== Layer {layer} ({hook_point}) ===")

            # Procedure 1: context_only
            logger.info("-> procedure: context_only")
            ctx_acts = collect_context_only(model, hook_point, samples, want_entity_pos=want_entity_pos, batch_size=args.batch_size)
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
            #logger.info("-> procedure: ab_choice")
            #ab_acts = collect_ab_choice(model, hook_point, samples, batch_size=args.batch_size)
            #for pos_name, stacks in ab_acts.items():
            #    direction = diff_in_means(stacks["pos"], stacks["neg"], normalize=False)
            #    save_direction(
            #        out_root / "ab_choice" / f"layer_{layer}" / pos_name,
            #        direction, stacks["pos"], stacks["neg"],
            #        {"model": args.model, "dataset": args.dataset, "layer": layer, "seed": seed,
            #         "procedure": "ab_choice", "position": pos_name, "ab_seed": AB_CHOICE_SEED},
            #    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
