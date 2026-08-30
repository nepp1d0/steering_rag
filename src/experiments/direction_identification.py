"""
Step 1 - Direction identification via diff-in-means on the residual stream.

Two procedures are run for every call:

  - "context_only": positive activations come from the factual context, negative ones
    from the non-factual evidence. Two positions for every dataset:
      * last_pos:   last token of the chunk.
      * entity_pos: last token of the first occurrence of the answer entity inside the
        chunk. The entity is located from the evidence side only (the conflictqa
        templated answer strings are never used for locating):
          - nq_swap:    factual_answer[0] / non_factual_answer[0] are verbatim spans.
          - conflictqa: pos = earliest ground_truth alias inside factual_context;
                        neg = extractive-QA span (QA_MODEL_ID) for
                              (question, non_factual_evidence).
          - longfact:   verified-entity texts attached by add_longfact_entities.py
                        (verbatim substrings of the sentence).
        Samples whose entity cannot be located are dropped from entity_pos (they
        still contribute to last_pos).

  - "ab_choice": each sample is turned into a single A/B-choice prompt where both
    chunks are shown and a label ("A" or "B") is appended. The pos prompt ends with
    the label of the factual chunk; the neg prompt ends with the wrong label. The
    A/B ordering is shuffled (seed 42) so neither side is always "A".
    Position: `choice_token` (last token, i.e. right after the appended label).

Outputs:
    results/direction_identification/<model>/<dataset>/<procedure>/layer_<L>/<position>/{direction.pt,meta.json}

Usage:
    python src/experiments/direction_identification.py --automated
    python src/experiments/direction_identification.py --automated --force-recompute
    # --automated over every dataset, but only this model and these layers:
    python src/experiments/direction_identification.py --automated \
        --model Qwen/Qwen2-7B-Instruct --layers 0,18,22,24,26 --seed 42 --force-recompute
    python src/experiments/direction_identification.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --dataset nq_swap \
        --layers 10,15,20
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from utils import (
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


# Same model list as retrieval_evaluation.py (keep in sync).
MODELS = ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-1B-Instruct", "google/gemma-3-4b-it", "Qwen/Qwen2-7B-Instruct"]

# Per-dataset positions used by the "context_only" procedure.
CONTEXT_ONLY_POSITIONS: Dict[str, List[str]] = {
    "nq_swap":    ["last_pos", "entity_pos"],
    "conflictqa": ["last_pos", "entity_pos"],
    "longfact":   ["last_pos", "entity_pos"],
}

AB_CHOICE_SEED = 42

# How many layers share one forward pass. Larger = fewer passes but more RAM, since every
# layer in the group holds activations for the whole side (roughly
# n_texts * n_layers * d_model * 2 bytes). 8 keeps a full-depth run near the old per-layer
# memory profile while still cutting a 5-6 layer run to a single pass.
LAYERS_PER_PASS = 8

# Extractive QA model used to locate the claimed answer inside conflictqa's
# non-factual evidence (the wrong entity exists nowhere else in the data).
QA_MODEL_ID = "deepset/roberta-base-squad2"
QA_MAX_LENGTH = 384
QA_MAX_ANSWER_TOKENS = 30


def parse_layers(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def discover_seeds(dataset: str) -> List[int]:
    dirs = sorted((NORMALIZED_DIR / dataset).glob("seed_*"), key=lambda d: int(d.name.split("_")[1]))
    return [int(d.name.split("_")[1]) for d in dirs if d.is_dir()]


# ---------------------------------------------------------------------------
# Entity-span resolution (char spans; token indices are derived at collection time)
# ---------------------------------------------------------------------------

def find_earliest_span(text: str, needles: List[str], ignore_case: bool = False,
                       word_boundary: bool = False) -> Optional[Tuple[int, int]]:
    """Char span (start, end) of the earliest occurrence of any needle in text."""
    hay = text.lower() if ignore_case else text
    best: Optional[Tuple[int, int]] = None
    for needle in needles:
        if not needle:
            continue
        nd = needle.lower() if ignore_case else needle
        if word_boundary:
            m = re.search(r"\b" + re.escape(nd) + r"\b", hay)
            start = m.start() if m else -1
        else:
            start = hay.find(nd)
        if start >= 0 and (best is None or start < best[0]):
            best = (start, start + len(nd))
    return best


def compute_conflictqa_qa_spans(samples: List[Dict], batch_size: int = 64) -> List[Optional[Tuple[int, int]]]:
    """Char span of the claimed (wrong) answer inside each non_factual_evidence.

    Extractive QA on (question, evidence): the span the evidence itself offers as
    the answer to the question. Loads and frees the QA model internally; call this
    before the big model is loaded.
    """
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading QA model {QA_MODEL_ID} on {device} ...")
    qa_tok = AutoTokenizer.from_pretrained(QA_MODEL_ID)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_ID).to(device).eval()

    spans: List[Optional[Tuple[int, int]]] = []
    for i in tqdm(range(0, len(samples), batch_size), desc="conflictqa QA spans"):
        b = samples[i:i + batch_size]
        enc = qa_tok([s["question"] for s in b], [s["non_factual_evidence"] for s in b],
                     return_tensors="pt", truncation="only_second", max_length=QA_MAX_LENGTH,
                     padding=True, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        seq_ids = [enc.sequence_ids(j) for j in range(len(b))]
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            res = qa_model(**enc)
        start_logits, end_logits = res.start_logits.cpu(), res.end_logits.cpu()
        for j in range(len(b)):
            # Restrict the span to context tokens (sequence id 1).
            mask = torch.tensor([sid != 1 for sid in seq_ids[j]])
            s_log = start_logits[j].masked_fill(mask, -1e9)
            e_log = end_logits[j].masked_fill(mask, -1e9)
            si = int(s_log.argmax())
            ei = int(e_log[si:si + QA_MAX_ANSWER_TOKENS].argmax()) + si
            c_start, c_end = int(offsets[j][si][0]), int(offsets[j][ei][1])
            spans.append((c_start, c_end) if c_end > c_start else None)

    del qa_model
    torch.cuda.empty_cache()
    return spans


def resolve_side_spans(dataset: str, samples: List[Dict], side: str,
                       qa_spans: Optional[List[Optional[Tuple[int, int]]]] = None,
                       ) -> List[Tuple[str, Optional[Tuple[int, int]]]]:
    """[(text, char_span_or_None)] for one side; rows with an empty chunk are skipped.

    The char span marks the first occurrence of the answer entity in the text; None
    means the sample is dropped from entity_pos (it still contributes to last_pos).
    `qa_spans` must be aligned with `samples` (conflictqa neg side only).
    """
    key_text = "factual_context" if side == "pos" else "non_factual_evidence"
    key_ans = "factual_answer" if side == "pos" else "non_factual_answer"
    out: List[Tuple[str, Optional[Tuple[int, int]]]] = []
    for i, s in enumerate(samples):
        text = s[key_text]
        if not text:
            continue
        span = None
        if dataset == "nq_swap":
            ans = s.get(key_ans) or []
            span = find_earliest_span(text, ans[:1]) or find_earliest_span(text, ans[:1], ignore_case=True)
        elif dataset == "conflictqa":
            if side == "pos":
                aliases = s.get("ground_truth") or []
                span = (find_earliest_span(text, aliases, ignore_case=True, word_boundary=True)
                        or find_earliest_span(text, aliases, ignore_case=True))
            else:
                span = qa_spans[i] if qa_spans is not None else None
        elif dataset == "longfact":
            ents = s.get(key_ans) or []
            span = find_earliest_span(text, ents) or find_earliest_span(text, ents, ignore_case=True)
        out.append((text, span))
    return out


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

def collect_side_acts_layers(
    model,
    layers: List[int],
    items: List[Tuple[str, Optional[Tuple[int, int]]]],
    batch_size: int = 8,
    desc: str = "acts",
) -> Dict[int, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
    """Residual-stream activations for one side, at several layers in ONE forward pass.

    `items` is [(text, char_span_or_None)]. Returns {layer: (last [n, d_model],
    entity [m, d_model])} where m counts the resolved spans; either tensor is None if
    nothing was collected.

    Collecting several layers at once costs one pass to max(layers) instead of one pass
    per layer. Memory scales with len(layers): activations for every requested layer are
    held for the whole side, so callers should chunk long layer lists (main() uses
    LAYERS_PER_PASS).
    """
    tok = model.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    layers = sorted(set(layers))
    hooks = {lay: tl_utils.get_act_name("resid_post", lay) for lay in layers}
    wanted = set(hooks.values())

    last_acts: Dict[int, List[torch.Tensor]] = {lay: [] for lay in layers}
    ent_acts: Dict[int, List[torch.Tensor]] = {lay: [] for lay in layers}
    for i in tqdm(range(0, len(items), batch_size), desc=desc):
        chunk = items[i:i + batch_size]
        texts = [t for t, _ in chunk]
        # Tokenize individually with BOS to know each sequence's original length.
        enc = [tok(t, return_tensors="pt", add_special_tokens=True).input_ids[0] for t in texts]
        orig_lens = [e.shape[0] for e in enc]
        maxlen = max(orig_lens)
        # Left-pad the batch.
        batch = torch.full((len(texts), maxlen), tok.pad_token_id, dtype=torch.long, device=model.cfg.device)
        # The mask must be passed explicitly: TransformerLens only derives one when
        # tokenizer.padding_side == "left" (false by default for Llama and Qwen), and its
        # derivation masks every occurrence of pad_token_id -- which equals eos_token here,
        # so real end-of-text tokens inside a chunk would be masked too. Without a mask the
        # real tokens attend to the pad tokens, making a text's activation depend on how
        # much padding it received, i.e. on which other texts share its batch.
        mask = torch.zeros((len(texts), maxlen), dtype=torch.long, device=model.cfg.device)
        for r, e in enumerate(enc):
            batch[r, maxlen - e.shape[0]:] = e.to(model.cfg.device)
            mask[r, maxlen - e.shape[0]:] = 1
        with torch.no_grad():
            # stop_at_layer skips every block above the deepest requested layer and the
            # unembed. Without it the logits tensor ([B, L, d_vocab]) is built and thrown
            # away: GiB-scale on large-vocab models, which fragments the allocator.
            _, cache = model.run_with_cache(batch, names_filter=lambda n: n in wanted,
                                            attention_mask=mask,
                                            stop_at_layer=max(layers) + 1, prepend_bos=False)

        # Entity token indices depend only on the tokenization, not on the layer, so they
        # are resolved once per batch and reused for every requested layer.
        ent_positions: List[Tuple[int, int]] = []
        for r, (text, span) in enumerate(chunk):
            if span is None:
                continue
            prefix = text[:span[1]]
            # entity_tok_idx: 0-based index into the BOS-prepended token sequence of the
            # last token covering the entity span.
            entity_tok_idx = tok(prefix, return_tensors="pt", add_special_tokens=True).input_ids.shape[1] - 1
            # Map into left-padded batch position.
            padded_idx = maxlen - orig_lens[r] + entity_tok_idx
            ent_positions.append((r, max(0, min(padded_idx, maxlen - 1))))

        for lay in layers:
            resid = cache[hooks[lay]]  # [B, L, d_model]
            last_acts[lay].extend(resid[:, -1, :].detach().cpu().unbind(0))
            for r, padded_idx in ent_positions:
                ent_acts[lay].append(resid[r, padded_idx, :].detach().cpu())
        del cache

    return {lay: (torch.stack(last_acts[lay]) if last_acts[lay] else None,
                  torch.stack(ent_acts[lay]) if ent_acts[lay] else None)
            for lay in layers}


def collect_side_acts(
    model,
    hook_point: str,
    layer: int,
    items: List[Tuple[str, Optional[Tuple[int, int]]]],
    batch_size: int = 8,
    desc: str = "acts",
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Single-layer wrapper around `collect_side_acts_layers` (kept for existing callers).

    `hook_point` is redundant with `layer` and is ignored; it is retained so the
    signature stays compatible with mixed_direction_identification.py and
    probe_direction_identification.py.
    """
    return collect_side_acts_layers(model, [layer], items, batch_size, desc)[layer]


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
    parser.add_argument("--model", default=None, help="HuggingFace model id (e.g. meta-llama/Llama-3.1-8B-Instruct).")
    parser.add_argument("--dataset", default=None, choices=list(CONTEXT_ONLY_POSITIONS.keys()),
                        help="Normalized dataset id.")
    parser.add_argument("--automated", action="store_true",
                        help="Run all models x datasets (all seeds and layers; both positions are always "
                             "computed). --model and/or --dataset narrow it to that model/dataset.")
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
        # --model / --dataset narrow the batch when given; omit them to run everything.
        # (They used to be silently ignored here, which made it easy to think a run was
        # restricted to one model when it was in fact processing all of them.)
        models = [args.model] if args.model else MODELS
        datasets = [args.dataset] if args.dataset else list(CONTEXT_ONLY_POSITIONS)
    else:
        if not args.model or not args.dataset:
            parser.error("--model and --dataset are required without --automated")
        models, datasets = [args.model], [args.dataset]

    setup_logging("direction_identification", RESULTS_DIR / "direction_identification")
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
                out_root = RESULTS_DIR / "direction_identification" / safe_model_id(model_name) / dataset / f"seed_{seed}"
                todo = [L for L in layers if args.force_recompute or not all(
                    (out_root / "context_only" / f"layer_{L}" / pos / "direction.pt").exists()
                    for pos in ctx_positions)]
                if not todo:
                    logger.info(f"Skip {model_name} | {dataset} | seed {seed} (all layers computed).")
                    continue
                logger.info(f"=== {model_name} | {dataset} | seed {seed} | layers to do: {todo} ===")

                # Entity spans are model-independent; resolved only when there is work to do.
                # (The conflictqa QA locator is a 125M model, fine next to the loaded LLM.)
                samples = load_normalized(dataset, seed)["train"]
                logger.info(f"Loaded {len(samples)} train samples.")
                qa_spans = compute_conflictqa_qa_spans(samples) if dataset == "conflictqa" else None
                side_data = {side: resolve_side_spans(dataset, samples, side, qa_spans) for side in ("pos", "neg")}
                for side in ("pos", "neg"):
                    n_texts = len(side_data[side])
                    n_resolved = sum(1 for _, sp in side_data[side] if sp is not None)
                    logger.info(f"Seed {seed} {side}: {n_texts} texts | entity spans resolved: "
                                f"{n_resolved} ({n_texts - n_resolved} dropped from entity_pos)")

                # Layers are processed in groups: one forward pass serves every layer in
                # the group (cost of the deepest), instead of one pass per layer. The group
                # is bounded because activations for all its layers are held in RAM.
                for gi in range(0, len(todo), LAYERS_PER_PASS):
                    group = todo[gi:gi + LAYERS_PER_PASS]
                    logger.info(f"=== Layers {group} (single pass to layer {max(group)}) ===")

                    # Procedure 1: context_only
                    logger.info("-> procedure: context_only")
                    pos_by_layer = collect_side_acts_layers(model, group, side_data["pos"],
                                                            batch_size=args.batch_size, desc="pos acts")
                    neg_by_layer = collect_side_acts_layers(model, group, side_data["neg"],
                                                            batch_size=args.batch_size, desc="neg acts")
                    for layer in group:
                        pos_last, pos_ent = pos_by_layer[layer]
                        neg_last, neg_ent = neg_by_layer[layer]
                        stacks_by_pos = {"last_pos": (pos_last, neg_last), "entity_pos": (pos_ent, neg_ent)}
                        for pos_name in ctx_positions:
                            p_stack, n_stack = stacks_by_pos[pos_name]
                            if p_stack is None or n_stack is None:
                                logger.warning(f"Skipping layer {layer} {pos_name}: no activations collected.")
                                continue
                            direction = diff_in_means(p_stack, n_stack, normalize=False)
                            save_direction(
                                out_root / "context_only" / f"layer_{layer}" / pos_name,
                                direction, p_stack, n_stack,
                                {"model": model_name, "dataset": dataset, "layer": layer, "seed": seed,
                                 "procedure": "context_only", "position": pos_name,
                                 "n_pos": int(p_stack.shape[0]), "n_neg": int(n_stack.shape[0])},
                            )
                    del pos_by_layer, neg_by_layer

                    # Procedure 2: ab_choice (disabled)
                    #logger.info("-> procedure: ab_choice")
                    #ab_acts = collect_ab_choice(model, hook_point, samples, batch_size=args.batch_size)
                    #for pos_name, stacks in ab_acts.items():
                    #    direction = diff_in_means(stacks["pos"], stacks["neg"], normalize=False)
                    #    save_direction(
                    #        out_root / "ab_choice" / f"layer_{layer}" / pos_name,
                    #        direction, stacks["pos"], stacks["neg"],
                    #        {"model": model_name, "dataset": dataset, "layer": layer, "seed": seed,
                    #         "procedure": "ab_choice", "position": pos_name, "ab_seed": AB_CHOICE_SEED},
                    #    )

        del model
        torch.cuda.empty_cache()

    logger.info("Done.")


if __name__ == "__main__":
    main()
