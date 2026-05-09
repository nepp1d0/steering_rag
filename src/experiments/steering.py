"""
Step 2 - Steering experiment.

For every direction found under
    results/direction_identification/<model>/<id_dataset>/<procedure>/layer_<L>/<position>/direction.pt
we generate answers on the eval dataset both without (baseline) and with steering,
and dump them to a per-direction `runs.jsonl`. Annotation/judging is a separate step
(`judge.py`).

Steering = add `alpha * d` to the residual stream at every position of the chosen
layer (via a TransformerLens forward hook on `blocks.{L}.hook_resid_post`).

Generation is BATCHED: prompts are sorted by token length and grouped into buckets
of similar length (so the amount of left-padding inside a batch is small). This
matters because TransformerLens 2.x does NOT mask padding tokens during batched
`generate` (issue #1005), so heavy padding could bias outputs vs single-prompt
generation. Tight buckets keep that bias negligible.

Usage:
    python -m src.experiments.steering \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --eval-dataset nq_swap \
        --alpha 4.0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import (
    RESULTS_DIR,
    load_normalized,
    logger,
    safe_model_id,
    setup_logging,
)

import transformer_lens.utils as tl_utils
from transformer_lens import HookedTransformer


SHUFFLE_SEED = 42
MAX_NEW_TOKENS_DEFAULT = 32
BATCH_SIZE_DEFAULT = 16


def build_rag_prompt(model: HookedTransformer, sample: Dict, rng: random.Random) -> Tuple[str, List[str]]:
    """Build a chat-formatted RAG prompt with the two chunks shuffled. Returns (prompt, doc_order)."""
    chunks = [("factual", sample["factual_context"]), ("non_factual", sample["non_factual_evidence"])]
    rng.shuffle(chunks)
    doc_order = [c[0] for c in chunks]
    user = (
        "You are given some retrieved evidence. Answer the question concisely using only the evidence.\n\n"
        "If the documents are conflicting, answer the question based on the factual evidence."
        f"[Document 1]\n{chunks[0][1]}\n\n"
        f"[Document 2]\n{chunks[1][1]}\n\n"
        f"Question: {sample['question']}\nAnswer:"
    )
    messages = [{"role": "user", "content": user}]
    prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt, doc_order


def batched_generate(model: HookedTransformer, prompts: List[str], max_new_tokens: int, batch_size: int, desc: str) -> List[str]:
    """Greedy batched generation. Sorts by length, left-pads inside each bucket, returns
    only the newly generated text per prompt, in the original input order."""
    tok = model.tokenizer
    # Llama tokenizer ships without a pad token; reuse EOS and left-pad so all rows end at the same position.
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Tokenize once. add_special_tokens=False because the chat template already includes <|begin_of_text|>.
    enc = [tok(p, return_tensors="pt", add_special_tokens=False).input_ids[0] for p in prompts]
    # Sort indices by token length so each batch contains similarly-sized prompts (minimal padding).
    order = sorted(range(len(prompts)), key=lambda i: enc[i].shape[0])
    out: List[str] = [""] * len(prompts)

    for s in tqdm(range(0, len(order), batch_size), desc=desc):
        idxs = order[s : s + batch_size]
        L = max(enc[i].shape[0] for i in idxs)
        logger.info(f"Batch size: {len(idxs)} | Max length: {L}")
        if L > 500: # max 500 tokens for GPU memory
            logger.warning(f"Skipping batch of size {len(idxs)} because it exceeds 500 tokens.")
            break
        # Left-pad with pad_token_id up to the bucket's max length.
        batch = torch.full((len(idxs), L), tok.pad_token_id, dtype=torch.long, device=model.cfg.device)
        for r, i in enumerate(idxs):
            t = enc[i]
            batch[r, L - t.shape[0]:] = t

        with torch.no_grad():
            gen = model.generate(
                batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                prepend_bos=False,
                verbose=False,
            )
        # `gen` has shape [B, L + new]; the new tokens are the tail.
        new_tokens = gen[:, L:]
        for r, i in enumerate(idxs):
            out[i] = tok.decode(new_tokens[r], skip_special_tokens=True).strip()
    return out


def find_directions(model_id: str) -> List[Path]:
    root = RESULTS_DIR / "direction_identification" / safe_model_id(model_id)
    return sorted(root.rglob("direction.pt"))


def parse_direction_path(p: Path, model_id: str) -> Dict:
    """results/direction_identification/<model>/<id_dataset>/<procedure>/layer_<L>/<position>/direction.pt"""
    rel = p.relative_to(RESULTS_DIR / "direction_identification" / safe_model_id(model_id))
    id_dataset, procedure, layer_dir, position = rel.parts[:4]
    return {
        "id_dataset": id_dataset,
        "procedure": procedure,
        "layer": int(layer_dir.split("_")[1]),
        "position": position,
    }


def load_direction(path: Path, normalize: bool, device, dtype) -> torch.Tensor:
    d = torch.load(path, map_location="cpu").float()
    if normalize:
        d = d / (d.norm() + 1e-8)
    return d.to(device=device, dtype=dtype)


def main() -> None:
    ap = argparse.ArgumentParser(description="Steering experiment over all identified directions.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval-dataset", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--normalize", action="store_true", help="L2-normalize directions before applying alpha (default: false).")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    args = ap.parse_args()

    out_root = RESULTS_DIR / "steering" / safe_model_id(args.model) / args.eval_dataset
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging("steering", out_root)
    logger.info(f"model={args.model} | eval={args.eval_dataset} | alpha={args.alpha} | normalize={args.normalize} | bs={args.batch_size}")

    samples = load_normalized(args.eval_dataset)
    logger.info(f"Loaded {len(samples)} samples for '{args.eval_dataset}'.")

    direction_files = find_directions(args.model)
    if not direction_files:
        logger.error(f"No directions found for model {args.model}.")
        return
    logger.info(f"Found {len(direction_files)} directions.")

    device = tl_utils.get_device()
    logger.info(f"Loading model on {device} ...")
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

        # Hook adds alpha*direction to every position of the residual stream at this layer.
        def steer_hook(resid, hook, _v=d, _a=args.alpha):
            return resid + _a * _v

        with model.hooks(fwd_hooks=[(hook_point, steer_hook)]):
            steered = batched_generate(model, prompts, args.max_new_tokens, args.batch_size, f"steer L{info['layer']}/{info['position']}")

        with runs_path.open("w") as f:
            for i, (sample, (prompt, order), b, s) in enumerate(zip(samples, prompts_and_orders, baseline, steered)):
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
