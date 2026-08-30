"""
RAGuard step B - last-token residual-stream activations for every RAGuard claim.

One forward pass per batch caching ALL layers at once (names_filter on resid_post), so a
model is traversed a single time regardless of how many layers we want. Claims are fed as
BARE TEXT, no prompt template, matching how the directions were extracted (direction
identification feeds raw `factual_context` / `non_factual_evidence` strings).

Outputs, per model:
    results/raguard/<model>/hidden_states/layer_<L>.pt   float32 [n_claims, d_model]
    results/raguard/<model>/hidden_states/meta.json      claim_ids in row order + shapes

The `--variant` flag selects the input text: `raw` (default, the bare claim) or
`statement` (claim wrapped in a minimal declarative frame), used only to check whether the
direction needs passage-shaped input. Non-raw variants write to a suffixed directory
(`hidden_states_<variant>`), so the two never overwrite each other.

Usage:
    python src/exploratory/raguard_claim_activations.py
    python src/exploratory/raguard_claim_activations.py --models meta-llama/Llama-3.2-1B-Instruct
    python src/exploratory/raguard_claim_activations.py --variant statement
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
from utils import REPO_ROOT, RESULTS_DIR, logger, safe_model_id, setup_logging

import transformer_lens.utils as tl_utils
from transformer_lens import HookedTransformer

MODELS = ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-1B-Instruct",
          "google/gemma-3-4b-it", "Qwen/Qwen2-7B-Instruct"]
CLAIMS_PATH = REPO_ROOT / "data" / "raguard" / "claims.jsonl"
OUT_ROOT = RESULTS_DIR / "raguard"
BATCH_SIZE = 32


def hidden_dir(model_name: str, variant: str) -> Path:
    suffix = "" if variant == "raw" else f"_{variant}"
    return OUT_ROOT / safe_model_id(model_name) / f"hidden_states{suffix}"


def load_claims() -> List[Dict]:
    if not CLAIMS_PATH.exists():
        raise FileNotFoundError(f"{CLAIMS_PATH} not found. Run raguard_normalization.py first.")
    with CLAIMS_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def build_text(claim: str, variant: str) -> str:
    if variant == "raw":
        return claim
    if variant == "statement":
        return f"The following statement is a matter of fact: {claim}"
    raise ValueError(f"Unknown variant '{variant}'")


def compute_all_layer_hidden(model: HookedTransformer, texts: List[str], batch_size: int) -> Dict[int, torch.Tensor]:
    """Last-token resid_post for every layer. Returns {layer: [n_texts, d_model] float32}."""
    tok = model.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    enc = [tok(t, return_tensors="pt", add_special_tokens=True).input_ids[0] for t in texts]
    # Length-sorted batches keep padding (and therefore compute) minimal.
    order = sorted(range(len(texts)), key=lambda i: enc[i].shape[0])
    n_layers = model.cfg.n_layers
    hidden = {L: torch.zeros(len(texts), model.cfg.d_model) for L in range(n_layers)}

    for s in tqdm(range(0, len(order), batch_size), desc="claim activations"):
        idxs = order[s: s + batch_size]
        maxlen = max(enc[i].shape[0] for i in idxs)
        batch = torch.full((len(idxs), maxlen), tok.pad_token_id, dtype=torch.long, device=model.cfg.device)
        # Explicit mask, same reasoning as collect_side_acts in direction_identification.py:
        # without it the real tokens attend to the pad tokens and a claim's activation
        # depends on which other claims share its batch.
        mask = torch.zeros((len(idxs), maxlen), dtype=torch.long, device=model.cfg.device)
        for r, i in enumerate(idxs):  # left-pad so index -1 is the true last token
            t = enc[i]
            batch[r, maxlen - t.shape[0]:] = t.to(model.cfg.device)
            mask[r, maxlen - t.shape[0]:] = 1
        with torch.no_grad():
            _, cache = model.run_with_cache(
                batch, names_filter=lambda n: n.endswith("resid_post"),
                attention_mask=mask, prepend_bos=False)
        for L in range(n_layers):
            resid = cache[tl_utils.get_act_name("resid_post", L)][:, -1, :].detach().float().cpu()
            for r, i in enumerate(idxs):
                hidden[L][i] = resid[r]
        del cache
    return hidden


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGuard claim activations (all layers, last token).")
    parser.add_argument("--models", nargs="+", default=MODELS, help="HuggingFace model ids.")
    parser.add_argument("--variant", default="raw", choices=["raw", "statement"],
                        help="Input text form fed to the model.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--force-recompute", action="store_true",
                        help="Recompute even if layer tensors already exist.")
    args = parser.parse_args()

    setup_logging("raguard_claim_activations", OUT_ROOT)
    claims = load_claims()
    texts = [build_text(c["claim"], args.variant) for c in claims]
    logger.info(f"{len(claims)} claims | variant={args.variant} | models={args.models}")

    device = tl_utils.get_device()
    for model_name in args.models:
        out_dir = hidden_dir(model_name, args.variant)
        meta_path = out_dir / "meta.json"
        if meta_path.exists() and not args.force_recompute:
            meta = json.loads(meta_path.read_text())
            if all((out_dir / f"layer_{L}.pt").exists() for L in range(meta["n_layers"])):
                logger.info(f"Skip {model_name} (all {meta['n_layers']} layers cached in {out_dir}).")
                continue

        logger.info(f"##### {model_name} (loading on {device}) #####")
        model = HookedTransformer.from_pretrained_no_processing(model_name, device=device, dtype="bfloat16")
        hidden = compute_all_layer_hidden(model, texts, args.batch_size)

        out_dir.mkdir(parents=True, exist_ok=True)
        for L, H in hidden.items():
            if not torch.isfinite(H).all():
                raise ValueError(f"Non-finite activations for {model_name} layer {L}")
            torch.save(H, out_dir / f"layer_{L}.pt")
        meta_path.write_text(json.dumps({
            "model": model_name,
            "variant": args.variant,
            "n_claims": len(claims),
            "n_layers": model.cfg.n_layers,
            "d_model": model.cfg.d_model,
            "claim_ids": [c["claim_id"] for c in claims],
        }, indent=2))
        logger.info(f"Saved {model.cfg.n_layers} layers of [{len(claims)}, {model.cfg.d_model}] -> {out_dir}")

        del model, hidden
        torch.cuda.empty_cache()

    logger.info("Done.")


if __name__ == "__main__":
    main()
