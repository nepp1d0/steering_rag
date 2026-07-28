"""
Retrieval evaluation for the MIXED-dataset directions (see mixed_direction_identification.py).

Same fusion as retrieval_evaluation.py:

    score(d, q) = (1 - alpha) * zscore(s_cos) + alpha * zscore(s_proj)

but the direction comes from results/mixed_directions/ and is indexed by a dataset
*combo* ("conflictqa", "conflictqa+nq_swap", ...) instead of a single dataset.

Loop shape differs from retrieval_evaluation.py on purpose: `llm_hidden_states` depends
only on (model, eval_dataset, layer) — the direction enters afterwards as a dot product —
so it is computed once per layer and reused across all 7 combos. Same for the SBERT
embeddings. This avoids recomputing identical tensors 7x (and reloading the model inside
the layer loop, as the original does in automated mode).

Mixed directions exist for seed 42 only, so only that split is evaluated.

Outputs:
    <root>/<model>/<eval_ds>/cache/layer_<L>/{llm_hidden_states.pt,docs.jsonl}   (shared)
    <root>/<model>/<eval_ds>/cache/sbert_embeddings.pt                           (shared)
    <root>/<model>/<eval_ds>/<combo>/unnormalized/seed_42/context_only/layer_<L>/results.jsonl

The retrieved documents are not inlined into results.jsonl (only topk_indices); resolve
them against the corpus in cache/layer_<L>/docs.jsonl.

where <root> = results/mixed_directions_retrieval_evaluation.

Usage:
    python -m src.experiments.mixed_directions_retrieval_evaluation
    python -m src.experiments.mixed_directions_retrieval_evaluation --models google/gemma-3-4b-it
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import RESULTS_DIR, load_normalized, logger, safe_model_id, setup_logging, write_jsonl

import transformer_lens.utils as tl_utils
from transformer_lens import HookedTransformer
from sentence_transformers import SentenceTransformer

MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "google/gemma-3-4b-it",
    "Qwen/Qwen2-7B-Instruct",
]
EVAL_DATASETS = ["nq_swap", "conflictqa"]

# Direction sources: the combos written by mixed_direction_identification.py.
COMBOS = [
    "conflictqa",
    "nq_swap",
    "longfact",
    "conflictqa+nq_swap",
    "conflictqa+longfact",
    "nq_swap+longfact",
    "conflictqa+nq_swap+longfact",
]

SEED = 42                  # mixed directions are computed for seed 42 only
PROCEDURE = "context_only"
POSITION = "last_pos"
NORMALIZE_PATH = "unnormalized"   # directions are used raw, as in retrieval_evaluation.py
ALPHAS = [0.0, 0.3, 0.5, 1.0]
KS = [2, 5, 10]
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 4

OUT_ROOT = RESULTS_DIR / "mixed_directions_retrieval_evaluation"
DIRECTIONS_ROOT = RESULTS_DIR / "mixed_directions"


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std())


def direction_path(model_id: str, combo: str, layer: int) -> Path:
    return (DIRECTIONS_ROOT / safe_model_id(model_id) / combo / f"seed_{SEED}"
            / PROCEDURE / f"layer_{layer}" / POSITION / "direction.pt")


def compute_llm_hidden_states(model: HookedTransformer, docs: List[str], layer: int, batch_size: int) -> torch.Tensor:
    hook_point = tl_utils.get_act_name("resid_post", layer)
    tok = model.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    enc = [tok(d, return_tensors="pt", add_special_tokens=True).input_ids[0] for d in docs]
    order = sorted(range(len(docs)), key=lambda i: enc[i].shape[0])
    hidden = torch.zeros(len(docs), model.cfg.d_model)

    for s in tqdm(range(0, len(order), batch_size), desc=f"hidden states L{layer}"):
        idxs = order[s: s + batch_size]
        L = max(enc[i].shape[0] for i in idxs)
        batch = torch.full((len(idxs), L), tok.pad_token_id, dtype=torch.long, device=model.cfg.device)
        for r, i in enumerate(idxs):
            t = enc[i]
            batch[r, L - t.shape[0]:] = t.to(model.cfg.device)
        with torch.no_grad():
            # stop_at_layer skips the blocks above `layer` and the unembed, whose [B, L, d_vocab]
            # logits are built and discarded otherwise (5 GiB/batch on gemma's 262k vocab).
            _, cache = model.run_with_cache(batch, names_filter=hook_point,
                                            stop_at_layer=layer + 1, prepend_bos=False)
        resid = cache[hook_point][:, -1, :].detach().float().cpu()
        for r, i in enumerate(idxs):
            hidden[i] = resid[r]
    return hidden


def evaluate_combo(
    llm_hidden: torch.Tensor,
    direction: torch.Tensor,
    sbert_norm: torch.Tensor,
    q_embs_norm: torch.Tensor,
    samples: List[dict],
    doc_idx: dict,
    out_dir: Path,
) -> None:
    """Rank the corpus for every query and write results.jsonl. Same scoring as the original."""
    s_proj_all = (llm_hidden @ direction).numpy()  # [N]
    s_proj_norm_global = zscore(s_proj_all)

    records = []
    for si, sample in enumerate(tqdm(samples, desc="Evaluating", leave=False)):
        gold_idx = doc_idx[sample["factual_context"]]
        nf_idx = doc_idx[sample["non_factual_evidence"]]

        s_cos = (sbert_norm @ q_embs_norm[si]).numpy()  # [N]
        s_cos_norm = zscore(s_cos)

        for alpha in ALPHAS:
            scores = (1 - alpha) * s_cos_norm + alpha * s_proj_norm_global
            sorted_indices = np.argsort(-scores)  # descending
            gold_rank = np.where(sorted_indices == gold_idx)[0][0] + 1
            nf_rank = np.where(sorted_indices == nf_idx)[0][0] + 1
            for k in KS:
                topk_indices = sorted_indices[:k].tolist()
                records.append({
                    "sample_idx": si,
                    "question": sample["question"],
                    "alpha": alpha,
                    "k": k,
                    "gold_in_topk": bool(gold_rank <= k),
                    "nonfactual_in_topk": bool(nf_rank <= k),
                    "gold_rank": int(gold_rank),
                    "nonfactual_rank": int(nf_rank),
                    # No topk_text: the retrieved documents are recoverable from
                    # topk_indices + the cache's docs.jsonl, and inlining them here made
                    # each results.jsonl ~75 MB (62 GB over the full grid, filling the disk).
                    "topk_indices": topk_indices,
                })

    write_jsonl(out_dir / "results.jsonl", records)
    logger.info(f"Wrote {len(records)} records -> {out_dir / 'results.jsonl'}")

    for alpha in ALPHAS:
        for k in KS:
            rows = [r for r in records if r["alpha"] == alpha and r["k"] == k]
            gold_rate = sum(r["gold_in_topk"] for r in rows) / len(rows)
            nf_rate = sum(r["nonfactual_in_topk"] for r in rows) / len(rows)
            logger.info(f"  alpha={alpha:.1f} k={k:2d} | gold_rate@k={gold_rate:.3f} | nonfactual_rate@k={nf_rate:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieval evaluation with mixed-dataset directions.")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--eval-datasets", nargs="+", default=EVAL_DATASETS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    setup_logging("mixed_directions_retrieval_evaluation", OUT_ROOT)
    logger.info(f"models={args.models} | eval_datasets={args.eval_datasets} | combos={COMBOS} | seed={SEED}")

    sbert_enc = SentenceTransformer(SBERT_MODEL)
    device = tl_utils.get_device()

    for model_name in args.models:
        # Skip models whose mixed directions have not been computed yet.
        model_dir = DIRECTIONS_ROOT / safe_model_id(model_name)
        if not model_dir.is_dir():
            logger.warning(f"No mixed directions for {model_name} ({model_dir}), skipping model.")
            continue

        logger.info(f"=== Model: {model_name} ===")
        model = HookedTransformer.from_pretrained_no_processing(model_name, device=device, dtype="bfloat16")
        model.eval()
        layers = list(range(model.cfg.n_layers))

        for eval_ds in args.eval_datasets:
            logger.info(f"--- Eval dataset: {eval_ds} (seed {SEED}) ---")
            samples = load_normalized(eval_ds, SEED)["test"]
            all_docs = sorted(set(s["factual_context"] for s in samples)
                              | set(s["non_factual_evidence"] for s in samples))
            doc_idx = {d: i for i, d in enumerate(all_docs)}
            logger.info(f"Corpus size: {len(all_docs)} unique documents, {len(samples)} queries")

            cache_dir = OUT_ROOT / safe_model_id(model_name) / eval_ds / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # SBERT doc + query embeddings: identical for every layer and combo.
            sbert_cache = cache_dir / "sbert_embeddings.pt"
            if sbert_cache.exists():
                sbert_emb = torch.load(sbert_cache, map_location="cpu")
            else:
                logger.info("Computing SBERT embeddings ...")
                emb = sbert_enc.encode(all_docs, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
                sbert_emb = torch.tensor(emb, dtype=torch.float32)
                torch.save(sbert_emb, sbert_cache)
            sbert_norm = sbert_emb / sbert_emb.norm(dim=1, keepdim=True)

            q_embs = sbert_enc.encode([s["question"] for s in samples], batch_size=64,
                                      show_progress_bar=True, convert_to_numpy=True)
            q_embs_norm = torch.tensor(q_embs, dtype=torch.float32)
            q_embs_norm = q_embs_norm / q_embs_norm.norm(dim=1, keepdim=True)

            for layer in layers:
                out_dirs = {
                    c: (OUT_ROOT / safe_model_id(model_name) / eval_ds / c / NORMALIZE_PATH
                        / f"seed_{SEED}" / PROCEDURE / f"layer_{layer}")
                    for c in COMBOS
                }
                todo = [c for c in COMBOS
                        if not (out_dirs[c] / "results.jsonl").exists() and direction_path(model_name, c, layer).exists()]
                missing = [c for c in COMBOS if not direction_path(model_name, c, layer).exists()]
                if missing:
                    logger.warning(f"Layer {layer}: no direction for combos {missing}, skipping those.")
                if not todo:
                    logger.info(f"Skip layer {layer} (all combos done or unavailable).")
                    continue

                # Hidden states: shared by every combo at this layer.
                layer_cache = cache_dir / f"layer_{layer}"
                layer_cache.mkdir(parents=True, exist_ok=True)
                hidden_cache = layer_cache / "llm_hidden_states.pt"
                if hidden_cache.exists():
                    llm_hidden = torch.load(hidden_cache, map_location="cpu")
                else:
                    llm_hidden = compute_llm_hidden_states(model, all_docs, layer, args.batch_size)
                    torch.save(llm_hidden, hidden_cache)
                    write_jsonl(layer_cache / "docs.jsonl", list(enumerate(all_docs)))

                for combo in todo:
                    logger.info(f"model={model_name} | eval={eval_ds} | combo={combo} | layer={layer}")
                    out_dir = out_dirs[combo]
                    out_dir.mkdir(parents=True, exist_ok=True)
                    direction = torch.load(direction_path(model_name, combo, layer), map_location="cpu").float()
                    evaluate_combo(llm_hidden, direction, sbert_norm, q_embs_norm,
                                   samples, doc_idx, out_dir)

                del llm_hidden

        del model

    logger.info("Done.")


if __name__ == "__main__":
    main()
