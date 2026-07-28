"""
Retrieval evaluation using the ICLR "LLMs Know More Than They Show" probe directions.

Same fusion as src/experiments/retrieval_evaluation.py:
    score(d, q) = (1 - alpha) * zscore(s_cos) + alpha * zscore(s_proj)

Two deliberate differences from our mean-diff retrieval eval:
  1. The direction is a logistic-regression probe coef (see adapt_iclr_directions.py),
     trained on `mlp` output activations -- NOT resid_post. So documents are projected
     at the MATCHING activation location: TransformerLens `blocks.L.hook_mlp_out`
     (the d_model output of the MLP block, equivalent to their HF `model.layers.L.mlp`).
     Using from_pretrained_no_processing keeps the activation space identical to the HF
     model the probe was trained on.
  2. The model is fixed to Meta-Llama-3-8B-Instruct (the probe is model-specific), and
     the direction is selected by <source_dataset> (the probe's training dataset), which
     is the LEAF of the output path -- playing the role `position` played in our eval.

Documents are always projected at their LAST token.

Output tree (separate from resid_post retrieval_evaluation, never mixed):
    results/iclr_retrieval_evaluation/<model>/<eval>/<probe_at>/seed_<S>/layer_<L>/<source_dataset>/results.jsonl

llm_hidden_states.pt / sbert_embeddings.pt / docs.jsonl live at the layer level
(<...>/seed_<S>/layer_<L>/) and are shared across every source_dataset (they depend
on the corpus + hook, not the direction). Cached tensors are reused only when
docs.jsonl matches the current corpus.

Usage:
    python -m src.experiments.ICLR_paper_retrieval_evaluation --automated
    python -m src.experiments.ICLR_paper_retrieval_evaluation --automated --force-recompute
    python -m src.experiments.ICLR_paper_retrieval_evaluation \
        --dataset nq_swap --source-dataset triviaqa --layer 15
"""

from __future__ import annotations
import itertools
import argparse
import json
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

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"  # fixed: probe is model-specific
EVAL_DATASETS = ["nq_swap", "conflictqa"]
SOURCE_DATASETS = ["natural_questions_with_context", "triviaqa"]
PROBE_AT = "mlp"
LAYERS = [13, 14, 15, 16]
ALPHAS = [0.0, 0.3, 0.5, 1.0]
KS = [2, 5, 10]
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 4


def discover_eval_seeds(dataset: str) -> List[int]:
    """Seeds are our normalized-dataset test splits (independent of the probe)."""
    from src.utils import NORMALIZED_DIR
    root = NORMALIZED_DIR / dataset
    dirs = sorted(root.glob("seed_*"), key=lambda d: int(d.name.split("_")[1]))
    return [int(d.name.split("_")[1]) for d in dirs if (d / "test.jsonl").exists()]


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std())


def compute_llm_hidden_states(model: HookedTransformer, docs: list[str], hook_point: str, batch_size: int) -> torch.Tensor:
    tok = model.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    enc = [tok(d, return_tensors="pt", add_special_tokens=True).input_ids[0] for d in docs]
    order = sorted(range(len(docs)), key=lambda i: enc[i].shape[0])
    hidden = torch.zeros(len(docs), model.cfg.d_model)

    for s in tqdm(range(0, len(order), batch_size), desc="LLM hidden states"):
        idxs = order[s: s + batch_size]
        L = max(enc[i].shape[0] for i in idxs)
        batch = torch.full((len(idxs), L), tok.pad_token_id, dtype=torch.long, device=model.cfg.device)
        for r, i in enumerate(idxs):
            t = enc[i]
            batch[r, L - t.shape[0]:] = t.to(model.cfg.device)
        with torch.no_grad():
            _, cache = model.run_with_cache(batch, names_filter=hook_point, prepend_bos=False)
        resid = cache[hook_point][:, -1, :].detach().float().cpu()
        for r, i in enumerate(idxs):
            hidden[i] = resid[r]
    return hidden


def compute_evaluation(llm_hidden: torch.Tensor,
                    direction: torch.Tensor,
                    sbert_emb: torch.Tensor,
                    samples: list[dict],
                    all_docs: list[str],
                    doc_idx: dict[str, int],
                    sbert_enc: SentenceTransformer,
                    alphas: list[float],
                    ks: list[int],
                    out_dir: Path):
    """Compute ranking given the embedding similarities and the llm hidden states."""
    s_proj_all = (llm_hidden @ direction).numpy()  # [N]
    sbert_norm = sbert_emb / (sbert_emb.norm(dim=1, keepdim=True))  # [N, dim]

    logger.info("Encoding queries with SBERT ...")
    q_embs = sbert_enc.encode([s["question"] for s in samples], batch_size=64,
                            show_progress_bar=True, convert_to_numpy=True)
    q_embs_norm = torch.tensor(q_embs, dtype=torch.float32)
    q_embs_norm = q_embs_norm / (q_embs_norm.norm(dim=1, keepdim=True))

    s_proj_norm_global = zscore(s_proj_all)

    records = []
    for si, sample in enumerate(tqdm(samples, desc="Evaluating")):
        gold_idx = doc_idx[sample["factual_context"]]
        nf_idx = doc_idx[sample["non_factual_evidence"]]

        s_cos = (sbert_norm @ q_embs_norm[si]).numpy()  # [N]
        s_cos_norm = zscore(s_cos)

        for alpha in alphas:
            scores = (1 - alpha) * s_cos_norm + alpha * s_proj_norm_global
            sorted_indices = np.argsort(-scores)  # descending
            gold_rank = np.where(sorted_indices == gold_idx)[0][0] + 1
            nf_rank   = np.where(sorted_indices == nf_idx)[0][0] + 1
            for k in ks:
                records.append({
                    "sample_idx": si,
                    "question": sample["question"],
                    "alpha": alpha,
                    "k": k,
                    "gold_in_topk": bool(gold_rank <= k),
                    "nonfactual_in_topk": bool(nf_rank <= k),
                    "gold_rank": int(gold_rank),
                    "nonfactual_rank": int(nf_rank),
                    "topk_indices": sorted_indices[:k].tolist(),
                })

    results_path = out_dir / "results.jsonl"
    docs_path = out_dir.parent / "docs.jsonl"  # layer level: describes the shared tensors

    write_jsonl(results_path, records)
    id_to_doc = {doc_id: text for text, doc_id in doc_idx.items()}
    assert len(id_to_doc) == len(all_docs)
    write_jsonl(docs_path, list(id_to_doc.items()))

    logger.info(f"Wrote {len(records)} records -> {results_path}")
    logger.info(f"Wrote {len(all_docs)} docs -> {docs_path}")

    for alpha in alphas:
        for k in ks:
            rows = [r for r in records if r["alpha"] == alpha and r["k"] == k]
            gold_rate = sum(r["gold_in_topk"] for r in rows) / len(rows)
            nf_rate = sum(r["nonfactual_in_topk"] for r in rows) / len(rows)
            logger.info(f"alpha={alpha:.1f} k={k:2d} | gold_rate@k={gold_rate:.3f} | nonfactual_rate@k={nf_rate:.3f}")


def corpus_of(samples: list[dict]) -> list[str]:
    return sorted(set(s["factual_context"] for s in samples) | set(s["non_factual_evidence"] for s in samples))


def cache_matches_corpus(layer_dir: Path, all_docs: list[str]) -> bool:
    """True if the layer-level tensor caches exist and were computed on `all_docs`."""
    docs_path = layer_dir / "docs.jsonl"
    if not ((layer_dir / "llm_hidden_states.pt").exists()
            and (layer_dir / "sbert_embeddings.pt").exists() and docs_path.exists()):
        return False
    pairs = [json.loads(line) for line in docs_path.open() if line.strip()]
    id_to_doc = {i: t for i, t in pairs}
    return [id_to_doc.get(i) for i in range(len(id_to_doc))] == all_docs


def evaluate_combo(eval_dataset: str, source_dataset: str, probe_at: str, seed: int, layer: int,
                   alphas: list[float], ks: list[int], batch_size: int, sbert_model: str,
                   force: bool = False) -> None:
    layer_dir = (RESULTS_DIR / "iclr_retrieval_evaluation" / safe_model_id(MODEL)
                 / eval_dataset / probe_at / f"seed_{seed}" / f"layer_{layer}")
    out_dir = layer_dir / source_dataset  # leaf selects the direction
    if not force and (out_dir / "results.jsonl").exists():
        logger.info(f"Skip (exists): {out_dir}")
        return

    dir_path = (RESULTS_DIR / "iclr_directions" / safe_model_id(MODEL)
                / source_dataset / probe_at / f"layer_{layer}" / "direction.pt")
    if not dir_path.exists():
        logger.warning(f"Skip (no direction): {dir_path}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging("ICLR_paper_retrieval_evaluation", out_dir)
    logger.info(f"model={MODEL} | eval_dataset={eval_dataset} | source_dataset={source_dataset} | "
                f"probe_at={probe_at} | seed={seed} | layer={layer}")

    samples = load_normalized(eval_dataset, seed)["test"]
    all_docs = corpus_of(samples)
    doc_idx = {d: i for i, d in enumerate(all_docs)}
    logger.info(f"Corpus size: {len(all_docs)} unique documents")

    direction = torch.load(dir_path, map_location="cpu").float()
    logger.info(f"Loaded ICLR probe direction from {dir_path} (dim={direction.shape[0]})")

    sbert_enc = SentenceTransformer(sbert_model)

    if cache_matches_corpus(layer_dir, all_docs):
        logger.info("Loading cached SBERT embeddings + LLM hidden states (corpus verified)")
        sbert_emb = torch.load(layer_dir / "sbert_embeddings.pt", map_location="cpu")
        llm_hidden = torch.load(layer_dir / "llm_hidden_states.pt", map_location="cpu")
    else:
        logger.info("Computing SBERT embeddings ...")
        emb = sbert_enc.encode(all_docs, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        sbert_emb = torch.tensor(emb, dtype=torch.float32)
        torch.save(sbert_emb, layer_dir / "sbert_embeddings.pt")

        logger.info("Computing LLM hidden states (hook_mlp_out) ...")
        device = tl_utils.get_device()
        model = HookedTransformer.from_pretrained_no_processing(MODEL, device=device, dtype="bfloat16")
        model.eval()
        hook_point = tl_utils.get_act_name("mlp_out", layer)  # blocks.L.hook_mlp_out (d_model)
        llm_hidden = compute_llm_hidden_states(model, all_docs, hook_point, batch_size)
        torch.save(llm_hidden, layer_dir / "llm_hidden_states.pt")
        del model
        torch.cuda.empty_cache()

    if direction.shape[0] != llm_hidden.shape[1]:
        logger.error(f"Dim mismatch: direction {direction.shape[0]} vs activation {llm_hidden.shape[1]} "
                     f"-- probe_at/model mismatch, skipping {out_dir}")
        return

    compute_evaluation(llm_hidden, direction, sbert_emb, samples, all_docs, doc_idx,
                       sbert_enc, alphas, ks, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--automated", action="store_true", help="Run all eval x source x layer combinations.")
    ap.add_argument("--dataset", default="nq_swap", help="Our eval dataset (nq_swap / conflictqa).")
    ap.add_argument("--source-dataset", default="triviaqa", help="Dataset the ICLR probe was trained on.")
    ap.add_argument("--probe-at", default=PROBE_AT)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    ap.add_argument("--ks", type=int, nargs="+", default=KS)
    ap.add_argument("--sbert-model", default=SBERT_MODEL)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--seed", type=int, default=None,
                    help="Eval test-split seed. If omitted, runs all seeds found in the normalized dataset.")
    ap.add_argument("--force-recompute", action="store_true",
                    help="Recompute results even if they exist. Cached tensors are reused when the corpus matches.")
    args = ap.parse_args()

    if args.automated:
        logger.info("Running automated mode ...")
        for eval_dataset, source_dataset, layer in itertools.product(EVAL_DATASETS, SOURCE_DATASETS, LAYERS):
            seeds = [args.seed] if args.seed is not None else discover_eval_seeds(eval_dataset)
            if not seeds:
                logger.warning(f"No eval seeds found for {eval_dataset}, skipping.")
                continue
            for seed in seeds:
                evaluate_combo(eval_dataset, source_dataset, args.probe_at, seed, layer,
                               ALPHAS, KS, BATCH_SIZE, SBERT_MODEL, force=args.force_recompute)
        logger.info("Done computing automated ICLR evaluation.")
    else:
        seeds = [args.seed] if args.seed is not None else discover_eval_seeds(args.dataset)
        setup_logging("ICLR_paper_retrieval_evaluation", RESULTS_DIR / "iclr_retrieval_evaluation")
        for seed in seeds:
            evaluate_combo(args.dataset, args.source_dataset, args.probe_at, seed, args.layer,
                           args.alphas, args.ks, args.batch_size, args.sbert_model,
                           force=args.force_recompute)
        logger.info("Done.")


if __name__ == "__main__":
    main()
