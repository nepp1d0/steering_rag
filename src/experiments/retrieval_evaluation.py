"""
Retrieval evaluation: fuse SBERT cosine similarity with LLM factuality-direction projection.

score(d, q) = (1 - alpha) * zscore(s_cos) + alpha * zscore(s_proj)

Usage:
    python -m src.experiments.retrieval_evaluation \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --dataset nq_swap \
        --layer 15
"""

from __future__ import annotations
import itertools
import argparse
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import RESULTS_DIR, load_normalized, logger, safe_model_id, setup_logging, write_jsonl

import transformer_lens.utils as tl_utils
from transformer_lens import HookedTransformer
from sentence_transformers import SentenceTransformer

MODELS = ["meta-llama/Llama-3.1-8B-Instruct", "google/gemma-3-4b-it"]
DATASETS = ["nq_swap", "conflictqa"]
DIRECTION_DATASETS = ["nq_swap", "conflictqa"]
PROCEDURES = ["context_only"] #, "ab_choice"]
LAYERS = [10, 15, 20]
POSITIONS = ["last_pos"]#, "entity_pos"]
ALPHAS = [0.0, 0.3, 0.5, 1.0]
KS = [2, 5, 10]
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 4


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
    """
    Compute ranking given the embedding similairities and the llm hidden states.
    """
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
        # Get the index of the gold document and the non-factual document in the corpus from the all_docs list
        gold_idx = doc_idx[sample["factual_context"]]
        nf_idx = doc_idx[sample["non_factual_evidence"]]

        s_cos = (sbert_norm @ q_embs_norm[si]).numpy()  # [N]
        s_cos_norm = zscore(s_cos)

        for alpha in alphas:
            scores = (1 - alpha) * s_cos_norm + alpha * s_proj_norm_global
            sorted_indices = np.argsort(-scores)  # descending, no Python sort. List of indices of documents (index coming from all_docs-> embedded in SBERT and hidden states -> combination of scores against the query)
            gold_rank = np.where(sorted_indices == gold_idx)[0][0] + 1
            nf_rank   = np.where(sorted_indices == nf_idx)[0][0] + 1
            # [OLD] now we need to compute the rank of the gold document and the non-factual document
            # in the sorted scores the first element is the rank 1 element, the second element is the rank 2 element, etc.
            # we just neet to find the position in sorted_score where x[1] == gold_idx or x[1] == nf_idx
            #gold_rank = int((scores > scores[gold_idx]).sum()) + 1
            #nf_rank = int((scores > scores[nf_idx]).sum()) + 1
            for k in ks:
                gold_in_topk = bool(gold_rank <= k)
                nf_in_topk = bool(nf_rank <= k)
                topk_indices = sorted_indices[:k].tolist()
                records.append({
                    "sample_idx": si,
                    "question": sample["question"],
                    "alpha": alpha,
                    "k": k,
                    "gold_in_topk": gold_in_topk,
                    "nonfactual_in_topk": nf_in_topk,
                    "gold_rank": int(gold_rank),
                    "nonfactual_rank": int(nf_rank),
                    "topk_text": [text for idx in topk_indices for text, doc_id in doc_idx.items() if idx == doc_id],
                    "topk_indices": topk_indices,
                })

    results_path = out_dir / "results.jsonl"
    docs_path = out_dir / "docs.jsonl"

    write_jsonl(results_path, records)
    # Save also the mapping from document index to document text, with assert to ensure ids are unique
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

    logger.info("Done.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--automated", action="store_true", help="Compute experiments on all datasets automatically (combinations are hardcode here).")
    ap.add_argument("--dataset", default="nq_swap")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--direction-dataset", default=None, help="Dataset used for direction (defaults to --dataset)")
    ap.add_argument("--procedure", default="context_only")
    ap.add_argument("--position", default="last_pos")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.3, 0.5, 1.0])
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    
    normalize_literals = ["unnormalized", "normalized"]
    normalize_direction = False # Change here to normalize the directions
    normalize_path = normalize_literals[1] if normalize_direction else normalize_literals[0]

    # Automated mode
    if args.automated:
        logger.info("Running automated mode ...")
        
        combinations = itertools.product(MODELS, DATASETS, DIRECTION_DATASETS, PROCEDURES, POSITIONS, LAYERS)
        for model, dataset, direction_dataset, procedure, position, layer in combinations:
            # Create output directory
            out_dir = (RESULTS_DIR / "retrieval_evaluation" / safe_model_id(model)
                / dataset / direction_dataset / normalize_path / procedure / f"layer_{layer}")
            out_dir.mkdir(parents=True, exist_ok=True)
            setup_logging("retrieval_evaluation", out_dir)
            logger.info(f"model={model} | dataset={dataset} | direction_dataset={direction_dataset} | normalize_direction={normalize_direction} | layer={layer} | procedure={procedure}")
            
            #Load samples
            samples = load_normalized(dataset)
            all_docs = sorted(set(s["factual_context"] for s in samples) | set(s["non_factual_evidence"] for s in samples))
            doc_idx = {d: i for i, d in enumerate(all_docs)}
            logger.info(f"Corpus size: {len(all_docs)} unique documents")


            #Load direction
            dir_path = (RESULTS_DIR / "direction_identification" / safe_model_id(args.model)
                    / direction_dataset / procedure / f"layer_{layer}" / position / "direction.pt")
            direction = torch.load(dir_path, map_location="cpu").float()
            if normalize_direction:
                direction = direction / (direction.norm() + 1e-8)
            logger.info(f"Loaded direction from {dir_path}")

            # Load embedding model
            sbert_enc = SentenceTransformer(SBERT_MODEL)

            # Compute SBERT embeddings
            logger.info("Computing SBERT embeddings ...")
            sbert_cache = out_dir / "sbert_embeddings.pt"
            emb = sbert_enc.encode(all_docs, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
            sbert_emb = torch.tensor(emb, dtype=torch.float32)
            torch.save(sbert_emb, sbert_cache)
            logger.info(f"Saved SBERT embeddings -> {sbert_cache}")

            # Compute LLM hidden states
            logger.info("Computing LLM hidden states ...")
            llm_cache = out_dir / "llm_hidden_states.pt"
            device = tl_utils.get_device()
            model = HookedTransformer.from_pretrained(model, device=device, dtype="bfloat16")
            model.eval()
            hook_point = tl_utils.get_act_name("resid_post", layer)
            llm_hidden = compute_llm_hidden_states(model, all_docs, hook_point, BATCH_SIZE)
            torch.save(llm_hidden, llm_cache)
            logger.info(f"Saved LLM hidden states -> {llm_cache}")
            del model

            compute_evaluation(llm_hidden, direction, sbert_emb, samples, all_docs, doc_idx, sbert_enc, ALPHAS, KS, out_dir)
            del llm_hidden, sbert_emb, sbert_enc
        logger.info("Done computing automatedevaluation.")
    else:
        direction_dataset = args.direction_dataset or args.dataset
        out_dir = (RESULTS_DIR / "retrieval_evaluation" / safe_model_id(args.model)
                / args.dataset / direction_dataset / normalize_path / args.procedure / f"layer_{args.layer}")
        out_dir.mkdir(parents=True, exist_ok=True)
        setup_logging("retrieval_evaluation", out_dir)
        logger.info(f"model={args.model} | dataset={args.dataset} | direction_dataset={direction_dataset} | layer={args.layer} | procedure={args.procedure}")
        samples = load_normalized(args.dataset)

        all_docs = sorted(set(s["factual_context"] for s in samples) | set(s["non_factual_evidence"] for s in samples))
        doc_idx = {d: i for i, d in enumerate(all_docs)}
        logger.info(f"Corpus size: {len(all_docs)} unique documents")

        dir_path = (RESULTS_DIR / "direction_identification" / safe_model_id(args.model)
                    / direction_dataset / args.procedure / f"layer_{args.layer}" / args.position / "direction.pt")
        direction = torch.load(dir_path, map_location="cpu").float()
        if normalize_direction:
            direction = direction / (direction.norm())
        logger.info(f"Loaded direction from {dir_path}")

        sbert_cache = out_dir / "sbert_embeddings.pt"
        sbert_enc = SentenceTransformer(args.sbert_model)
        if sbert_cache.exists():
            logger.info("Loading cached SBERT embeddings")
            sbert_emb = torch.load(sbert_cache, map_location="cpu")
        else:
            logger.info("Computing SBERT embeddings ...")
            emb = sbert_enc.encode(all_docs, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
            sbert_emb = torch.tensor(emb, dtype=torch.float32)
            torch.save(sbert_emb, sbert_cache)
            logger.info(f"Saved SBERT embeddings -> {sbert_cache}")

        llm_cache = out_dir / "llm_hidden_states.pt"
        if llm_cache.exists():
            logger.info("Loading cached LLM hidden states")
            llm_hidden = torch.load(llm_cache, map_location="cpu")
        else:
            logger.info("Computing LLM hidden states ...")
            device = tl_utils.get_device()
            model = HookedTransformer.from_pretrained(args.model, device=device, dtype="bfloat16")
            model.eval()
            hook_point = tl_utils.get_act_name("resid_post", args.layer)
            llm_hidden = compute_llm_hidden_states(model, all_docs, hook_point, args.batch_size)
            torch.save(llm_hidden, llm_cache)
            logger.info(f"Saved LLM hidden states -> {llm_cache}")
            del model

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
            # Get the index of the gold document and the non-factual document in the corpus from the all_docs list
            gold_idx = doc_idx[sample["factual_context"]]
            nf_idx = doc_idx[sample["non_factual_evidence"]]

            s_cos = (sbert_norm @ q_embs_norm[si]).numpy()  # [N]
            s_cos_norm = zscore(s_cos)

            for alpha in args.alphas:
                scores = (1 - alpha) * s_cos_norm + alpha * s_proj_norm_global
                sorted_indices = np.argsort(-scores)  # descending, no Python sort. List of indices of documents (index coming from all_docs-> embedded in SBERT and hidden states -> combination of scores against the query)
                gold_rank = np.where(sorted_indices == gold_idx)[0][0] + 1
                nf_rank   = np.where(sorted_indices == nf_idx)[0][0] + 1
                # [OLD] now we need to compute the rank of the gold document and the non-factual document
                # in the sorted scores the first element is the rank 1 element, the second element is the rank 2 element, etc.
                # we just neet to find the position in sorted_score where x[1] == gold_idx or x[1] == nf_idx
                #gold_rank = int((scores > scores[gold_idx]).sum()) + 1
                #nf_rank = int((scores > scores[nf_idx]).sum()) + 1
                for k in args.ks:
                    gold_in_topk = bool(gold_rank <= k)
                    nf_in_topk = bool(nf_rank <= k)
                    topk_indices = sorted_indices[:k].tolist()
                    records.append({
                        "sample_idx": si,
                        "question": sample["question"],
                        "alpha": alpha,
                        "k": k,
                        "gold_in_topk": gold_in_topk,
                        "nonfactual_in_topk": nf_in_topk,
                        "gold_rank": int(gold_rank),
                        "nonfactual_rank": int(nf_rank),
                        "topk_text": [text for idx in topk_indices for text, doc_id in doc_idx.items() if idx == doc_id],
                        "topk_indices": topk_indices,
                    })

        results_path = out_dir / "results.jsonl"
        docs_path = out_dir / "docs.jsonl"

        write_jsonl(results_path, records)
        # Save also the mapping from document index to document text, with assert to ensure ids are unique
        id_to_doc = {doc_id: text for text, doc_id in doc_idx.items()}
        assert len(id_to_doc) == len(all_docs)
        write_jsonl(docs_path, list(id_to_doc.items()))

        logger.info(f"Wrote {len(records)} records -> {results_path}")
        logger.info(f"Wrote {len(all_docs)} docs -> {docs_path}")

        for alpha in args.alphas:
            for k in args.ks:
                rows = [r for r in records if r["alpha"] == alpha and r["k"] == k]
                gold_rate = sum(r["gold_in_topk"] for r in rows) / len(rows)
                nf_rate = sum(r["nonfactual_in_topk"] for r in rows) / len(rows)
                logger.info(f"alpha={alpha:.1f} k={k:2d} | gold_rate@k={gold_rate:.3f} | nonfactual_rate@k={nf_rate:.3f}")

        logger.info("Done.")


if __name__ == "__main__":
    main()
