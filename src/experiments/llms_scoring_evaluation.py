"""
LLM-as-judge factuality baseline for retrieval re-ranking.

Replaces the factuality-direction projection of retrieval_evaluation.py with a
query-independent LLM-generated factuality score in [0, 1] per document:

    score(d, q) = (1 - alpha) * zscore(s_cos) + alpha * zscore(s_llm)

Per-record output schema matches retrieval_evaluation.py exactly so downstream
plotting can be shared.

Usage:
    python -m src.experiments.llms_scoring_evaluation --automated
    python -m src.experiments.llms_scoring_evaluation \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --dataset nq_swap
"""

from __future__ import annotations
import argparse
import gc
import itertools
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import NORMALIZED_DIR, RESULTS_DIR, load_normalized, logger, safe_model_id, setup_logging, write_jsonl

from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams

MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "google/gemma-3-4b-it",
    "Qwen/Qwen2-7B-Instruct",
]
DATASETS = ["nq_swap", "conflictqa"]
ALPHAS = [0.0, 0.3, 0.5, 1.0]
KS = [2, 5, 10]
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_NEW_TOKENS = 8
MAX_MODEL_LEN = 8192

FACTUALITY_PROMPT = (
    "Rate the factuality of the following passage as a single number "
    "between 0 (completely false / fabricated) and 1 (factually correct). "
    "Reply with ONLY the number, nothing else.\n\n"
    "Passage: {doc}\nScore:"
)

_SCORE_RE = re.compile(r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?|\.\d+)(?![\d.])")


def discover_seeds(dataset: str) -> list[int]:
    dirs = sorted((NORMALIZED_DIR / dataset).glob("seed_*"), key=lambda d: int(d.name.split("_")[1]))
    return [int(d.name.split("_")[1]) for d in dirs if d.is_dir()]


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std())


def parse_score(text: str) -> tuple[float, bool]:
    """Returns (score, parse_ok). Falls back to 0.5 on failure."""
    m = _SCORE_RE.search(text)
    if not m:
        return 0.5, False
    try:
        val = float(m.group())
    except ValueError:
        return 0.5, False
    return max(0.0, min(1.0, val)), True


def compute_llm_scores(llm: LLM, docs: list[str], sampling: SamplingParams) -> tuple[torch.Tensor, list[dict]]:
    convs = [[{"role": "user", "content": FACTUALITY_PROMPT.format(doc=d)}] for d in docs]
    outs = llm.chat(convs, sampling, use_tqdm=True)
    raw_rows = []
    scores = []
    n_failed = 0
    for i, o in enumerate(outs):
        text = o.outputs[0].text.strip()
        score, ok = parse_score(text)
        if not ok:
            n_failed += 1
        scores.append(score)
        raw_rows.append({"doc_idx": i, "raw_text": text, "parsed_score": score, "parse_ok": ok})
    if n_failed:
        logger.warning(f"{n_failed}/{len(docs)} documents had unparseable LLM output (fell back to 0.5)")
    return torch.tensor(scores, dtype=torch.float32), raw_rows


def compute_evaluation(llm_scores: torch.Tensor,
                       sbert_emb: torch.Tensor,
                       samples: list[dict],
                       all_docs: list[str],
                       doc_idx: dict[str, int],
                       sbert_enc: SentenceTransformer,
                       alphas: list[float],
                       ks: list[int],
                       out_dir: Path) -> None:
    s_proj_all = llm_scores.numpy()  # [N], already in [0,1]
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
            sorted_indices = np.argsort(-scores)
            gold_rank = np.where(sorted_indices == gold_idx)[0][0] + 1
            nf_rank = np.where(sorted_indices == nf_idx)[0][0] + 1
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


def run_one(model_name: str, dataset: str, seed: int, llm: LLM | None,
            sbert_enc: SentenceTransformer, sampling: SamplingParams,
            alphas: list[float], ks: list[int]) -> LLM | None:
    out_dir = RESULTS_DIR / "llms_scoring_evaluation" / safe_model_id(model_name) / dataset / f"seed_{seed}"
    if (out_dir / "results.jsonl").exists():
        logger.info(f"Skip (exists): {out_dir}")
        return llm
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging("llms_scoring_evaluation", out_dir)
    logger.info(f"model={model_name} | dataset={dataset} | seed={seed}")

    samples = load_normalized(dataset, seed)["test"]
    all_docs = sorted(set(s["factual_context"] for s in samples) | set(s["non_factual_evidence"] for s in samples))
    doc_idx = {d: i for i, d in enumerate(all_docs)}
    logger.info(f"Corpus size: {len(all_docs)} unique documents")

    sbert_cache = out_dir / "sbert_embeddings.pt"
    if sbert_cache.exists():
        logger.info("Loading cached SBERT embeddings")
        sbert_emb = torch.load(sbert_cache, map_location="cpu")
    else:
        logger.info("Computing SBERT embeddings ...")
        emb = sbert_enc.encode(all_docs, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        sbert_emb = torch.tensor(emb, dtype=torch.float32)
        torch.save(sbert_emb, sbert_cache)
        logger.info(f"Saved SBERT embeddings -> {sbert_cache}")

    scores_cache = out_dir / "llm_scores.pt"
    raw_cache = out_dir / "llm_raw_outputs.jsonl"
    if scores_cache.exists():
        logger.info("Loading cached LLM scores")
        llm_scores = torch.load(scores_cache, map_location="cpu")
    else:
        logger.info("Computing LLM factuality scores ...")
        if llm is None:
            llm = LLM(model=model_name, dtype="bfloat16", max_model_len=MAX_MODEL_LEN)
        llm_scores, raw_rows = compute_llm_scores(llm, all_docs, sampling)
        torch.save(llm_scores, scores_cache)
        write_jsonl(raw_cache, raw_rows)
        logger.info(f"Saved LLM scores -> {scores_cache}")
        logger.info(f"Saved raw LLM outputs -> {raw_cache}")

    compute_evaluation(llm_scores, sbert_emb, samples, all_docs, doc_idx, sbert_enc, alphas, ks, out_dir)
    return llm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--automated", action="store_true",
                    help="Run all (model, dataset, seed) combinations hardcoded above.")
    ap.add_argument("--dataset", default="nq_swap")
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    ap.add_argument("--ks", type=int, nargs="+", default=KS)
    ap.add_argument("--sbert-model", default=SBERT_MODEL)
    ap.add_argument("--seed", type=int, default=None,
                    help="Split seed. If omitted, runs for all seeds found in data/normalized_dataset/.")
    args = ap.parse_args()

    sbert_enc = SentenceTransformer(args.sbert_model, device="cpu")
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    if args.automated:
        logger.info("Running automated mode ...")
        for model_name in MODELS:
            llm = None
            for dataset in DATASETS:
                seeds = [args.seed] if args.seed is not None else discover_seeds(dataset)
                if not seeds:
                    logger.warning(f"No seeds found for dataset={dataset}, skipping.")
                    continue
                for seed in seeds:
                    llm = run_one(model_name, dataset, seed, llm, sbert_enc, sampling, ALPHAS, KS)
            del llm
            gc.collect()
            torch.cuda.empty_cache()
        logger.info("Done computing automated evaluation.")
    else:
        setup_logging("llms_scoring_evaluation", RESULTS_DIR / "llms_scoring_evaluation")
        seeds = [args.seed] if args.seed is not None else discover_seeds(args.dataset)
        llm = None
        for seed in seeds:
            llm = run_one(args.model, args.dataset, seed, llm, sbert_enc, sampling, args.alphas, args.ks)
        del llm
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
