"""
Prompt-only RAG evaluation baseline: retrieve top-k docs with SBERT and instruct
the model to identify and attend to the most factual context chunk.

Evaluates on ConflictQA (ground-truth aliases) across all seeds and models.
No direction scoring — pure SBERT retrieval with a refined generation prompt.

Usage:
    python src/exploratory/prompt_evaluation.py
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
from utils import NORMALIZED_DIR, RESULTS_DIR, load_normalized, logger, safe_model_id, setup_logging, write_jsonl

from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams

MODELS = ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-1B-Instruct", "google/gemma-3-4b-it", "Qwen/Qwen2-7B-Instruct"]
EVAL_DATASET = "conflictqa"
KS = [2, 5, 10]
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_NEW_TOKENS = 64
MAX_MODEL_LEN = 8192
MIN_ALIAS_LEN = 4


def discover_seeds(dataset: str) -> list[int]:
    dirs = sorted((NORMALIZED_DIR / dataset).glob("seed_*"), key=lambda d: int(d.name.split("_")[1]))
    return [int(d.name.split("_")[1]) for d in dirs if d.is_dir()]


def build_prompt(docs: list[str], question: str) -> str:
    docs_block = "\n".join(f"[{i + 1}] {d}" for i, d in enumerate(docs))
    return (
        "The following context chunks may conflict with each other. "
        "Only one is factually correct — carefully identify it and use it to answer concisely. "
        "Ignore any chunk that contradicts established facts.\n\n"
        f"Context:\n{docs_block}\n\n"
        f"Question: {question}\nAnswer:"
    )


def is_answer_correct(answer: str, ground_truth: list[str]) -> bool:
    answer_lower = answer.lower()
    aliases = [a for a in ground_truth if len(a.strip()) >= MIN_ALIAS_LEN]
    return any(a.strip().lower() in answer_lower for a in aliases)


def generate(llm: LLM, contents: list[str], sampling: SamplingParams) -> tuple[list[str], list[str]]:
    convs = [[{"role": "user", "content": c}] for c in contents]
    outs = llm.chat(convs, sampling, use_tqdm=True)
    return [o.prompt for o in outs], [o.outputs[0].text.strip() for o in outs]


def main() -> None:
    setup_logging("prompt_evaluation", RESULTS_DIR / "prompt_evaluation")
    sbert_enc = SentenceTransformer(SBERT_MODEL, device="cpu")
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
    seeds = discover_seeds(EVAL_DATASET)
    logger.info(f"Seeds: {seeds}")

    for model_name in MODELS:
        llm = None

        for seed in seeds:
            out_dir = RESULTS_DIR / "prompt_evaluation" / safe_model_id(model_name) / f"seed_{seed}"
            if (out_dir / "results.jsonl").exists():
                logger.info(f"Skip (exists): {out_dir}")
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"model={model_name} | seed={seed}")

            samples = load_normalized(EVAL_DATASET, seed)["test"]
            if "ground_truth" not in samples[0]:
                raise KeyError("ConflictQA samples lack 'ground_truth'; run add_conflictqa_ground_truth first.")

            all_docs = sorted(set(s["factual_context"] for s in samples) | set(s["non_factual_evidence"] for s in samples))
            doc_idx = {d: i for i, d in enumerate(all_docs)}
            logger.info(f"Corpus: {len(all_docs)} unique docs, {len(samples)} test samples")

            emb = sbert_enc.encode(all_docs, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
            sbert_norm = torch.tensor(emb, dtype=torch.float32)
            sbert_norm = sbert_norm / sbert_norm.norm(dim=1, keepdim=True)

            q_emb = sbert_enc.encode([s["question"] for s in samples], batch_size=64,
                                     show_progress_bar=False, convert_to_numpy=True)
            q_norm = torch.tensor(q_emb, dtype=torch.float32)
            q_norm = q_norm / q_norm.norm(dim=1, keepdim=True)

            jobs = []
            for si, sample in enumerate(samples):
                gold_idx = doc_idx[sample["factual_context"]]
                nf_idx = doc_idx[sample["non_factual_evidence"]]
                s_cos = (sbert_norm @ q_norm[si]).numpy()
                ranked = np.argsort(-s_cos)
                for k in KS:
                    topk = [{"text": all_docs[i], "score": float(s_cos[i]),
                             "is_gold": i == gold_idx, "is_nonfactual": i == nf_idx}
                            for i in ranked[:k].tolist()]
                    content = build_prompt([d["text"] for d in topk], sample["question"])
                    jobs.append({"sample_idx": si, "question": sample["question"],
                                 "k": k, "topk": topk,
                                 "ground_truth": sample["ground_truth"], "content": content})

            unique_contents = list(dict.fromkeys(j["content"] for j in jobs))
            logger.info(f"{len(jobs)} jobs -> {len(unique_contents)} unique prompts to generate")

            if llm is None:
                llm = LLM(model=model_name, dtype="bfloat16", max_model_len=MAX_MODEL_LEN)
            prompts, answers = generate(llm, unique_contents, sampling)
            prompt_map = dict(zip(unique_contents, prompts))
            answer_map = dict(zip(unique_contents, answers))

            records = []
            for j in jobs:
                answer = answer_map[j["content"]]
                records.append({
                    "sample_idx": j["sample_idx"], "question": j["question"],
                    "k": j["k"], "prompt": prompt_map[j["content"]], "topk": j["topk"],
                    "generated_answer": answer, "ground_truth": j["ground_truth"],
                    "is_correct": is_answer_correct(answer, j["ground_truth"]),
                })

            write_jsonl(out_dir / "results.jsonl", records)
            logger.info(f"Wrote {len(records)} records -> {out_dir / 'results.jsonl'}")

            for k in KS:
                rows = [r for r in records if r["k"] == k]
                acc = sum(r["is_correct"] for r in rows) / len(rows)
                logger.info(f"k={k:2d} | accuracy={acc:.3f}")

        del llm
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Done.")


if __name__ == "__main__":
    main()
