"""
Preliminary analysis: prompt each model on both normalized datasets under 5 conditions.

Conditions per sample:
    no_context          – question only
    factual_only        – question + factual_context
    non_factual_only    – question + non_factual_evidence
    both_factual_first  – question + factual_context + non_factual_evidence
    both_non_factual_first – question + non_factual_evidence + factual_context

Output:
    results/preliminary_analysis/<model>/<dataset>/results.jsonl

Usage:
    python -m src.experiments.preliminary_analysis
"""

from __future__ import annotations

import gc
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import RESULTS_DIR, load_normalized, logger, safe_model_id, setup_logging, write_jsonl

from vllm import LLM, SamplingParams

MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "google/gemma-3-4b-it",
    "Qwen/Qwen2-7B-Instruct",
]
DATASETS = ["nq_swap", "conflictqa"]
MAX_SAMPLES = 500
SEED = 42
MAX_NEW_TOKENS = 64
MAX_MODEL_LEN = 4096


def build_prompt(question: str, *contexts: str) -> str:
    if contexts:
        ctx_block = "\n\n".join(f"Context {i+1}:\n{c}" for i, c in enumerate(contexts))
        return f"{ctx_block}\n\nAnswer the following question as factually as possible. Answer concisely.\n\nQuestion: {question}\nAnswer:"
    return f"Answer the following question as factually as possible. Answer concisely.\n\nQuestion: {question}\nAnswer:"


def conditions(sample: dict) -> list[tuple[str, str]]:
    q = sample["question"]
    fc = sample["factual_context"]
    nf = sample["non_factual_evidence"]
    return [
        ("no_context", build_prompt(q)),
        ("factual_only", build_prompt(q, fc)),
        ("non_factual_only", build_prompt(q, nf)),
        ("both_factual_first", build_prompt(q, fc, nf)),
        ("both_non_factual_first", build_prompt(q, nf, fc)),
    ]


def main() -> None:
    setup_logging("preliminary_analysis", RESULTS_DIR / "preliminary_analysis")
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    for model_name in MODELS:
        llm = None

        for dataset_id in DATASETS:
            out_path = RESULTS_DIR / "preliminary_analysis" / safe_model_id(model_name) / dataset_id / "results.jsonl"
            if out_path.exists():
                logger.info(f"Skip (exists): {out_path}")
                continue

            samples = load_normalized(dataset_id, seed=SEED)["train"]
            rng = random.Random(SEED)
            if len(samples) > MAX_SAMPLES:
                samples = rng.sample(samples, MAX_SAMPLES)
            logger.info(f"model={model_name} | dataset={dataset_id} | n={len(samples)}")

            # Build jobs and collect unique contents for batched generation.
            jobs = []
            for sample in samples:
                gt = sample.get("ground_truth") or sample["factual_answer"]
                for cond_name, content in conditions(sample):
                    jobs.append({**sample, "condition": cond_name, "ground_truth": gt, "content": content})

            unique_contents = list(dict.fromkeys(j["content"] for j in jobs))
            logger.info(f"{len(jobs)} jobs -> {len(unique_contents)} unique prompts")

            if llm is None:
                llm = LLM(model=model_name, dtype="bfloat16", max_model_len=MAX_MODEL_LEN)

            convs = [[{"role": "user", "content": c}] for c in unique_contents]
            outs = llm.chat(convs, sampling, use_tqdm=True)
            answer_map = {c: o.outputs[0].text.strip() for c, o in zip(unique_contents, outs)}

            records = []
            for j in jobs:
                records.append({
                    "question": j["question"],
                    "factual_context": j["factual_context"],
                    "non_factual_evidence": j["non_factual_evidence"],
                    "factual_answer": j["factual_answer"],
                    "non_factual_answer": j["non_factual_answer"],
                    "ground_truth": j["ground_truth"],
                    "original_dataset_id": j["original_dataset_id"],
                    "condition": j["condition"],
                    "generated_answer": answer_map[j["content"]],
                })

            write_jsonl(out_path, records)
            logger.info(f"Wrote {len(records)} records -> {out_path}")

        del llm
        gc.collect()

    logger.info("Done.")


if __name__ == "__main__":
    main()
