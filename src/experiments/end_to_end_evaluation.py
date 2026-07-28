"""
End-to-end RAG evaluation: retrieve top-k docs (same fused scoring as
retrieval_evaluation), feed them to the generative model, and check whether the
generated answer contains any ConflictQA ground-truth alias.

Runs only the top layers selected by plot_retrieval_evaluation.py
(top_layers_<procedure>.json), each at its best alpha plus the alpha=0 baseline.
Generation uses vLLM (no steering hooks during decoding, so TransformerLens is not
needed); the per-document hidden states come from the cached tensors that
retrieval_evaluation.py already wrote (errors if a cache is missing).

For each (sample, alpha, k) we store the prompt, the retrieved docs with scores,
the generated answer, the ground-truth aliases and an is_correct flag.

Usage:
    python -m src.experiments.end_to_end_evaluation
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# vLLM resolves device_config=cuda in the parent (initializing a CUDA context), then
# launches its EngineCore worker. Force that worker to spawn (fresh process) rather than
# fork, otherwise the forked child inherits the context and dies on cudaErrorInitializationError.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import RESULTS_DIR, load_normalized, logger, safe_model_id, setup_logging, write_jsonl

from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams

MODELS = ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-1B-Instruct", "google/gemma-3-4b-it", "Qwen/Qwen2-7B-Instruct"]
EVAL_DATASET = "conflictqa"            # ground-truth aliases are ConflictQA-specific
DIRECTION_DATASETS = ["nq_swap", "conflictqa"]
PROCEDURE = "context_only"
POSITION = "last_pos"
NORMALIZE = "unnormalized"             # mirror retrieval_evaluation default (directions unnormalized)
KS = [2, 5, 10]
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_NEW_TOKENS = 64
MAX_MODEL_LEN = 8192
MIN_ALIAS_LEN = 4                      # drop very short aliases (e.g. "pol") to avoid false matches


def discover_direction_seeds(model_id: str, dataset: str) -> list[int]:
    root = RESULTS_DIR / "direction_identification" / safe_model_id(model_id) / dataset
    return [int(d.name.split("_")[1]) for d in sorted(root.glob("seed_*")) if d.is_dir()]


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std())


def build_prompt(docs: list[str], question: str) -> str:
    """User-message content (single turn, no system role: gemma/llama template compatible)."""
    docs_block = "\n".join(f"[{i + 1}] {d}" for i, d in enumerate(docs))
    return (
        "Use only the information in the context below to answer the question as "
        "factually as possible. Answer concisely.\n\n"
        f"Context:\n{docs_block}\n\n"
        f"Question: {question}\nAnswer:"
    )


def is_answer_correct(answer: str, ground_truth: list[str]) -> bool:
    answer_lower = answer.lower()
    aliases = [a for a in ground_truth if len(a.strip()) >= MIN_ALIAS_LEN]
    return any(a.strip().lower() in answer_lower for a in aliases)


def load_cache(retrieval_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Require the tensors retrieval_evaluation cached; error (don't recompute) if absent."""
    sbert_cache = retrieval_dir / "sbert_embeddings.pt"
    llm_cache = retrieval_dir / "llm_hidden_states.pt"
    if not (sbert_cache.exists() and llm_cache.exists()):
        raise FileNotFoundError(
            f"Missing cached tensors in {retrieval_dir}. Run retrieval_evaluation for this "
            f"(model, eval, direction, seed, layer) before the end-to-end evaluation.")
    return torch.load(sbert_cache, map_location="cpu"), torch.load(llm_cache, map_location="cpu")


def load_top_layers(model_name: str, direction_dataset: str) -> list[dict]:
    """Read top_layers_<procedure>_<position>.json from top_retrieval_evaluation; return only the single best layer."""
    path = (RESULTS_DIR / "top_retrieval_evaluation" / safe_model_id(model_name) / EVAL_DATASET
            / direction_dataset / NORMALIZE / f"top_layers_{PROCEDURE}_{POSITION}.json")
    if not path.exists():
        logger.warning(f"No top-layers file ({path}); run plot_retrieval_evaluation first. Skipping.")
        return []
    return json.loads(path.read_text())["ranking"][:1]


def generate(llm: LLM, contents: list[str], sampling: SamplingParams) -> tuple[list[str], list[str]]:
    """Batched greedy generation over unique prompts; returns (rendered_prompts, answers)."""
    convs = [[{"role": "user", "content": c}] for c in contents]
    outs = llm.chat(convs, sampling, use_tqdm=True)
    return [o.prompt for o in outs], [o.outputs[0].text.strip() for o in outs]


def main() -> None:
    setup_logging("end_to_end_evaluation", RESULTS_DIR / "end_to_end_evaluation")
    sbert_enc = SentenceTransformer(SBERT_MODEL, device="cpu")  # CPU: leave the GPU to vLLM
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    for model_name in MODELS:
        llm = None  # vLLM model, loaded lazily and reused across all this model's combinations

        for direction_dataset in DIRECTION_DATASETS:
            top_layers = load_top_layers(model_name, direction_dataset)
            seeds = discover_direction_seeds(model_name, direction_dataset)
            if not top_layers or not seeds:
                continue

            for entry in top_layers:
                layer, best_alpha = entry["layer"], entry["best_alpha"]
                alphas = [0.0, best_alpha]  # baseline + chosen steering config

                for seed in seeds:
                    out_dir = (RESULTS_DIR / "end_to_end_evaluation" / safe_model_id(model_name)
                               / EVAL_DATASET / direction_dataset / NORMALIZE
                               / f"seed_{seed}" / PROCEDURE / f"layer_{layer}")
                    if (out_dir / "results.jsonl").exists():
                        logger.info(f"Skip (exists): {out_dir}")
                        continue
                    out_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"model={model_name} | direction={direction_dataset} | seed={seed} | "
                                f"layer={layer} | alphas={alphas}")

                    samples = load_normalized(EVAL_DATASET, seed)["test"]
                    if "ground_truth" not in samples[0]:
                        raise KeyError("ConflictQA samples lack 'ground_truth'; attach it before running.")
                    all_docs = sorted(set(s["factual_context"] for s in samples)
                                      | set(s["non_factual_evidence"] for s in samples))
                    doc_idx = {d: i for i, d in enumerate(all_docs)}

                    dir_path = (RESULTS_DIR / "direction_identification" / safe_model_id(model_name)
                                / direction_dataset / f"seed_{seed}" / PROCEDURE
                                / f"layer_{layer}" / POSITION / "direction.pt")
                    direction = torch.load(dir_path, map_location="cpu").float()

                    retrieval_dir = (RESULTS_DIR / "top_retrieval_evaluation" / safe_model_id(model_name)
                                     / EVAL_DATASET / direction_dataset / NORMALIZE
                                     / f"seed_{seed}" / PROCEDURE / f"layer_{layer}")
                    sbert_emb, llm_hidden = load_cache(retrieval_dir)
                    if len(all_docs) != sbert_emb.shape[0]:
                        raise ValueError(f"Corpus/cache mismatch: {len(all_docs)} docs vs "
                                         f"{sbert_emb.shape[0]} cached rows in {retrieval_dir}.")

                    # Same fused scoring as retrieval_evaluation.
                    s_proj_norm = zscore((llm_hidden @ direction).numpy())          # [N]
                    sbert_norm = sbert_emb / sbert_emb.norm(dim=1, keepdim=True)     # [N, d]
                    q_emb = sbert_enc.encode([s["question"] for s in samples], batch_size=64,
                                             show_progress_bar=False, convert_to_numpy=True)
                    q_norm = torch.tensor(q_emb, dtype=torch.float32)
                    q_norm = q_norm / q_norm.norm(dim=1, keepdim=True)

                    # Build all (sample, alpha, k) jobs; deduplicate identical prompts.
                    jobs = []
                    for si, sample in enumerate(samples):
                        gold_idx = doc_idx[sample["factual_context"]]
                        nf_idx = doc_idx[sample["non_factual_evidence"]]
                        s_cos_norm = zscore((sbert_norm @ q_norm[si]).numpy())       # [N]
                        for alpha in alphas:
                            scores = (1 - alpha) * s_cos_norm + alpha * s_proj_norm
                            ranked = np.argsort(-scores)                            # descending
                            for k in KS:
                                topk = [{"text": all_docs[i], "score": float(scores[i]),
                                         "is_gold": i == gold_idx, "is_nonfactual": i == nf_idx}
                                        for i in ranked[:k].tolist()]
                                content = build_prompt([d["text"] for d in topk], sample["question"])
                                jobs.append({"sample_idx": si, "question": sample["question"],
                                             "alpha": alpha, "k": k, "topk": topk,
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
                            "alpha": j["alpha"], "k": j["k"],
                            "prompt": prompt_map[j["content"]], "topk": j["topk"],
                            "generated_answer": answer, "ground_truth": j["ground_truth"],
                            "is_correct": is_answer_correct(answer, j["ground_truth"]),
                        })

                    results_path = out_dir / "results.jsonl"
                    write_jsonl(results_path, records)
                    logger.info(f"Wrote {len(records)} records -> {results_path}")

                    for alpha in alphas:
                        for k in KS:
                            rows = [r for r in records if r["alpha"] == alpha and r["k"] == k]
                            acc = sum(r["is_correct"] for r in rows) / len(rows)
                            logger.info(f"alpha={alpha:.2f} k={k:2d} | accuracy={acc:.3f}")

        del llm  # free the GPU before loading the next model
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Done computing end-to-end evaluation.")


if __name__ == "__main__":
    main()
