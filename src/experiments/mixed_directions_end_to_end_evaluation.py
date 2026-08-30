"""
End-to-end RAG evaluation for the MIXED-dataset directions (see mixed_direction_identification.py).

Sibling of end_to_end_evaluation.py: same fused retrieval, same prompt, same alias
metric, but the direction is indexed by a dataset *combo* ("conflictqa",
"conflictqa+nq_swap", ...) instead of a single dataset, and both ConflictQA and
NQ-Swap are used as eval sets (both carry `ground_truth` aliases).

Only the best configuration per (model, eval, combo) is run: the layer and alpha
already selected by mixed_directions_plot_retrieval_evaluation.py in
top_layers_<procedure>.json, plus the alpha=0 baseline. If a cell has no selection
for its own eval dataset, the other eval dataset's selection for the same
(model, combo) is used and the fallback is logged.

Mixed directions exist for seed 42 only, so a single seed is run: one point estimate per
cell, no error bars.

Generation uses vLLM; the per-document hidden states come from the cached tensors
mixed_directions_retrieval_evaluation.py already wrote (errors if a cache is
missing — no TransformerLens pass here). Those caches live one level up from the
original layout: sbert_embeddings.pt is shared by every layer.

The alpha=0 baseline depends only on SBERT, so it is identical for all 7 combos of a
given (model, eval). Answers are cached by prompt text across the combo loop, which avoids
regenerating that baseline 7x (~43% of the generation work); decoding is greedy, so a
reused answer is what a rerun would produce.

Outputs:
    results/mixed_directions_end_to_end_evaluation/<model>/<eval>/<combo>/<normalize>/
        seed_42/<procedure>/layer_<L>/{results.jsonl,config.json}

Usage:
    python src/experiments/mixed_directions_end_to_end_evaluation.py
    python src/experiments/mixed_directions_end_to_end_evaluation.py --models Qwen/Qwen2-7B-Instruct
    python src/experiments/mixed_directions_end_to_end_evaluation.py --eval-datasets conflictqa
"""

from __future__ import annotations

import argparse
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

sys.path.append(str(Path(__file__).resolve().parent))
from utils import RESULTS_DIR, load_normalized, logger, safe_model_id, setup_logging, write_jsonl

from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams

MODELS = ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-1B-Instruct",
          "google/gemma-3-4b-it", "Qwen/Qwen2-7B-Instruct"]
EVAL_DATASETS = ["conflictqa", "nq_swap"]
# Ordered singles -> pairs -> triple, same order as mixed_directions_plot_combos.py.
COMBOS = ["conflictqa", "nq_swap", "longfact",
          "conflictqa+nq_swap", "conflictqa+longfact", "nq_swap+longfact",
          "conflictqa+nq_swap+longfact"]
SEED = 42                              # mixed directions exist for this seed only
PROCEDURE = "context_only"
POSITION = "last_pos"                  # the mixed retrieval evaluation is last_pos only
NORMALIZE = "unnormalized"             # mirror retrieval_evaluation default (directions unnormalized)
KS = [2, 5, 10]
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_NEW_TOKENS = 64
MAX_MODEL_LEN = 8192
MIN_ALIAS_LEN = 4                      # drop very short aliases (e.g. "pol") to avoid false matches

TOP_DIR = RESULTS_DIR / "mixed_directions_top_retrieval_evaluation"
EVAL_DIR = RESULTS_DIR / "mixed_directions_retrieval_evaluation"
DIRECTIONS_DIR = RESULTS_DIR / "mixed_directions"
OUT_ROOT = RESULTS_DIR / "mixed_directions_end_to_end_evaluation"


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


def load_best_config(model_name: str, eval_dataset: str, combo: str) -> tuple[dict | None, str | None]:
    """Best (layer, alpha) for this cell, from the mixed retrieval evaluation.

    Falls back to the other eval dataset's selection for the same (model, combo) when
    this cell has none of its own, so a combo is skipped only when the model has no
    selection for it at all. Returns (entry, source_eval_dataset).
    """
    others = [e for e in EVAL_DATASETS if e != eval_dataset]
    for source in [eval_dataset] + others:
        path = (TOP_DIR / safe_model_id(model_name) / source / combo / NORMALIZE
                / f"top_layers_{PROCEDURE}.json")
        if not path.exists():
            continue
        entry = json.loads(path.read_text())["ranking"][0]
        if entry["best_alpha"] is None:
            continue
        return entry, source
    return None, None


def load_sbert_cache(model_name: str, eval_dataset: str) -> torch.Tensor:
    """SBERT document embeddings. Unlike the original layout these are shared by every
    layer, so they live one level above the per-layer hidden states."""
    path = EVAL_DIR / safe_model_id(model_name) / eval_dataset / "cache" / "sbert_embeddings.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing SBERT cache {path}. Run mixed_directions_retrieval_evaluation first.")
    return torch.load(path, map_location="cpu")


def load_hidden_cache(model_name: str, eval_dataset: str, layer: int) -> torch.Tensor:
    """Per-layer document hidden states; error (don't recompute) if absent."""
    path = (EVAL_DIR / safe_model_id(model_name) / eval_dataset / "cache"
            / f"layer_{layer}" / "llm_hidden_states.pt")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing hidden-state cache {path}. Run mixed_directions_retrieval_evaluation first.")
    return torch.load(path, map_location="cpu")


def generate(llm: LLM, contents: list[str], sampling: SamplingParams) -> tuple[list[str], list[str]]:
    """Batched greedy generation over unique prompts; returns (rendered_prompts, answers)."""
    convs = [[{"role": "user", "content": c}] for c in contents]
    outs = llm.chat(convs, sampling, use_tqdm=True)
    return [o.prompt for o in outs], [o.outputs[0].text.strip() for o in outs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None, help=f"Subset of {MODELS}.")
    ap.add_argument("--eval-datasets", nargs="+", default=None, help=f"Subset of {EVAL_DATASETS}.")
    ap.add_argument("--combos", nargs="+", default=None, help=f"Subset of {COMBOS}.")
    ap.add_argument("--force", action="store_true", help="Recompute cells that already have results.")
    args = ap.parse_args()

    models = args.models or MODELS
    eval_datasets = args.eval_datasets or EVAL_DATASETS
    combos = args.combos or COMBOS

    setup_logging("mixed_directions_end_to_end_evaluation", OUT_ROOT)
    sbert_enc = SentenceTransformer(SBERT_MODEL, device="cpu")  # CPU: leave the GPU to vLLM
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    for model_name in models:
        llm = None  # vLLM model, loaded lazily and reused across all this model's combinations

        for eval_dataset in eval_datasets:
            samples = load_normalized(eval_dataset, SEED)["test"]
            if "ground_truth" not in samples[0]:
                raise KeyError(f"{eval_dataset} samples lack 'ground_truth'; attach it before running.")
            all_docs = sorted(set(s["factual_context"] for s in samples)
                              | set(s["non_factual_evidence"] for s in samples))
            doc_idx = {d: i for i, d in enumerate(all_docs)}

            configs = {}
            for combo in combos:
                entry, source = load_best_config(model_name, eval_dataset, combo)
                if entry is None:
                    logger.warning(f"No top-layers selection for {model_name} / {combo}; skipping.")
                    continue
                if source != eval_dataset:
                    logger.warning(f"{model_name} / {eval_dataset} / {combo}: no selection for this "
                                   f"eval set, falling back to the {source} selection "
                                   f"(layer {entry['layer']}, alpha {entry['best_alpha']}).")
                configs[combo] = {"layer": entry["layer"], "alpha": entry["best_alpha"],
                                  "selection_source": source}
            if not configs:
                continue

            logger.info(f"=== {model_name} | eval={eval_dataset} | {len(samples)} samples | "
                        f"{len(all_docs)} docs | seed {SEED}")
            for combo, cfg in configs.items():
                logger.info(f"    {combo:<30} layer={cfg['layer']:<4} alpha={cfg['alpha']} "
                            f"(selected on {cfg['selection_source']})")

            # Query embeddings and the SBERT half of the score depend only on (model, eval),
            # so they are computed once and reused by every combo.
            q_emb = sbert_enc.encode([s["question"] for s in samples], batch_size=64,
                                     show_progress_bar=False, convert_to_numpy=True)
            q_norm = torch.tensor(q_emb, dtype=torch.float32)
            q_norm = q_norm / q_norm.norm(dim=1, keepdim=True)

            sbert_emb = load_sbert_cache(model_name, eval_dataset)
            if len(all_docs) != sbert_emb.shape[0]:
                raise ValueError(f"Corpus/cache mismatch: {len(all_docs)} docs vs "
                                 f"{sbert_emb.shape[0]} cached rows for {model_name}/{eval_dataset}.")
            sbert_norm = sbert_emb / sbert_emb.norm(dim=1, keepdim=True)          # [N, d]
            s_cos = (q_norm @ sbert_norm.T).numpy()                              # [n_samples, N]
            s_cos_norm = (s_cos - s_cos.mean(axis=1, keepdims=True)) / s_cos.std(axis=1, keepdims=True)

            # content -> (rendered_prompt, answer). Shared across combos: the alpha=0 baseline
            # is identical for all of them, so it is generated once instead of 7 times.
            answer_cache: dict[str, tuple[str, str]] = {}

            for combo, cfg in configs.items():
                layer, best_alpha = cfg["layer"], cfg["alpha"]
                alphas = [0.0, best_alpha]  # baseline + chosen steering config

                out_dir = (OUT_ROOT / safe_model_id(model_name) / eval_dataset / combo / NORMALIZE
                           / f"seed_{SEED}" / PROCEDURE / f"layer_{layer}")
                if not args.force and (out_dir / "results.jsonl").exists():
                    logger.info(f"Skip (exists): {out_dir}")
                    continue
                out_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"model={model_name} | eval={eval_dataset} | combo={combo} | "
                            f"seed={SEED} | layer={layer} | alphas={alphas}")

                dir_path = (DIRECTIONS_DIR / safe_model_id(model_name) / combo / f"seed_{SEED}"
                            / PROCEDURE / f"layer_{layer}" / POSITION / "direction.pt")
                if not dir_path.exists():
                    logger.warning(f"Skip (no direction): {dir_path}")
                    continue
                direction = torch.load(dir_path, map_location="cpu").float()

                llm_hidden = load_hidden_cache(model_name, eval_dataset, layer)
                if llm_hidden.shape[0] != len(all_docs):
                    raise ValueError(f"Corpus/cache mismatch: {len(all_docs)} docs vs "
                                     f"{llm_hidden.shape[0]} cached rows at layer {layer}.")
                s_proj_norm = zscore((llm_hidden @ direction).numpy())            # [N]

                # Build all (sample, alpha, k) jobs; deduplicate identical prompts.
                jobs = []
                for si, sample in enumerate(samples):
                    gold_idx = doc_idx[sample["factual_context"]]
                    nf_idx = doc_idx[sample["non_factual_evidence"]]
                    for alpha in alphas:
                        scores = (1 - alpha) * s_cos_norm[si] + alpha * s_proj_norm
                        ranked = np.argsort(-scores)                              # descending
                        for k in KS:
                            topk = [{"text": all_docs[i], "score": float(scores[i]),
                                     "is_gold": i == gold_idx, "is_nonfactual": i == nf_idx}
                                    for i in ranked[:k].tolist()]
                            content = build_prompt([d["text"] for d in topk], sample["question"])
                            jobs.append({"sample_idx": si, "question": sample["question"],
                                         "alpha": alpha, "k": k, "topk": topk,
                                         "ground_truth": sample["ground_truth"], "content": content})

                unique_contents = list(dict.fromkeys(j["content"] for j in jobs))
                todo = [c for c in unique_contents if c not in answer_cache]
                logger.info(f"{len(jobs)} jobs -> {len(unique_contents)} unique prompts "
                            f"({len(unique_contents) - len(todo)} already generated, {len(todo)} to go)")

                if todo:
                    if llm is None:
                        llm = LLM(model=model_name, dtype="bfloat16", max_model_len=MAX_MODEL_LEN)
                    prompts, answers = generate(llm, todo, sampling)
                    answer_cache.update(zip(todo, zip(prompts, answers)))

                records = []
                for j in jobs:
                    prompt, answer = answer_cache[j["content"]]
                    records.append({
                        "sample_idx": j["sample_idx"], "question": j["question"],
                        "alpha": j["alpha"], "k": j["k"],
                        "prompt": prompt, "topk": j["topk"],
                        "generated_answer": answer, "ground_truth": j["ground_truth"],
                        "is_correct": is_answer_correct(answer, j["ground_truth"]),
                    })

                results_path = out_dir / "results.jsonl"
                write_jsonl(results_path, records)
                (out_dir / "config.json").write_text(json.dumps(
                    {"model": model_name, "eval_dataset": eval_dataset, "combo": combo,
                     "seed": SEED, "layer": layer, "alphas": alphas, "ks": KS,
                     "procedure": PROCEDURE, "position": POSITION, "normalize": NORMALIZE,
                     "selection_source": cfg["selection_source"]}, indent=2))
                logger.info(f"Wrote {len(records)} records -> {results_path}")

                for alpha in alphas:
                    for k in KS:
                        rows = [r for r in records if r["alpha"] == alpha and r["k"] == k]
                        acc = sum(r["is_correct"] for r in rows) / len(rows)
                        logger.info(f"alpha={alpha:.2f} k={k:2d} | accuracy={acc:.3f}")

            del answer_cache  # prompts are large; don't carry them into the next eval dataset
            gc.collect()

        del llm  # free the GPU before loading the next model
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Done computing mixed-direction end-to-end evaluation.")


if __name__ == "__main__":
    main()
