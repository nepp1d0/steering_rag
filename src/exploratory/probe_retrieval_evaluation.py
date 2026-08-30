"""
Retrieval evaluation for the LOGISTIC-REGRESSION PROBE directions
(see probe_direction_identification.py).

Identical to retrieval_evaluation.py in every respect (same fusion, same caching,
same per-record schema, same position handling) except the two roots:

  - directions are read from results/probes/direction_identification/...
  - results are written to  results/probes/retrieval_evaluation/...

    <OUT_ROOT>/<model>/<eval>/<direction>/<normalize>/seed_<S>/<procedure>/layer_<L>/<position>/results.jsonl

llm_hidden_states.pt / sbert_embeddings.pt / docs.jsonl live at the layer level and
are shared by every position (they do not depend on the direction). Cached tensors are
reused only when docs.jsonl matches the current corpus; otherwise they are recomputed
(--force-recompute re-does the results, not matching tensors).

Usage:
    python src/exploratory/probe_retrieval_evaluation.py --automated
    python src/exploratory/probe_retrieval_evaluation.py --automated --force-recompute
    python src/exploratory/probe_retrieval_evaluation.py \
        --model meta-llama/Llama-3.1-8B-Instruct --dataset nq_swap --layer 15 --position entity_pos
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

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
from utils import RESULTS_DIR, load_normalized, logger, safe_model_id, setup_logging, write_jsonl

import transformer_lens.utils as tl_utils
from transformer_lens import HookedTransformer
from sentence_transformers import SentenceTransformer

MODELS = ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-1B-Instruct", "google/gemma-3-4b-it", "Qwen/Qwen2-7B-Instruct"]
DATASETS = ["nq_swap", "conflictqa"]
DIRECTION_DATASETS = ["nq_swap", "conflictqa", "longfact"]
PROCEDURES = ["context_only"]
POSITIONS = ["last_pos", "entity_pos"]
ALPHAS = [0.0, 0.3, 0.5, 1.0]
KS = [2, 5, 10]
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 4

# Probe roots: directions produced by probe_direction_identification.py, results kept
# in a parallel tree so they never collide with the diff-in-means evaluation.
DIRECTIONS_ROOT = RESULTS_DIR / "probes" / "direction_identification"
OUT_ROOT = RESULTS_DIR / "probes" / "retrieval_evaluation"


def discover_direction_seeds(model_id: str, dataset: str) -> List[int]:
    root = DIRECTIONS_ROOT / safe_model_id(model_id) / dataset
    dirs = sorted(root.glob("seed_*"), key=lambda d: int(d.name.split("_")[1]))
    return [int(d.name.split("_")[1]) for d in dirs if d.is_dir()]


def discover_layers(model_id: str, dataset: str, procedure: str, position: str, seed: int) -> List[int]:
    root = DIRECTIONS_ROOT / safe_model_id(model_id) / dataset / f"seed_{seed}" / procedure
    return sorted(int(d.name.split("_")[1]) for d in root.glob("layer_*") if (d / position / "direction.pt").exists())


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
            sorted_indices = np.argsort(-scores)  # descending
            gold_rank = np.where(sorted_indices == gold_idx)[0][0] + 1
            nf_rank   = np.where(sorted_indices == nf_idx)[0][0] + 1
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
                    # No topk_text: the retrieved documents are recoverable from
                    # topk_indices + the layer-level docs.jsonl.
                    "topk_indices": topk_indices,
                })

    results_path = out_dir / "results.jsonl"
    # docs.jsonl lives at the layer level: it describes the shared tensors, not the position.
    docs_path = out_dir.parent / "docs.jsonl"

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


def evaluate_combo(model_name: str, dataset: str, direction_dataset: str, procedure: str,
                   position: str, seed: int, layer: int, alphas: list[float], ks: list[int],
                   batch_size: int, normalize_direction: bool, normalize_path: str,
                   sbert_model: str, force: bool = False) -> None:
    layer_dir = (OUT_ROOT / safe_model_id(model_name)
                 / dataset / direction_dataset / normalize_path / f"seed_{seed}"
                 / procedure / f"layer_{layer}")
    out_dir = layer_dir / position
    if not force and (out_dir / "results.jsonl").exists():
        logger.info(f"Skip (exists): {out_dir}")
        return

    dir_path = (DIRECTIONS_ROOT / safe_model_id(model_name)
                / direction_dataset / f"seed_{seed}" / procedure / f"layer_{layer}"
                / position / "direction.pt")
    if not dir_path.exists():
        logger.warning(f"Skip (no direction): {dir_path}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging("probe_retrieval_evaluation", out_dir)
    logger.info(f"model={model_name} | dataset={dataset} | direction_dataset={direction_dataset} | "
                f"seed={seed} | layer={layer} | procedure={procedure} | position={position} | "
                f"normalize_direction={normalize_direction}")

    samples = load_normalized(dataset, seed)["test"]
    all_docs = corpus_of(samples)
    doc_idx = {d: i for i, d in enumerate(all_docs)}
    logger.info(f"Corpus size: {len(all_docs)} unique documents")

    direction = torch.load(dir_path, map_location="cpu").float()
    if normalize_direction:
        direction = direction / (direction.norm() + 1e-8)
    logger.info(f"Loaded direction from {dir_path}")

    sbert_enc = SentenceTransformer(sbert_model)

    # Tensors are direction-independent: even with --force-recompute they are reused
    # when they match the current corpus (a mismatch recomputes them regardless).
    if cache_matches_corpus(layer_dir, all_docs):
        logger.info("Loading cached SBERT embeddings + LLM hidden states (corpus verified)")
        sbert_emb = torch.load(layer_dir / "sbert_embeddings.pt", map_location="cpu")
        llm_hidden = torch.load(layer_dir / "llm_hidden_states.pt", map_location="cpu")
    else:
        logger.info("Computing SBERT embeddings ...")
        emb = sbert_enc.encode(all_docs, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        sbert_emb = torch.tensor(emb, dtype=torch.float32)
        torch.save(sbert_emb, layer_dir / "sbert_embeddings.pt")

        logger.info("Computing LLM hidden states ...")
        device = tl_utils.get_device()
        model = HookedTransformer.from_pretrained_no_processing(model_name, device=device, dtype="bfloat16")
        model.eval()
        hook_point = tl_utils.get_act_name("resid_post", layer)
        llm_hidden = compute_llm_hidden_states(model, all_docs, hook_point, batch_size)
        torch.save(llm_hidden, layer_dir / "llm_hidden_states.pt")
        del model
        torch.cuda.empty_cache()

    compute_evaluation(llm_hidden, direction, sbert_emb, samples, all_docs, doc_idx,
                       sbert_enc, alphas, ks, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--automated", action="store_true", help="Run all hardcoded combinations (models x datasets x directions x positions).")
    ap.add_argument("--dataset", default="nq_swap")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--direction-dataset", default=None, help="Dataset used for direction (defaults to --dataset)")
    ap.add_argument("--procedure", default="context_only")
    ap.add_argument("--position", default="last_pos")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.3, 0.5, 1.0])
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None,
                    help="Split seed. If omitted, runs for all seeds found in probe direction results.")
    ap.add_argument("--force-recompute", action="store_true",
                    help="Recompute results even if they already exist. Cached tensors are still "
                         "reused when they match the current corpus.")
    args = ap.parse_args()

    normalize_direction = False  # Change here to normalize the directions
    normalize_path = "normalized" if normalize_direction else "unnormalized"

    if args.automated:
        logger.info("Running automated mode ...")
        combinations = itertools.product(MODELS, DATASETS, DIRECTION_DATASETS, PROCEDURES, POSITIONS)
        for model_name, dataset, direction_dataset, procedure, position in combinations:
            seeds = [args.seed] if args.seed is not None else discover_direction_seeds(model_name, direction_dataset)
            if not seeds:
                logger.warning(f"No direction seeds found for {model_name}/{direction_dataset}, skipping.")
                continue
            for seed in seeds:
                for layer in discover_layers(model_name, direction_dataset, procedure, position, seed):
                    evaluate_combo(model_name, dataset, direction_dataset, procedure, position,
                                   seed, layer, ALPHAS, KS, BATCH_SIZE,
                                   normalize_direction, normalize_path, SBERT_MODEL,
                                   force=args.force_recompute)
        logger.info("Done computing automated evaluation.")
    else:
        direction_dataset = args.direction_dataset or args.dataset
        seeds = [args.seed] if args.seed is not None else discover_direction_seeds(args.model, direction_dataset)
        setup_logging("probe_retrieval_evaluation", OUT_ROOT)
        for seed in seeds:
            evaluate_combo(args.model, args.dataset, direction_dataset, args.procedure, args.position,
                           seed, args.layer, args.alphas, args.ks, args.batch_size,
                           normalize_direction, normalize_path, args.sbert_model,
                           force=args.force_recompute)
        logger.info("Done.")


if __name__ == "__main__":
    main()
