"""
ClashEval stage 3: generate answers from the re-ranked pool and score them (figure 7).

Stage 2 shows whether the fused score retrieves the uncorrupted document; this asks what that
buys downstream, once the top-k documents are handed to the generator.

Pipeline, per model:
  1. Rebuild the frozen 12-document pools from frozen_pools.json (one draw, POOL_DRAW).
  2. For each identification dataset in DIRECTION_DATASETS, score every pool at every alpha:
         (1-alpha) * z_pool(cos_SBERT(q,d)) + alpha * z_pool(h_L(d).v_L)
     with the projection z-scored per layer, then averaged over all layers (no layer selection).
  3. Take the top-1 and top-2 documents, build one prompt per (question, alpha, k).
  4. Deduplicate identical prompts before generating: decoding is greedy, so the same prompt
     yields the same answer. Every (question, alpha, k) cell still keeps its own record.
  5. Score each answer against ClashEval's numeric ground truth (see label_answer).

Ground truth: drugs+news answers are numeric, with one gold value per question
(`answer_original`) and a wrong value on every corrupted document (`answer_mod`), so scoring is
an exact numeric comparison. It does not reuse end_to_end_evaluation.is_answer_correct, whose
MIN_ALIAS_LEN=4 substring rule would discard two-character answers like "30".

Labels: `correct` matches the gold answer; `misled` matches the answer_mod of a corrupted
document actually shown in this cell's top-k; `other` is neither, or no number could be parsed.
Only `correct` is plotted -- `misled` is logged as the evidence that accuracy moved because
corrupted documents stopped reaching the generator.

GPU for generation only, steps 1-3 are CPU. One model at a time (vLLM takes the whole card).

Usage:
    python src/experiments/clasheval_end_to_end_generation.py --model meta-llama/Llama-3.1-8B-Instruct
    ... --dry-run       # CPU only: report unique-prompt count and exit, generate nothing
    ... --force         # overwrite an existing results file
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from math import isclose
from pathlib import Path

# vLLM defaults to forking its EngineCore subprocess (envs.py: VLLM_WORKER_MULTIPROC_METHOD
# = "fork"). This process does CPU torch work (SBERT, torch.load, einsum) before vLLM starts, and
# a fork after anything has touched the CUDA driver dies with "CUDA error: initialization error".
# vLLM's own _maybe_force_spawn() is supposed to catch this but its cuda_is_initialized() probe
# did not fire here. setdefault, so an explicit env var from the caller still wins.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clasheval_pipeline import (       # noqa: E402
    HEADLINE_DOMAINS, MODEL, OUT_ROOT, SEEDS, discover_direction_layers, load_eval_pairs,
    load_real_direction, safe_model_id, select_headline,
)
from clasheval_pool_ranking import zscore                       # noqa: E402
from clasheval_pool_ranking_v2 import build_eligible_questions  # noqa: E402
from utils import RESULTS_DIR                                   # noqa: E402

POOL_DIR = RESULTS_DIR / "clasheval_pool_ranking_v2"
OUT_DIR = RESULTS_DIR / "clasheval_end_to_end"

# ---- configuration (all frozen before any generation) -----------------------------------------
POOL_DRAW = 101                 # ONE of the five frozen draws; the pool is fixed across alpha, so
                                # every alpha-to-alpha comparison stays paired within the same pool
DIRECTION_DATASETS = ["nq_swap", "conflictqa", "longfact"]   # same three series as figure 6
SEED_MODE = "mean"              # "mean": average the 5 identification seeds' direction vectors
                                # "single": use SINGLE_SEED alone
SINGLE_SEED = 42
ALPHAS = [round(0.1 * i, 1) for i in range(11)]
KS = [1, 2]
POOL_SIZE = 12
MAX_NEW_TOKENS = 16
MAX_MODEL_LEN = 8192            # 2 documents ~ 2,400 tokens; ample, and within Qwen2-7B's 32k
TIE_JITTER_SEED = 54321         # same recipe as clasheval_pool_ranking_v2 (breaks exact ties only)

NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def build_prompt(docs: list[str], question: str) -> str:
    """Single user turn (no system role: gemma/llama template compatible).

    Asks for a bare number: ClashEval's own stored GPT-4o responses are bare numbers 98.5-99.5%
    of the time, so the task supports it, and it makes first-number extraction unambiguous.
    """
    docs_block = "\n".join(f"[{i + 1}] {d}" for i, d in enumerate(docs))
    return (
        "Use only the information in the context below to answer the question.\n"
        "Answer with a single number and nothing else.\n\n"
        f"Context:\n{docs_block}\n\n"
        f"Question: {question}\nAnswer:"
    )


def first_number(text: str) -> float | None:
    """First number in the answer, commas stripped. None if the model emitted no number."""
    m = NUM_RE.search(text)
    if m is None:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def label_answer(pred: float | None, gold: float, shown_corrupted: list[float]) -> str:
    """correct / misled / other. `correct` takes precedence: a distractor can coincidentally
    carry the gold number (measured: 0.13 of the 11 non-target documents per pool)."""
    if pred is None:
        return "other"
    if isclose(pred, gold, rel_tol=1e-6):
        return "correct"
    if any(isclose(pred, v, rel_tol=1e-6) for v in shown_corrupted):
        return "misled"
    return "other"


def load_direction_stack(model: str, dataset: str, layers: list[int]) -> torch.Tensor:
    """[len(layers), d_model]. SEED_MODE="mean" averages the 5 identification seeds per layer.

    The nq_swap direction is the least seed-stable of the three (mean across-seed cosine 0.80 on
    Llama-3.1-8B and Qwen2-7B, 0.73 on Llama-3.2-1B, 0.32 on gemma-3-4b), so a single seed is a
    real gamble on gemma. Averaging costs nothing here -- one direction means one ranking per
    (pool, alpha), the same generation budget as one seed. Each layer's projection is z-scored
    within the pool before the layers are averaged, so the mean vector's norm is irrelevant.
    """
    if SEED_MODE == "single":
        return torch.stack([load_real_direction(model, dataset, SINGLE_SEED, L)
                            for L in layers])
    return torch.stack([
        torch.stack([load_real_direction(model, dataset, s, L) for s in SEEDS]).mean(0)
        for L in layers
    ])


def load_answer_cache(path: Path, build_content) -> dict[str, str]:
    """Reuse generations from a previous run of this script.

    The prompt is a deterministic function of (question, top_rows, top_sides), all of which are
    stored in the records, so a previous file can be replayed into a prompt -> answer map. This
    makes the nq_swap pass free on a re-run, and also covers every conflictqa/longfact cell whose
    top-k happens to coincide -- which is ALL of them at alpha=0, where the score has no direction
    term. Keyed on the full prompt string, so a changed prompt template self-invalidates.
    """
    if not path.exists():
        return {}
    cache = {}
    for line in path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        cache[build_content(r["question"], r["top_rows"], r["top_sides"])] = r["generated_answer"]
    print(f"[cache] {len(cache)} prompt->answer pairs replayed from {path.name}")
    return cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dry-run", action="store_true", help="CPU only: report prompt counts, generate nothing.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    model = args.model
    ms = safe_model_id(model)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"end_to_end__{ms}.jsonl"
    summary_path = OUT_DIR / f"end_to_end_summary__{ms}.json"
    if out_path.exists() and not args.force and not args.dry_run:
        # A file from the single-direction run does not cover the current DIRECTION_DATASETS, so
        # only skip when every requested direction is already in it.
        have = {json.loads(l).get("direction") for l in out_path.open() if l.strip()}
        if have >= set(DIRECTION_DATASETS):
            print(f"[skip] {out_path} already covers {sorted(have)} (use --force to overwrite).")
            return
        print(f"[rerun] {out_path} covers {sorted(have)}; need {DIRECTION_DATASETS} "
              f"-- its generations will be reused as a cache.")

    # ---- data, pools -------------------------------------------------------------------------
    headline = select_headline(load_eval_pairs(), HEADLINE_DOMAINS)
    by_domain = build_eligible_questions(headline)

    frozen = json.loads((POOL_DIR / "frozen_pools.json").read_text())
    pools = []
    for dom in HEADLINE_DOMAINS:
        for p in frozen[str(POOL_DRAW)][dom]:
            pools.append({"question": p["question"], "domain": dom,
                          "rows": p["rows"], "sides": p["sides"]})
    n_pools = len(pools)
    assert n_pools == sum(len(v) for v in by_domain.values()), "pool count != eligible questions"
    print(f"[pools] draw {POOL_DRAW}: {n_pools} pools (one per question), {POOL_SIZE} docs each")

    orig_text = headline["context_original"].tolist()
    mod_text = headline["context_mod"].tolist()
    ans_orig = headline["answer_original"].astype(float).to_numpy()
    ans_mod = headline["answer_mod"].astype(float).to_numpy()

    def doc_text(r: int, s: int) -> str:
        return orig_text[r] if s == 0 else mod_text[r]

    def doc_answer(r: int, s: int) -> float:
        return float(ans_orig[r]) if s == 0 else float(ans_mod[r])

    # ---- relevance axis ----------------------------------------------------------------------
    from sentence_transformers import SentenceTransformer
    print("[sbert] loading all-MiniLM-L6-v2 (CPU) ...")
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    questions = [p["question"] for p in pools]
    q_emb = dict(zip(questions, sbert.encode(questions, batch_size=64, normalize_embeddings=True,
                                             convert_to_numpy=True, show_progress_bar=True)))
    doc_keys = sorted({(r, s) for p in pools for r, s in zip(p["rows"], p["sides"])})
    print(f"[sbert] encoding {len(doc_keys)} unique pool documents ...")
    doc_emb = dict(zip(doc_keys, sbert.encode([doc_text(r, s) for r, s in doc_keys], batch_size=64,
                                              normalize_embeddings=True, convert_to_numpy=True,
                                              show_progress_bar=True)))
    del sbert

    def content_for(question: str, top_rows: list[int], top_sides: list[int]) -> str:
        return build_prompt([doc_text(r, s) for r, s in zip(top_rows, top_sides)], question)

    # ---- pool-level quantities that do not depend on the direction ---------------------------
    tie_rng = np.random.default_rng(TIE_JITTER_SEED)
    z_cos_by_pool, jitter_by_pool = [], []
    for p in pools:
        cos = np.stack([doc_emb[(r, s)] for r, s in zip(p["rows"], p["sides"])]) @ q_emb[p["question"]]
        z_cos_by_pool.append(zscore(cos))
        jitter_by_pool.append(tie_rng.uniform(-1e-9, 1e-9, size=POOL_SIZE))

    # ---- factuality axis, one direction dataset at a time ------------------------------------
    H_doc = torch.load(OUT_ROOT / ms / "headline_doc_repr.pt", map_location="cpu").float()
    layers = discover_direction_layers(model)
    print(f"[layers] all-layer mean runs over {len(layers)}/{H_doc.shape[0]} layers present "
          f"for every (dataset, seed): {layers}")
    H_doc = H_doc[layers]
    n_layers = H_doc.shape[0]

    jobs = []
    for ds in DIRECTION_DATASETS:
        v = load_direction_stack(model, ds, layers)
        # one einsum over the whole cached tensor, then index per pool -- far cheaper than
        # slicing H_doc 477 times per dataset
        proj_all = torch.einsum("lpsd,ld->lps", H_doc, v).numpy()      # [n_layers, n_pairs, 2]
        print(f"[direction] {ds}, seed_mode={SEED_MODE}, {n_layers} layers")

        for pi, p in enumerate(pools):
            proj = proj_all[:, p["rows"], p["sides"]]                  # [n_layers, 12]
            z_proj = np.stack([zscore(proj[L]) for L in range(n_layers)]).mean(axis=0)
            z_cos, jitter = z_cos_by_pool[pi], jitter_by_pool[pi]
            gold = doc_answer(p["rows"][0], p["sides"][0])

            for a in ALPHAS:
                ranked = np.argsort(-((1 - a) * z_cos + a * z_proj + jitter))
                for k in KS:
                    top = ranked[:k].tolist()
                    top_rows = [p["rows"][i] for i in top]
                    top_sides = [p["sides"][i] for i in top]
                    jobs.append({
                        "direction": ds,
                        "question": p["question"], "domain": p["domain"], "alpha": a, "k": k,
                        "top_rows": top_rows, "top_sides": top_sides,
                        "target_shown": 0 in top,
                        "gold": gold,
                        # answers carried by the corrupted documents actually shown in this cell
                        "shown_corrupted": [doc_answer(r, s) for r, s in zip(top_rows, top_sides) if s == 1],
                        "content": content_for(p["question"], top_rows, top_sides),
                    })
        del v, proj_all

    unique_contents = list(dict.fromkeys(j["content"] for j in jobs))
    cache = load_answer_cache(out_path, content_for)
    todo = [c for c in unique_contents if c not in cache]
    print(f"[jobs] {len(jobs)} cells ({len(DIRECTION_DATASETS)} directions x {n_pools} pools "
          f"x {len(ALPHAS)} alphas x {len(KS)} k) -> {len(unique_contents)} unique prompts "
          f"({len(unique_contents) / len(jobs):.1%} of cells), {len(cache)} cached, "
          f"{len(todo)} to generate")

    if args.dry_run:
        print("[dry-run] no generation, nothing written.")
        return

    # Everything the ranking needed is now materialised into `jobs` (prompt text, gold, shown
    # corrupted answers), so drop the big CPU tensors before vLLM takes the card -- H_doc alone is
    # ~5 GB of host RAM that would otherwise sit there for the whole generation pass.
    del H_doc, doc_emb, q_emb
    gc.collect()

    # ---- generate ----------------------------------------------------------------------------
    answer_map = dict(cache)
    if todo:
        from vllm import LLM, SamplingParams
        llm = LLM(model=model, dtype="bfloat16", max_model_len=MAX_MODEL_LEN)
        sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
        outs = llm.chat([[{"role": "user", "content": c}] for c in todo], sampling, use_tqdm=True)
        answer_map.update(zip(todo, [o.outputs[0].text.strip() for o in outs]))
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("[generate] every prompt was cached; no GPU needed.")

    # ---- score -------------------------------------------------------------------------------
    records = []
    for j in jobs:
        answer = answer_map[j["content"]]
        pred = first_number(answer)
        records.append({
            "direction": j["direction"],
            "question": j["question"], "domain": j["domain"], "alpha": j["alpha"], "k": j["k"],
            "top_rows": j["top_rows"], "top_sides": j["top_sides"],
            "target_shown": j["target_shown"], "n_corrupted_shown": len(j["shown_corrupted"]),
            "generated_answer": answer, "pred_number": pred, "parsed": pred is not None,
            "gold": j["gold"],
            "label": label_answer(pred, j["gold"], j["shown_corrupted"]),
        })
    # write via a temp file: the previous run's file is this run's cache, so it must stay intact
    # until the replacement is complete
    tmp_path = out_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    tmp_path.replace(out_path)
    print(f"[write] {len(records)} records -> {out_path}")

    # ---- summary -----------------------------------------------------------------------------
    summary = {"model": model, "pool_draw": POOL_DRAW, "direction_datasets": DIRECTION_DATASETS,
               "seed_mode": SEED_MODE, "n_questions": n_pools, "alphas": ALPHAS, "ks": KS,
               "n_unique_prompts": len(unique_contents), "n_generated": len(todo), "cells": {}}
    for ds in DIRECTION_DATASETS:
        for k in KS:
            for a in ALPHAS:
                rows = [r for r in records if r["direction"] == ds and r["k"] == k and r["alpha"] == a]
                n = len(rows)
                cell = {lab: sum(r["label"] == lab for r in rows) / n
                        for lab in ("correct", "misled", "other")}
                cell["parsed_rate"] = sum(r["parsed"] for r in rows) / n
                cell["target_shown"] = sum(r["target_shown"] for r in rows) / n
                cell["n"] = n
                summary["cells"][f"{ds}_k{k}_a{a}"] = cell
        for k in KS:
            curve = " ".join(f"{summary['cells'][f'{ds}_k{k}_a{a}']['correct']:.3f}" for a in ALPHAS)
            print(f"  {ds:<11} k={k} correct by alpha: {curve}")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[write] {summary_path}")


if __name__ == "__main__":
    main()
