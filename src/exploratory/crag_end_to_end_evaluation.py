"""
End-to-end RAG evaluation on CRAG Task 3 (up to 50 real web pages per question).

Tests the fused factuality scoring in a realistic retrieve-then-rerank setting:

    score(chunk, q) = (1 - alpha) * z(cos_SBERT(q, chunk)) + alpha * z(proj(h_L(chunk), v_fact))

where z(.) is a z-score computed OVER THAT QUESTION'S OWN CANDIDATE POOL (not a global
corpus, unlike retrieval_evaluation), v_fact is the diff-in-means factuality direction and
h_L is the chunk's last-token resid_post at layer L. We rank the pool, take top-k, feed the
chunks to the generator and check whether the answer matches a CRAG gold answer.

Three sequential phases (TransformerLens and vLLM cannot co-reside on one GPU):

  1. Shortlist  (model-independent, GPU=SBERT): parse HTML -> blingfire sentences ->
                pack to <=512-token chunks -> SBERT-recall top-M per question.
                Cached to _shortlist/<cfg>/shortlist.jsonl (resumable).
  2. Residuals  (per model, GPU=TransformerLens): last-token resid_post at the top
                layer(s) for the M chunks -> _hidden/<model>/layer_<L>.pt.
  3. Generate   (per model, GPU=vLLM): project, fuse, rank, top-k, generate, score.

Uses the top-layer configs already selected by plot_retrieval_evaluation
(top_layers_<procedure>_<position>.json), each at its best alpha plus the alpha=0 baseline.

Metric is a proxy (normalized match vs answer + alt_ans). results.jsonl stores every field
needed to re-score later with CRAG's official LLM judge.

Usage:
    python src/exploratory/crag_end_to_end_evaluation.py --only-shortlist   # build/inspect stage 1
    python src/exploratory/crag_end_to_end_evaluation.py
    python src/exploratory/crag_end_to_end_evaluation.py --models meta-llama/Llama-3.2-1B-Instruct --num-questions 200
    python src/exploratory/crag_end_to_end_evaluation.py --mixed --models Qwen/Qwen2-7B-Instruct
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

# Many CRAG pages are XML/broken; we parse best-effort and don't need the noisy heuristic warning.
warnings.filterwarnings("ignore", message="It looks like you're parsing an XML document")

# vLLM's EngineCore worker must spawn (fresh CUDA context), not fork; see end_to_end_evaluation.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
from utils import REPO_ROOT, RESULTS_DIR, logger, safe_model_id, setup_logging, write_jsonl

MODELS = ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-1B-Instruct",
          "google/gemma-3-4b-it", "Qwen/Qwen2-7B-Instruct"]
DIRECTION_DATASETS = ["nq_swap", "conflictqa"]
# --mixed: directions pooled over several datasets (mixed_direction_identification.py).
# Only the true mixtures; the single-dataset combos are already covered by DIRECTION_DATASETS.
MIXED_COMBOS = ["conflictqa+nq_swap", "conflictqa+longfact", "nq_swap+longfact",
                "conflictqa+nq_swap+longfact"]
EVAL_TASK = "task3"
PROCEDURE = "context_only"
POSITION = "last_pos"                   # CRAG chunks have no entity; only last_pos directions apply
NORMALIZE = "unnormalized"             # mirror retrieval_evaluation default
KS = [2, 5, 10]
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_TOKENIZER = "meta-llama/Llama-3.1-8B-Instruct"  # fixed reference so chunking is model-independent
CRAG_DIR = REPO_ROOT / "benchmark" / "CRAG" / "data" / "crag_task_3_dev_v5"
OUT_ROOT = RESULTS_DIR / "crag_end_to_end_evaluation"

MAX_NEW_TOKENS = 64
MAX_MODEL_LEN = 8192
TL_BATCH = 8                           # chunks per TransformerLens forward
SBERT_BATCH = 256
GPU_UTIL = 0.88                        # leaves room for the parent process's CUDA context
GEN_CHUNK = 2000                       # prompts per vLLM call; the cache is flushed after each


# ----------------------------------------------------------------------------- data / sampling

def free_gpu(tag: str) -> None:
    """Collect + empty the cache, then log what is actually free (stages hand the GPU over)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        logger.info(f"GPU free {free / 2**30:.1f}/{total / 2**30:.1f} GiB ({tag})")


def crag_shards() -> list[Path]:
    shards = sorted(CRAG_DIR.glob("crag_task_3_dev_v5_*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"No CRAG Task 3 shards in {CRAG_DIR}. Extract crag_task_3_dev_v5 first.")
    return shards


def build_offset_index(shards: list[Path]) -> list[tuple[str, int]]:
    """(shard_path, byte_offset) for every non-empty line; cached (reads ~48 GB once)."""
    cache = OUT_ROOT / "_offsets.json"
    if cache.exists():
        return [(p, o) for p, o in json.loads(cache.read_text())]
    logger.info("Indexing CRAG line offsets (one-time scan of all shards) ...")
    index: list[tuple[str, int]] = []
    for shard in shards:
        with shard.open("rb") as f:
            while True:
                off = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    index.append((str(shard), off))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(index))
    logger.info(f"Indexed {len(index)} questions -> {cache}")
    return index


def read_record(shard_path: str, offset: int) -> dict:
    with open(shard_path, "rb") as f:
        f.seek(offset)
        return json.loads(f.readline())


def sample_records(num_questions: int, subsample_seed: int) -> list[dict]:
    index = build_offset_index(crag_shards())
    if num_questions < len(index):
        index = random.Random(subsample_seed).sample(index, num_questions)
    return [read_record(p, o) for p, o in index]


def golds_of(record: dict) -> list[str]:
    """Gold answer plus alternatives (Task 3 uses 'alt_ans').

    CRAG stores numeric answers (counts, years) as ints, so coerce everything to str."""
    raw = [record.get("answer")] + list(record.get("alt_ans") or [])
    return [str(g) for g in raw if g is not None and str(g).strip()]


# ----------------------------------------------------------------------------- stage 1: shortlist

def html_to_text(html: str) -> str:
    """Visible text from (often malformed) HTML; lxml is fast but chokes on some XML/broken
    pages, so fall back to the lenient stdlib parser, then to empty."""
    from bs4 import BeautifulSoup

    for parser in ("lxml", "html.parser"):
        try:
            return BeautifulSoup(html or "", parser).get_text(" ", strip=True)
        except Exception:
            continue
    return ""


def html_to_chunks(html: str, tok, max_tokens: int) -> list[str]:
    """HTML -> visible text -> sentences (blingfire) -> greedy-packed <=max_tokens chunks."""
    from blingfire import text_to_sentences_and_offsets

    text = html_to_text(html)
    if not text:
        return []
    _, offsets = text_to_sentences_and_offsets(text)
    sentences = [text[a:b] for a, b in offsets]

    chunks: list[str] = []
    cur: list[str] = []
    cur_n = 0
    for s in sentences:
        n = len(tok.encode(s, add_special_tokens=False))
        if cur and cur_n + n > max_tokens:
            chunks.append(" ".join(cur))
            cur, cur_n = [], 0
        cur.append(s)
        cur_n += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def build_shortlist(records: list[dict], shortlist_path: Path, recall_m: int,
                    max_chunk_tokens: int, max_pages: int) -> None:
    """Parse+chunk each question's pages, SBERT-recall top-M chunks; append per question (resumable)."""
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    done = set()
    if shortlist_path.exists():
        done = {json.loads(l)["interaction_id"] for l in shortlist_path.open() if l.strip()}
    todo = [r for r in records if r["interaction_id"] not in done]
    if not todo:
        logger.info(f"Shortlist complete ({len(done)} questions): {shortlist_path}")
        return
    logger.info(f"Building shortlist for {len(todo)} questions ({len(done)} cached) -> {shortlist_path}")

    tok = AutoTokenizer.from_pretrained(CHUNK_TOKENIZER)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sbert = SentenceTransformer(SBERT_MODEL, device=device)
    shortlist_path.parent.mkdir(parents=True, exist_ok=True)

    with shortlist_path.open("a") as out:
        for ri, rec in enumerate(todo):
            chunks: list[str] = []
            for sr in rec["search_results"][:max_pages]:
                chunks.extend(html_to_chunks(sr.get("page_result", ""), tok, max_chunk_tokens))
            if not chunks:
                logger.warning(f"No chunks for {rec['interaction_id']} ('{rec['query'][:60]}'); skipping.")
                continue

            q_emb = sbert.encode([rec["query"]], convert_to_numpy=True, normalize_embeddings=True)[0]
            c_emb = sbert.encode(chunks, batch_size=SBERT_BATCH, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)
            cos = c_emb @ q_emb                                   # [n_chunks]
            keep = np.argsort(-cos)[:recall_m]
            out.write(json.dumps({
                "interaction_id": rec["interaction_id"],
                "query": rec["query"],
                "golds": golds_of(rec),
                "domain": rec.get("domain", ""),
                "question_type": rec.get("question_type", ""),
                "static_or_dynamic": rec.get("static_or_dynamic", ""),
                "popularity": rec.get("popularity", ""),
                "chunks": [chunks[i] for i in keep],
                "cosine": [float(cos[i]) for i in keep],
            }, ensure_ascii=False) + "\n")
            out.flush()
            if (ri + 1) % 50 == 0:
                logger.info(f"  shortlisted {ri + 1}/{len(todo)}")

    del sbert
    free_gpu("after SBERT")


def load_shortlist(shortlist_path: Path) -> list[dict]:
    return [json.loads(l) for l in shortlist_path.open() if l.strip()]


def truncate_chunks(shortlist: list[dict], max_tokens: int) -> None:
    """Cap every chunk at max_tokens (reference tokenizer), in place.

    Sentence packing never splits a single sentence, so junk pages (minified JS, tables with
    no punctuation) can emit one multi-thousand-token 'sentence' as a single chunk. Those blow
    up attention (O(seq^2)) and can overflow the generator context. We keep the FIRST
    max_tokens: that head is also the span SBERT scored, so the cosine and factuality terms
    keep reading overlapping text.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CHUNK_TOKENIZER)
    n_cut = 0
    for q in shortlist:
        for i, c in enumerate(q["chunks"]):
            ids = tok.encode(c, add_special_tokens=False)
            if len(ids) > max_tokens:
                q["chunks"][i] = tok.decode(ids[:max_tokens])
                n_cut += 1
    logger.info(f"Truncated {n_cut} over-long chunks to {max_tokens} tokens.")


# ----------------------------------------------------------------------------- stage 2: residuals

def directions_root(mixed: bool) -> Path:
    """Where direction.pt lives. Both trees share the layout <root>/<model>/<dataset>/seed_<S>/..."""
    return RESULTS_DIR / ("mixed_directions" if mixed else "direction_identification")


def needed_layers(model_name: str, mixed: bool) -> dict[str, dict]:
    """direction_dataset -> {'layer', 'best_alpha'} from the top-layers config (best layer only).

    Selection is always made on the conflictqa retrieval evaluation. The mixed configs live in
    their own tree and their filename carries no position suffix (they are last_pos only)."""
    out: dict[str, dict] = {}
    for dd in (MIXED_COMBOS if mixed else DIRECTION_DATASETS):
        if mixed:
            path = (RESULTS_DIR / "mixed_directions_top_retrieval_evaluation" / safe_model_id(model_name)
                    / "conflictqa" / dd / NORMALIZE / f"top_layers_{PROCEDURE}.json")
        else:
            path = (RESULTS_DIR / "top_retrieval_evaluation" / safe_model_id(model_name) / "conflictqa"
                    / dd / NORMALIZE / f"top_layers_{PROCEDURE}_{POSITION}.json")
        if not path.exists():
            logger.warning(f"No top-layers file ({path}); skipping direction={dd} for {model_name}.")
            continue
        top = json.loads(path.read_text())["ranking"][:1]
        if top:
            out[dd] = {"layer": top[0]["layer"], "best_alpha": top[0]["best_alpha"]}
    return out


def compute_hidden(model_name: str, shortlist: list[dict], layers: list[int], max_len: int) -> None:
    """Last-token resid_post at each layer for every shortlisted chunk -> _hidden/<model>/layer_<L>.pt.

    Chunks are already text-truncated with the reference tokenizer; max_len re-caps here in the
    model's own tokenization so attention memory stays bounded for every model."""
    import transformer_lens.utils as tl_utils
    from transformer_lens import HookedTransformer

    hidden_dir = OUT_ROOT / "_hidden" / safe_model_id(model_name)
    layers = [L for L in layers if not (hidden_dir / f"layer_{L}.pt").exists()]
    if not layers:
        logger.info(f"Hidden states cached for {model_name}: {[str(p.name) for p in hidden_dir.glob('layer_*.pt')]}")
        return
    hidden_dir.mkdir(parents=True, exist_ok=True)

    # Flatten (question, chunk) pairs; sort by length for efficient left-padded batches.
    pairs = [(qi, ci, c) for qi, q in enumerate(shortlist) for ci, c in enumerate(q["chunks"])]
    logger.info(f"Computing resid_post at layers {layers} for {len(pairs)} chunks ({model_name}) ...")

    device = tl_utils.get_device()
    model = HookedTransformer.from_pretrained_no_processing(model_name, device=device, dtype="bfloat16")
    tok = model.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    hooks = {L: tl_utils.get_act_name("resid_post", L) for L in layers}
    stop = max(layers) + 1  # stop after the last needed layer: skips remaining blocks + the huge unembed

    enc = [tok(c, return_tensors="pt", add_special_tokens=True).input_ids[0][:max_len] for _, _, c in pairs]
    order = sorted(range(len(pairs)), key=lambda i: enc[i].shape[0])
    d_model = model.cfg.d_model
    flat = {L: torch.zeros(len(pairs), d_model) for L in layers}

    from tqdm import tqdm
    for s in tqdm(range(0, len(order), TL_BATCH), desc="resid_post"):
        idxs = order[s: s + TL_BATCH]
        width = max(enc[i].shape[0] for i in idxs)
        batch = torch.full((len(idxs), width), tok.pad_token_id, dtype=torch.long, device=device)
        for r, i in enumerate(idxs):
            batch[r, width - enc[i].shape[0]:] = enc[i].to(device)
        with torch.no_grad():
            out, cache = model.run_with_cache(batch, names_filter=list(hooks.values()),
                                              prepend_bos=False, stop_at_layer=stop)
        for L in layers:
            resid = cache[hooks[L]][:, -1, :].detach().float().cpu()
            for r, i in enumerate(idxs):
                flat[L][i] = resid[r]
        # ActivationCache keeps a reference to the model, so a surviving `cache` pins all the
        # weights on the GPU and vLLM later fails to allocate. Drop every GPU ref each batch.
        del out, cache, batch

    # Regroup rows back into per-question [n_chunks, d_model] tensors keyed by interaction_id.
    for L in layers:
        by_q: dict[str, torch.Tensor] = {}
        row = 0
        for q in shortlist:
            n = len(q["chunks"])
            by_q[q["interaction_id"]] = flat[L][row: row + n].clone()
            row += n
        # reorder rows: flat is in original pair order (qi, ci), not `order`, so index directly
        torch.save(by_q, hidden_dir / f"layer_{L}.pt")
        logger.info(f"Wrote hidden states -> {hidden_dir / f'layer_{L}.pt'}")

    del model
    free_gpu("after TransformerLens")


# ----------------------------------------------------------------------------- stage 3: generate + score

def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-8)


def build_prompt(docs: list[str], question: str) -> str:
    docs_block = "\n".join(f"[{i + 1}] {d}" for i, d in enumerate(docs))
    return (
        "Use only the information in the context below to answer the question as "
        "factually as possible. Answer concisely.\n\n"
        f"Context:\n{docs_block}\n\n"
        f"Question: {question}\nAnswer:"
    )


def normalize_text(s) -> str:
    """str() guard: golds cached in an existing shortlist may still be ints."""
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def is_correct_proxy(answer: str, golds: list[str]) -> bool:
    """Rough proxy: a gold appears in the answer.

    Short golds ("yes") and numeric ones ("1913", "43") are matched on word boundaries so they
    cannot hit inside a longer token ("11913", "430"); CRAG has many count/year answers."""
    a = normalize_text(answer)
    for g in golds:
        g = normalize_text(g)
        if not g:
            continue
        if len(g) < 4 or g.replace(",", "").replace(".", "").isdigit():
            if re.search(rf"\b{re.escape(g)}\b", a):
                return True
        elif g in a:
            return True
    return False


def sha(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def load_generations(path: Path) -> dict[str, tuple[str, str]]:
    """sha(content) -> (rendered_prompt, answer) produced by earlier runs.

    Generation is the expensive step (~1h/model), so it is cached on its own and keyed by the
    prompt content: any later failure (scoring, IO) costs seconds instead of the whole run."""
    if not path.exists():
        return {}
    out = {}
    for line in path.open():
        if line.strip():
            r = json.loads(line)
            out[r["content_sha"]] = (r["prompt"], r["answer"])
    return out


def generate(llm, contents: list[str], sampling) -> tuple[list[str], list[str]]:
    convs = [[{"role": "user", "content": c}] for c in contents]
    outs = llm.chat(convs, sampling, use_tqdm=True)
    return [o.prompt for o in outs], [o.outputs[0].text.strip() for o in outs]


def evaluate_model(model_name: str, shortlist: list[dict], configs: dict[str, dict],
                   subsample_seed: int, force: bool, mixed: bool) -> None:
    from vllm import LLM, SamplingParams

    hidden_dir = OUT_ROOT / "_hidden" / safe_model_id(model_name)
    hidden_cache = {c["layer"]: torch.load(hidden_dir / f"layer_{c['layer']}.pt", map_location="cpu")
                    for c in configs.values()}

    # Assemble all (direction_dataset, seed) jobs, dedup prompts across the whole model.
    jobs = []            # one per (dd, seed, question, alpha, k)
    for dd, cfg in configs.items():
        layer, best_alpha = cfg["layer"], cfg["best_alpha"]
        for seed in discover_direction_seeds(model_name, dd, mixed):
            # mix_ prefix keeps the mixture results out of the per-dataset trees (a combo can
            # share a name with a direction dataset, e.g. "conflictqa").
            out_dir = (OUT_ROOT / safe_model_id(model_name) / EVAL_TASK
                       / (f"mix_{dd}" if mixed else dd) / NORMALIZE
                       / f"seed_{seed}" / PROCEDURE / f"layer_{layer}")
            if not force and (out_dir / "results.jsonl").exists():
                logger.info(f"Skip (exists): {out_dir}")
                continue
            dir_path = (directions_root(mixed) / safe_model_id(model_name)
                        / dd / f"seed_{seed}" / PROCEDURE / f"layer_{layer}" / POSITION / "direction.pt")
            if not dir_path.exists():
                logger.warning(f"Missing direction {dir_path}; skipping seed {seed}.")
                continue
            direction = torch.load(dir_path, map_location="cpu").float()
            hid = hidden_cache[layer]

            for q in shortlist:
                h = hid[q["interaction_id"]]                       # [n_chunks, d_model]
                s_proj = zscore((h @ direction).numpy())
                s_cos = zscore(np.asarray(q["cosine"], dtype=np.float32))
                for alpha in (0.0, best_alpha):
                    scores = (1 - alpha) * s_cos + alpha * s_proj
                    ranked = np.argsort(-scores)
                    for k in KS:
                        topk = [{"text": q["chunks"][i], "score": float(scores[i])}
                                for i in ranked[:k].tolist()]
                        content = build_prompt([d["text"] for d in topk], q["query"])
                        jobs.append({"dd": dd, "seed": seed, "layer": layer, "out_dir": out_dir,
                                     "interaction_id": q["interaction_id"], "query": q["query"],
                                     "golds": q["golds"], "domain": q["domain"],
                                     "question_type": q["question_type"],
                                     "static_or_dynamic": q["static_or_dynamic"],
                                     "popularity": q["popularity"], "alpha": alpha, "k": k,
                                     "topk": topk, "content": content})
    if not jobs:
        logger.info(f"Nothing to generate for {model_name}.")
        return

    unique = list(dict.fromkeys(j["content"] for j in jobs))
    sha_of = {c: sha(c) for c in unique}
    cache_path = OUT_ROOT / "_generations" / f"{safe_model_id(model_name)}.jsonl"
    cached = load_generations(cache_path)
    todo = [c for c in unique if sha_of[c] not in cached]
    logger.info(f"{model_name}: {len(jobs)} jobs -> {len(unique)} unique prompts "
                f"({len(unique) - len(todo)} cached, {len(todo)} to generate)")

    if todo:
        free_gpu("before vLLM")
        llm = LLM(model=model_name, dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
                  gpu_memory_utilization=GPU_UTIL)
        sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        for s in range(0, len(todo), GEN_CHUNK):
            part = todo[s: s + GEN_CHUNK]
            prompts, answers = generate(llm, part, sampling)
            with cache_path.open("a") as f:  # flush per chunk so an interrupt keeps the work
                for c, p, a in zip(part, prompts, answers):
                    f.write(json.dumps({"content_sha": sha_of[c], "prompt": p, "answer": a},
                                       ensure_ascii=False) + "\n")
            cached.update({sha_of[c]: (p, a) for c, p, a in zip(part, prompts, answers)})
            logger.info(f"  generated {min(s + GEN_CHUNK, len(todo))}/{len(todo)} -> {cache_path.name}")
        del llm
        free_gpu("after vLLM")

    # Group finished jobs back per output dir and write results.jsonl.
    by_out: dict[Path, list[dict]] = {}
    for j in jobs:
        prompt, ans = cached[sha_of[j["content"]]]
        by_out.setdefault(j["out_dir"], []).append({
            "interaction_id": j["interaction_id"], "query": j["query"],
            "domain": j["domain"], "question_type": j["question_type"],
            "static_or_dynamic": j["static_or_dynamic"], "popularity": j["popularity"],
            "alpha": j["alpha"], "k": j["k"], "prompt": prompt,
            "topk": j["topk"], "generated_answer": ans, "ground_truth": j["golds"],
            "is_correct": is_correct_proxy(ans, j["golds"]),
        })
    for out_dir, records in by_out.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_dir / "results.jsonl", records)
        logger.info(f"Wrote {len(records)} records -> {out_dir / 'results.jsonl'}")
        for alpha in sorted({r["alpha"] for r in records}):
            for k in KS:
                rows = [r for r in records if r["alpha"] == alpha and r["k"] == k]
                acc = sum(r["is_correct"] for r in rows) / len(rows)
                logger.info(f"  {out_dir.relative_to(OUT_ROOT)} | alpha={alpha:.2f} k={k:2d} | proxy_acc={acc:.3f}")


def discover_direction_seeds(model_id: str, dataset: str, mixed: bool) -> list[int]:
    """Mixed directions exist for seed 42 only, so this returns a single seed there."""
    root = directions_root(mixed) / safe_model_id(model_id) / dataset
    return [int(d.name.split("_")[1]) for d in sorted(root.glob("seed_*")) if d.is_dir()]


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--num-questions", type=int, default=1000)
    ap.add_argument("--subsample-seed", type=int, default=42)
    ap.add_argument("--recall-m", type=int, default=50)
    ap.add_argument("--max-chunk-tokens", type=int, default=512)
    ap.add_argument("--max-pages", type=int, default=50)
    ap.add_argument("--only-shortlist", action="store_true", help="Build/inspect stage 1 and exit.")
    ap.add_argument("--mixed", action="store_true",
                    help="Use the mixed-dataset directions (results/mixed_directions) instead of "
                         "the per-dataset ones.")
    ap.add_argument("--force", action="store_true", help="Recompute results.jsonl even if present.")
    args = ap.parse_args()

    setup_logging("crag_end_to_end_evaluation", OUT_ROOT)
    cfg_tag = f"n{args.num_questions}_sub{args.subsample_seed}_m{args.recall_m}_tok{args.max_chunk_tokens}_pg{args.max_pages}"
    shortlist_path = OUT_ROOT / "_shortlist" / cfg_tag / "shortlist.jsonl"

    records = sample_records(args.num_questions, args.subsample_seed)
    logger.info(f"Sampled {len(records)} CRAG questions (config {cfg_tag}).")
    build_shortlist(records, shortlist_path, args.recall_m, args.max_chunk_tokens, args.max_pages)
    if args.only_shortlist:
        logger.info("Done (shortlist only).")
        return

    shortlist = load_shortlist(shortlist_path)
    logger.info(f"Loaded shortlist: {len(shortlist)} questions.")
    truncate_chunks(shortlist, args.max_chunk_tokens)

    for model_name in args.models:
        configs = needed_layers(model_name, args.mixed)
        if not configs:
            continue
        compute_hidden(model_name, shortlist, sorted({c["layer"] for c in configs.values()}),
                       args.max_chunk_tokens)
        evaluate_model(model_name, shortlist, configs, args.subsample_seed, args.force, args.mixed)

    logger.info("Done computing CRAG end-to-end evaluation.")


if __name__ == "__main__":
    main()
