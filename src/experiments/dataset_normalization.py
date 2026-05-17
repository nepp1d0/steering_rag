"""
Normalize NQ-Swap and ConflictQA into a single common schema (see `src/utils.py`).

Outputs:
    data/normalized_dataset/nq_swap/data.jsonl
    data/normalized_dataset/conflictqa/data.jsonl

Usage:
    python -m src.experiments.dataset_normalization
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import load_dataset

# Allow running both as a module and as a plain script.
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import NORMALIZED_DIR, REPO_ROOT, logger, setup_logging, write_jsonl


CONFLICTQA_DEFAULT_CSV = REPO_ROOT / "data" / "conflictQA-popQA-gpt4_is_memory_correct_non_ambiguous.csv"
NQ_SWAP_HF_ID = "younanna/NQ-Swap"


def normalize_nq_swap(split: str = "validation") -> List[Dict]:
    """Keep only `substitution_type == 'corpus'` and map fields."""
    logger.info(f"Loading NQ-Swap from HF ({NQ_SWAP_HF_ID}, split={split}) ...")
    ds = load_dataset(NQ_SWAP_HF_ID, split=split)
    rows: List[Dict] = []
    for row in ds:
        if row.get("substitution_type") != "corpus":
            continue
        original = row.get("original_answers") or []
        substituted = row.get("substituted_answers") or []
        if not row.get("original_context") or not row.get("substituted_context"):
            continue
        rows.append({
            "question": row["question"],
            "factual_answer": list(original),
            "factual_context": row["original_context"],
            "non_factual_answer": list(substituted),
            "non_factual_evidence": row["substituted_context"],
            "ground_truth": row["original_answers"],
            "original_dataset_id": "nq_swap",
        })
    logger.info(f"NQ-Swap: kept {len(rows)} samples (substitution_type==corpus).")
    return rows


def normalize_conflictqa(csv_path: Path) -> List[Dict]:
    """Keep XOR rows (memory_is_correct != counter_is_correct) and orient pos/neg accordingly."""
    logger.info(f"Loading ConflictQA from {csv_path} ...")
    df = pd.read_csv(csv_path)
    rows: List[Dict] = []
    for r in df.to_dict(orient="records"):
        mem_ok, ctr_ok = bool(r.get("memory_is_correct")), bool(r.get("counter_is_correct"))
        if mem_ok == ctr_ok:
            continue  # skip ambiguous (both true or both false)
        if mem_ok:
            factual_ans = r["memory_answer"]
            factual_ctx = r["parametric_memory_aligned_evidence"]
            non_factual_ans = r["counter_answer"]
            non_factual_ctx = r["counter_memory_aligned_evidence"]
        else:
            factual_ans = r["counter_answer"]
            factual_ctx = r["counter_memory_aligned_evidence"]
            non_factual_ans = r["memory_answer"]
            non_factual_ctx = r["parametric_memory_aligned_evidence"]
        if not factual_ctx or not non_factual_ctx:
            continue
        rows.append({
            "question": r["question"],
            "factual_answer": [str(factual_ans)],
            "factual_context": str(factual_ctx),
            "non_factual_answer": [str(non_factual_ans)],
            "non_factual_evidence": str(non_factual_ctx),
            "original_dataset_id": "conflictqa",
        })
    logger.info(f"ConflictQA: kept {len(rows)} XOR samples.")
    return rows


def make_split(rows: List[Dict], train_frac: float = 0.8, seed: int = 42) -> Dict[str, List[Dict]]:
    """Split rows 80/20 on unique questions (all rows sharing a question go to the same split)."""
    questions = list({r["question"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(questions)
    n_train = int(len(questions) * train_frac)
    train_qs = set(questions[:n_train])
    return {
        "train": [r for r in rows if r["question"] in train_qs],
        "test": [r for r in rows if r["question"] not in train_qs],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize NQ-Swap and ConflictQA into a common schema.")
    parser.add_argument("--nq_swap_split", type=str, default="dev",
                        help="Split to use for the HF NQ-Swap dataset.")
    parser.add_argument("--conflictqa_csv", type=str, default=str(CONFLICTQA_DEFAULT_CSV),
                        help="Path to the ConflictQA CSV file.")
    parser.add_argument("--train_frac", type=float, default=0.8,
                        help="Fraction of unique questions assigned to train split.")
    args = parser.parse_args()

    setup_logging("dataset_normalization", NORMALIZED_DIR)

    for dataset_id, rows in [
        ("nq_swap", normalize_nq_swap(split=args.nq_swap_split)),
        ("conflictqa", normalize_conflictqa(Path(args.conflictqa_csv))),
    ]:
        splits = make_split(rows, train_frac=args.train_frac)
        for split_name, split_rows in splits.items():
            path = NORMALIZED_DIR / dataset_id / f"{split_name}.jsonl"
            write_jsonl(path, split_rows)
            logger.info(f"Wrote {len(split_rows)} {split_name} samples -> {path}")


if __name__ == "__main__":
    main()
