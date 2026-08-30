"""
Attach ConflictQA ground-truth keyword lists to the already-split normalized files.

Reads the original ConflictQA CSV -> {question: ground_truth}, then adds a
"ground_truth" field to every sample of each conflictqa train/test split, in place.
Run this instead of regenerating, so the existing splits (and the directions /
retrieval results computed on them) stay untouched.

Usage:
    python src/experiments/add_conflictqa_ground_truth.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from utils import DATA_DIR, NORMALIZED_DIR, logger, write_jsonl

CSV_PATH = DATA_DIR / "conflictQA-popQA-gpt4_is_memory_correct_non_ambiguous.csv"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    qmap = {r["question"]: ast.literal_eval(r["ground_truth"]) for r in df.to_dict(orient="records")}
    logger.info(f"Built ground_truth map for {len(qmap)} questions from {CSV_PATH}")

    for path in sorted((NORMALIZED_DIR / "conflictqa").rglob("*.jsonl")):
        rows = [json.loads(line) for line in path.open() if line.strip()]
        missing = 0
        for r in rows:
            gt = qmap.get(r["question"])
            if gt is None:
                missing += 1
            else:
                r["ground_truth"] = gt
        write_jsonl(path, rows)
        logger.info(f"{path}: {len(rows)} samples updated ({missing} questions missing from CSV)")


if __name__ == "__main__":
    main()
