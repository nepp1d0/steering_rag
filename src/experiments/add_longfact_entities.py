"""
Attach LongFact verified-entity texts to the already-split normalized files.

`dataset_normalization.py` keeps only the sentence and its label, so the entity
strings (verbatim substrings of the sentence, needed for entity_pos direction
identification) are lost. This script rebuilds the sentence -> entities map from
the raw HF dataset and attaches it to every longfact train/test split in place:
factual rows get `factual_answer`, non-factual rows get `non_factual_answer`.
Run this instead of regenerating, so the existing splits (and the directions
computed on them) stay untouched.

Usage:
    python src/experiments/add_longfact_entities.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.append(str(Path(__file__).resolve().parent))
from utils import NORMALIZED_DIR, logger, write_jsonl
from dataset_normalization import LONGFACT_HF_ID, LONGFACT_MODEL_ID_HF

from datasets import load_dataset


def build_sentence_entity_map(split: str = "test") -> Dict[str, List[str]]:
    """Same sentence extraction as `normalize_longfact`: one entry per sentence whose
    entity labels all agree and are Supported / Not Supported. Entity texts are
    verbatim substrings of the sentence by construction."""
    import nltk
    nltk.download("punkt_tab", quiet=True)

    logger.info(f"Loading LongFact from HF ({LONGFACT_HF_ID}, split={split}) ...")
    ds = load_dataset(LONGFACT_HF_ID, LONGFACT_MODEL_ID_HF, split=split)

    per_sentence: Dict[str, Dict[str, List[str]]] = {}
    for row in ds:
        model_answer = [m["content"] for m in row["conversation"] if m["role"] == "assistant"][0]
        sentences = nltk.tokenize.sent_tokenize(model_answer)
        for entity in row["verified_entities"]:
            if entity["text"] not in model_answer:
                continue
            target = [s for s in sentences if entity["text"] in s]
            if len(target) != 1:
                continue
            rec = per_sentence.setdefault(target[0], {"labels": [], "entities": []})
            rec["labels"].append(entity["label"])
            rec["entities"].append(entity["text"])

    mapping: Dict[str, List[str]] = {}
    for sentence, rec in per_sentence.items():
        labels = set(rec["labels"])
        if len(labels) != 1 or labels.pop() not in ("Supported", "Not Supported"):
            continue
        mapping[sentence] = rec["entities"]
    logger.info(f"Built entity map for {len(mapping)} sentences.")
    return mapping


def main() -> None:
    mapping = build_sentence_entity_map()
    for path in sorted((NORMALIZED_DIR / "longfact").rglob("*.jsonl")):
        rows = [json.loads(line) for line in path.open() if line.strip()]
        missing = 0
        for r in rows:
            sentence = r["factual_context"] or r["non_factual_evidence"]
            ents = mapping.get(sentence)
            if ents is None:
                missing += 1
                continue
            if r["factual_context"]:
                r["factual_answer"] = ents
            else:
                r["non_factual_answer"] = ents
        write_jsonl(path, rows)
        logger.info(f"{path}: {len(rows)} samples updated ({missing} sentences missing from map)")


if __name__ == "__main__":
    main()
