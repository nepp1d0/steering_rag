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
import ast
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
LONGFACT_HF_ID = "obalcells/hallucination-heads-longfact"
LONGFACT_MODEL_ID_HF = "Llama-3.3-70B-Instruct"



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
            "ground_truth": ast.literal_eval(r["ground_truth"]),
            "original_dataset_id": "conflictqa",
        })
    logger.info(f"ConflictQA: kept {len(rows)} XOR samples.")
    return rows


def normalize_longfact(split: str = "test") -> List[Dict]:
    """Load LongFact from HF and normalize to the common schema."""
    logger.info(f"Loading LongFact from HF ({LONGFACT_HF_ID}, split={split}) ...")
    ds = load_dataset(LONGFACT_HF_ID, LONGFACT_MODEL_ID_HF, split=split)
    # `split` already selects the HF experimental split, so use `ds` directly.
    df = pd.DataFrame(ds)

    import nltk
    nltk.download('punkt_tab')
    
    # Get data ONLY FOR IDENTIFICATION TODO: fix this to be also for reranking
    # Next code is to split each assistant message into sentences 
    # Keep only the sentences with only one entity ("supported2, not supported, not specified)
    list_of_conversations = df['conversation'].tolist()
    list_of_entities_objects = df['verified_entities'].tolist()
    list_of_sentences = []
    list_of_entities = []
    list_of_labels = []
    list_of_queries = []


    for conversation, entities in zip(list_of_conversations, list_of_entities_objects):
        
        for entity in entities:

            # find entity in the conversation, they sould be present in the model answer
            model_answer = [message['content'] for message in conversation if message['role'] == 'assistant'][0]
            
            # if the entity not in model answer, skip
            if entity['text'] not in model_answer:
                print(f"Entity {entity['text']} not in model answer")
                print(f"Model answer: {model_answer}")
                continue
            # if the entity in model answer, get the sentence around the entity
            # split into sentences
            sentences = nltk.tokenize.sent_tokenize(model_answer)
            target_sentence = [sentence for sentence in sentences if entity['text'] in sentence]
            if len(target_sentence) == 0:
                continue
            if len(target_sentence) > 1:
                print(f"More than one sentence found for entity {entity}")
                continue
            target_sentence = target_sentence[0]
            # get the label
            label = entity['label']
            entity_name = entity['text']
            # add to the list
            list_of_sentences.append(target_sentence)
            list_of_entities.append(entity_name)
            list_of_labels.append(label)

    df_sentences = pd.DataFrame({"sentence": list_of_sentences, "label": list_of_labels, "entity": list_of_entities})

    # Filter out duplicated sentences and balance the dataset 

    # --- Step 1: dedupe — keep one row per sentence only if all entity labels agree ---
    records = []
    for sentence, group in df_sentences.groupby('sentence', sort=False):
        labels = group['label'].unique()
        if len(labels) == 1:
            records.append({
                'sentence': sentence,
                'label': labels[0],
                'entities': list(group['entity'])
            })
        # else: mixed labels for this sentence -> discard entirely

    df_dedup = pd.DataFrame(records)

    # --- Step 2: keep only the two classes we use, then balance them ---
    df_dedup = df_dedup[df_dedup['label'].isin(["Supported", "Not Supported"])]
    print("Before balancing:")
    print(df_dedup['label'].value_counts())

    min_count = df_dedup['label'].value_counts().min()
    df_balanced = (
        df_dedup
        .groupby('label', group_keys=False)
        .apply(lambda g: g.sample(n=min_count, random_state=42))
        .reset_index(drop=True)
    )

    rows: List[Dict] = []
    for row in df_balanced.to_dict(orient="records"):
        supported = row["label"] == "Supported"
        rows.append({
            "question": "",
            "factual_answer": [],
            "factual_context": str(row["sentence"]) if supported else "",
            "non_factual_answer": [],
            "non_factual_evidence": "" if supported else str(row["sentence"]),
            "ground_truth": [],
            "original_dataset_id": "longfact",
        })
    logger.info(f"LongFact: kept {len(rows)} samples ({min_count} per class).")
    return rows

def make_split(rows: List[Dict], train_frac: float = 0.8, seed: int = 42,
               key=lambda r: r["question"]) -> Dict[str, List[Dict]]:
    """Split rows 80/20 on the unique values of `key` (all rows sharing a key go together)."""
    groups = list({key(r) for r in rows})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_train = int(len(groups) * train_frac)
    train_keys = set(groups[:n_train])
    return {
        "train": [r for r in rows if key(r) in train_keys],
        "test": [r for r in rows if key(r) not in train_keys],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize NQ-Swap and ConflictQA into a common schema.")
    parser.add_argument("--nq_swap_split", type=str, default="dev",
                        help="Split to use for the HF NQ-Swap dataset.")
    parser.add_argument("--conflictqa_csv", type=str, default=str(CONFLICTQA_DEFAULT_CSV),
                        help="Path to the ConflictQA CSV file.")
    parser.add_argument("--train_frac", type=float, default=0.8,
                        help="Fraction of unique questions assigned to train split.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 67, 89, 90],
                        help="List of random seeds for train/test splitting.")
    args = parser.parse_args()

    setup_logging("dataset_normalization", NORMALIZED_DIR)

    # Normalize once, then split for each seed.
    nq_rows = normalize_nq_swap(split=args.nq_swap_split)
    cqa_rows = normalize_conflictqa(Path(args.conflictqa_csv))
    lf_rows = normalize_longfact()

    # longfact rows carry no question, so split on the sentence (its single non-empty chunk).
    lf_key = lambda r: r["factual_context"] or r["non_factual_evidence"]

    for seed in args.seeds:
        logger.info(f"=== Seed {seed} ===")
        for dataset_id, rows, key in [
            ("nq_swap", nq_rows, None),
            ("conflictqa", cqa_rows, None),
            ("longfact", lf_rows, lf_key),
        ]:
            split_kwargs = {"key": key} if key is not None else {}
            splits = make_split(rows, train_frac=args.train_frac, seed=seed, **split_kwargs)
            for split_name, split_rows in splits.items():
                path = NORMALIZED_DIR / dataset_id / f"seed_{seed}" / f"{split_name}.jsonl"
                write_jsonl(path, split_rows)
                logger.info(f"Wrote {len(split_rows)} {split_name} samples -> {path}")


if __name__ == "__main__":
    main()
