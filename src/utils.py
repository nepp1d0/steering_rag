"""
Common utilities shared across experiments.

Normalized sample schema (used by every experiment):
    {
        "question": str,
        "factual_answer":      List[str],   # one or more accepted strings
        "factual_context":     str,
        "non_factual_answer":  List[str],
        "non_factual_evidence": str,
        "original_dataset_id": "nq_swap" | "conflictqa",
    }
"""

from __future__ import annotations

import json
from re import X
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from loguru import logger


# Where normalized datasets live, relative to the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
NORMALIZED_DIR = REPO_ROOT / "data" / "normalized_dataset"
RESULTS_DIR = REPO_ROOT / "results"


def setup_logging(name: str, log_dir: Path) -> None:
    """Logs to both stdout and `<log_dir>/<name>.log`."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(log_dir / f"{name}.log", rotation="10 MB", level="INFO")


def safe_model_id(hf_id: str) -> str:
    """Filesystem-friendly version of an HF model id (`org/name` -> `org__name`)."""
    return hf_id.replace("/", "__")


class NormalizedDataset:
    """Holds train/test splits; access via dataset['train'] or dataset['test']."""

    def __init__(self, splits: Dict[str, List[Dict]]):
        self._splits = splits

    def __getitem__(self, split: str) -> List[Dict]:
        if split not in self._splits:
            raise KeyError(f"Unknown split '{split}'. Available: {list(self._splits)}")
        return self._splits[split]


def load_normalized(dataset_id: str, seed: int = 42) -> NormalizedDataset:
    """Loads train/test splits from `data/normalized_dataset/<id>/seed_<seed>/{train,test}.jsonl`."""
    base = NORMALIZED_DIR / dataset_id / f"seed_{seed}"
    splits: Dict[str, List[Dict]] = {}
    for name in ("train", "test"):
        path = base / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Split not found: {path}. Run dataset_normalization.py first.")
        with path.open() as f:
            splits[name] = [json.loads(line) for line in f if line.strip()]
    return NormalizedDataset(splits)


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_ab_choice_prompt(context_a: str, context_b: str, correct_label: str) -> str:
    """A/B-choice prompt: shows two chunks and the (correct or wrong) label."""
    return (
        "Among these two chunks of text, choose the one which is more factual and truthful. "
        "Answer with a single letter: A or B.\n"
        "------- CHUNK A ----------\n"
        f"{context_a}\n"
        "------- CHUNK B ----------\n"
        f"{context_b}\n"
        "-----------------------------\n"
        f"The correct chunk is: {correct_label}."
    )


def get_last_residual(model, hook_point: str, text: str) -> torch.Tensor:
    """Residual activation at the last token of `text`. Returns 1D `[d_model]` on CPU."""
    with torch.no_grad():
        toks = model.to_tokens(text, prepend_bos=True)
        _, cache = model.run_with_cache(toks, names_filter=hook_point, prepend_bos=True)
    return cache[hook_point][0, -1].detach().cpu()


def get_residual_at_positions(
    model,
    hook_point: str,
    text: str,
    entity: Optional[str],
) -> Dict[str, torch.Tensor]:
    """
    Runs `text` through the model and returns residual-stream activations at:
      - "last_pos":   last token of the text
      - "entity_pos": last token of the first `entity` occurrence (only if entity is given
                      and found in the text; otherwise this key is omitted)

    Returned tensors are 1D `[d_model]`, on CPU.
    """
    with torch.no_grad():
        tokens = model.to_tokens(text, prepend_bos=True)
        _, cache = model.run_with_cache(tokens, names_filter=hook_point, prepend_bos=True)
    resid = cache[hook_point][0]  # [seq, d_model]

    out: Dict[str, torch.Tensor] = {"last_pos": resid[-1].detach().cpu()}

    if entity:
        idx_in_text = text.find(entity)
        if idx_in_text >= 0:
            prefix = text[: idx_in_text + len(entity)]
            tok_idx = model.to_tokens(prefix, prepend_bos=True).shape[1] - 1
            tok_idx = max(0, min(tok_idx, resid.shape[0] - 1))
            out["entity_pos"] = resid[tok_idx].detach().cpu()
    return out


def diff_in_means(pos_acts: torch.Tensor, neg_acts: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Diff-in-means direction: mean(pos) - mean(neg), optionally L2-normalized.
    `pos_acts`, `neg_acts` are tensors of shape [n, d_model].
    """
    direction = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)
    direction = direction.detach().float().cpu()
    if normalize:
        direction = direction / (direction.norm() + 1e-8)
    return direction


