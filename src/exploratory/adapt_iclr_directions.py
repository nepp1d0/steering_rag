"""
Adapt LLMsKnow (ICLR "LLMs Know More Than They Show") probes into our direction template.

Their "direction" is a logistic-regression probe weight vector: for a saved
sklearn classifier, `clf.coef_[0]` is a single vector in activation space whose
sign already points toward the "truthful / correct" class (higher coef.x -> more
truthful). We drop the intercept: our retrieval eval z-scores projections
globally, so a constant offset does not change any ranking.

Their checkpoints live at:
    <LLMsKnow>/checkpoints/clf_<friendly_model>_<dataset>_layer-<L>_token-<token>.pkl

NOTE: the probe location (--probe_at, e.g. mlp) is NOT encoded in the filename,
so we take it as a CLI argument and trust the caller ran the probe with that
location. We assert the coef dimensionality matches the expected activation size
(d_model=4096 for a `mlp` / `attention_output` probe on Llama-3-8B) to guard
against accidentally adapting a d_ff probe (mlp_last_layer_only_input).

Output (separate tree, never mixed with our mean-diff directions):
    results/iclr_directions/<safe_model>/<dataset>/<probe_at>/layer_<L>/direction.pt
                                                          .../layer_<L>/meta.json

Usage:
    python src/exploratory/adapt_iclr_directions.py
    python src/exploratory/adapt_iclr_directions.py --probe-at mlp --expected-dim 4096
"""

from __future__ import annotations
import argparse
import pickle
import re
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
from utils import RESULTS_DIR, logger, safe_model_id, setup_logging, write_jsonl

# Their friendly model names (probe.py filenames) -> full HF ids we use elsewhere.
FRIENDLY_TO_HF = {
    "llama-3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
    "mistral-7b": "mistralai/Mistral-7B-v0.3",
}

# clf_<friendly_model>_<dataset>_layer-<L>_token-<token>.pkl
# <dataset> may contain underscores (natural_questions_with_context); `_layer-`
# and `_token-` are unambiguous markers, so a greedy dataset capture is safe.
FRIENDLY_ALT = "|".join(re.escape(k) for k in FRIENDLY_TO_HF)
CKPT_RE = re.compile(
    rf"^clf_(?P<model>{FRIENDLY_ALT})_(?P<dataset>.+)_layer-(?P<layer>\d+)_token-(?P<token>.+)\.pkl$"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints-dir", default=str(
        Path(__file__).resolve().parents[1] / "other_methods" / "LLMsKnow" / "checkpoints"))
    ap.add_argument("--probe-at", default="mlp",
                    help="Activation location the probes were trained at (not stored in the filename).")
    ap.add_argument("--expected-dim", type=int, default=4096,
                    help="Expected coef length; asserts we did not grab a d_ff probe. Set to -1 to skip.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing adapted directions.")
    args = ap.parse_args()

    ckpt_dir = Path(args.checkpoints_dir)
    out_root = RESULTS_DIR / "iclr_directions"
    setup_logging("adapt_iclr_directions", out_root)

    if not ckpt_dir.is_dir():
        logger.error(f"Checkpoints dir not found: {ckpt_dir}. Run their probe.py --save_clf first.")
        return

    pkls = sorted(ckpt_dir.glob("clf_*.pkl"))
    if not pkls:
        logger.error(f"No clf_*.pkl found in {ckpt_dir}")
        return
    logger.info(f"Found {len(pkls)} checkpoint(s) in {ckpt_dir}")

    n_written = 0
    for pkl in pkls:
        m = CKPT_RE.match(pkl.name)
        if not m:
            logger.warning(f"Skip (unrecognized name): {pkl.name}")
            continue
        friendly, dataset, layer, token = m["model"], m["dataset"], int(m["layer"]), m["token"]
        hf_id = FRIENDLY_TO_HF[friendly]

        out_dir = out_root / safe_model_id(hf_id) / dataset / args.probe_at / f"layer_{layer}"
        dir_path = out_dir / "direction.pt"
        if dir_path.exists() and not args.force:
            logger.info(f"Skip (exists): {dir_path}")
            continue

        with pkl.open("rb") as f:
            clf = pickle.load(f)
        coef = clf.coef_[0]  # [n_features]; binary LR -> shape [1, d]
        if args.expected_dim > 0 and coef.shape[0] != args.expected_dim:
            logger.error(f"Skip {pkl.name}: coef dim {coef.shape[0]} != expected {args.expected_dim} "
                         f"(wrong probe_at? d_ff probe?)")
            continue

        direction = torch.tensor(coef, dtype=torch.float32)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(direction, dir_path)
        write_jsonl(out_dir / "meta.json", [{
            "source": "LLMsKnow_probe_coef",
            "checkpoint": pkl.name,
            "hf_model": hf_id,
            "dataset": dataset,
            "probe_at": args.probe_at,
            "layer": layer,
            "token": token,
            "n_features": int(coef.shape[0]),
            "coef_norm": float(direction.norm()),
            "intercept": float(clf.intercept_[0]) if hasattr(clf, "intercept_") else None,
        }])
        logger.info(f"Wrote {dir_path} (dim={coef.shape[0]}, norm={direction.norm():.3f})")
        n_written += 1

    logger.info(f"Done. Adapted {n_written} direction(s) into {out_root}")


if __name__ == "__main__":
    main()
