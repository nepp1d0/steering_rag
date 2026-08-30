"""
RAGuard step A - normalize the RAGuard claim set.

Downloads `claims.csv` / `documents.csv` from the HuggingFace repo `UCSC-IRKM/RAGuard`
into `data/raguard/` (cached; skipped if already there) and writes the claim probe set:

    data/raguard/claims.jsonl
        {"claim_id": int, "claim": str, "verdict": bool,
         "original_verdict": str, "truthiness": int}

`truthiness` ranks the 6-way original verdict pants-on-fire=0 .. true=5, used for the
graded (monotonicity) analysis in raguard_direction_evaluation.py.

No seeded splits: this is a zero-shot eval set used in full. The supervised probe
baseline does its own stratified CV internally.

Note on the document side: RAGuard's supporting/misleading/unrelated labels are stance
*relative to the claim's verdict* (supporting is 82% verdict=False, misleading is 75%
verdict=True), not intrinsic factuality labels, so documents are not used here.
`documents.csv` is still downloaded for reference.

Usage:
    python src/exploratory/raguard_normalization.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
from utils import REPO_ROOT, logger, setup_logging, write_jsonl

RAGUARD_DIR = REPO_ROOT / "data" / "raguard"
HF_BASE = "https://huggingface.co/datasets/UCSC-IRKM/RAGuard/resolve/main"
FILES = ["claims.csv", "documents.csv"]

# 6-way original verdict -> ordinal truthiness rank.
TRUTHINESS = {
    "pants-on-fire": 0,
    "false": 1,
    "mostly-false": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5,
}

# Sanity constants from the published dataset.
EXPECTED_N = 2648
EXPECTED_TRUE = 1333
EXPECTED_FALSE = 1315


def download(force: bool) -> None:
    RAGUARD_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = RAGUARD_DIR / name
        if dest.exists() and not force:
            logger.info(f"Cached: {dest}")
            continue
        url = f"{HF_BASE}/{name}"
        logger.info(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize the RAGuard claim set.")
    parser.add_argument("--force-download", action="store_true", help="Re-download the CSVs.")
    args = parser.parse_args()

    setup_logging("raguard_normalization", RAGUARD_DIR)
    download(args.force_download)

    # Some document rows exceed the default field limit; claims.csv is read with the same reader.
    csv.field_size_limit(10 ** 9)
    with (RAGUARD_DIR / "claims.csv").open(newline="") as f:
        raw = list(csv.DictReader(f))

    rows = []
    for r in raw:
        ov = r["Original Verdict"].strip().lower()
        if ov not in TRUTHINESS:
            raise ValueError(f"Unknown Original Verdict '{ov}' for claim {r['Claim ID']}")
        rows.append({
            "claim_id": int(r["Claim ID"]),
            # Every claim is wrapped in double quotes in the CSV; the directions were
            # extracted from bare passage text, so strip them.
            "claim": r["Claim"].strip().strip('"').strip(),
            "verdict": r["Verdict"].strip() == "True",
            "original_verdict": ov,
            "truthiness": TRUTHINESS[ov],
        })

    n_true = sum(r["verdict"] for r in rows)
    n_false = len(rows) - n_true
    logger.info(f"Claims: {len(rows)} | True: {n_true} | False: {n_false}")
    logger.info(f"Original verdicts: {dict(Counter(r['original_verdict'] for r in rows))}")

    assert len(rows) == EXPECTED_N, f"Expected {EXPECTED_N} claims, got {len(rows)}"
    assert (n_true, n_false) == (EXPECTED_TRUE, EXPECTED_FALSE), \
        f"Expected {EXPECTED_TRUE}/{EXPECTED_FALSE} True/False, got {n_true}/{n_false}"
    assert all(r["claim"] for r in rows), "Empty claim text after quote stripping"

    out = RAGUARD_DIR / "claims.jsonl"
    write_jsonl(out, rows)
    logger.info(f"Wrote {out}")


if __name__ == "__main__":
    main()
