# Adding Qwen2.5-32B-Instruct to the paper's results

Instructions for a collaborator with a large GPU. You only need to **run commands** — no code
reading or editing is required. Everything below is copy-paste in order.

At the end you produce a results directory and send it back; the figures are generated on our side.

---

## 0. What this produces

Three figures in the paper currently show four models (1B → 8B). We want a fifth, larger model
added to each, to test the paper's central claim that the factuality signal gets cleaner with scale.

| Paper figure | What your run contributes |
| --- | --- |
| Figure 3 — re-ranking | mean gold / non-factual document rank vs α for the 32B model |
| Figure 6 — cross-domain | rank-separation gain and Δaccuracy per identification set |
| Figure 7 — ClashEval | out-of-distribution end-to-end accuracy |

**Model: `Qwen/Qwen2.5-32B-Instruct`** (64 layers, d_model 5120). It is not gated, so no HuggingFace
token is needed.

Please do not substitute another model without asking us. It was chosen because it is a true 32B
instruct model supported by the pinned TransformerLens 3.2.1, and because it extends the Qwen family
already in the paper (which has Qwen2-7B), so the size comparison stays within one family. Note that
`Qwen3-32B` is **not** supported by this TransformerLens version — Qwen3 stops at 14B.

---

## 1. Requirements

| | |
| --- | --- |
| GPU | one card with **≥ 80 GB** (H100 / A100 80GB). The code is single-GPU; it does not shard. |
| Disk | **≥ 200 GB free** for cached activation tensors under `results/`. |
| Time | roughly **21 h** of GPU time in total. Per-stage estimates below are rough. |
| Python | 3.12 |

Stages are resumable: every script skips work that is already on disk. If a run dies, re-run the
exact same command and it continues.

---

## 2. Setup (once)

```bash
git clone <repo-url> steering_rag
cd steering_rag

conda create -n bias_rag python=3.12 -y
conda activate bias_rag

# Install torch first, matching your CUDA version (this is the cu130 build):
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements_32b.txt
```

**Run every command from the repo root** (`steering_rag/`). Paths are derived from it.

The datasets are already in the repository — you do **not** need to build them. Verify:

```bash
python - <<'EOF'
from pathlib import Path
import json
for ds in ["conflictqa", "nq_swap", "longfact"]:
    p = Path(f"data/normalized_dataset/{ds}/seed_42/train.jsonl")
    n = sum(1 for _ in p.open())
    print(f"{ds:11s} {n:5d} train samples  ok")
print("clasheval parquet:", Path("data/clasheval_gpt4.pqt").stat().st_size // 1_000_000, "MB")
EOF
```

Expected: `conflictqa ~7314`, `nq_swap ~1345`, `longfact ~1974`, and a 28 MB parquet.

---

## 3. Smoke test (~15 minutes) — do this before the long runs

This downloads the model (~62 GB) and proves the GPU, data and code paths all work. The two
layers below are part of the stage 1 set, so nothing computed here is wasted.

```bash
python src/experiments/direction_identification.py \
    --model Qwen/Qwen2.5-32B-Instruct --dataset nq_swap --layers 0,8 --seed 42
```

Then:

```bash
python src/experiments/retrieval_evaluation.py \
    --model Qwen/Qwen2.5-32B-Instruct --dataset nq_swap --direction-dataset nq_swap \
    --layer 0 --seed 42
```

**Check:** both commands finish without error and these exist:

```bash
ls results/direction_identification/Qwen__Qwen2.5-32B-Instruct/nq_swap/seed_42/context_only/layer_0/last_pos/direction.pt
ls results/retrieval_evaluation/Qwen__Qwen2.5-32B-Instruct/nq_swap/nq_swap/unnormalized/seed_42/context_only/layer_0/last_pos/results.jsonl
```

If you hit a CUDA out-of-memory error, add `--batch-size 2` to the first command and tell us — do
not continue to the long runs.

---

## 4. Stage 1 — Direction identification (~2.5 h)

The foundation for all three figures. We identify **8 layers strided by 8** (`0, 8, 16, ..., 56`)
rather than all 64. Stage 2 then evaluates every one of those 8 on all 5 seeds, so the layer that
carries the signal is picked from real measurements in one pass. No stage ever needs the other
layers — the ClashEval scripts average over whatever layers are on disk.

**Use exactly this layer set everywhere in this document.** The three datasets must end up with the
same layers or Figure 7's three series would not be comparable.

Note the layer list here is **comma-separated**:

```bash
python src/experiments/direction_identification.py --automated \
    --model Qwen/Qwen2.5-32B-Instruct --dataset conflictqa \
    --layers 0,8,16,24,32,40,48,56

python src/experiments/direction_identification.py --automated \
    --model Qwen/Qwen2.5-32B-Instruct --dataset nq_swap \
    --layers 0,8,16,24,32,40,48,56
```

Each covers all 5 seeds. ConflictQA is the long one (~2 h); NQ-Swap ~25 min.

> `--model` is required. Without it, `--automated` would start running the other four models too.

**Check:**

```bash
find results/direction_identification/Qwen__Qwen2.5-32B-Instruct -name direction.pt | wc -l
```
Expected: **160** (2 datasets × 5 seeds × 8 layers × 2 positions). LongFact is absent on
purpose — it is only needed for Figure 7, in stage 4.

---

## 5. Stage 2 — Retrieval evaluation (~10 h)

The layer that carries the factuality signal is chosen empirically, from this stage's numbers. All
8 layers on all 5 seeds, in one command. **Note the layer syntax differs from stage 1** — commas
above, spaces here:

```bash
python src/experiments/retrieval_evaluation.py --automated \
    --model Qwen/Qwen2.5-32B-Instruct \
    --layers 0 8 16 24 32 40 48 56
```

Omitting `--seed` is deliberate: the script then runs every seed that has a direction on disk, i.e.
all 5. Those 5 seeds are what produce the error bands in Figure 3.

Then the layer selection (CPU, a few minutes — **required**, later stages read what it writes):

```bash
python src/experiments/plot_retrieval_evaluation.py
```

This scores every layer across all 5 seeds, writes `top_layers_context_only_last_pos.json`, and
copies the winning layer's results into `results/top_retrieval_evaluation/`. Nothing to wait for —
the choice is made from your own measurements.

**Check:** `results/top_retrieval_evaluation/Qwen__Qwen2.5-32B-Instruct/` exists and contains
`results.jsonl` files under both `conflictqa/` and `nq_swap/`.

The winner is picked among layers spaced 8 apart, so the true optimum may sit a few layers away.
For a 64-layer model the signal sits in a broad mid-depth band rather than one spiking layer, so
this is expected to be immaterial — if the curve turns out sharply peaked we may ask you for a
handful of neighbouring layers afterwards. Do not wait for that; it is not part of the plan.

It costs us nothing to see which layer won, so please paste us the output of this while stage 3
runs — no need to pause for our reply:

```bash
cat results/retrieval_evaluation/Qwen__Qwen2.5-32B-Instruct/conflictqa/conflictqa/unnormalized/top_layers_context_only_last_pos.json
```

*Figure 3 is now covered.*

---

## 6. Stage 3 — Cross-domain / mixed directions (~3 h)

Directions estimated on each dataset and on all 7 combinations of the three. Seed 42 only, by design.

Same 8 layers again. This track picks its own layer per combination, from its own sweep — that is
the protocol the other four models were run under, so it is not replaced by stage 2's winner.

```bash
python src/experiments/mixed_direction_identification.py \
    --model Qwen/Qwen2.5-32B-Instruct --seed 42 \
    --layers 0,8,16,24,32,40,48,56

python src/experiments/mixed_directions_retrieval_evaluation.py \
    --models Qwen/Qwen2.5-32B-Instruct

python src/experiments/mixed_directions_plot_retrieval_evaluation.py

python src/experiments/mixed_directions_end_to_end_evaluation.py \
    --models Qwen/Qwen2.5-32B-Instruct
```

Run them in exactly this order: the third command chooses the layer and α that the fourth one uses.
The fourth loads the model in vLLM and generates answers.

**Check:**
```bash
find results/mixed_directions_end_to_end_evaluation/Qwen__Qwen2.5-32B-Instruct -name results.jsonl | wc -l
```
Expected: **14** (7 dataset combinations × 2 evaluation sets).

*Figure 6 is now covered.*

---

## 7. Stage 4 — ClashEval (~4.5 h)

Figure 7 has three panels — NQ-Swap, ConflictQA and **LongFact** — and averages each direction's
projection across layers with no layer selection. The scripts average over the layers that have a
direction on disk for **every** dataset and seed, so the stage 1 set is already enough and nothing
has to be backfilled.

LongFact is the exception: it has no directions yet and is used only here, so it needs one
identification run (~30 min). Same 8 layers as stage 1 — the layer sets must match across the three
datasets, or the three series in the figure would not be comparable:

```bash
python src/experiments/direction_identification.py --automated \
    --model Qwen/Qwen2.5-32B-Instruct --dataset longfact \
    --layers 0,8,16,24,32,40,48,56
```

**Check:**
```bash
find results/direction_identification/Qwen__Qwen2.5-32B-Instruct/longfact -name direction.pt | wc -l
```
Expected: **80** (5 seeds × 8 layers × 2 positions).

---

**Before running the rest of this stage**, ask us for the two small files
`frozen_pools.json` and `frozen_pools.sha256` (together ~650 KB) and put them here:

```bash
mkdir -p results/clasheval_pool_ranking_v2
# copy the two files we send you into results/clasheval_pool_ranking_v2/
ls results/clasheval_pool_ranking_v2/
```

These freeze the 12-document pools that all four existing models were evaluated on. With them in
place your run is guaranteed comparable; without them the script would draw its own pools and the
numbers could not be put on the same figure.

Then the ClashEval pipeline itself:

```bash
python src/experiments/clasheval_pipeline.py --model Qwen/Qwen2.5-32B-Instruct

python src/experiments/clasheval_pool_ranking_v2.py --model Qwen/Qwen2.5-32B-Instruct

python src/experiments/clasheval_end_to_end_generation.py --model Qwen/Qwen2.5-32B-Instruct
```

Each of the three prints a line like

```
[layers] all-layer mean runs over 8/64 layers present for every (dataset, seed): [0, 8, 16, ...]
```

It should say **8/64** and list `0, 8, 16, ..., 56`. Fewer than 8 means a direction is missing —
stop and tell us which.

(For the four models already in the paper this mean runs over every layer they have. The 32B
averages over a uniform 1-in-8 stride instead, which is an unbiased subsample of the same depth
range — slightly noisier, and noted as such in the paper.)

`clasheval_pool_ranking_v2.py` prints two further build checks that matter:

- `[BUILD CHECK PASS] frozen_pools.json sha256 ...` — or, on a first run, `recorded reference hash`.
- a check on the α=0 accuracy, which must equal `0.213836`.

α=0 is pure retrieval similarity and carries no model term, so that number is identical for every
model. **If that check fails, stop and send us the output** — it means the document pools were
drawn differently and the result would not be comparable to the other four models.

**Check:**
```bash
ls results/clasheval_end_to_end/
```
Expected: `end_to_end__Qwen__Qwen2.5-32B-Instruct*.jsonl`.

*Figure 7 is now covered.*

---

## 8. Packaging and sending the results

The large files are cached activation tensors (`.pt`) that we do **not** need — they are
regenerable and total well over 100 GB. Send only the small result files.

```bash
tar czf qwen32b_results.tar.gz \
    --exclude='llm_hidden_states.pt' \
    --exclude='sbert_embeddings.pt' \
    results/direction_identification/Qwen__Qwen2.5-32B-Instruct \
    results/top_retrieval_evaluation/Qwen__Qwen2.5-32B-Instruct \
    results/mixed_directions_retrieval_evaluation/Qwen__Qwen2.5-32B-Instruct \
    results/mixed_directions_end_to_end_evaluation/Qwen__Qwen2.5-32B-Instruct \
    results/clasheval_end_to_end \
    results/clasheval_pool_ranking_v2 \
    $(find results/retrieval_evaluation/Qwen__Qwen2.5-32B-Instruct -name 'top_layers_*.json')

ls -lh qwen32b_results.tar.gz
```

The two excluded names are the big activation caches (tens of GB). Everything else is small:
the `direction.pt` files are only ~46 MB in total and we do want those.

Expect roughly **200–400 MB** compressed. This is too large for a git push, so please send it via
file transfer instead — Google Drive, WeTransfer, or `rsync` if we can reach your machine.

Please also include the run logs, which are small and record every configuration used:

```bash
tar czf qwen32b_logs.tar.gz $(find results -name '*.log' -newermt '-30 days')
```

Do **not** delete `results/` on your side until we confirm the transfer is complete and readable —
if anything is missing, re-deriving it from the cached tensors is far cheaper than a full re-run.

---

## 9. If something goes wrong

| Symptom | What to do |
| --- | --- |
| CUDA out of memory in stage 1, 2 or 3 | Add `--batch-size 2` to the command and re-run it; it resumes. |
| CUDA out of memory in a vLLM stage (last command of stage 3, last of stage 4) | Make sure no other process holds GPU memory (`nvidia-smi`), then re-run. Tell us if it persists — it needs a code change we would make on our side. |
| `No direction seeds found ... skipping` | Expected for `longfact` in stage 2 — its directions are only estimated later, in stage 4. Harmless. |
| `Layer N: no direction for combos [...], skipping those` | Expected in stage 3, roughly 56 times per dataset. Stage 1 computes 8 layers, and this script checks all 64; it skips the others before doing any GPU work. Harmless. |
| A run died partway | Re-run the identical command. Everything is resumable. |
| The α=0 ClashEval check fails | Stop and send us the output. Do not continue. |

Every script also writes a log under `results/<stage>/<name>.log` — please send those along if you
report a problem.
