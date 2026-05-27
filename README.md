# steering_rag

**Factuality-direction steering for retrieval-augmented generation.**

We extract a *factuality direction* from the residual stream of a frozen LLM (diff-in-means between factual and non-factual context activations) and use it to re-rank retrieved documents at inference time — no fine-tuning, no architectural changes.

---

## Repository layout

```
steering_rag/
│
├── data/
│   ├── conflictQA-popQA-gpt4_is_memory_correct_non_ambiguous.csv   ← in git
│   └── normalized_dataset/                                          ← in git
│       ├── nq_swap/
│       │   └── seed_{7,42,67,89,90}/
│       │       ├── train.jsonl
│       │       └── test.jsonl
│       └── conflictqa/
│           └── seed_{7,42,67,89,90}/
│               ├── train.jsonl
│               └── test.jsonl
│
├── src/
│   ├── utils.py                           ← shared schema, path constants, helpers
│   └── experiments/
│       │
│       ├── ── ACTIVE PIPELINE ──────────────────────────────────────────────────
│       ├── dataset_normalization.py        [step 1a]
│       ├── add_conflictqa_ground_truth.py  [step 1b]
│       ├── preliminary_analysis.py         [step 2a]  exploratory
│       ├── plot_preliminary_analysis.py    [step 2b]  exploratory
│       ├── direction_identification.py     [step 3]
│       ├── retrieval_evaluation.py         [step 4]
│       ├── plot_retrieval_evaluation.py    [step 5]   also selects top layer + create top_retrieval_evaluation/
│       ├── end_to_end_evaluation.py        [step 6]
│       ├── plot_end_to_end_evaluation.py   [step 7]
│       ├── plot_figure3.py                 [step 8]   paper figure 3
│       │
│       └── ── LEGACY (not part of main pipeline) ───────────────────────────────
│           ├── direction_identification_sae.py
│           ├── steering.py
│           ├── steering_sae.py
│           └── evaluation_steering.py
│
└── results/
    ├── direction_identification/           ← in git
    ├── retrieval_evaluation/               ← NOT in git (100+ GB, *.pt excluded)
    ├── top_retrieval_evaluation/           ← in git  (best layer per group only)
    ├── end_to_end_evaluation/              ← in git
    ├── preliminary_analysis/               ← in git
    └── figures/                            ← in git  (paper PDFs)
```

---

## Environment

```bash
conda activate bias_rag   # all dependencies + HF_TOKEN already set
```

Models are downloaded from HuggingFace on first use. A valid `HF_TOKEN` is required for gated models (Llama).

**Models used throughout:**


| Short name   | HuggingFace ID                     |
| ------------ | ---------------------------------- |
| Llama-3.2-1B | `meta-llama/Llama-3.2-1B-Instruct` |
| Gemma-3-4B   | `google/gemma-3-4b-it`             |
| Qwen2-7B     | `Qwen/Qwen2-7B-Instruct`           |
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B-Instruct` |


---

## Normalized sample schema

Every experiment consumes `.jsonl` files where each line is:

```json
{
  "question":             "...",
  "factual_context":      "...",   // passage that supports the correct answer
  "factual_answer":       ["..."], // accepted correct answer strings
  "non_factual_evidence": "...",   // passage that supports a wrong answer
  "non_factual_answer":   ["..."], // wrong answer strings
  "ground_truth":         ["..."], // ConflictQA only – official answer aliases
  "original_dataset_id":  "nq_swap" | "conflictqa"
}
```

---

## Full pipeline — reproduction guide

Run the steps below in order. Each step is idempotent (skips already-computed outputs).

---

### Step 1 — Dataset normalization

**1a. Normalize NQ-Swap and ConflictQA into the common schema and create 5 seeded splits.**

- *Reads:* NQ-Swap from HuggingFace (`younanna/NQ-Swap`, split `dev`); ConflictQA from `data/conflictQA-popQA-gpt4_is_memory_correct_non_ambiguous.csv` (both in git)
- *Writes:* `data/normalized_dataset/{nq_swap,conflictqa}/seed_{7,42,67,89,90}/{train,test}.jsonl`

```bash
python -m src.experiments.dataset_normalization
```

**1b. Attach ConflictQA ground-truth keyword lists** (required for the end-to-end evaluation accuracy metric).

- *Reads:* `data/conflictQA-popQA-gpt4_is_memory_correct_non_ambiguous.csv`; existing `data/normalized_dataset/conflictqa/**/*.jsonl`
- *Writes:* same `.jsonl` files in-place, adding a `"ground_truth"` field to every row

```bash
python -m src.experiments.add_conflictqa_ground_truth
```

---

### Step 2 — Preliminary analysis *(exploratory, for Figure 1)*

Tests each model's sensitivity to context position across 5 prompt conditions (no context / factual only / non-factual only / both factual-first / both NF-first). Uses the `train` split, subsampled to 500 questions per dataset.

**2a. Generate answers for all models × datasets × conditions.**

- *Reads:* `data/normalized_dataset/{nq_swap,conflictqa}/seed_42/train.jsonl`
- *Uses:* vLLM for batched generation
- *Writes:* `results/preliminary_analysis/<model>/<dataset>/results.jsonl`

```bash
python -m src.experiments.preliminary_analysis
```

**2b. Plot grouped-bar accuracy figures.**

- *Reads:* `results/preliminary_analysis/<model>/<dataset>/results.jsonl`
- *Writes:* `results/preliminary_analysis/<dataset>/accuracy_all.pdf`, `accuracy_no_context_correct.pdf`, `accuracy_grid.pdf`, `figure_2_motivation.pdf`

```bash
python -m src.experiments.plot_preliminary_analysis
```

---

### Step 3 — Direction identification

Extracts a per-layer *factuality direction* via diff-in-means in the residual stream. Runs the `context_only` procedure: positive activations come from `factual_context`, negative from `non_factual_evidence`, at the last token position (`last_pos`); NQ-Swap also extracts at the answer entity position (`entity_pos`). Uses the `train` split of each seed.

- *Reads:* `data/normalized_dataset/<dataset>/seed_<N>/train.jsonl`; the LLM (via TransformerLens)
- *Writes:* `results/direction_identification/<model>/<dataset>/seed_<N>/context_only/layer_<L>/last_pos/{direction.pt,meta.json}`

Run for **each model × direction-dataset** combination. Omitting `--layers` runs all layers of the model automatically:

```bash
MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.2-1B-Instruct"
  "google/gemma-3-4b-it"
  "Qwen/Qwen2-7B-Instruct"
)
DATASETS=("nq_swap" "conflictqa")

for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    python -m src.experiments.direction_identification \
        --model "$MODEL" \
        --dataset "$DATASET"
        # seeds are auto-discovered from data/normalized_dataset/
        # layers default to all layers of the model
  done
done
```

---

### Step 4 — Retrieval evaluation

Scores each document in the test corpus using a fused ranking:

```
score(d, q) = (1 − α) · zscore(SBERT_cos(d, q)) + α · zscore(llm_hidden(d) · direction)
```

Sweeps `α ∈ {0.0, 0.3, 0.5, 1.0}` and `k ∈ {2, 5, 10}`. Caches per-document LLM hidden states (`llm_hidden_states.pt`) and SBERT embeddings (`sbert_embeddings.pt`) on disk for reuse by the end-to-end evaluation.

- *Reads:* `data/normalized_dataset/<eval_dataset>/seed_<N>/test.jsonl`; `results/direction_identification/...`
- *Writes:* `results/retrieval_evaluation/<model>/<eval>/<direction>/<normalize>/seed_<N>/context_only/layer_<L>/{results.jsonl, sbert_embeddings.pt, llm_hidden_states.pt}`

**Before running, set `MODELS` at the top of `retrieval_evaluation.py` to all four models** (it defaults to a single model from the last run). Then use `--automated` to run all (model, eval-dataset, direction-dataset) combinations:

```bash
python -m src.experiments.retrieval_evaluation --automated
```

---

### Step 5 — Score layers, plot, and extract the top layer

For each (model, eval-dataset, direction-dataset) group, scores every layer by `gold_recall_lift + nonfactual_rate_drop` (improvement over the α=0 baseline, averaged across seeds and k values), selects the best α per layer, writes `top_layers_context_only.json`, generates per-file and aggregated plots, and **copies the single best-scoring layer's seed directories to `results/top_retrieval_evaluation/`** (the compact, git-trackable version).

- *Reads:* all `results/retrieval_evaluation/***/results.jsonl`
- *Writes:*
  - `results/retrieval_evaluation/***/retrieval_plot.png` (per-file)
  - `results/retrieval_evaluation/***/aggregated_plots/*.png` (mean ± std across seeds)
  - `results/retrieval_evaluation/.../<normalize>/top_layers_context_only.json`
  - `results/top_retrieval_evaluation/...` (mirrored best-layer dirs)

```bash
python -m src.experiments.plot_retrieval_evaluation
```

---

### Step 6 — End-to-end evaluation

For each model, loads the best layer from `top_retrieval_evaluation`, re-ranks the ConflictQA test corpus using the fused score, feeds the top-k documents to the model via vLLM, and checks whether the generated answer contains a ground-truth alias.

- *Reads:* `results/top_retrieval_evaluation/.../{sbert_embeddings.pt,llm_hidden_states.pt,top_layers_context_only.json}`; `results/direction_identification/.../direction.pt`; `data/normalized_dataset/conflictqa/seed_<N>/test.jsonl`
- *Writes:* `results/end_to_end_evaluation/<model>/conflictqa/<direction>/unnormalized/seed_<N>/context_only/layer_<L>/results.jsonl`

```bash
python -m src.experiments.end_to_end_evaluation
```

---

### Step 7 — Plot end-to-end results

Generates accuracy-vs-k line plots (baseline vs steered) with mean ± std across seeds.

- *Reads:* `results/end_to_end_evaluation/***/results.jsonl`
- *Writes:* `results/end_to_end_evaluation/***/end_to_end_plot.png`

```bash
python -m src.experiments.plot_end_to_end_evaluation
```

---

### Step 8 — Paper Figure 3 (re-ranking figure)

Paper-ready 2×2 PDF: mean document rank vs α (top row) and rank-delta scatter / rank-separation vs model size (bottom row). Reads directly from `top_retrieval_evaluation`.

- *Reads:* `results/top_retrieval_evaluation/***/results.jsonl`
- *Writes:* `results/figures/figure_3_reranking.pdf`

```bash
python -m src.experiments.plot_figure3
```

---

## Script reference


| Script                           | Step | Key inputs                                                   | Key outputs                                                     |
| -------------------------------- | ---- | ------------------------------------------------------------ | --------------------------------------------------------------- |
| `dataset_normalization.py`       | 1a   | NQ-Swap (HF), ConflictQA CSV                                 | `data/normalized_dataset/`                                      |
| `add_conflictqa_ground_truth.py` | 1b   | ConflictQA CSV, normalized splits                            | patches `conflictqa` splits in-place                            |
| `preliminary_analysis.py`        | 2a   | normalized train splits (seed 42)                            | `results/preliminary_analysis/<model>/<dataset>/results.jsonl`  |
| `plot_preliminary_analysis.py`   | 2b   | preliminary analysis jsonl                                   | PDF plots in `results/preliminary_analysis/`                    |
| `direction_identification.py`    | 3    | normalized train splits, LLM                                 | `results/direction_identification/…/direction.pt`               |
| `retrieval_evaluation.py`        | 4    | normalized test splits, directions, LLM, SBERT               | `results/retrieval_evaluation/…/{results.jsonl,*.pt}`           |
| `plot_retrieval_evaluation.py`   | 5    | retrieval results                                            | plots, `top_layers_*.json`, `results/top_retrieval_evaluation/` |
| `end_to_end_evaluation.py`       | 6    | top_retrieval_evaluation caches, directions, conflictqa test | `results/end_to_end_evaluation/…/results.jsonl`                 |
| `plot_end_to_end_evaluation.py`  | 7    | end-to-end results                                           | `results/end_to_end_evaluation/…/end_to_end_plot.png`           |
| `plot_figure3.py`                | 8    | top_retrieval_evaluation results                             | `results/figures/figure_3_reranking.pdf`                        |


---

## Results directory structure

```
results/
│
├── direction_identification/
│   └── <model>/
│       └── <direction_dataset>/
│           └── seed_<N>/
│               └── context_only/
│                   └── layer_<L>/
│                       └── last_pos/
│                           ├── direction.pt    ← diff-in-means direction [d_model]
│                           └── meta.json       ← n_samples, norm, config
│
├── retrieval_evaluation/          ← NOT in git (*.pt excluded)
│   └── <model>/<eval>/<direction>/unnormalized/
│       ├── top_layers_context_only.json        ← layer ranking + best α per layer
│       └── seed_<N>/context_only/layer_<L>/
│           ├── results.jsonl                   ← per-(sample,α,k) gold/nf recall
│           ├── sbert_embeddings.pt             ← cached doc SBERT embeddings
│           └── llm_hidden_states.pt            ← cached doc LLM hidden states
│
├── top_retrieval_evaluation/      ← in git (best layer only, same file structure)
│   └── <model>/<eval>/<direction>/unnormalized/
│       ├── top_layers_context_only.json
│       └── seed_<N>/context_only/layer_<L>/
│           ├── results.jsonl
│           ├── sbert_embeddings.pt
│           └── llm_hidden_states.pt
│
├── end_to_end_evaluation/
│   └── <model>/conflictqa/<direction>/unnormalized/seed_<N>/context_only/layer_<L>/
│       └── results.jsonl          ← per-(sample,α,k): prompt, topk docs, answer, is_correct
│
├── preliminary_analysis/
│   ├── <model>/<dataset>/results.jsonl   ← per-(sample,condition): generated answer
│   └── <dataset>/
│       ├── accuracy_all.pdf
│       ├── accuracy_no_context_correct.pdf
│       ├── accuracy_grid.pdf
│       └── figure_2_motivation.pdf
│
└── figures/
    └── figure_3_reranking.pdf
```

---

## Legacy scripts

These scripts were used in earlier experiments and are **not part of the main pipeline**. They are kept for reference.


| Script                            | Description                                                                                                                                                                                                                                    |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `direction_identification_sae.py` | Identifies directions using sparse-autoencoder feature activations (SAE diff-in-means) rather than raw residual diff-in-means. Only works for Llama (public SAEs available via `sae_lens`). Outputs `sae_context_only/` procedure directories. |
| `steering.py`                     | Applies a steering vector (`+α·direction`) at every token position during generation using TransformerLens hooks. Outputs `results/steering/<model>/<eval>/<direction>/<procedure>/layer_<L>/<pos>/runs.jsonl`.                                |
| `steering_sae.py`                 | Same as `steering.py` but restricted to SAE-derived directions. Supports a `--last-token-only` flag to steer only the last position.                                                                                                           |
| `evaluation_steering.py`          | Evaluates `runs.jsonl` files produced by `steering.py`/`steering_sae.py`; computes and plots accuracy split by document order (factual-first vs non-factual-first).                                                                            |


