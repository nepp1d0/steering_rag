# steering_rag

**Factuality-direction re-ranking for retrieval-augmented generation.**

We extract a *factuality direction* from the residual stream of a frozen LLM (diff-in-means between factual and non-factual context activations), turn it into a per-document factuality score, and fuse that score with retrieval similarity to re-rank retrieved documents at inference time — no fine-tuning, no architectural changes.

This README documents the paper pipeline: the scripts in `src/experiments/` and the data they consume.

---

## Repository layout

```
steering_rag/
│
├── data/
│   ├── normalized_dataset/                                          ← in git
│   │   ├── nq_swap/     seed_{7,42,67,89,90}/{train,test}.jsonl
│   │   ├── conflictqa/  seed_{7,42,67,89,90}/{train,test}.jsonl
│   │   └── longfact/    seed_{7,42,67,89,90}/{train,test}.jsonl
│   ├── conflictQA-popQA-gpt4_is_memory_correct_non_ambiguous.csv    ← in git
│   └── clasheval_gpt4.pqt                                           ← NOT in git (28 MB)
│
├── src/
│   ├── experiments/     ← the paper pipeline (this README)
│   └── exploratory/     ← work not used in the paper; see its own README
│
└── results/             ← created on first run, NOT in git
```

`data/` also holds inputs used only by `src/exploratory/` (RAGuard, HotpotQA, AVeriTeC and other ConflictQA CSV variants); they are not needed for anything below.

### Paths

`src/experiments/utils.py` derives every path from the repo root, taken as two levels above itself. Nothing outside `data/`, `src/experiments/` and `results/` is required, so those two directories are enough to port the pipeline to another repository. Two overrides:

| Variable | Effect |
| --- | --- |
| `STEERING_RAG_ROOT` | repo root, if the code sits at a different depth |
| `RESULTS_DIR` | write results to another tree |

---

## Environment

```bash
conda activate bias_rag   # all dependencies + HF_TOKEN already set
```

Run everything **from the repo root**:

```bash
python src/experiments/<script>.py          # `python -m src.experiments.<script>` also works
```

Models are downloaded from HuggingFace on first use; a valid `HF_TOKEN` is required for the gated Llama models. Direction identification, retrieval evaluation, generation and the ClashEval stages need a GPU; every `plot_*` script is CPU-only.

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
  "ground_truth":         ["..."], // official answer aliases (QA datasets only)
  "original_dataset_id":  "nq_swap" | "conflictqa" | "longfact"
}
```

LongFact carries no question or short answer: it is single-sided (one supported or unsupported sentence per sample) and is therefore used only as a source of directions, never as an evaluation target.

---

## Pipeline — reproduction guide

Steps 1–5 are shared. After that each figure has its own block; Figures 3, 4 and 5 all reuse the layer selection from step 5. Every step is idempotent (skips already-computed outputs).

---

### Step 1 — Dataset construction *(Table 1)*

**1a. Normalize NQ-Swap, ConflictQA and LongFact into the common schema and create 5 seeded 80/20 splits**, grouped by question so that all evidence for a question stays on one side.

- *Reads:* NQ-Swap and LongFact from HuggingFace; ConflictQA from `data/conflictQA-popQA-gpt4_is_memory_correct_non_ambiguous.csv`
- *Writes:* `data/normalized_dataset/{nq_swap,conflictqa,longfact}/seed_{7,42,67,89,90}/{train,test}.jsonl`

```bash
python src/experiments/dataset_normalization.py
```

**1b. Attach ConflictQA ground-truth keyword lists** (required for the end-to-end accuracy metric).

- *Writes:* the same `conflictqa` `.jsonl` files in place, adding a `ground_truth` field

```bash
python src/experiments/add_conflictqa_ground_truth.py
```

**1c. Attach LongFact verified-entity spans** (required to read activations at `entity_pos`).

- *Writes:* the same `longfact` `.jsonl` files in place

```bash
python src/experiments/add_longfact_entities.py
```

---

### Step 2 — Preliminary analysis *(Figure 2, Appendix A)*

Measures each model's behaviour under conflicting evidence across 5 conditions (no context / factual only / non-factual only / both, in either order), on the `train` split subsampled to 500 questions per dataset.

- *Reads:* `data/normalized_dataset/{nq_swap,conflictqa}/seed_42/train.jsonl`
- *Writes:* `results/preliminary_analysis/<model>/<dataset>/results.jsonl`, then the PDFs

```bash
python src/experiments/preliminary_analysis.py        # GPU (vLLM)
python src/experiments/plot_preliminary_analysis.py
```

Paper figures, all under `results/preliminary_analysis/`: `main_figure_2.pdf` (Figure 2, all questions), `appendix_1_figure_1.pdf` (no-context-correct subset), `appendix_1_figure_2.pdf` (positional bias). The other PDFs it writes are earlier layouts.

---

### Step 3 — Direction identification

Per-layer diff-in-means direction in the residual stream: positives from `factual_context`, negatives from `non_factual_evidence`, read at the last token (`last_pos`) and at the answer-entity token (`entity_pos`). Uses the `train` split of each seed.

- *Reads:* `data/normalized_dataset/<dataset>/seed_<N>/train.jsonl`; the LLM via TransformerLens
- *Writes:* `results/direction_identification/<model>/<dataset>/seed_<N>/context_only/layer_<L>/<position>/{direction.pt,meta.json}`

```bash
python src/experiments/direction_identification.py --automated
```

`--automated` covers every model × dataset × seed × layer. To run one cell: `--model <hf-id> --dataset <name> --layers 10,15,20`.

---

### Step 4 — Retrieval evaluation

Builds a synthetic knowledge base per query and re-ranks it with the fused score:

```
score(d, q) = (1 − α) · zscore(SBERT_cos(d, q)) + α · zscore(llm_hidden(d) · direction)
```

Sweeps `α ∈ {0.0, 0.3, 0.5, 1.0}` and `k ∈ {2, 5, 10}`. Caches per-document LLM hidden states and SBERT embeddings for reuse by the end-to-end evaluation.

- *Reads:* `data/normalized_dataset/<eval>/seed_<N>/test.jsonl`; `results/direction_identification/…`
- *Writes:* `results/retrieval_evaluation/<model>/<eval>/<direction>/<normalize>/seed_<N>/context_only/layer_<L>/<position>/results.jsonl`, plus `llm_hidden_states.pt` / `sbert_embeddings.pt` / `docs.jsonl` at the layer level

```bash
python src/experiments/retrieval_evaluation.py --automated
```

---

### Step 5 — Layer selection

Scores every layer by `gold_recall_lift + nonfactual_rate_drop` over the α=0 baseline, picks the best α per layer, and copies the winning layer's seed directories into a compact tree. **Required before steps 6–9** — they all read the selection it writes.

- *Reads:* every `results/retrieval_evaluation/**/results.jsonl`
- *Writes:* per-cell `retrieval_plot.png`, `top_layers_<procedure>_<position>.json`, and `results/top_retrieval_evaluation/…`

```bash
python src/experiments/plot_retrieval_evaluation.py
```

---

### Step 6 — Figure 3 *(re-ranking)*

2×2 PDF; the paper uses its **top row** only (mean gold / non-factual rank vs α), from two runs stacked.

Set at the top of `plot_figure3.py`: `DIRECTION_DATASET = "same"` (in-domain, as reported), `POSITION = "last_pos"`, and `TOP_ROW_DATASET` to `"conflictqa"`, then `"nq_swap"`.

- *Writes:* `results/figures/<DIRECTION_DATASET>/figure_3_reranking.pdf`

```bash
python src/experiments/plot_figure3.py
```

---

### Step 7 — End-to-end evaluation *(Figure 4)*

Re-ranks the ConflictQA test corpus, feeds the top-k to the generator via vLLM, and checks whether the answer contains a ground-truth alias. Runs the selected layer at its best α plus the α=0 baseline.

- *Reads:* `results/top_retrieval_evaluation/…` (including the cached tensors), `results/direction_identification/…/direction.pt`, `data/normalized_dataset/conflictqa/seed_<N>/test.jsonl`
- *Writes:* `results/end_to_end_evaluation/…/results.jsonl`, then `results/end_to_end_evaluation/figures/figure_4_end_to_end.pdf`

```bash
python src/experiments/end_to_end_evaluation.py
python src/experiments/plot_figure_4.py
```

---

### Step 8 — LLM-as-judge baseline *(Figure 5, Appendix D)*

Replaces the projection with a verbalized factuality rating in [0, 1]: each model is prompted to rate every document, and the parsed score is fused with SBERT similarity exactly as in step 4, over the same α and k grids and with the same per-record schema.

- *Reads:* `data/normalized_dataset/<eval>/seed_<N>/test.jsonl`; the LLM via vLLM
- *Writes:* `results/llms_scoring_evaluation/<model>/<eval>/seed_<N>/{results.jsonl,docs.jsonl,llm_scores.pt,llm_raw_outputs.jsonl,sbert_embeddings.pt}`, then `results/figures/[<DIRECTION_DATASET>/]figure_3b_judge_comparison.pdf`

```bash
python src/experiments/llms_scoring_evaluation.py --automated
python src/experiments/plot_figure_3b.py
```

Set `DIRECTION_DATASET = "same"` and `SCATTER_ALPHA = 0.5` to match the paper. The output is a 2×2 panel: panel B (top-right) is Figure 5, panel A (top-left) is the Appendix D figure.

---

### Step 9 — Cross-domain generalization *(Figure 6)*

Directions estimated on each dataset and on all 7 combinations of the three, evaluated on both QA test sets. Mixed directions exist for **seed 42 only**, so each cell is a point estimate with no error bars.

- *Writes:* `results/mixed_directions/…`, `results/mixed_directions_retrieval_evaluation/…`, `results/mixed_directions_end_to_end_evaluation/…`, and the figures in `results/figures/mixed/`

```bash
python src/experiments/mixed_direction_identification.py --automated    # GPU
python src/experiments/mixed_directions_retrieval_evaluation.py         # GPU
python src/experiments/mixed_directions_plot_retrieval_evaluation.py    # layer/alpha selection
python src/experiments/mixed_directions_end_to_end_evaluation.py        # GPU
python src/experiments/mixed_directions_plot_combos.py                  # figure_combos.pdf
python src/experiments/mixed_directions_plot_end_to_end_combos.py       # figure_combos_end_to_end_k2.pdf
```

The paper figure stacks the **bottom row** of `figure_combos.pdf` on top of `figure_combos_end_to_end_k2.pdf`.

The rank-separation gains quoted in the text come from:

```bash
python src/experiments/recap_rank_separation.py    # → results/figures/recap/rank_separation.{json,csv,md}
```

---

### Step 10 — ClashEval generalization *(Figure 7)*

Out-of-distribution test on the drugs+news subset of ClashEval (477 questions): each question gets a frozen 12-document pool, re-ranked with the same fused score, and the top-k are handed to the generator and scored against a numeric ground truth. Needs the directions from step 3.

- *Reads:* `data/clasheval_gpt4.pqt`; `results/direction_identification/…`
- *Writes:* `results/clasheval_hidden/`, `results/clasheval_pool_ranking_v2/`, `results/clasheval_end_to_end/`, then `results/figures/clasheval/`

```bash
python src/experiments/clasheval_pipeline.py                                # GPU
python src/experiments/clasheval_pool_ranking_v2.py --model <hf-id>         # GPU, per model
python src/experiments/clasheval_end_to_end_generation.py --model <hf-id>   # GPU, per model
python src/experiments/plot_figure_7_end_to_end_accuracy_bars.py
```

`figure_7_end_to_end_accuracy_alpha03.pdf` is Figure 7; the full α sweep and a line variant are written next to it. `clasheval_pool_ranking.py` is imported by the v2 script for its pool helpers; running it directly reproduces the earlier pool design, which the paper does not use. Each figure gets a `.md` sidecar recording its settings.

---

## Script reference

| Script | Step | Key inputs | Key outputs |
| --- | --- | --- | --- |
| `dataset_normalization.py` | 1a | NQ-Swap + LongFact (HF), ConflictQA CSV | `data/normalized_dataset/` |
| `add_conflictqa_ground_truth.py` | 1b | ConflictQA CSV, normalized splits | patches `conflictqa` splits in place |
| `add_longfact_entities.py` | 1c | LongFact (HF), normalized splits | patches `longfact` splits in place |
| `preliminary_analysis.py` | 2 | normalized train splits (seed 42) | `results/preliminary_analysis/…/results.jsonl` |
| `plot_preliminary_analysis.py` | 2 | preliminary-analysis jsonl | Figure 2, Appendix A figures |
| `direction_identification.py` | 3 | normalized train splits, LLM | `results/direction_identification/…/direction.pt` |
| `retrieval_evaluation.py` | 4 | normalized test splits, directions, SBERT | `results/retrieval_evaluation/…/{results.jsonl,*.pt}` |
| `plot_retrieval_evaluation.py` | 5 | retrieval results | plots, `top_layers_*.json`, `results/top_retrieval_evaluation/` |
| `plot_figure3.py` | 6 | `top_retrieval_evaluation` results | `results/figures/…/figure_3_reranking.pdf` |
| `end_to_end_evaluation.py` | 7 | top-layer caches, directions, conflictqa test | `results/end_to_end_evaluation/…/results.jsonl` |
| `plot_figure_4.py` | 7 | end-to-end results | `results/end_to_end_evaluation/figures/figure_4_end_to_end.pdf` |
| `llms_scoring_evaluation.py` | 8 | normalized test splits, LLM (vLLM), SBERT | `results/llms_scoring_evaluation/…` |
| `plot_figure_3b.py` | 8 | judge scores + `top_retrieval_evaluation` | `results/figures/…/figure_3b_judge_comparison.pdf` |
| `mixed_direction_identification.py` | 9 | normalized train splits, LLM | `results/mixed_directions/…/direction.pt` |
| `mixed_directions_retrieval_evaluation.py` | 9 | mixed directions, test splits | `results/mixed_directions_retrieval_evaluation/…` |
| `mixed_directions_plot_retrieval_evaluation.py` | 9 | mixed retrieval results | plots + `top_layers_<procedure>.json` |
| `mixed_directions_end_to_end_evaluation.py` | 9 | mixed selection + cached tensors | `results/mixed_directions_end_to_end_evaluation/…` |
| `mixed_directions_plot_combos.py` | 9 | mixed retrieval results | `results/figures/mixed/figure_combos.pdf` |
| `mixed_directions_plot_end_to_end_combos.py` | 9 | mixed end-to-end results | `results/figures/mixed/figure_combos_end_to_end_k2.pdf` |
| `recap_rank_separation.py` | 9 | `top_retrieval_evaluation` results | `results/figures/recap/rank_separation.{json,csv,md}` |
| `clasheval_pipeline.py` | 10 | `data/clasheval_gpt4.pqt`, directions | `results/clasheval_hidden/` |
| `clasheval_pool_ranking.py` | 10 | pool helpers for the v2 script | `results/clasheval_pool_ranking/` if run directly |
| `clasheval_pool_ranking_v2.py` | 10 | ClashEval activations | `results/clasheval_pool_ranking_v2/` |
| `clasheval_end_to_end_generation.py` | 10 | frozen pools, LLM (vLLM) | `results/clasheval_end_to_end/` |
| `plot_figure_7_end_to_end_accuracy.py` | 10 | ClashEval generations | line variant of Figure 7 |
| `plot_figure_7_end_to_end_accuracy_bars.py` | 10 | ClashEval generations | `results/figures/clasheval/figure_7_end_to_end_accuracy_alpha03.pdf` |
| `utils.py` | — | — | shared schema, path constants, helpers |

Keep these files in one directory: several of them import siblings by module name.

---

## Results directory structure

```
results/                                   ← NOT in git
│
├── direction_identification/
│   └── <model>/<dataset>/seed_<N>/context_only/layer_<L>/<position>/
│       ├── direction.pt                   ← diff-in-means direction [d_model]
│       └── meta.json                      ← n_samples, norm, config
│
├── retrieval_evaluation/
│   └── <model>/<eval>/<direction>/<normalize>/
│       ├── top_layers_<procedure>_<position>.json   ← layer ranking + best α per layer
│       └── seed_<N>/context_only/layer_<L>/
│           ├── <position>/results.jsonl   ← per-(sample,α,k) gold/non-factual ranks
│           ├── sbert_embeddings.pt        ← cached doc SBERT embeddings
│           ├── llm_hidden_states.pt       ← cached doc LLM hidden states
│           └── docs.jsonl                 ← doc_idx → text
│
├── top_retrieval_evaluation/              ← best layer only, same structure
│
├── end_to_end_evaluation/
│   ├── <model>/<eval>/<direction>/…/results.jsonl  ← prompt, top-k docs, answer, is_correct
│   └── figures/figure_4_end_to_end.pdf
│
├── llms_scoring_evaluation/
│   └── <model>/<eval>/seed_<N>/           ← judge scores, same record schema as step 4
│
├── mixed_directions/                      ← directions from dataset combos (seed 42)
├── mixed_directions_retrieval_evaluation/
├── mixed_directions_end_to_end_evaluation/
│
├── clasheval_hidden/                      ← per-layer document activations + null controls
├── clasheval_pool_ranking_v2/             ← frozen pools + pool-ranking metrics
├── clasheval_end_to_end/                  ← generations + numeric scoring
│
├── preliminary_analysis/                  ← results.jsonl + Figure 2 / Appendix A PDFs
│
└── figures/
    ├── <direction_dataset>/figure_3_reranking.pdf, figure_3b_judge_comparison.pdf
    ├── mixed/figure_combos.pdf, figure_combos_end_to_end_k2.pdf
    ├── clasheval/figure_7_end_to_end_accuracy*.pdf
    └── recap/rank_separation.{json,csv,md}
```

---

## Figure map

| Paper element | Step | Producing script |
| --- | --- | --- |
| Figure 2 | 2 | `plot_preliminary_analysis.py` |
| Figure 3 | 6 | `plot_figure3.py` (top row, two runs stacked) |
| Figure 4 | 7 | `plot_figure_4.py` |
| Figure 5 | 8 | `plot_figure_3b.py` (panel B) |
| Figure 6 | 9 | `mixed_directions_plot_combos.py` + `mixed_directions_plot_end_to_end_combos.py` |
| Figure 7 | 10 | `plot_figure_7_end_to_end_accuracy_bars.py` |
| Appendix A | 2 | `plot_preliminary_analysis.py` |
| Appendix D | 8 | `plot_figure_3b.py` (panel A) |
| Table 1 | 1 | `dataset_normalization.py` |

Figure 1 is a hand-drawn diagram with no producing script.
