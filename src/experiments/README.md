# Reproducing the paper's figures

All scripts for the figures and tables of the paper. Exploratory work is in
[`../exploratory/`](../exploratory/README.md). Figure 1 is a hand-drawn diagram, no script.

## Setup

```bash
conda activate bias_rag        # dependencies + HF_TOKEN
cd <repo root>                 # run everything from here
python src/experiments/<script>.py
```

Layout assumed by the code:

```
<repo root>/
├── data/         normalized datasets + raw inputs        (required)
├── src/experiments/                                      (this folder)
└── results/      created on the first run
```

`utils.py` derives every path from the repo root, taken as two levels above this folder.
Override with `STEERING_RAG_ROOT=/path/to/root` if the code sits elsewhere, or `RESULTS_DIR=...`
to write results to another tree. Nothing else in the repo is needed.

GPU is required for direction identification, retrieval evaluation, generation and the ClashEval
stages. Every `plot_*` script is CPU-only.

## Order

Stage A first, then each figure's block. Figures 3–5 all reuse stage C's layer selection.

### A. Datasets (Table 1)

```bash
python src/experiments/dataset_normalization.py         # -> data/normalized_dataset/
python src/experiments/add_conflictqa_ground_truth.py
python src/experiments/add_longfact_entities.py
```

### B. Figure 2 + Appendix A (Figures 8, 9)

```bash
python src/experiments/preliminary_analysis.py          # GPU
python src/experiments/plot_preliminary_analysis.py
```

Writes to `results/preliminary_analysis/`: `main_figure_2.pdf` (Fig. 2),
`appendix_1_figure_1.pdf` (Fig. 8), `appendix_1_figure_2.pdf` (Fig. 9). Used as generated.

### C. Figure 3

```bash
python src/experiments/direction_identification.py --automated   # GPU
python src/experiments/retrieval_evaluation.py --automated       # GPU
python src/experiments/plot_retrieval_evaluation.py              # layer selection, required
python src/experiments/plot_figure3.py
```

Set in `plot_figure3.py` before running: `DIRECTION_DATASET = "same"` (in-domain, as reported),
`POSITION = "last_pos"`, and `TOP_ROW_DATASET` to `"conflictqa"`, then `"nq_swap"`. The paper
figure is the top row of those two runs, cropped and stacked.

> The committed values are `DIRECTION_DATASET = "nq_swap"` / `TOP_ROW_DATASET = "nq_swap"`, left
> over from a cross-domain check.

Output: `results/figures/<DIRECTION_DATASET>/figure_3_reranking.pdf`.

### D. Figure 4

```bash
python src/experiments/end_to_end_evaluation.py         # GPU, needs stage C
python src/experiments/plot_figure_4.py
```

Output: `results/end_to_end_evaluation/figures/figure_4_end_to_end.pdf`. Used as generated.

### E. Figure 5 + Appendix D (Figure 10)

```bash
python src/experiments/llms_scoring_evaluation.py --automated    # GPU, needs stage C
python src/experiments/plot_figure_3b.py
```

Set `DIRECTION_DATASET = "same"` and `SCATTER_ALPHA = 0.5` (committed: `"longfact"` and `0.3`).

Output: `results/figures/figure_3b_judge_comparison.pdf`, a 2×2 panel. Panel B (top-right) is
Figure 5, panel A (top-left) is Figure 10; both are cropped out of it.

### F. Figure 6

```bash
python src/experiments/mixed_direction_identification.py --automated   # GPU
python src/experiments/mixed_directions_retrieval_evaluation.py        # GPU
python src/experiments/mixed_directions_plot_retrieval_evaluation.py   # layer/alpha selection
python src/experiments/mixed_directions_end_to_end_evaluation.py       # GPU
python src/experiments/mixed_directions_plot_combos.py
python src/experiments/mixed_directions_plot_end_to_end_combos.py
```

The paper figure stacks two outputs from `results/figures/mixed/`: the **bottom row** of
`figure_combos.pdf` on top, and `figure_combos_end_to_end.pdf` below.

Rank-separation numbers quoted in §4.2: `python src/experiments/recap_rank_separation.py`
(`--alpha` to change the operating point) → `results/figures/recap/`.

### G. Figure 7

```bash
python src/experiments/clasheval_pipeline.py                                        # GPU
python src/experiments/clasheval_pool_ranking_v2.py --model <hf-id>                 # GPU, per model
python src/experiments/clasheval_end_to_end_generation.py --model <hf-id>           # GPU, per model
python src/experiments/plot_figure_7_end_to_end_accuracy_bars.py
```

Needs stage C's directions and `data/clasheval_gpt4.pqt`. `clasheval_pool_ranking.py` is imported
by the v2 script for its pool helpers; its own main reproduces the earlier, unused pool design.

Output: `results/figures/clasheval/figure_7_end_to_end_accuracy_alpha03.pdf` (Fig. 7, used as
generated), plus the full alpha sweep and a line variant next to it. Each figure gets a `.md`
sidecar recording its settings.

## Notes

- Keep these files in one directory: several import siblings by module name.
- Both `python src/experiments/<script>.py` and `python -m src.experiments.<script>` work.
