# `src/exploratory/` — work not used in the paper

28 scripts that were run during the project but produce no figure, table or number in the
submission. Kept for provenance. Everything the paper needs is in
[`../experiments/`](../experiments/README.md).

Run them the same way, from the repo root:

```bash
python src/exploratory/<script>.py
```

They import `utils.py` from `../experiments/`, so they are not self-contained: porting them means
taking `src/experiments/` along. Keep them two levels below the repo root and in one directory —
the `probe_*` scripts import each other by module name.

## Alternative direction estimators

**Logistic probes** — the §4 pipeline with the diff-in-means direction replaced by a
logistic-regression probe.

| File | Role |
|---|---|
| `probe_direction_identification.py` | probe directions, per layer/seed/position |
| `mixed_probe_direction_identification.py` | probe directions from dataset combinations |
| `probe_retrieval_evaluation.py` | re-ranking sweep using probe directions |
| `probe_plot_retrieval_evaluation.py` | per-cell plots + layer selection (imported by the two below) |
| `probe_plot_figure3.py` | probe counterpart of Figure 3 |
| `probe_plot_figure_3b.py` | probe counterpart of the judge comparison |
| `probe_direction_analysis_heatmap.py` | layer × dataset separability heatmaps |

**ICLR "LLMs Know More Than They Show" probes** — their released classifier weights in our
direction format. Llama-3-8B-Instruct only, read at `hook_mlp_out` rather than `resid_post`.

| File | Role |
|---|---|
| `adapt_iclr_directions.py` | converts their `clf.coef_` checkpoints into `direction.pt` |
| `ICLR_paper_retrieval_evaluation.py` | retrieval evaluation with those directions |

**SAEs and steering** — the earlier steering line, before the work moved to re-ranking.

| File | Role |
|---|---|
| `direction_identification_sae.py` | SAE-latent directions (see `REPORT_SAE.md`) |
| `steering.py` | activation steering during generation |
| `steering_sae.py` | steering with SAE latents |
| `evaluation_steering.py` | accuracy plots for the steering runs |

## Other datasets

| File | Role |
|---|---|
| `raguard_normalization.py` | RAGuard → normalized schema |
| `raguard_claim_activations.py` | per-claim activations on RAGuard |
| `raguard_direction_evaluation.py` | AUROC of each direction source on RAGuard claims |
| `raguard_retrieval_evaluation.py` | fused re-ranking on RAGuard pools |
| `plot_raguard_claim_separation.py` | AUROC by layer / summary heatmap |
| `plot_raguard_retrieval.py` | α sweeps, net contrast, controls |
| `crag_end_to_end_evaluation.py` | end-to-end on CRAG Task 3 (real web pages) |
| `plot_figure_6_end_to_end_generalization.py` | Accuracy@1 of the target document in the ClashEval pool vs α; the retrieval-side companion to Figure 7, dropped from the paper. Data comes from `../experiments/clasheval_pool_ranking_v2.py`. |

## Diagnostics and one-offs

| File | Role |
|---|---|
| `new_identification_reranking_script.py` | regression harness for the attention-mask fix in `collect_side_acts`: re-runs identification + re-ranking for one cell and diffs every metric against the results on disk |
| `plot_figure3_maskfix.py` | Figure 3's top row from the mask-fixed results |
| `direction_analysis_heatmap.py` | layer × dataset separability heatmaps |
| `mixed_directions_direction_analysis_heatmap.py` | same, for the mixed-combo directions |
| `plot_end_to_end_evaluation.py` | per-cell end-to-end diagnostics (superseded by `plot_figure_4.py`) |
| `prompt_evaluation.py` | prompt ablation: baseline vs. a prompt warning that one source is false |
| `migrate_results_to_position_layout.py` | one-off migration to the position-as-leaf layout |
