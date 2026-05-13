# SAE-Based Steering for RAG Factuality — Research Report

## 1. Sources Consulted

| Source | Why |
|---|---|
| ACL 2025 #228 — "Improving RAG Factuality via Activation Steering" | Primary paper; establishes the task, datasets, and diff-in-means baseline we extend |
| `jbloomAus/SAELens` GitHub + docs | Library for loading pre-trained SAEs; de-facto community standard |
| Llama Scope (`fnlp/Llama-Scope`) | Pre-trained SAEs for Llama-3.1-8B, all 32 layers |
| Gemma Scope (`google/gemma-scope-*`) | Pre-trained SAEs for Gemma-2 (NOT Gemma-3) |
| Zou et al. (2023) — Representation Engineering | Theoretical grounding for linear feature directions |
| Bills et al. / Templeton et al. (Anthropic) | SAE feature steering methodology |

> **Note**: During research, web access was unavailable in the agent session. All library and paper details were sourced from training knowledge (cutoff August 2025). Release names should be verified at runtime — see §5.

---

## 2. What the ACL 2025 Paper Does

**Task**: In a RAG scenario, two documents are retrieved for a question — one factual, one fabricated. LLMs frequently follow the wrong document. The paper shows that injecting a learned direction into the residual stream at inference time (no fine-tuning) reliably improves factual grounding.

**Method**: Diff-in-means activation steering.
- Collect residual stream activations from factual vs. non-factual contexts
- Direction = `mean(factual_acts) - mean(nonfactual_acts)` at layer L, token position P
- At inference: `resid_L += alpha * direction` at every token position

**Datasets**: NQ-Swap (`substitution_type == corpus`) and ConflictQA (PopQA variant).

**Models**: Llama-3.1-8B-Instruct and Gemma-3-4b-it (from the results directory structure in this repo).

**This repo IS the implementation of that paper.** The scripts `direction_identification.py` and `steering.py` faithfully reproduce it.

---

## 3. Why Extend to Sparse Autoencoders?

Diff-in-means works in raw residual stream space. The direction is a dense vector that mixes many overlapping features. SAEs offer a better basis:

- An SAE learns a sparse dictionary `x ≈ b + Σᵢ aᵢ dᵢ` where each `dᵢ` (decoder column) is a near-monosemantic feature direction.
- SAE feature activations `aᵢ` are interpretable: you can look up what feature `i` represents.
- Steering with a single decoder column `dᵢ` is the minimal, most targeted intervention.
- Using contrastive examples to *select* which features to steer is a principled generalisation of diff-in-means.

**Key insight**: diff-in-means in residual space ≈ diff-in-means in SAE *feature* space projected back through the decoder. The SAE version adds sparsity — only the K most discriminative features contribute, filtering noise.

---

## 4. SAE Steering Methodology

### Algorithm (implemented in `direction_identification_sae.py`)

1. **Collect residuals** (identical to diff-in-means):
   - For each sample, run the model forward and cache `hook_resid_post` at layer L
   - Two procedures:
     - `sae_context_only`: positive = factual_context text, negative = non_factual_evidence text
     - `sae_ab_choice`: positive = A/B prompt with correct label, negative = wrong label
   - Same token positions as the original: `last_pos`, `entity_pos` (nq_swap only), `choice_token`

2. **Encode through SAE**:
   ```
   pos_feats = SAE.encode(pos_residuals)   # [n_samples, n_sae_features]
   neg_feats = SAE.encode(neg_residuals)   # [n_samples, n_sae_features]
   ```

3. **Feature-level diff-in-means**:
   ```
   feat_diff = mean(pos_feats, axis=0) - mean(neg_feats, axis=0)   # [n_sae_features]
   ```

4. **Select top-K features** (by positive diff score):
   ```
   top_idx = argsort(feat_diff)[-K:]
   ```

5. **Build direction** as weighted sum of decoder columns:
   ```
   direction = Σᵢ feat_diff[i] * W_dec[i]   for i in top_idx
   ```

6. **Save** as `direction.pt` (un-normalised, same convention as diff-in-means) + `meta.json` with `method: "sae"` and extra fields `top_k_features`, `top_feature_indices`.

### Steering (implemented in `steering_sae.py`)

Exactly identical to `steering.py` — load direction, register a `hook_resid_post` forward hook that adds `alpha * direction` to every token position. The only difference is that `find_sae_directions()` filters for `sae_*` procedure directories instead of returning all directions.

---

## 5. Library Choice: `sae_lens`

**Chosen**: `sae_lens` by Joseph Bloom (`pip install sae-lens`)

**Why**: 
- De-facto community standard for SAE interpretability (>1800 GitHub stars as of mid-2025)
- Native TransformerLens integration — hook names are identical (`blocks.{L}.hook_resid_post`)
- Largest pre-trained model hub on HuggingFace
- Supports TopK (Llama Scope), JumpReLU (Gemma Scope), and standard ReLU architectures
- Simple API: `SAE.from_pretrained(release, sae_id)` + `sae.encode(x)` + `sae.W_dec`

**Alternatives considered and rejected**:
- `EleutherAI/sae`: older, less pre-trained coverage
- `Goodfire SDK`: commercial API, not self-hostable
- `SAEBench`: evaluation harness only, not a loading library

### Pre-trained SAE Availability

| Model | SAE Available? | Release | Notes |
|---|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | ✅ | `llama_scope_lxr_8x` | Trained on base model; instruct works in practice |
| `google/gemma-3-4b-it` | ❌ | — | Too new; no public release as of 2025 |

**For Llama**, the Llama Scope release covers all 32 layers with TopK SAEs at multiple widths. The SAE was trained on the base model (`Meta-Llama-3.1-8B`), not the instruct fine-tune. This is the standard practice — the residual stream geometry is largely preserved through RLHF/SFT, so base-model SAE features generalise.

**For Gemma-3-4b-it**, no public SAEs exist. Options:
1. Train your own using `sae_lens`'s `LanguageModelSAERunnerConfig` pipeline
2. Pivot to `google/gemma-2-2b` which has Google's Gemma Scope SAEs (JumpReLU, all layers)

To verify the exact release name at runtime:
```python
from sae_lens.pretrained_saes import get_pretrained_saes_directory
d = get_pretrained_saes_directory()
for k in d:
    if "llama" in k.lower():
        print(k, list(d[k].saes_map.keys())[:3])
```

---

## 6. Implementation Decisions

### Code reuse
Both new scripts import directly from the existing ones:
- `direction_identification_sae.py` imports `collect_context_only`, `collect_ab_choice` from `direction_identification.py` — zero code duplication for activation collection
- `steering_sae.py` imports `build_rag_prompt`, `batched_generate`, `parse_direction_path`, `load_direction` from `steering.py` — zero code duplication for generation

### Direction format compatibility
Directions are saved **un-normalised** (same as `direction_identification.py`, which calls `diff_in_means(..., normalize=False)`). This means:
- `steering_sae.py`'s `--normalize` flag works identically to `steering.py`'s
- `evaluation_steering.py` needs no changes — it discovers all `runs.jsonl` under `results/steering/` recursively

### Procedure naming
SAE procedures are named `sae_context_only` and `sae_ab_choice` (prefixed with `sae_`). This makes them discoverable as a distinct family, allowing `find_sae_directions()` to filter cleanly with `startswith("sae_")` while keeping them in the same directory tree.

### Alpha scaling
Because SAE decoder columns are near-unit-norm and we sum K of them with discriminative weights, the direction norm is typically much smaller than a raw diff-in-means vector (e.g., diff-in-means norm ≈ 9.3 for Llama conflictqa/layer_20). A larger alpha (e.g., 20–50) is likely needed. Recommended: sweep `--alpha` values as a hyperparameter when running `steering_sae.py`.

### Top-K default
Default `--top-k 10`. This is a reasonable middle ground:
- K=1: most interpretable (single monosemantic feature), potentially noisy
- K=10: balances interpretability and signal strength
- K=50+: approaches diff-in-means in character (loses sparsity advantage)

### SAE dtype handling
SAEs from sae_lens are stored in float32. The model runs in bfloat16. Activations are cast to the SAE's dtype before encoding, and directions are stored as float32. `load_direction()` (from `steering.py`) casts to the model's dtype (bfloat16) at load time — no change needed.

---

## 7. How SAE Differs from Diff-in-Means

| Aspect | Diff-in-means (current) | SAE steering (new) |
|---|---|---|
| Direction source | `mean(pos_resid) - mean(neg_resid)` | Weighted sum of top-K SAE decoder columns |
| Space | Residual stream directly | Feature space → projected back via decoder |
| Sparsity | Dense (uses all d_model dimensions) | Sparse (only K features contribute) |
| Interpretability | Low (mixed features) | High (each top_feature_indices entry is monosemantic) |
| Requires pre-trained SAE | No | Yes |
| Model coverage | Any model | Only models with public SAEs |
| Expected alpha range | ~4–10 (raw diff-in-means scale) | ~20–50 (decoder column scale) |
| Cross-dataset transfer | Implicit | Explicit: features selected on one dataset may generalise |

---

## 8. Running the Pipeline

```bash
# Step 0: install sae_lens
pip install sae-lens

# Step 1: identify directions (Llama on nq_swap, layers 10/15/20)
python -m src.experiments.direction_identification_sae \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset nq_swap \
    --layers 10,15,20 \
    --top-k 10

# Also on conflictqa
python -m src.experiments.direction_identification_sae \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset conflictqa \
    --layers 10,15,20 \
    --top-k 10

# Step 2: run steering (eval on nq_swap, steer with all identified directions)
python -m src.experiments.steering_sae \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --eval-dataset nq_swap \
    --alpha 20.0

# Step 3: evaluate (no changes needed — discovers SAE runs automatically)
python -m src.experiments.evaluation_steering
```

> If the default SAE release name (`llama_scope_lxr_8x`) fails, check the correct name via `get_pretrained_saes_directory()` and pass `--sae-release <correct_name>`.
