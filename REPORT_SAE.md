# SAE-Based Steering for RAG Factuality — Research Report

## 1. Sources Consulted

| Source | How accessed | Purpose |
|---|---|---|
| Xin et al. (ACL 2025) — "Sparse Latents Steer Retrieval-Augmented Generation" | ACL Anthology page (HTML); chatpaper.com summary | Primary paper — methodology, models, datasets |
| arXiv:2512.08892 — "Toward Faithful RAG with Sparse Autoencoders" (RAGLens, ICLR 2026) | arXiv abstract page | Related work — different paper, frequency-based feature selection |
| `jbloomAus/SAELens` docs & supported SAEs table | decoderesearch.github.io/SAELens | Library API, release names, sae_id formats |
| Llama Scope (`fnlp/Llama-Scope`) | sae_lens pretrained registry | Pre-trained SAEs for Llama-3.1-8B |

> **Correction from v1 of this report**: The initial version was written from agent training knowledge without live web access. The paper title was wrong (it was guessed as "Improving RAG Factuality via Activation Steering"). The actual title is "Sparse Latents Steer Retrieval-Augmented Generation" and it is directly about SAE-based RAG steering — not diff-in-means. The methodology has been updated accordingly.

---

## 2. What the ACL 2025 Paper Does

**Title**: Sparse Latents Steer Retrieval-Augmented Generation  
**Authors**: Chunlei Xin, Shuheng Zhou, Huijia Zhu, Weiqiang Wang, Xuanang Chen, Xinyan Guan, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun  
**Venue**: ACL 2025 Long Papers, pp. 4547–4562, Vienna

**Core idea**: Use SAEs trained on LLaMA-3.1-8B to identify sparse, interpretable latents that govern two RAG decisions: (1) *context vs. memory* prioritization, and (2) *generate vs. reject* decisions. Then steer those specific latents to control model behavior.

**Key findings**:
- Specific SAE features are causally responsible for whether the model follows retrieved context or its own parametric memory
- These features concentrate in **middle layers**
- Manipulating them reconfigures attention patterns of retrieval heads
- SAEs provide a principled, interpretable handle on RAG faithfulness without fine-tuning

---

## 3. Paper Methodology (Step by Step)

### SAE Used
- Llama Scope SAEs trained on post-MLP residual streams of **LLaMA-3.1-8B** (base model)
- **8x expansion**, **TopK** variant (sparsity enforced by keeping top-k activations per token)
- Applied to both `Llama-3.1-8B` (base) and `Llama-3.1-8B-Instruct`

### Feature Identification
Features are selected using a **frequency-based separation score**:

```
sep(i) = freq(target, i) - freq(baseline, i)
```

where `freq(S, i)` = proportion of samples in set S where feature `i` activates (fires with activation > 0). High `sep(i)` means feature `i` fires much more often on context-following inputs than on memory-following ones.

This is more robust than mean activation difference for TopK SAEs, since exactly K features fire per token regardless of magnitude — making frequency a more stable discriminant.

### Steering Mechanism
**Activation addition** (not clamping or ablation):

```
resid_post_L += alpha * W_dec[i]
```

where `W_dec[i]` is the decoder column for feature `i` (the direction in residual stream space that feature `i` points in). Alpha is tuned to be "effective and stable".

### Layer and Token Position
- Interventions target **post-MLP residual stream** (`hook_resid_post`) — same as this repo
- Token position: **final token of the input**
- Most discriminative features found in **middle layers**

---

## 4. Our Implementation

### What we implement
We follow the paper's methodology with two generalizations:

1. **Two identification procedures** (matching the diff-in-means scripts in this repo):
   - `sae_context_only`: contrastive pair = (factual_context, non_factual_evidence) — plain text, no chat template
   - `sae_ab_choice`: contrastive pair = (A/B prompt with correct label, same prompt with wrong label)

2. **Combined direction** from top-K features (not just top-1):
   ```
   direction = Σᵢ sep(i) * W_dec[i]   for i in top-K by sep score
   ```
   This is a weighted sum of decoder columns, more signal than a single feature and still interpretable via `top_feature_indices` in meta.json.

### Separation score implementation (matches paper)
```python
pos_freq = (pos_feats > 0).float().mean(0)
neg_freq = (neg_feats > 0).float().mean(0)
sep_score = pos_freq - neg_freq
top_idx = sep_score.topk(top_k).indices
```

### Direction format
Directions are saved **un-normalised** (same convention as `direction_identification.py`). The `norm_pre_normalize` field in meta.json records the L2 norm before any normalization. `steering_sae.py`'s `--normalize` flag works identically to `steering.py`'s.

---

## 5. Library Choice: `sae_lens`

**Chosen**: `sae_lens` (`pip install sae-lens`)

The de-facto community standard for loading and using pre-trained SAEs. The ACL 2025 paper itself uses Llama Scope, which is distributed via sae_lens. Native TransformerLens hook-name alignment, active maintenance, largest pre-trained model hub.

### Pre-trained SAE Availability

| Model | Available? | Release | sae_id format |
|---|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | ✅ | `llama_scope_lxr_8x` | `l{layer}r_8x` (e.g. `l10r_8x`) |
| `google/gemma-3-4b-it` | ❌ | — | No public SAEs as of 2025 |

**Note on instruct vs base**: The Llama Scope SAEs are trained on `Meta-Llama-3.1-8B` (base). The paper itself also applies them to the instruct variant. Residual stream geometry is largely preserved through SFT/RLHF, so this works in practice.

**Note on Gemma-3-4b-it**: Google's Gemma Scope covers Gemma-2 (2B and 9B), not Gemma-3. No public SAEs exist for Gemma-3-4b-it. Options: train your own with `sae_lens`, or pivot to Gemma-2-2b.

To verify available release names at runtime:
```python
from sae_lens.pretrained_saes import get_pretrained_saes_directory
d = get_pretrained_saes_directory()
for k in d:
    if "llama_scope" in k:
        print(k, list(d[k].saes_map.keys())[:3])
```

---

## 6. How SAE Differs from Diff-in-Means

| Aspect | Diff-in-means (existing) | SAE steering (new) |
|---|---|---|
| Direction source | `mean(pos_resid) - mean(neg_resid)` | Weighted sum of top-K SAE decoder columns |
| Feature selection | Implicit (dense over all d_model dims) | Explicit frequency-based separation scores |
| Sparsity | Dense direction | Sparse: only K features contribute |
| Interpretability | Low | High — `top_feature_indices` in meta.json are monosemantic |
| Requires pre-trained SAE | No | Yes |
| Model coverage | Any model | Only models with public SAEs (Llama ✅, Gemma-3 ❌) |
| Alpha scale | ~4–10 (raw diff-in-means norm ≈ 9) | ~20–50 (sep-score weights ∈ [0,1], sum of K decoder cols) |

---

## 7. Running the Pipeline

```bash
pip install sae-lens

# Step 1: identify SAE directions
python -m src.experiments.direction_identification_sae \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset nq_swap \
    --layers 10,15,20 \
    --top-k 10

python -m src.experiments.direction_identification_sae \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset conflictqa \
    --layers 10,15,20 \
    --top-k 10

# Step 2: run steering (try larger alpha than diff-in-means)
python -m src.experiments.steering_sae \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --eval-dataset nq_swap \
    --alpha 20.0

# Step 3: evaluate — no changes needed, discovers SAE runs automatically
python -m src.experiments.evaluation_steering
```
