# PhD Report — Literature Review & Proposed Fusion Method

## Literature Findings

### Score fusion / hybrid retrieval

- **Saad-Falcon et al., "An Analysis of Fusion Functions for Hybrid Retrieval," arXiv:2210.11934 / ACM TOIS 2023** — authoritative empirical study. Key takeaways:
  - Convex Combination (CC) with a single tuned α consistently outperforms Reciprocal Rank Fusion (RRF) in in-domain and out-of-domain settings
  - CC is agnostic to normalization choice (min-max vs. z-score): any linear rescaling is absorbed into α
  - Grid search over α ∈ {0, 0.1, …, 1.0} is standard
- **Cormack et al., 2009 (RRF)**: operates on ranks, sidesteps scale-mismatch entirely; lower variance than CC but less flexible — not recommended here since we want interpretable α
- **Z-score vs. min-max**: z-score is more robust when score distributions are heavy-tailed (e.g., unbounded projection scores); preferred for this setting

### LLM internal representations as retrieval / reranking signal

- **Probing-RAG (Baek et al., arXiv:2410.13339, NAACL 2025)**: trains a lightweight linear probe on LLM hidden states (~1/3 depth) to gate whether to retrieve at all. Closest prior art — uses the hidden state as a binary classifier, not a dot product onto a pre-computed direction
- **CAR: Query-Guided Confidence-Aware Reranking (arXiv:2605.04495, 2026)**: uses the generator's sampling variance as a reranking signal — training-free, internal LLM signal, but relies on generation sampling rather than a fixed direction
- **Probing Ranking LLMs (arXiv:2410.18527)**: probes neuron activations inside a fine-tuned ranking LLM to identify features used for ranking; does not use pre-computed directions
- **Gap**: no prior work uses a pre-computed factuality direction (diff-in-means or SAE probe) projected via dot product with the document's last-token hidden state as a retrieval score. The proposed method is novel in this specific combination.

### Linear probes / geometry of truth

- **Marks & Tegmark, "The Geometry of Truth," arXiv:2310.06824**: truth/falsity of factual claims is linearly encoded in LLM residual streams at sufficient scale. Mass-mean (diff-in-means) probing generalizes as well as logistic regression and identifies *causally implicated* directions. Direct theoretical grounding for using a diff-in-means direction as a document score.
- **"Probing the Geometry of Truth" (arXiv:2506.00823, 2025)**: confirms the direction generalizes across dataset types and logical variants, supporting use of a single direction across diverse documents.

---

## Proposed Fusion Method

**Formula:**

```
s_cos(d, q)  = cosine_similarity(embed_SBERT(q), embed_SBERT(d))
s_proj(d)    = h_d · v_fact          # dot product of last-token hidden state with factuality direction

# Z-score normalize over corpus D:
s_cos_norm   = zscore_{d∈D}(s_cos(d, q))   # per query
s_proj_norm  = zscore_{d∈D}(s_proj(d))     # once globally (query-independent)

score(d, q)  = (1 - α) · s_cos_norm + α · s_proj_norm
```

- **α = 0**: pure SBERT baseline
- **α = 1**: pure factuality-direction projection
- Sweep α ∈ {0.0, 0.3, 0.5, 1.0}

**Rationale:** CC is the best-performing simple fusion per Saad-Falcon et al. TOIS 2023. Z-score is preferred over min-max because `s_proj` has unbounded range. The projection score is a linear probe in the sense of Marks & Tegmark 2310.06824.

---

## Justification and Known Limitations

**Justification:**
- CC with α is equivalent to any linear normalization choice up to α reparametrization (Saad-Falcon et al.) — no additional design choices needed
- The `h_d · v_fact` projection measures alignment with the same causal direction that controls factuality behavior in the LLM

**Limitations:**
- **Transductive normalization**: z-score statistics computed over the corpus; noisy if corpus is small
- **Direction generalization**: direction found on a specific question distribution; may degrade on very different document styles
- **Last-token choice**: may miss information for long documents; averaging last few tokens is an alternative
- **Additive fusion**: does not enforce both signals being high simultaneously; a geometric mean would, but is harder to tune

---

## Sources

- [Saad-Falcon et al., arXiv:2210.11934](https://arxiv.org/abs/2210.11934) / [ACM TOIS](https://dl.acm.org/doi/full/10.1145/3596512)
- [Probing-RAG, arXiv:2410.13339](https://arxiv.org/abs/2410.13339)
- [Marks & Tegmark, arXiv:2310.06824](https://arxiv.org/abs/2310.06824)
- [Geometry of Truth (2025), arXiv:2506.00823](https://arxiv.org/html/2506.00823v1)
- [CAR, arXiv:2605.04495](https://arxiv.org/abs/2605.04495v1)
- [Probing Ranking LLMs, arXiv:2410.18527](https://arxiv.org/abs/2410.18527)
