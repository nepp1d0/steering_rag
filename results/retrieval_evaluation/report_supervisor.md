# Supervisor Report — Review of Implementation and Methodology

**Files reviewed:** `src/experiments/retrieval_evaluation.py`, `src/experiments/direction_identification.py`, `src/utils.py`, `data/normalized_dataset/nq_swap/data.jsonl`  
**Dataset statistics confirmed:** 1685 samples, 3304 unique documents (0 cross-role docs, 0 identical gold/nonfactual pairs)

---

## Implementation Review

### Bug 1 (HIGH — fixed): Non-deterministic corpus ordering across cached files

**Original code:** `list({s["factual_context"] for s in samples} | {s["non_factual_evidence"] for s in samples})`

Python's `PYTHONHASHSEED` randomization makes `set` iteration order non-deterministic across interpreter sessions. Both cache files (`sbert_embeddings.pt`, `llm_hidden_states.pt`) are plain tensors with no companion ordering file. If one cache existed from a prior run and the other was computed fresh in a new session, the two tensors would silently index different documents — every `s_proj` score assigned to the wrong document.

**Fix applied:** Changed to `sorted(set(...) | set(...))`.

### Bug 2 (LOW — fixed): SBERT model instantiated twice

`SentenceTransformer` was constructed once for document encoding (conditionally skipped on cache hit) and again for query encoding (always). Fixed by hoisting a single instance before the cache branch.

### Non-issues confirmed

- **`s_proj` global z-score:** Correct. Since `s_proj` is query-independent, z-scoring it once over the full corpus produces the same scale (std≈1, mean≈0) as the per-query z-scored `s_cos`. The weighted sum is numerically valid.
- **BOS token handling:** `direction_identification.py` uses `model.to_tokens(prepend_bos=True)`; `retrieval_evaluation.py` uses `tokenizer(add_special_tokens=True)` + `run_with_cache(prepend_bos=False)`. Both paths add BOS exactly once. Consistent.
- **Left-padding + `[:, -1, :]`:** Correct. Left-padding means position `[-1]` is always the last real token.
- **`np.argpartition`:** Correct for O(N) top-k selection.

---

## Theoretical Assessment

**The fusion formula is internally sound.** Z-scored convex combination is a well-grounded simple fusion baseline (Saad-Falcon et al. TOIS 2023). The PhD's citations are appropriate.

### Critical methodological concern: direction leakage (not in PhD's flagged limitations)

The factuality direction is `mean(factual_hidden) - mean(nonfactual_hidden)` computed over all 1685 samples. The retrieval evaluation runs over those same 1685 samples — there is no train/test split.

By construction, `doc @ direction` is systematically biased toward documents that contributed to the positive pole of the direction. At `α=1` the ranking essentially uses a label-derived signal on the same data it was fit from. Per-document influence is small (~1/1685), so this is not a catastrophic shortcut, but any improvement of `α > 0` over `α = 0` cannot cleanly be attributed to direction generalization.

**Recommended fix:** Hold out at least 20% of samples for direction learning and evaluate only on the held-out set. **Practical workaround without recomputing directions:** compute direction on one dataset (e.g. ConflictQA) and evaluate on the other (NQ-Swap) using the `--direction-dataset` / `--dataset` split — this gives a clean cross-dataset generalization test that avoids the leakage entirely.

### Corpus composition note

The corpus is exactly 50% factual and 50% non-factual documents (one of each per question, pooled). A direction that generically promotes "factual-style text" will boost *other questions'* factual docs into the top-k for query i. The `nonfactual_rate@k` metric therefore conflates "direction depresses query i's non-factual doc" with "direction promotes other questions' factual docs above it." This is not a bug but should be stated when interpreting results.

### Minor theoretical note

The diff-in-means direction was validated as a probing/classification tool in Marks & Tegmark 2310.06824. Using it as a continuous ranking score assumes factual documents form a compact, direction-separated cluster in residual space — a stronger assumption that should be stated explicitly in any write-up.

---

## Verdict

| Issue | Severity | Status |
|---|---|---|
| Non-deterministic set ordering across cache files | HIGH — silent data corruption | Fixed |
| No train/test split for direction (label leakage) | HIGH — methodological | Workaround: cross-dataset evaluation |
| SBERT double-instantiation | LOW — efficiency | Fixed |
| Corpus composition interpretation | LOW — reporting | Acknowledge in write-up |

**Implementation mechanics (batching, cosine similarity, top-k, scoring loop): correct.**  
**Ready to run.** Leakage concern best addressed by evaluating with direction from one dataset applied to the other.
