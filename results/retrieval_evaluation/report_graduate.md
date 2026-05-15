# Graduate Student Report — Implementation Notes

## Libraries Chosen

- **`sentence-transformers`**: standard library for SBERT; `encode()` returns batched float32 arrays and handles tokenization, pooling, and normalization internally. Pre-normalizing all doc embeddings once enables per-query cosine similarity as a single matrix multiply.
- **`transformer_lens`**: already used throughout the codebase; `run_with_cache` with `names_filter` runs a forward pass returning only the named activation, keeping peak memory low.
- **`numpy.argpartition`**: O(N) top-k selection vs. O(N log N) full sort.

## Implementation Decisions

**Corpus ordering (determinism fix):** `sorted(set(...) | set(...))` instead of `list(set(...) | set(...))`. Python's hash randomization (`PYTHONHASHSEED`) makes `set` iteration non-deterministic across sessions. Both cache files (`sbert_embeddings.pt`, `llm_hidden_states.pt`) are stored as plain tensors without the ordering — if one was cached from a prior session and the other recomputed fresh, the two tensors would index different documents. `sorted()` makes the ordering deterministic and reproducible.

**`s_proj` z-scored once globally:** The projection score `s_proj(d) = h_d · v_fact` is query-independent — same value for all queries. Z-scoring it once over the corpus produces std≈1, mean≈0, matching the per-query z-scored `s_cos`. No need to recompute inside the per-query loop.

**Left-padding + `[:, -1, :]`:** Documents are tokenized individually, sorted by length, then left-padded inside each batch. Because all sequences are right-aligned, position `[-1]` is always the last real token regardless of padding length — no per-row length tracking needed.

**BOS token handling:** `tokenizer(add_special_tokens=True)` adds BOS; `run_with_cache(prepend_bos=False)` tells TransformerLens not to add another one. Double-prepending would shift the last-token index. This matches the pattern in `utils.get_last_residual` (which uses `model.to_tokens(prepend_bos=True)` — one BOS either way).

**SBERT single instance:** One `SentenceTransformer` object is instantiated before the cache branch and reused for both document encoding (conditional on cache miss) and query encoding (always needed).

**Caching strategy:** SBERT embeddings and LLM hidden states are cached to `out_dir` (same directory as results, keyed by model/dataset/procedure/layer). Subsequent runs skip the expensive compute. The corpus ordering determinism fix ensures caches remain valid across sessions.

## Tricky Parts

- `torch.load` without `weights_only=True` produces a deprecation warning in PyTorch ≥ 2.0; benign here since all saved objects are plain tensors, consistent with the rest of the codebase.
- Cosine similarity for all docs against a single query: pre-normalize doc embeddings to `sbert_norm` once, then `sbert_norm @ q_norm[si]` gives all N cosine similarities in one vectorized call.
