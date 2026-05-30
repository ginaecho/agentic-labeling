# TextPreparerAgent

**File:** `agents/text_preparer.py`
**Class:** `TextPreparerAgent`

## Role

Text-modality analog of `FeatureEngineerAgent`. When the pipeline runs on a
text-dominant dataset (articles, reviews, support tickets, documents), this
agent replaces FeatureEngineer:

1. Locates the text column (from `UserIntent.text_column`,
   `DatasetProfile.text_column`, or via the agent's own `detect_text_column`
   heuristic).
2. Asks the `text_vectorizer` skill to **recommend** an embedding method
   (`tfidf_svd` for short text, `transformer` for long-form prose), and
   lets the LLM (via OrchestratorBus) confirm or override the choice.
3. **Vectorizes** the documents into a dense numeric matrix.
4. Returns that matrix as a DataFrame of columns `emb_0..emb_n` — a plain
   numeric feature table that flows through the existing FeatureSelector →
   Clusterer → Classifier → PersonaNamer stages unchanged.

The raw documents (aligned to the embedding rows by index) and the TF-IDF
artifacts are stashed on the result so a later, text-aware labelling step
can compute distinctive terms + representative documents per cluster.

## Skills used

- [orchestrator_bus](../skills/orchestrator_bus.md) — `ask()`, `report()`
- `text_vectorizer` (`skills/text_vectorizer.py`) —
  `recommend_text_vectorizer()`, `vectorize_text()`

## Inputs

- `raw_df: pd.DataFrame` — the table containing the text column
- `user_intent: UserIntent` — `.text_column`, `.modality`, `.business_purpose`
- `dataset_profile: DatasetProfile | None` — uses `.text_column` if set
- `output_path: str` — where to save the embeddings parquet
- `method: str | None` — force `'tfidf_svd'` or `'transformer'`; `None`
  triggers the recommend → LLM-override flow
- `feedback: str` — orchestrator hint (e.g. "switch to transformer next iter")
- `iteration: int`

## Outputs

- `TextPreparationResult`:
  - `n_docs: int`, `n_dims: int`
  - `method: str` — actually-used method (may differ from request on fallback)
  - `feature_names: list[str]` — `emb_0..emb_n`
  - `text_column: str`
  - `output_path: str`
  - `reasoning: str`
  - `artifacts: dict` — `tfidf` vectorizer, `tfidf_matrix`, `feature_names`,
    `svd`, `explained_variance`, `transformer_model`, `method`, `fallback`
  - `raw_docs: list[str]` — aligned to embedding rows

Also writes:
- `data/processed/text_embeddings.parquet` — the embedding matrix (best-effort)
- A `success` or `blocked` `OrchestratorMessage` on the bus

## Failure modes

| Condition | Action |
|-----------|--------|
| No free-text column found | `blocked` status; raises `RuntimeError`. Caller should set `text_column` in `UserIntent` / config and retry. |
| Fewer than 20 usable documents (after empty-row filtering) | `blocked`; the corpus is too small to cluster. |
| `transformer` requested but `sentence-transformers` not installed | Silent fallback to `tfidf_svd` with `artifacts['fallback'] = True`. |
| Transformer encoding raises any exception | Same fallback path. |

## Cross-modal loopback

The orchestrator's failure-tuning LLM may set
`state.tuning_params['text_vectorizer']` to a different method on a failed
iteration. The orchestrator detects the change at the top of the next loop
iteration and re-invokes `TextPreparerAgent.run(method=<new>)` —
the text-mode analog of `FeatureEngineer` re-engineering.
