# text_vectorizer

**File:** `skills/text_vectorizer.py`

## Role

Turns a list of raw documents into a dense numeric embedding matrix. Mirrors
`algo_recommender`'s interface: it (a) **recommends** an embedding method
from corpus characteristics, and (b) **vectorizes** the documents with the
chosen method.

Used by `TextPreparerAgent`.

## Public API

### `recommend_text_vectorizer(n_docs, avg_doc_len, business_purpose='', offline_only=None, verbose=True)`

Returns `VectorizerRecommendation(method, reasoning, confidence, factors)`.

**Heuristics**
- `avg_doc_len ≥ 40` tokens → `transformer` (long prose benefits from
  semantic embeddings)
- `avg_doc_len < 40` → `tfidf_svd` (short text clusters well on lexical
  overlap and is cheaper)
- `n_docs > 50_000` adds a TF-IDF point (transformers cost more at scale)
- Semantic-intent keywords in `business_purpose` (`semantic`, `meaning`,
  `topic`, `theme`, `intent`) add a transformer point
- **Hard constraint:** if `sentence-transformers` isn't importable (or
  `offline_only=True`), the recommendation is forced to `tfidf_svd`.

### `vectorize_text(docs, method='tfidf_svd', svd_dims=100, max_features=20_000, transformer_model='sentence-transformers/all-MiniLM-L6-v2', verbose=True)`

Returns `(X: np.ndarray, artifacts: dict)`.

**`tfidf_svd` path** — `TfidfVectorizer(ngram_range=(1,2), min_df=2,
sublinear_tf=True)` → `TruncatedSVD(n_components≤svd_dims)` → L2-normalize.
Artifacts: `tfidf`, `tfidf_matrix`, `feature_names`, `svd`,
`explained_variance`.

**`transformer` path** — lazy-imports `sentence_transformers`, loads the
model, encodes. On any failure (package missing, model download blocked,
runtime exception), **falls back to `tfidf_svd`** and sets
`artifacts['fallback'] = True`.

## Failure modes

| Condition | Behaviour |
|-----------|-----------|
| Empty `docs` | Raises `ValueError`. |
| Unknown `method` | Raises `ValueError` listing valid options. |
| `transformer` requested but unavailable | Silent fallback to `tfidf_svd`. |
| `transformer` encoding raises | Same fallback path. |

The fallback is deliberate: the pipeline must run offline on plain
scikit-learn even when transformers aren't installed.

## Why TF-IDF artifacts are retained

The `tfidf` vectorizer + `tfidf_matrix` are stashed on `artifacts` so the
`ClusteringAgent` text branch can compute **c-TF-IDF distinctive terms**
per cluster — terms common in cluster c but rare in the rest. This is the
text analog of "feature deviations from the global mean" and feeds the
PersonaNamer text prompt block.
