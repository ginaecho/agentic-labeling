# Design Plan — Extending the Pipeline to Text / Document Clustering

Status: **PROPOSED — awaiting approval before implementation**
Branch: `feat/text-clustering`
Date: 2026-05-30

## 1. Goal

Today the multi-agent pipeline only clusters **tabular** data. Extend it to cluster
**text** (articles, documents, reviews, support tickets, …) using the **same
architecture**: ML-savvy agents + tools, a single Orchestrator coordinating them,
and **hard control gates** that decide when to loop back, escalate, or proceed.

Text becomes a **modality** the pipeline routes into — not a separate fork. The
Orchestrator, its loop, and its gates are reused; only the data-shaping and
profiling steps gain text-aware implementations.

### Decisions locked with the user
- **Embeddings:** the agent **picks per dataset** — TF-IDF + TruncatedSVD (short
  text, offline, no heavy deps) vs. local sentence-transformer (long-form,
  semantic). Chosen by an `algo_recommender`-style tool; degrades to TF-IDF if
  transformers aren't installed.
- **Labelling:** **c-TF-IDF distinctive terms + representative docs → LLM** names
  the theme (the text analog of `PersonaNamer`). No heavy topic-model deps.
- **Routing:** **auto-detect** a text-dominant dataset in `DatasetExaminer`, with a
  `config.yaml` / CLI override.

## 2. Current architecture (what we keep)

```
UserInput → DatasetExaminer → [FeatureEngineer] → ┌─ loop ───────────────────┐
                                                  │ FeatureSelector          │
                                                  │ Clusterer  (gate)        │
                                                  │ PersonaNamer (gate)      │
                                                  │ Classifier  (gate)       │
                                                  │ Human checkpoint         │
                                                  └──────────────────────────┘
```

**Control gates (all reused):**
| Gate | Where | Condition | Action on fail |
|------|-------|-----------|----------------|
| Dataset usable | `dataset_examiner.py` | needs usable signal | `blocked` → escalate |
| Silhouette | `clusterer.py` | `sil < min_silhouette` | `reselect_features` |
| Oversized cluster | `clusterer.py` | cluster > `max_cluster_size_pct` | subcluster / reselect |
| Clarity Gate | `persona_namer.py` | `avg_conf ≥ 6` & unique names | `recluster` |
| Classifier F1 | `classifier.py` | `cv_f1_macro ≥ 0.70` | `recluster` / `reselect_features` |
| Parameter tuning | `orchestrator._ask_parameter_tuning` | after any fail | adjusts `tuning_params` |

These gates are **modality-agnostic** and stay exactly as they are. We only change
*what data flows through them*.

## 3. Where tabular is hard-assumed (the surfaces to change)

| File | Tabular assumption | Line(s) |
|------|--------------------|---------|
| `agents/dataset_examiner.py` | **BLOCKS if no numeric columns** | ~167–174 |
| `agents/feature_engineer.py` | entity × time × category builders; needs entity/amount/timestamp | whole file |
| `agents/feature_selector.py` | PCA + autoencoder + VIF on a numeric matrix | whole file |
| `agents/clusterer.py` | `StandardScaler`, log-transform, Euclidean silhouette, profiles = numeric feature means | 244–257, 407, 506, `_extract_profiles` |
| `agents/classifier.py` | RandomForest on numeric matrix | — (works as-is on any numeric matrix) |
| `agents/persona_namer.py` | reads `top_above_average` / `feature_means` | generic — already OK |

Key insight: **once text is turned into a numeric embedding matrix**, the
Clusterer (with a cosine tweak), Classifier, and PersonaNamer need only small,
additive changes. The real work is *replacing FeatureEngineer/FeatureSelector with
a text vectorizer* and *making cluster profiles term-based instead of mean-based*.

## 4. Proposed changes

### 4.1 Modality routing (small, central)
- **`agents/state.py`** — add `modality: str = "auto"` and `text_column: str | None`
  to `UserIntent`; add `text_vectorizer` to `tuning_params`.
- **`agents/dataset_examiner.py`** — detect text: an `object` column whose mean
  token count is high and cardinality ≈ row count → text-dominant. Set
  `DatasetProfile.modality` + `text_column`. **Only block on "no numeric columns"
  when `modality == 'tabular'`.** Honour a config/CLI override.
- **`config.yaml`** — add `modality: auto`, `text_column: ~`, `text_vectorizer: auto`,
  `embedding_model: sentence-transformers/all-MiniLM-L6-v2`.
- **`run_pipeline.py`** — add `--modality` and `--text-column` flags.

### 4.2 New tool — `skills/text_vectorizer.py`
- `recommend_text_vectorizer(n_docs, avg_doc_len, offline, purpose) -> Recommendation`
  (mirrors `algo_recommender`): short docs → `tfidf_svd`; long-form & transformers
  available → `transformer`.
- `vectorize_text(docs, method, ...) -> (X: np.ndarray, artifacts)` where artifacts
  retain the fitted TF-IDF vocabulary so we can compute **c-TF-IDF** per cluster.
- Lazy imports; **fallback to TF-IDF** if `sentence-transformers`/`torch` missing.

### 4.3 New agent — `agents/text_preparer.py` (text analog of FeatureEngineer)
- Input: raw docs from `text_column`. Cleans, calls `text_vectorizer`, returns an
  embedding matrix as a DataFrame `emb_0…emb_n` (one row per document), and keeps
  the raw text aligned by index for labelling.
- Reports to the bus (success/warning/blocked) exactly like FeatureEngineer.
- The Orchestrator calls this **instead of** FeatureEngineer when `modality=text`.

### 4.4 FeatureSelector — modality-aware short-circuit
- For embeddings, PCA/VIF/autoencoder don't apply cleanly. When `modality=text`,
  FeatureSelector returns all embedding dims (TruncatedSVD already compressed
  them), skipping autoencoder + VIF. The `FeatureSelectionResult` shape is
  unchanged so the loop and history keep working.

### 4.5 Clusterer — cosine + term-based profiles
- When `modality=text`: **L2-normalize** embeddings (KMeans on normalized vectors ≈
  spherical k-means), skip log-transform/StandardScaler, compute silhouette with
  `metric='cosine'`. Deepening loop unchanged.
- `_extract_profiles` (text branch) builds, per cluster:
  - `top_above_average` ← **c-TF-IDF distinctive terms** `{term: score}` (so the
    existing UI bars and PersonaNamer keep working unchanged),
  - `feature_means` ← term weights,
  - **new** `top_terms` and `representative_docs` (docs nearest the centroid).
- Reusing `top_above_average` means the **named-cluster UI, cluster-chat, and the
  cross-cluster comparison built earlier all work for text with zero UI changes.**

### 4.6 PersonaNamer — text-aware prompt block
- Add a text branch to `_format_cluster_block` that shows **top terms +
  representative document snippets** instead of numeric means. The Clarity Gate
  (confidence + uniqueness) is unchanged. Output schema (name/tagline/description/
  traits/confidence) is unchanged → UI unaffected.

### 4.7 Classifier — no change needed
- It trains a RandomForest on whatever numeric matrix it's given; feeding it the
  embedding matrix validates text clusters with the **same F1 ≥ 0.70 gate**.

### 4.8 Cross-modal loopback (the agentic control story)
- Add `text_vectorizer` to `tuning_params`. On a failed gate (e.g. low silhouette
  or low F1 on text), `_ask_parameter_tuning` can switch the embedding method
  (TF-IDF → transformer), change `k`, or switch algorithm — the **text analog of
  "reselect features"**. This keeps the "loop back / take action" behaviour intact.

## 5. Files added / touched

**New**
- `skills/text_vectorizer.py`
- `agents/text_preparer.py`
- `data/raw/text_articles/…` (small sample text dataset + README)
- `experiments/test_text_clustering.py` (offline end-to-end test, mocked LLM bus)
- this design doc

**Modified**
- `agents/state.py`, `agents/dataset_examiner.py`, `agents/clusterer.py`,
  `agents/persona_namer.py`, `agents/feature_selector.py`,
  `agents/orchestrator.py`, `config.yaml`, `run_pipeline.py`, `requirements.txt`

## 6. Dependencies
- **Already present:** scikit-learn (`TfidfVectorizer`, `TruncatedSVD`, `KMeans`,
  `silhouette_score(metric='cosine')`), numpy, pandas.
- **Optional (lazy + fallback):** `sentence-transformers` (+ `torch`) for semantic
  embeddings; `hdbscan` for density text clustering. Pipeline runs fully offline on
  TF-IDF if neither is installed.

## 7. Testing & verification
- Unit: `text_vectorizer` returns a finite matrix for both methods; fallback path
  works without transformers.
- Integration: run the text path on the sample dataset with a mocked LLM bus;
  assert clusters form, profiles carry `top_terms`/`representative_docs`, gates
  fire, and `personas.json` is produced.
- Regression: tabular path unchanged (existing datasets still cluster).

## 8. Risks / open questions
- **Silhouette on text** is often lower than on tabular RFM features; we may relax
  the text-mode `min_silhouette` default and lean on the F1 + Clarity gates.
- **Classifier F1 ≥ 0.70** can be strict for fuzzy topics; consider a text-mode
  threshold (e.g. 0.60) — flagged for your call during build.
- **Model download** for transformers needs network on first run; TF-IDF is the
  safe offline default.

## 9. Rollout
1. Land routing + `state`/`config`/CLI scaffolding (no behaviour change for tabular).
2. Add `text_vectorizer` + `text_preparer` + sample data.
3. Make Clusterer/PersonaNamer/FeatureSelector modality-aware.
4. Wire cross-modal loopback tuning.
5. Tests + docs; open PR into `feat/text-clustering`.
```
```
