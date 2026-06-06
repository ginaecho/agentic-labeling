# Skill Registry

This file catalogues every reusable skill available to agents in the
agentic clustering pipeline. Skills are atomic, testable Python functions
or classes. They live in `skills/` and are imported by agents.

To add a new skill: implement it in the appropriate `skills/*.py` file,
then register it here.

---

## `orchestrator_bus` — Agent → Orchestrator Communication

**File**: `skills/orchestrator_bus.py`
**Used by**: All agents

### Purpose
Provides a shared message bus so every agent can report structured status
messages to the orchestrator without tight coupling. Also exposes `ask()` for
agents that need LLM reasoning via the orchestrator.

### API

```python
from skills.orchestrator_bus import OrchestratorMessage, OrchestratorBus

msg = OrchestratorMessage(
    agent="FeatureSelector",
    iteration=2,
    status="success",           # "success" | "warning" | "blocked" | "failure"
    what_was_done="...",
    what_was_not_done="...",
    doubts="...",
    issues=[],
    metrics={"n_features": 45},
    recommendation="proceed",   # "proceed" | "retry" | "escalate"
    context={},                 # arbitrary agent-specific payload
)

bus = OrchestratorBus()
bus.report(msg)
log = bus.get_log()             # list[OrchestratorMessage]
bus.save_log("outputs/pipeline_log.json")

# LLM call (routed through orchestrator)
raw_response = bus.ask(
    agent="FeatureSelector",
    purpose="select best feature subset",
    prompt="...",
    max_tokens=2048,
)
```

### Status meanings

| Status | When to use |
|--------|-------------|
| `success` | Agent completed its task with no issues |
| `warning` | Agent completed, but with caveats (e.g. low silhouette, sparse features) |
| `blocked` | Agent cannot proceed; orchestrator must reroute |
| `failure` | Unexpected exception; pipeline should halt or retry |

### Recommendation meanings

| Recommendation | When to use |
|----------------|-------------|
| `proceed` | Orchestrator should move to the next agent |
| `retry` | Orchestrator should re-run this agent with adjusted params |
| `escalate` | Trigger human checkpoint immediately |

---

## `vif_checker` — Multicollinearity & Feature Quality Gates

**File**: `skills/vif_checker.py`
**Used by**: `FeatureSelectionAgent`

### Purpose
Computes Variance Inflation Factor (VIF) for each feature to detect
multicollinearity. Also flags high pairwise correlations. Provides iterative
removal to bring all VIFs below a threshold.

### Reference
- VIF interpretation: VIF = 1 (no correlation); VIF 1–5 (moderate, acceptable);
  VIF 5–10 (high, consider removing); VIF > 10 (severe collinearity)
- Default threshold is **10.0** — tuned dynamically by the Orchestrator each iteration.

### API

```python
from skills.vif_checker import compute_vif, remove_high_vif, flag_high_correlation

# Compute VIF for all columns
vif_df = compute_vif(df)
# Returns pd.DataFrame with columns: feature, vif

# Iteratively remove features with VIF above threshold until all pass
clean_df, removed = remove_high_vif(df, threshold=10.0, min_features=10, verbose=True)
# Returns: (cleaned DataFrame, list of removed feature names)

# Flag feature pairs with |correlation| > threshold
pairs = flag_high_correlation(df, threshold=0.85)
# Returns: list of (feature_a, feature_b, correlation) tuples
```

### Thresholds (defaults)

| Gate | Default threshold | Configurable |
|------|------------------|--------------|
| VIF | < 10.0 | Yes (Orchestrator tunes dynamically) |
| Pairwise correlation | \|r\| < 0.85 | Yes (`config.yaml`: `corr_threshold`) |
| Minimum features after filtering | ≥ 10 | Yes |

---

## `silhouette_optimizer` — Data-Driven Cluster Count Selection

**File**: `skills/silhouette_optimizer.py`
**Used by**: `ClusteringAgent`

### Purpose
Tries a range of k values, fits the chosen clustering algorithm for each,
computes the silhouette score, and returns the k that maximises it.
Also computes the elbow-method inertia curve for K-Means as a secondary signal.

### API

```python
from skills.silhouette_optimizer import optimize_k, SilhouetteResult

result = optimize_k(
    X_scaled,                        # np.ndarray, preprocessed features
    algorithm="hierarchical",        # "hierarchical" | "kmeans" | "dbscan" | "gmm" | "fuzzy_cmeans"
    k_range=[3, 4, 5, 6, 7, 8, 10, 12, 15],
    random_state=42,
)

result.best_k          # int — k with highest silhouette
result.best_silhouette # float — silhouette at best_k
result.scores          # dict[int, float] — silhouette for each k tried
result.inertias        # dict[int, float] — inertia for each k (K-Means only)
result.reasoning       # str — human-readable summary
```

### Silhouette interpretation

| Score | Quality |
|-------|---------|
| ≥ 0.50 | Strong structure |
| 0.25 – 0.50 | Reasonable structure |
| 0.10 – 0.25 | Weak structure (consider different features) |
| < 0.10 | No meaningful structure |

---

## `algo_recommender` — Clustering Algorithm Recommendation

**File**: `skills/algo_recommender.py`
**Used by**: `ClusteringAgent`

### Purpose
Analyses dataset size, feature distribution shape, and business purpose
to score and recommend the most appropriate clustering algorithm from the
5 supported options.

### API

```python
from skills.algo_recommender import recommend_algorithm, AlgoRecommendation

rec = recommend_algorithm(
    n_rows=10000,
    n_features=45,
    feature_skewness={"feat_a": 3.2, "feat_b": 1.1, ...},
    dataset_profile=profile,   # DatasetProfile from DatasetExaminerAgent
    user_intent=intent,        # UserIntent
)

rec.algorithm    # "hierarchical" | "kmeans" | "dbscan" | "gmm" | "fuzzy_cmeans"
rec.reasoning    # str — explanation of the choice
rec.confidence   # float 0–1 — how confident the recommendation is
```

### Scoring rules (all 5 algorithms scored; highest wins)

| Condition | Effect |
|-----------|--------|
| `n_rows > 100_000` | +2 K-Means (speed) |
| Mean feature skewness > 2.0 | +1 Hierarchical (robust to skew after log-transform) |
| Business purpose mentions "nested" or "sub-group" | +2 Hierarchical |
| Business purpose mentions "overlap" or "fuzzy" | +2 Fuzzy C-Means |
| Business purpose mentions "outlier" or "noise" | +2 DBSCAN |
| Business purpose mentions "soft" or "probabilistic" | +2 GMM |
| Default (no strong signal) | Hierarchical / Ward |

---

## `automl_candidate_search` — AutoML Search As A Skill

**File**: `skills/automl_candidate_search.py`
**Used by**: `ClusteringAgent`

### Purpose
Runs a bounded tournament over clustering algorithm/k candidates and returns
evidence for the best option. This makes AutoML-style search a deterministic
tool inside the agentic workflow: agents decide when to use it, while the skill
does the reproducible scoring.

### API

```python
from skills.automl_candidate_search import search_clustering_candidates

result = search_clustering_candidates(
    X_scaled,
    algorithms=["kmeans", "hierarchical", "gmm"],
    k_range=[3, 4, 5, 6, 7, 8, 10, 12],
    metric="euclidean",
    max_cluster_size_pct=0.40,
)

result.best.algorithm
result.best.k
result.best.silhouette
result.best.stability_ari
result.best.max_cluster_pct
result.best.composite_score
```

### Ranking
Candidates are scored by:

```text
max(0, silhouette) * 70
+ bootstrap_stability_ari * 25
- oversized_cluster_penalty
```

This intentionally optimises for usable clustering rather than one metric:
separation, repeatability, and cluster-size constraints all count.

---

## `score_pca` — PCA Importance Scoring

**File**: Implemented inline in `agents/feature_selector.py`
**Used by**: `FeatureSelectionAgent`

### Purpose
Scores each feature by its weighted contribution to the top PCA components.
Higher score = feature explains more variance across entities.

### Formula
```
pca_score[j] = Σ_i (loading[i,j]² × explained_variance_ratio[i])
```
where `i` ranges over retained PCA components.

---

## `score_autoencoder` — Autoencoder Reconstruction Error Scoring

**File**: Implemented inline in `agents/feature_selector.py`
**Used by**: `FeatureSelectionAgent`

### Purpose
Trains a bottleneck MLP autoencoder, then scores each feature by its
reconstruction error. High error = feature is unique and hard to compress
= likely carries distinct signal that simpler features do not.

### Architecture
- Encoder: input → 64 → bottleneck (n_features // 5, min 8)
- Decoder: bottleneck → 64 → input
- Activation: ReLU
- Early stopping: val_fraction=0.1, patience=10

---

## `dataset_readme_context` — README.md Domain Context

**File**: Implemented inline in `agents/dataset_examiner.py`
**Used by**: `DatasetExaminerAgent` (reads), `FeatureEngineerAgent` (receives via `DatasetProfile`), `FeatureSelectionAgent` (receives via `DatasetProfile`)

### Purpose
If the dataset folder contains a `README.md` (e.g. `data/raw/air_quality/README.md`),
`DatasetExaminerAgent` reads and stores it in `DatasetProfile.dataset_readme`.
This text is injected into LLM prompts in DatasetExaminer (feature group suggestions),
and should be passed on by FeatureEngineer and FeatureSelector when calling
`bus.ask()` so the LLM has full domain context from the data provider.

### Behaviour
- File checked: `<dataset_file_directory>/README.md`
- Capped at 3 000 chars (truncated with `...[README truncated]` if longer)
- If no README exists, `dataset_readme = ""`; all downstream logic is unchanged

---

## `feature_engineer_builders` — 8 Generic Feature Builders

**File**: Implemented inline in `agents/feature_engineer.py`
**Used by**: `FeatureEngineerAgent`

### Purpose
Builds entity-level features from raw event-level data using 8 generic
statistical operations. The LLM chooses which builders to apply to which
columns — no domain vocabulary is hard-coded.
If `DatasetProfile.dataset_readme` is non-empty, it is injected into the
LLM prompt so domain context from the data provider informs which builders
and columns to prioritise.

### Builders

| Builder | What it computes | Example output column names |
|---------|-----------------|------------------------------|
| `group_aggregate` | count/sum/mean/std/max per group value × window | `count_{col}_{val}_{w}` |
| `group_trend` | change in count or sum between two windows | `trend_count_{col}_{val}` |
| `group_streak` | consecutive active periods per group value | `streak_{col}_{val}` |
| `overall_aggregate` | aggregate over all events (no grouping) | `sum_{val_col}_{w}` |
| `frequency_recency` | event frequency, active periods, recency, gap | `event_count_{w}`, `days_since_last` |
| `entity_diversity` | number of unique values per column × window | `n_unique_{col}_{w}` |
| `temporal_patterns` | morning/evening/weekend ratios, peak hour | `pct_morning_{w}`, `pct_weekend_{w}` |
| `static_attributes` | copy entity-level static columns as-is | original column name |

Feature column names embed the actual data column names, not domain abbreviations.

---

## `clarity_gate` — Persona Name Quality Gate

**File**: Implemented inline in `agents/persona_namer.py`
**Used by**: `PersonaNamingAgent`

### Checks
1. `avg_confidence ≥ 6.0` — LLM's self-assessed naming confidence
2. `all names unique` — no duplicate persona names across clusters
3. **`must_have_clusters` covered** — if `user_intent.must_have_clusters` is non-empty,
   every required type must appear (case-insensitive substring match) in at least one
   persona name or description. Missing types are listed in `issues` and trigger `recluster`.

### Must-have cluster matching
The gate performs a case-insensitive substring match of each required type against the
concatenated name + description of all personas. Both hyphenated and space-separated
variants are tried (e.g. `"high-value"` also matches `"high value"`).
If a required type is not found, the gate adds a descriptive issue to the failure report.

---

## `train_classifier` — LLM-Selected Classifier

**File**: Implemented inline in `agents/classifier.py`
**Used by**: `ClassifierAgent`

### Model selection
LLM chooses from 4 supported models based on dataset size, class balance,
and interpretability requirements:

| Model | Sklearn class |
|-------|--------------|
| `random_forest` | `RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)` |
| `xgboost` | `XGBClassifier(n_estimators=200, random_state=42)` |
| `gradient_boosting` | `GradientBoostingClassifier(n_estimators=200, random_state=42)` |
| `logistic_regression` | `LogisticRegression(max_iter=1000, random_state=42)` |

- Stratified k-fold CV (k = min(5, min_class_size))
- Reports: accuracy, macro-F1, weighted-F1, per-class F1

---

## Adding a New Skill

1. Implement the function/class in `skills/<module>.py`
2. Add it to `skills/__init__.py` exports
3. Register it in this file (`skill.md`) with:
   - File location
   - Which agents use it
   - Purpose and API
   - Any configurable thresholds
4. Import and call it from the relevant agent(s)
