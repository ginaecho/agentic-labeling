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
messages to the orchestrator without tight coupling.

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
multicollinearity. Also flags high pairwise correlations and low-variance
features. Provides iterative removal to bring all VIFs below a threshold.

### Reference
- VIF interpretation: VIF < 5 = acceptable; VIF > 10 = severe collinearity
- See: https://medium.com/@rasdhar.panchal/feature-selection-using-p-values-and-vif-in-linear-regression-6bf25b652d99

### API

```python
from skills.vif_checker import compute_vif, remove_high_vif, flag_high_correlation

# Compute VIF for all columns
vif_df = compute_vif(df)
# Returns pd.DataFrame with columns: feature, vif

# Iteratively remove features with VIF above threshold until all pass
clean_df, removed = remove_high_vif(df, threshold=5.0, max_iterations=50)
# Returns: (cleaned DataFrame, list of removed feature names)

# Flag feature pairs with |correlation| > threshold
pairs = flag_high_correlation(df, threshold=0.85)
# Returns: list of (feature_a, feature_b, correlation) tuples
```

### Thresholds (defaults)

| Gate | Default threshold | Configurable |
|------|------------------|--------------|
| VIF | < 5.0 | Yes (`config.yaml`: `vif_threshold`) |
| Pairwise correlation | \|r\| < 0.85 | Yes (`config.yaml`: `corr_threshold`) |
| Minimum features after filtering | ≥ 10 | Yes |

---

## `silhouette_optimizer` — Data-Driven Cluster Count Selection

**File**: `skills/silhouette_optimizer.py`
**Used by**: `ClusteringAgent`

### Purpose
Tries a range of k values, fits the chosen clustering algorithm for each,
computes the silhouette score, and returns the k that maximises it.
Also computes the elbow-method inertia curve for K-Means as a secondary
signal.

### API

```python
from skills.silhouette_optimizer import optimize_k, SilhouetteResult

result = optimize_k(
    X_scaled,                        # np.ndarray, preprocessed features
    algorithm="hierarchical",        # "hierarchical" | "kmeans"
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
to recommend the most appropriate clustering algorithm.

### API

```python
from skills.algo_recommender import recommend_algorithm, AlgoRecommendation

rec = recommend_algorithm(
    n_rows=10000,
    n_features=45,
    feature_skewness={"travel_spend": 3.2, "grocery_spend": 1.1, ...},
    dataset_profile=profile,   # DatasetProfile from DatasetExaminerAgent
    user_intent=intent,        # UserIntent
)

rec.algorithm    # "hierarchical" | "kmeans"
rec.reasoning    # str — explanation of the choice
rec.confidence   # float 0–1 — how confident the recommendation is
```

### Decision rules

| Condition | Recommendation |
|-----------|----------------|
| `n_rows > 100_000` | K-Means (speed) |
| Mean feature skewness > 2.0 | Hierarchical (robust to skew after log-transform) |
| Business purpose mentions "segments within segments" | Hierarchical |
| Fewer than 5 candidate k values have silhouette > 0.25 | K-Means (try different shape) |
| Default | Hierarchical / Ward |

---

## `score_pca` — PCA Importance Scoring

**File**: Implemented inline in `agents/feature_selector.py` (can be extracted)
**Used by**: `FeatureSelectionAgent`

### Purpose
Scores each feature by its weighted contribution to the top PCA components.
Higher score = feature explains more variance across customers.

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

## `build_frequency_features` — Transaction Frequency Features

**File**: `skills/feature_engineer_skills.py` (or inline in `agents/feature_engineer.py`)
**Used by**: `FeatureEngineerAgent`

### Purpose
Builds per-customer, per-category transaction count features over rolling
time windows (6-month, 12-month).

### Output columns
- `n_txn_{category}_{w}m` — count of transactions in window
- `consec_months_{category}` — consecutive active months

---

## `build_spend_features` — Transaction Spend Features

**File**: `skills/feature_engineer_skills.py`
**Used by**: `FeatureEngineerAgent`

### Output columns
- `amt_{category}_{w}m` — total spend in window
- `avg_spend_{category}_{w}m` — mean spend per transaction

---

## `build_recency_features` — Recency & Engagement Features

**File**: `skills/feature_engineer_skills.py`
**Used by**: `FeatureEngineerAgent`

### Output columns
- `avg_days_between_txn` — average days between any two transactions
- `active_months` — number of months with ≥ 1 transaction
- `days_since_last_txn` — recency signal

---

## `build_interaction_features` — Cross-Category Interaction Features

**File**: `skills/feature_engineer_skills.py`
**Used by**: `FeatureEngineerAgent`

### Purpose
Builds ratio and interaction features that capture how a customer allocates
spending across categories (e.g. travel_share, grocery_vs_dining_ratio).

---

## `clarity_gate` — Persona Name Quality Gate

**File**: Implemented inline in `agents/persona_namer.py`
**Used by**: `PersonaNamingAgent`

### Checks
1. `avg_confidence ≥ 6.0` — Claude's self-assessed naming confidence
2. `all names unique` — no duplicate persona names across clusters
3. `description references numbers` — at least 2 quantitative values per description

---

## `train_classifier` — Random Forest Classifier

**File**: Implemented inline in `agents/classifier.py`
**Used by**: `ClassifierAgent`

### Model
- `RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)`
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
