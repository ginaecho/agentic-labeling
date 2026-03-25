# Agent Registry

This file defines every agent in the agentic clustering pipeline, its role,
skills, communication contract with the orchestrator, and retry/fallback
behaviour. Each agent implementation lives in `agents/<name>.py` and uses
skills from `skills/`.

---

## Orchestrator

**File**: `agents/orchestrator.py`
**Class**: `Orchestrator`

### Role
Central coordinator. Owns the pipeline state, routes feedback between agents,
maintains the orchestrator message log, and uses LLM reasoning to diagnose
complex failures. Presents a human checkpoint when the pipeline converges or
exhausts its retry budget.

### Inputs
- `config: dict` (from `config.yaml`)
- `user_intent: UserIntent`
- `features_path: str`

### Outputs
- `dict` with keys `status`, `personas`, `run_history`, `timing`, `llm_usage`

### Responsibilities
1. Receive `OrchestratorMessage` from every agent via `orchestrator_bus`
2. Log all messages to `pipeline_log` (saved to `outputs/pipeline_log.json`)
3. Use LLM to analyse failure reports and decide routing
4. After each failed iteration, call `_ask_parameter_tuning()` to let LLM propose
   new values for `vif_threshold`, `k_range`, `algorithm`, `min_silhouette`, and
   `feature_focus` — these are passed directly to FeatureSelector and Clusterer
5. Enforce per-loop retry budgets (default `max_total_iterations=10`)
6. At max iterations with no approved result, deliver a best-effort analysis by
   running PersonaNamer (force_proceed=True) and Classifier on the highest-silhouette
   clustering observed across all iterations
7. Present human checkpoint with full pipeline log summary

### Dynamic tuning parameters (managed by Orchestrator, not config.yaml)
| Parameter | Default | LLM tuning range |
|-----------|---------|-----------------|
| `vif_threshold` | 10.0 | 5 – 25 |
| `k_range` | `[3,4,5,6,7,8,10,12,15]` | any subset of k ∈ [2,20] |
| `algorithm` | null (auto) | `"kmeans"`, `"hierarchical"`, `"dbscan"`, `"gmm"`, `"fuzzy_cmeans"`, `null` |
| `min_silhouette` | 0.05 | 0.02 – 0.12 |
| `feature_focus` | `""` | free-text hint injected into FeatureSelector prompt |

### Routing decisions (LLM-assisted)
| Agent reports | Orchestrator action |
|---------------|---------------------|
| `FeatureSelector BLOCKED` | → route to FeatureEngineer (more features needed) |
| `Clusterer WARNING` (low silhouette) | → proceed with warning; tune params |
| `Clusterer BLOCKED` (sil < min_silhouette) | → tune params → route to FeatureSelector |
| `PersonaNamer BLOCKED` (Clarity Gate fail) | → tune params → route to Clusterer |
| `Classifier BLOCKED` (F1 < 0.70) | → tune params → route to FeatureSelector or Clusterer |
| Any `recommendation=escalate` | → trigger human checkpoint immediately |
| Max iterations reached, no approved result | → best-effort fallback (force_proceed=True) |

---

## UserInputAgent

**File**: `agents/user_input.py`
**Class**: `UserInputAgent`

### Role
Collects and validates the user's clustering intent before any computation.
This is the entry point of the pipeline.

### Skills used
- (none from `skills/` — pure interactive I/O)

### Inputs
- None (interactive terminal)

### Outputs
- `UserIntent` dataclass:
  - `target_entity: str` — what is being clustered
  - `business_purpose: str` — why we are clustering
  - `dataset_path: str` — path to raw data file
  - `constraints: str` — optional free-text constraints
  - `n_clusters_requested: int | None` — if set, ClusteringAgent uses this exact k (skips silhouette optimisation)
  - `must_have_clusters: list[str]` — cluster labels that MUST appear in the final personas (e.g. `['traveller', 'VIP']`); enforced by PersonaNamingAgent Clarity Gate

### Questions asked (interactive)
1. "What entity are you clustering?" — `target_entity`
2. "What is the business purpose?" — `business_purpose` (follow-up if < 20 chars)
3. "Dataset path?" — `dataset_path`
4. "Any constraints?" — `constraints`
5. "How many clusters? (Enter = data-driven)" — `n_clusters_requested`
6. "Must any specific types appear as clusters? (Enter = none)" — `must_have_clusters` (comma-separated)

### Communication protocol
```json
{
  "agent": "UserInput",
  "status": "success | blocked",
  "what_was_done": "Collected target entity and business purpose from user",
  "what_was_not_done": "",
  "doubts": "Business purpose may still be too vague",
  "issues": [],
  "metrics": {
    "n_clusters_requested": 5,
    "must_have_clusters": ["traveller", "VIP"]
  },
  "recommendation": "proceed"
}
```

### Retry behaviour
- If user gives a vague business purpose (< 20 chars), agent asks one
  clarifying follow-up question before proceeding.
- If running non-interactively (EOFError), agent uses sensible defaults from
  `config.yaml` and reports `doubts="running with defaults"`.

---

## DatasetExaminerAgent

**File**: `agents/dataset_examiner.py`
**Class**: `DatasetExaminerAgent`

### Role
Profiles the raw dataset and identifies feature engineering opportunities
aligned with the stated business purpose. Uses LLM with the schema and
business purpose to get suggested feature groups.

### Skills used
- `orchestrator_bus.report()`

### Inputs
- `user_intent: UserIntent`
- Raw DataFrame (loaded from `user_intent.dataset_path`)

### README.md awareness
If the dataset folder contains a `README.md` (e.g. `data/raw/air_quality/README.md`),
the agent reads it automatically and includes its content in:
1. The LLM prompt for feature group suggestions (so the LLM can use domain context from the data provider)
2. `DatasetProfile.dataset_readme` — passed downstream to `FeatureEngineerAgent` and `FeatureSelectionAgent`

The README text is capped at 3 000 chars to keep prompts manageable. If no README exists, the field is `""`.

### Outputs
- `DatasetProfile` dataclass:
  - `n_rows: int`, `n_cols: int`
  - `column_types: dict[str, str]`
  - `missing_rates: dict[str, float]`
  - `distribution_summary: dict[str, dict]` (skewness, kurtosis, min/max/mean)
  - `suggested_feature_groups: list[str]` (from LLM)
  - `warnings: list[str]`
  - `dataset_readme: str` — full text of `README.md` from the dataset folder (empty string if absent)

### Communication protocol
```json
{
  "agent": "DatasetExaminer",
  "status": "success | warning | blocked",
  "what_was_done": "Profiled schema, analysed distributions, called LLM for feature group suggestions",
  "what_was_not_done": "Did not load data subsets for validation",
  "doubts": "A high-cardinality column may need grouping before feature engineering",
  "issues": ["Column 'age' missing in 15% of rows"],
  "metrics": { "n_rows": 10000, "n_cols": 25, "n_suggested_groups": 5, "has_readme": true },
  "recommendation": "proceed"
}
```

### Failure modes
| Issue | Status | Recommendation |
|-------|--------|----------------|
| Dataset not found | `blocked` | `escalate` |
| No numeric columns | `blocked` | `escalate` |
| > 30% missing in key columns | `warning` | `proceed` (with imputation note) |
| All columns constant | `blocked` | `escalate` |

---

## FeatureEngineerAgent

**File**: `agents/feature_engineer.py`
**Class**: `FeatureEngineerAgent`

### Role
Builds an entity-level feature matrix from raw event-level data. The LLM
(via OrchestratorBus) reads the actual dataset schema and business purpose,
then plans which of 8 generic statistical operations to apply to which columns.
No domain vocabulary is hard-coded — the same agent handles transaction logs,
product catalogs, patient visits, sensor readings, or any other tabular event data.

### Skills used
- `orchestrator_bus.ask()`, `orchestrator_bus.report()`

### Inputs
- Raw DataFrame (event-level)
- `DatasetProfile` — schema, suggested feature groups, algo hint
- `UserIntent` — target entity, business purpose
- Orchestrator feedback (free-text, injected into LLM prompt on retry)

### Outputs
- Engineered feature DataFrame (entity-level, persisted to `data/processed/`)
- `FeatureEngineeringResult`:
  - `n_entities: int`
  - `n_features: int`
  - `feature_names: list[str]`
  - `groups_built: list[str]`
  - `output_path: str`
  - `reasoning: str`

### Column auto-detection

Before building features, the agent auto-detects four structural columns from
the schema using exact match first, then case-insensitive substring match:

| Role | What it identifies | Example column names detected |
|------|-------------------|-------------------------------|
| **entity / ID** | The column that identifies each entity being clustered (required) | `id`, `user_id`, `customer_id`, `patient_id`, `device_id`, `sensor_id`, `uuid`, … |
| **timestamp / date** | When each event occurred (optional) | `timestamp`, `date`, `time`, `ts`, `created_at`, `event_time`, `visit_date`, … |
| **value / amount** | The primary numeric measure per event (optional) | `amount`, `value`, `price`, `cost`, `qty`, `score`, `duration`, `reading`, … |
| **category / kind** | The column that groups events into types (optional) | `category`, `type`, `kind`, `label`, `class`, `group`, `tag`, `genre`, `department`, `sector`, `channel`, … |

**If auto-detection fails for any column, the agent asks the user interactively** —
it prints all available column names and waits for input. Required columns
(entity/ID) must be provided; optional columns can be skipped by pressing Enter.

### The 8 generic builders

| Builder | What it computes | Example column names |
|---------|-----------------|---------------------|
| `group_aggregate` | count/sum/mean/std/max/freq/pct_count/pct_sum per group value × window | `count_{col}_{val}_{w}` |
| `group_trend` | change in count or sum between two windows | `trend_count_{col}_{val}` |
| `group_streak` | consecutive active periods per group value | `streak_{col}_{val}` |
| `overall_aggregate` | aggregate over all events (no grouping) | `sum_{val_col}_{w}` |
| `frequency_recency` | event frequency, active periods, recency, gap | `event_count_{w}`, `days_since_last` |
| `entity_diversity` | number of unique values per column × window | `n_unique_{col}_{w}` |
| `temporal_patterns` | morning/evening/weekend ratios, peak hour | `pct_morning_{w}`, `pct_weekend_{w}` |
| `static_attributes` | copy entity-level static columns as-is | original column name |

The LLM is shown the actual column names from the dataset schema and chooses
which builders to apply to which columns. Feature column names embed the actual
data column names, not domain abbreviations.

### Communication protocol
```json
{
  "agent": "FeatureEngineer",
  "status": "success | warning | blocked",
  "what_was_done": "Built 108 features across 6 behavioral groups from LLM plan",
  "what_was_not_done": "Could not build temporal features (no timestamp column found)",
  "doubts": "Frequency features may overlap with diversity features",
  "issues": [],
  "metrics": { "n_features": 108, "n_entities": 983, "n_groups": 6 },
  "recommendation": "proceed"
}
```

### Failure modes
| Issue | Status | Recommendation |
|-------|--------|----------------|
| Entity/ID column not auto-detected | — | Agent asks user interactively (required) |
| Timestamp/value/category column not auto-detected | — | Agent asks user; pressing Enter skips that role |
| Fewer than 20 features built | `blocked` | `escalate` |
| All features are binary/constant | `blocked` | `escalate` |

---

## FeatureSelectionAgent

**File**: `agents/feature_selector.py`
**Class**: `FeatureSelectionAgent`

### Role
Scores all engineered features using PCA importance and autoencoder
reconstruction error, then applies VIF and correlation gates, and finally
asks LLM to select the optimal subset for clustering.

### Skills used
- `skills.vif_checker.compute_vif()`
- `skills.vif_checker.remove_high_vif()`
- `skills.vif_checker.flag_high_correlation()`
- `orchestrator_bus.report()`

### Inputs
- Engineered feature DataFrame
- `UserIntent`
- Orchestrator feedback (free-text)

### Outputs
- `FeatureSelectionResult`:
  - `selected_features: list[str]`
  - `n_features: int`
  - `pca_scores: dict[str, float]`
  - `ae_scores: dict[str, float]`
  - `vif_table: dict[str, float]`
  - `removed_by_vif: list[str]`
  - `reasoning: str`

### Quality gates (in order)
1. VIF < 10 for all features (iterative removal; threshold tuned dynamically by Orchestrator)
2. |r| < 0.85 between any two features
3. ≥ 10 features survive; ≤ 80 features retained

### Communication protocol
```json
{
  "agent": "FeatureSelector",
  "status": "success | warning | blocked",
  "what_was_done": "Scored 108 features, removed 22 via VIF gate, LLM selected 45",
  "what_was_not_done": "Did not apply p-value gate (sufficient features after VIF)",
  "doubts": "Several features have borderline VIF (~9.5)",
  "issues": [],
  "metrics": { "n_input": 108, "n_after_vif": 86, "n_selected": 45, "max_vif": 8.2 },
  "recommendation": "proceed"
}
```

### Failure modes
| Issue | Status | Recommendation |
|-------|--------|----------------|
| < 10 features survive VIF gate | `blocked` | `retry` (relax to VIF < 15) or `escalate` |
| LLM returns invalid JSON | `warning` | `retry` |

---

## ClusteringAgent

**File**: `agents/clusterer.py`
**Class**: `ClusteringAgent`

### Role
Selects the clustering algorithm and number of clusters data-driven
(via silhouette optimisation), fits the model, runs the deepening loop
for oversized clusters, and builds cluster profiles.

### Skills used
- `skills.algo_recommender.recommend_algorithm()`
- `skills.silhouette_optimizer.optimize_k()`
- `orchestrator_bus.report()`

### Inputs
- Engineered feature DataFrame
- `selected_features: list[str]`
- `UserIntent`
- `DatasetProfile` (for algorithm recommendation)
- Orchestrator feedback

### Outputs
- `ClusteringResult`:
  - `action: str` (proceed | reselect_features)
  - `cluster_labels: pd.Series`
  - `profiles: dict`
  - `lineage: dict`
  - `silhouette: float`
  - `n_leaf: int`
  - `k_scores: dict[int, float]` (silhouette for each k tried)
  - `algo_used: str`

### Cluster profile structure
Each cluster entry in `profiles` contains:
```json
{
  "n_entities": 142,
  "pct_total": 0.144,
  "top_above_average": [["feature_name", 3.2], ...],
  "top_below_average": [["feature_name", 0.4], ...],
  "feature_means": {"feature_name": 0.72, ...},
  "feature_relative": {"feature_name": 1.6, ...}
}
```

### Algorithm selection logic
Delegated to `skills.algo_recommender`. Scores all 5 supported algorithms
based on dataset characteristics; picks the highest-scoring one:

| Algorithm | Best for |
|-----------|----------|
| `kmeans` | Large datasets (> 100k rows), fast, convex clusters |
| `hierarchical` | Nested sub-groups, non-convex shapes, dendrogram insight |
| `dbscan` | Arbitrary shapes, noise/outlier detection |
| `gmm` | Soft, probabilistic cluster memberships |
| `fuzzy_cmeans` | Overlapping clusters where entities belong to multiple groups |

### K selection logic
Priority order:
1. **`user_intent.n_clusters_requested`** — if the user specified an exact count at Q5, use it directly (silhouette optimisation is skipped)
2. **`config.n_clusters`** — if set in `config.yaml`, use it directly
3. **Silhouette optimisation** (default) — delegated to `skills.silhouette_optimizer`:
   - Try k ∈ `k_search_range` (default: [3,4,5,6,7,8,10,12,15])
   - Fit model for each k; compute silhouette
   - Pick k with maximum silhouette; report curve to orchestrator

### Communication protocol
```json
{
  "agent": "Clusterer",
  "status": "success | warning | blocked",
  "what_was_done": "Tried k=3..15 via silhouette; selected k=7 (sil=0.38); ran deepening loop",
  "what_was_not_done": "Deepening loop skipped (no oversized clusters)",
  "doubts": "Silhouette improvement from k=6 to k=7 is marginal (0.01 diff)",
  "issues": [],
  "metrics": { "best_k": 7, "silhouette": 0.38, "n_leaf": 9, "algo": "hierarchical" },
  "recommendation": "proceed"
}
```

### Failure modes
| Issue | Status | Recommendation |
|-------|--------|----------------|
| Silhouette < 0.20 for all k | `warning` | `retry` with different algo |
| Silhouette < 0.15 after all algorithms tried | `blocked` | `reselect_features` |
| Deepening loop can't resolve oversized cluster | `warning` | `reselect_features` |

---

## PersonaNamingAgent

**File**: `agents/persona_namer.py`
**Class**: `PersonaNamingAgent`

### Role
Uses LLM to generate human-readable persona names, taglines, descriptions,
and traits from cluster profiles. Applies the Clarity Gate to validate
output quality before proceeding.

### Skills used
- `orchestrator_bus.report()`

### Inputs
- `profiles: dict` (from ClusteringAgent — uses `n_entities`, `pct_total`,
  `top_above_average`, `top_below_average`, `feature_means`)
- `lineage: dict`
- `tone: str`
- `UserIntent`
- Orchestrator feedback

### Outputs
- `NamingResult`:
  - `personas: dict` (cid → name, tagline, description, traits, confidence)
  - `passed: bool`
  - `avg_confidence: float`
  - `issues: list[str]`

### Inputs
- `profiles: dict` (from ClusteringAgent)
- `lineage: dict`
- `tone: str`
- `user_intent: UserIntent` — used to inject `must_have_clusters` constraint into the LLM prompt and Clarity Gate
- Orchestrator feedback

### Clarity Gate thresholds
1. Avg confidence ≥ 6/10
2. No duplicate persona names
3. **Must-have clusters covered** — if `user_intent.must_have_clusters` is non-empty, every required type must appear in at least one persona name or description. Failure triggers `recluster`.

### Communication protocol
```json
{
  "agent": "PersonaNamer",
  "status": "success | warning | blocked",
  "what_was_done": "Named 9 clusters; avg confidence=7.2; Clarity Gate passed",
  "what_was_not_done": "",
  "doubts": "Cluster 4 and 7 are similar; confidence 6/10 for both",
  "issues": [],
  "metrics": { "n_clusters": 9, "avg_confidence": 7.2, "gate_passed": true },
  "recommendation": "proceed"
}
```

---

## ClassifierAgent

**File**: `agents/classifier.py`
**Class**: `ClassifierAgent`

### Role
Treats persona labels as pseudo ground truth, trains an LLM-selected
classifier, evaluates cluster separability via stratified CV, and routes
the pipeline back to feature selection or clustering if performance is poor.

### Skills used
- `orchestrator_bus.report()`

### Inputs
- Feature DataFrame
- `cluster_labels: pd.Series`
- `personas: dict`
- `UserIntent`
- History + feedback

### Outputs
- `ClassifierResult`:
  - `action: str` (proceed | reselect_features | recluster)
  - `cv_accuracy, cv_f1_macro, cv_f1_weighted: float`
  - `per_class_f1: dict[str, float]`
  - `feature_importances: dict[str, float]`
  - `reasoning: str`

### LLM-selected classifier models
The LLM chooses the best model type based on dataset size, class balance,
and interpretability requirements:

| Model | When preferred |
|-------|---------------|
| `random_forest` | Medium datasets, need feature importances |
| `xgboost` | Large datasets, best accuracy |
| `gradient_boosting` | Balanced datasets, robust to outliers |
| `logistic_regression` | Small datasets, high interpretability |

### Quality gate
- CV macro-F1 ≥ 0.70 → proceed
- Below threshold → LLM diagnoses and routes

### Communication protocol
```json
{
  "agent": "Classifier",
  "status": "success | warning | blocked",
  "what_was_done": "LLM selected random_forest; trained with 5-fold CV; computed feature importances",
  "what_was_not_done": "Did not compute SHAP values (not in current skill set)",
  "doubts": "Persona 'Moderate All-Rounder' is borderline (F1=0.65)",
  "issues": [],
  "metrics": { "cv_f1_macro": 0.82, "cv_accuracy": 0.85, "n_classes": 9 },
  "recommendation": "proceed"
}
```

---

## Agent Interaction Diagram

```
UserInputAgent
    │ UserIntent
    ▼
DatasetExaminerAgent
    │ DatasetProfile
    ▼
FeatureEngineerAgent
    │ engineered DataFrame
    ▼
FeatureSelectionAgent ◄───────────────────────────────────┐
    │ selected_features                                    │
    ▼                                              (reselect_features)
ClusteringAgent ◄────────────────────────────┐            │
    │ cluster_labels, profiles                │            │
    ▼                                  (recluster)        │
PersonaNamingAgent ──(gate fail)──────────────┘            │
    │ personas                                             │
    ▼                                                      │
ClassifierAgent ──(low F1)────────────────────────────────┘
    │ ClassifierResult
    ▼
Orchestrator Human Checkpoint
    │
    └── approve → save outputs → DONE
```

---

## Adding a New Agent

1. Create `agents/<name>.py` with a class inheriting no base class (duck-typed)
2. Implement `run(self, ...) -> <ResultDataclass>`
3. Call `bus.report(OrchestratorMessage(...))` at the end of `run()`
4. Add the result dataclass to `agents/state.py`
5. Register the agent in this file (`agent.md`)
6. Register any new skills in `skill.md`
7. Wire the agent into `agents/orchestrator.py`
