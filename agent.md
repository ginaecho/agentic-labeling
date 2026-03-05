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
maintains the orchestrator message log, and calls Claude to diagnose complex
failures. Presents a human checkpoint when the pipeline converges or exhausts
its retry budget.

### Inputs
- `config: dict` (from `config.yaml`)
- `user_intent: UserIntent`
- `features_path: str`

### Outputs
- `dict` with keys `status`, `personas`, `run_history`, `timing`, `claude_usage`

### Responsibilities
1. Receive `OrchestratorMessage` from every agent via `orchestrator_bus`
2. Log all messages to `pipeline_log` (saved to `outputs/pipeline_log.json`)
3. Use Claude to analyse failure reports and decide routing
4. After each failed iteration, call `_ask_parameter_tuning()` to let Claude propose
   new values for `vif_threshold`, `k_range`, `algorithm`, `min_silhouette`, and
   `feature_focus` — these are passed directly to FeatureSelector and Clusterer
5. Enforce per-loop retry budgets (default `max_total_iterations=10`)
6. At max iterations with no approved result, deliver a best-effort analysis by
   running PersonaNamer (force_proceed=True) and Classifier on the highest-silhouette
   clustering observed across all iterations
7. Present human checkpoint with full pipeline log summary

### Dynamic tuning parameters (managed by Orchestrator, not config.yaml)
| Parameter | Default | Claude's tuning range |
|-----------|---------|----------------------|
| `vif_threshold` | 10.0 | 5 – 25 |
| `k_range` | `[3,4,5,6,7,8,10,12,15]` | any subset of k ∈ [2,20] |
| `algorithm` | null (auto) | `"kmeans"`, `"hierarchical"`, `null` |
| `min_silhouette` | 0.05 | 0.02 – 0.12 |
| `feature_focus` | `""` | free-text hint injected into FeatureSelector prompt |

### Routing decisions (Claude-assisted)
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

### Communication protocol
```json
{
  "agent": "UserInput",
  "status": "success | blocked",
  "what_was_done": "Collected target entity and business purpose from user",
  "what_was_not_done": "",
  "doubts": "Business purpose may still be too vague",
  "issues": [],
  "metrics": {},
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
aligned with the stated business purpose. Calls Claude with the schema +
business purpose to get suggested feature groups.

### Skills used
- `orchestrator_bus.report()`

### Inputs
- `user_intent: UserIntent`
- Raw DataFrame (loaded from `user_intent.dataset_path`)

### Outputs
- `DatasetProfile` dataclass:
  - `n_rows: int`, `n_cols: int`
  - `column_types: dict[str, str]`
  - `missing_rates: dict[str, float]`
  - `distribution_summary: dict[str, dict]` (skewness, kurtosis, min/max/mean)
  - `suggested_feature_groups: list[str]` (from Claude)
  - `warnings: list[str]`

### Communication protocol
```json
{
  "agent": "DatasetExaminer",
  "status": "success | warning | blocked",
  "what_was_done": "Profiled schema, analysed distributions, called Claude for feature group suggestions",
  "what_was_not_done": "Did not load data subsets for validation",
  "doubts": "Column 'merchant_category' has 300+ unique values — may need grouping",
  "issues": ["Column 'age' missing in 15% of rows"],
  "metrics": { "n_rows": 10000, "n_cols": 25, "n_suggested_groups": 5 },
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
Builds a rich feature matrix from raw data, guided by the `DatasetProfile`
and business purpose. Uses Claude to decide which transformations to apply.

### Skills used
- `orchestrator_bus.report()`

### Inputs
- Raw DataFrame
- `DatasetProfile`
- `UserIntent`
- Orchestrator feedback (free-text, from previous iteration if any)

### Outputs
- Engineered feature `DataFrame` (persisted to `data/processed/`)
- `FeatureEngineeringResult` dataclass with feature count and group coverage

### Communication protocol
```json
{
  "agent": "FeatureEngineer",
  "status": "success | warning | blocked",
  "what_was_done": "Built 108 features across 6 behavioral groups",
  "what_was_not_done": "Could not build loyalty/tenure features (no signup date column)",
  "doubts": "Recency features may overlap heavily with frequency features",
  "issues": [],
  "metrics": { "n_features": 108, "n_groups": 6, "missing_groups": ["loyalty"] },
  "recommendation": "proceed"
}
```

### Failure modes
| Issue | Status | Recommendation |
|-------|--------|----------------|
| Required columns missing | `warning` | `proceed` (fewer feature groups) |
| Fewer than 20 features built | `blocked` | `escalate` |
| All features are binary/constant | `blocked` | `escalate` |

---

## FeatureSelectionAgent

**File**: `agents/feature_selector.py`
**Class**: `FeatureSelectionAgent`

### Role
Scores all engineered features using PCA importance and autoencoder
reconstruction error, then applies VIF and correlation gates, and finally
asks Claude to select the optimal subset for clustering.

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
  - `reasoning: str`

### Quality gates (in order)
1. VIF < 5 for all features (iterative removal)
2. |r| < 0.85 between any two features
3. ≥ 10 features survive; ≤ 80 features retained

### Communication protocol
```json
{
  "agent": "FeatureSelector",
  "status": "success | warning | blocked",
  "what_was_done": "Scored 108 features, removed 22 via VIF gate, Claude selected 45",
  "what_was_not_done": "Did not apply p-value gate (sufficient features after VIF)",
  "doubts": "Several travel features have borderline VIF (~4.8)",
  "issues": [],
  "metrics": { "n_input": 108, "n_after_vif": 86, "n_selected": 45, "max_vif": 4.2 },
  "recommendation": "proceed"
}
```

### Failure modes
| Issue | Status | Recommendation |
|-------|--------|----------------|
| < 10 features survive VIF gate | `blocked` | `retry` (relax to VIF < 8) or `escalate` |
| Claude returns invalid JSON | `warning` | `retry` |

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

### Algorithm selection logic
Delegated to `skills.algo_recommender`. Factors considered:
- `n_rows` (> 100k → prefer K-Means)
- Mean skewness of selected features
- Multi-modality detected by DatasetExaminer

### K selection logic
Delegated to `skills.silhouette_optimizer`. Steps:
1. Try k ∈ config `k_search_range` (default: [3,4,5,6,7,8,10,12,15])
2. Fit model for each k; compute silhouette
3. Pick k with maximum silhouette; report curve to orchestrator

### Communication protocol
```json
{
  "agent": "Clusterer",
  "status": "success | warning | blocked",
  "what_was_done": "Tried k=3..15 via silhouette; selected k=7 (sil=0.38); ran deepening loop",
  "what_was_not_done": "Did not try DBSCAN (not in skill repertoire)",
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
| Silhouette < 0.15 after both algos | `blocked` | `reselect_features` |
| Deepening loop can't resolve oversized cluster | `warning` | `reselect_features` |

---

## PersonaNamingAgent

**File**: `agents/persona_namer.py`
**Class**: `PersonaNamingAgent`

### Role
Sends cluster profiles to Claude to generate human-readable persona names,
taglines, descriptions, and traits. Applies the Clarity Gate to validate
output quality before proceeding.

### Skills used
- `orchestrator_bus.report()`

### Inputs
- `profiles: dict` (from ClusteringAgent)
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

### Clarity Gate thresholds
- Avg confidence ≥ 6/10
- No duplicate persona names
- Every persona description references ≥ 2 quantitative signals (checked by regex)

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
Treats persona labels as pseudo ground truth, trains a Random Forest
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

### Quality gate
- CV macro-F1 ≥ 0.70 → proceed
- Below threshold → Claude diagnoses and routes

### Communication protocol
```json
{
  "agent": "Classifier",
  "status": "success | warning | blocked",
  "what_was_done": "Trained RF, 5-fold CV, computed feature importances",
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
