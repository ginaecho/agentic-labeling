# Agentic Clustering Pipeline — Master Plan

## Vision

An intelligent, self-correcting multi-agent system that autonomously segments any
dataset by first asking the user for business intent, then engineering the right
features, selecting an optimal cluster count, and labelling each cluster with a
human-readable persona — while every agent reports its status, doubts, and blockers
to an orchestrator at every step.

---

## Core Principles

### P1 — Intent First

Before any computation, the pipeline collects six inputs from the user:

1. **Target entity**: what are we clustering? (e.g. customers, products, employees)
2. **Business purpose**: why are we clustering?
   (e.g. "understand shopping behaviour to personalise offers")
3. **Dataset path**: path to raw data file (defaults to config)
4. **Constraints**: optional filters (e.g. "only last 12 months")
5. **Cluster count**: exact number of clusters, or blank for data-driven selection
6. **Must-have cluster types**: comma-separated labels that MUST appear as distinct personas
   (e.g. "traveller, high-value-customer") — enforced by the PersonaNaming Clarity Gate

All downstream agents shape their decisions — which features to build, which
algorithm to select, what makes a "good" cluster — around these inputs.
The intent is stored in `UserIntent` and passed through every agent's context.

---

### P2 — Feature Quality Over Quantity

Features fed into clustering must satisfy all three criteria:

| Criterion | Threshold | Enforced by |
|-----------|-----------|-------------|
| Low multicollinearity | VIF < 10 for all retained features (dynamically tuned) | `vif_checker` skill |
| Low pairwise correlation | \|r\| < 0.85 between any two features | `vif_checker` skill |
| Minimum coverage | ≥ 10 features survive all gates | `vif_checker` skill |

Agents additionally use PCA importance and autoencoder reconstruction error
to rank features before the VIF gate. The orchestrator routes back to feature
engineering if fewer than 10 features survive all gates.

---

### P3 — Cluster Count Selection

The number of clusters `k` is resolved in priority order:

| Priority | Source | How |
|----------|--------|-----|
| 1 (highest) | User interactive input (Q5) | `user_intent.n_clusters_requested` — exact k, silhouette optimisation skipped |
| 2 | `config.yaml` `n_clusters` | Fixed override in config |
| 3 (default) | Data-driven silhouette | Try k ∈ [3,4,5,6,7,8,10,12,15]; pick highest silhouette |

The user can request an exact count at startup. When they do, the ClusteringAgent
uses it directly and skips the silhouette search.
The chosen `k` and its silhouette score are always reported to the orchestrator.

---

### P4 — Algorithm Selection Based on Data Shape

The clustering algorithm is chosen by the AlgoRecommender skill, which scores
all 5 supported algorithms based on dataset characteristics and picks the best:

| Condition | Preferred algorithm |
|-----------|---------------------|
| Dataset > 100 k rows | K-Means (speed) |
| Key features are multi-modal or nested | Hierarchical / Ward |
| Business purpose mentions overlapping groups | Fuzzy C-Means |
| Business purpose mentions outlier/noise detection | DBSCAN |
| Business purpose mentions soft/probabilistic assignments | GMM |
| Default | Hierarchical / Ward |

The DatasetExaminerAgent analyses feature distributions and reports its
findings so the AlgoRecommender can make an informed choice.

---

### P5 — Cluster Quality Gates

Every clustering result must pass three sequential gates before proceeding
to persona labelling:

| Gate | Condition | Failure action |
|------|-----------|----------------|
| **Size Gate** | No cluster > 40 % of data | Run deepening loop; if still failing → re-cluster |
| **Silhouette Gate** | Silhouette ≥ 0.25 | Re-cluster with different k or algorithm |
| **Distinguishability Gate** | Avg LLM confidence ≥ 6/10 AND no duplicate names | Re-cluster |

If any gate fails, the agent sends a structured failure report to the
orchestrator. The orchestrator analyses the report with LLM reasoning and
routes the pipeline back to the appropriate step.

---

### P6 — Agents Always Report to the Orchestrator

Every agent sends a structured `OrchestratorMessage` at the end of each run.
The message contains:

```json
{
  "agent": "AgentName",
  "iteration": 1,
  "status": "success | warning | blocked | failure",
  "what_was_done": "...",
  "what_was_not_done": "...",
  "doubts": "...",
  "issues": ["specific issue 1", "specific issue 2"],
  "metrics": { "key": "value" },
  "recommendation": "proceed | retry | escalate",
  "context": { "agent-specific payload" }
}
```

The orchestrator logs all messages, passes the log to LLM when making
routing decisions, and prints a human-readable summary at each human
checkpoint.

---

### P7 — Minimal Human Intervention

The system resolves issues autonomously through orchestrator routing. A human
checkpoint is only triggered when:

- All agents report `status=success` and agree to proceed, **OR**
- The pipeline has exhausted its retry budget (`max_total_iterations`), **OR**
- An agent reports `recommendation=escalate` (a hard block: missing data,
  incompatible schema, irrecoverable error)

At the human checkpoint, the user sees the full persona table, silhouette
scores, classifier F1, and the orchestrator's pipeline log, then chooses:
approve / re-cluster / re-select features / quit.

---

### P8 — Modular Skills

Each agent is composed of **skills** — atomic, testable, reusable Python
functions. Skills are catalogued in `skill.md` and implemented in `skills/`.
Agents select which skills to invoke based on business context and data shape.
New skills can be added to `skills/` and registered in `skill.md` without
modifying any agent class.

---

### P9 — Domain-Agnostic Feature Engineering

The FeatureEngineerAgent applies 8 generic statistical builders to whatever
columns exist in the dataset. The LLM reads the actual column names and schema,
then decides which builders to apply to which columns. No domain vocabulary
(banking, healthcare, retail, etc.) is hard-coded. The same agent handles
transaction logs, patient visits, sensor readings, product catalogs, or any
other tabular event data.

When the agent cannot automatically detect structural columns (entity ID,
timestamp, value, category), it asks the user interactively rather than
failing silently.

---

## Pipeline Steps

```
UserInput → DataExaminer → FeatureEngineer → FeatureSelector → Clusterer
    ↑             ↑                ↑                 ↑              |
    |             |                |                 |              ↓
    |             |                |           ←─ Clusterer says "need better features"
    |             |          ←──────────── FeatureSelector says "need more raw features"
    |        ←──────────────────────────── DataExaminer finds missing feature opportunity
    |
Clusterer → PersonaNamer → Classifier → Human
    ↑              |              |
    ← Gate fail ←──┘              |
                   Clarity     F1 too low → LLM routes:
                   Gate fail      ↓              ↓
                             Recluster    Reselect or Recluster
```

---

### Step 1 — User Intent Collection (`UserInputAgent`)

| | |
|---|---|
| **Inputs** | None (interactive prompts) |
| **Outputs** | `UserIntent` (target_entity, business_purpose, dataset_path, constraints, n_clusters_requested, must_have_clusters) |
| **Skills** | `collect_intent`, `validate_intent` |
| **Reports** | Always — confirms captured intent or flags ambiguity |

Questions asked:
1. "What entity are you clustering?" (customers / products / employees / other)
2. "What is the business purpose of this clustering?" (follow-up if < 20 chars)
3. "Where is your dataset?" (file path or uses default)
4. "Any constraints?" (e.g. "ignore outlier events", "only last 12 months")
5. "How many clusters? (Enter = data-driven)" → sets `n_clusters_requested`
6. "Must any specific types appear as clusters?" (comma-separated, e.g. "traveller, VIP") → sets `must_have_clusters`

The agent validates that the answers are specific enough. If the business purpose
is vague, it asks a follow-up clarifying question before proceeding.

---

### Step 2 — Dataset Examination (`DatasetExaminerAgent`)

| | |
|---|---|
| **Inputs** | `UserIntent`, raw dataset path |
| **Outputs** | `DatasetProfile` (schema, distributions, missing rates, suggested feature groups, `dataset_readme`) |
| **Skills** | `profile_schema`, `analyse_distributions`, `suggest_feature_groups` |
| **Orchestrator status** | SUCCESS / WARNING / BLOCKED |

The agent:
1. **Checks for `README.md`** in the dataset's folder (e.g. `data/raw/air_quality/README.md`).
   If found, its content (capped at 3 000 chars) is stored in `DatasetProfile.dataset_readme`
   and injected into the LLM prompt as domain context from the data provider.
2. Loads the dataset and profiles column types, missing rates, cardinality
3. Analyses distribution shape (skewed, multi-modal, sparse)
4. Calls LLM with the schema + business purpose + README context → LLM suggests which
   column groups to use for feature engineering
5. Reports findings to orchestrator

The `dataset_readme` field in `DatasetProfile` flows through to `FeatureEngineerAgent`
and `FeatureSelectionAgent`, giving both agents the same domain context.

Failure modes:
- WARNING: high missing rate (> 30 %) in key columns → recommend imputation
- BLOCKED: no numeric columns, dataset empty, path not found

---

### Step 3 — Feature Engineering (`FeatureEngineerAgent`)

| | |
|---|---|
| **Inputs** | `DatasetProfile`, `UserIntent`, raw dataset |
| **Outputs** | Engineered feature `DataFrame` saved to `data/processed/` |
| **Skills** | 8 generic builders: `group_aggregate`, `group_trend`, `group_streak`, `overall_aggregate`, `frequency_recency`, `entity_diversity`, `temporal_patterns`, `static_attributes` |
| **Orchestrator status** | SUCCESS / WARNING / BLOCKED |

The agent:
1. Auto-detects structural columns (entity ID, timestamp, value, category);
   asks user interactively if any cannot be determined from the schema
2. Receives feature group suggestions from DatasetExaminer
3. Asks LLM which generic builders to apply to which columns
4. Builds behavioral features aligned with business purpose
5. Validates coverage: every suggested feature group must have ≥ 1 feature built
6. Reports entity count, feature count, group coverage, and any sparse groups

Feedback loop: orchestrator can request additional feature groups based on
business purpose.

---

### Step 4 — Feature Selection (`FeatureSelectionAgent`)

| | |
|---|---|
| **Inputs** | Engineered feature DataFrame, `UserIntent`, orchestrator feedback |
| **Outputs** | Selected feature list, VIF table, PCA/AE scores |
| **Skills** | `score_pca`, `score_autoencoder`, `vif_checker`, `orchestrator_bus.ask()` |
| **Quality gates** | VIF < 10 all features (tunable); \|r\| < 0.85; ≥ 10 features retained |
| **Orchestrator status** | SUCCESS / WARNING / BLOCKED |

Process:
1. Score all features via PCA importance + AE reconstruction error
2. Run VIF analysis; iteratively remove highest-VIF feature until all VIF < threshold
3. Run pairwise correlation check; flag features with \|r\| > 0.85
4. Pass ranked + filtered list to LLM; LLM selects final subset
5. Report selected features, VIF table, and reasoning

Feedback loop: orchestrator provides text guidance that is appended to the
LLM prompt in the next iteration.

---

### Step 5 — Clustering (`ClusteringAgent`)

| | |
|---|---|
| **Inputs** | Selected features, `UserIntent` (incl. `n_clusters_requested`), orchestrator feedback |
| **Outputs** | Cluster labels, profiles, lineage, silhouette scores |
| **Skills** | `recommend_algorithm`, `optimize_k_silhouette`, `cluster`, `deepen_oversized` |
| **Quality gates** | Silhouette ≥ 0.25; no cluster > 40 % |
| **Orchestrator status** | SUCCESS / WARNING / BLOCKED |

Algorithm selection (`recommend_algorithm` skill):
- Scores all 5 algorithms using DatasetProfile + feature stats
- Supported: `kmeans`, `hierarchical`, `dbscan`, `gmm`, `fuzzy_cmeans`
- Reasoning is included in the orchestrator report

K selection (priority order):
1. **`user_intent.n_clusters_requested`** — if set by user at Q5, used directly; silhouette optimisation is skipped
2. **`config.n_clusters`** — if set in config, used directly
3. **`optimize_k_silhouette` skill** (default) — tries k ∈ {3, 4, 5, 6, 7, 8, 10, 12, 15}; picks maximum silhouette; reports full silhouette curve to orchestrator

Deepening loop:
- Any cluster > 40 % is sub-clustered in-place
- LLM decides whether to sub-cluster or request new features

---

### Step 6 — Persona Labelling (`PersonaNamingAgent`)

| | |
|---|---|
| **Inputs** | Cluster profiles, lineage, tone setting, `UserIntent` (incl. `must_have_clusters`) |
| **Outputs** | Persona dict (name, tagline, description, traits, confidence) |
| **Skills** | `build_cluster_prompt`, `call_llm_naming`, `clarity_gate` |
| **Quality gates** | Avg confidence ≥ 6/10; no duplicate names; all must-have cluster types covered |
| **Orchestrator status** | SUCCESS / WARNING / BLOCKED |

**Must-have cluster constraint**: if `user_intent.must_have_clusters` is non-empty,
the LLM prompt contains a MANDATORY section listing the required types. The Clarity Gate
then checks that every required type appears in at least one persona name or description.
If any required type is missing, the gate fails and the pipeline re-clusters.

If gate fails: report to orchestrator with specific issues (which clusters
have low confidence, what makes them ambiguous, which required types are missing).
Orchestrator routes back to clustering with targeted feedback.

---

### Step 7 — Classifier Validation (`ClassifierAgent`)

| | |
|---|---|
| **Inputs** | Feature DataFrame, cluster labels, personas |
| **Outputs** | CV F1 scores, feature importances, routing decision |
| **Skills** | `train_classifier` (LLM-selected model), `evaluate_cv`, `route_failure` |
| **Quality gates** | CV macro-F1 ≥ 0.70 |
| **Orchestrator status** | SUCCESS / WARNING / BLOCKED |

If F1 < 0.70: LLM diagnoses root cause and recommends reselect_features
or recluster. Report includes worst-performing personas and top predictive
features as evidence.

---

### Step 8 — Human Checkpoint (`Orchestrator`)

Displayed to user:
- Silhouette score and k
- CV accuracy and F1 (macro, weighted, per-class)
- Full persona table: name, confidence, cv-F1, tagline
- Top 3 hardest-to-predict personas (bar chart)
- Pipeline log: all agent status messages from this run

User choices:
1. Approve → save all outputs and exit
2. Re-cluster → loop back to Step 5 with feedback
3. Re-select features → loop back to Step 4 with feedback
4. Quit → exit without saving

---

## Feedback Loop Budget

| Loop type | Max automatic retries | Then |
|-----------|----------------------|------|
| Feature selection | 3 | Human checkpoint |
| Clustering | 5 | Human checkpoint |
| Naming (clarity gate) | 3 | Re-cluster |
| Classifier routing | 3 | Human checkpoint |
| Total pipeline | 10 | Save best and exit |

---

## Output Files

| File | Content |
|------|---------|
| `outputs/cluster_profiles.json` | Per-cluster statistics + lineage |
| `outputs/cluster_lineage.json` | Full cluster tree (parent/child) |
| `outputs/personas.json` | Cluster → Persona mapping |
| `outputs/classifier_metrics.json` | CV scores, feature importances |
| `outputs/pipeline_log.json` | Full orchestrator message log |

---

## File Structure

```
agents/
  user_input.py             ← UserInputAgent
  dataset_examiner.py       ← DatasetExaminerAgent
  feature_engineer.py       ← FeatureEngineerAgent (LLM-guided, 8 generic builders)
  feature_selector.py       ← FeatureSelectionAgent (VIF + PCA + AE gates)
  clusterer.py              ← ClusteringAgent (5 algorithms, silhouette k-opt)
  persona_namer.py          ← PersonaNamingAgent (LLM naming + Clarity Gate)
  classifier.py             ← ClassifierAgent (LLM-selected model)
  orchestrator.py           ← Orchestrator (route, tune, LLM diagnose)
  state.py                  ← All result dataclasses

skills/
  __init__.py
  vif_checker.py            ← VIF and correlation analysis
  silhouette_optimizer.py   ← Auto k selection via silhouette
  algo_recommender.py       ← Algorithm recommendation from data shape
  orchestrator_bus.py       ← Agent → orchestrator message protocol

plan.md                     ← THIS FILE (pipeline design & principles)
agent.md                    ← Agent registry: roles, interfaces, protocols
skill.md                    ← Skill registry: descriptions, owners, APIs
config.yaml                 ← Pipeline configuration
```
