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

Before any computation, the pipeline collects two inputs from the user:

1. **Target entity**: what are we clustering? (e.g. customers, products, employees)
2. **Business purpose**: why are we clustering?
   (e.g. "understand shopping behaviour to personalise offers")

All downstream agents shape their decisions — which features to build, which
algorithm to select, what makes a "good" cluster — around these two inputs.
The intent is stored in `UserIntent` and passed through every agent's context.

---

### P2 — Feature Quality Over Quantity

Features fed into clustering must satisfy all three criteria:

| Criterion | Threshold | Enforced by |
|-----------|-----------|-------------|
| Low multicollinearity | VIF < 5 for all retained features | `vif_checker` skill |
| Low pairwise correlation | |r| < 0.85 between any two features | `vif_checker` skill |
| Statistically non-trivial | p-value < 0.05 in univariate F-test | `vif_checker` skill |

Agents can additionally use PCA importance and autoencoder reconstruction error
to rank features before the VIF gate. The orchestrator routes back to feature
engineering if fewer than 10 features survive all gates.

---

### P3 — Data-Driven Cluster Count

The number of clusters `k` is NOT fixed by the user. The ClusteringAgent:

1. Tries `k` from a configurable range (default: 3 – 15)
2. Fits the chosen algorithm for each `k` and computes the silhouette score
3. Selects the `k` that maximises silhouette
4. Falls back to elbow-method if silhouette is flat across the range

The user can override `k` via config, but the default is always data-driven.
The chosen `k` and its silhouette score are reported to the orchestrator.

---

### P4 — Algorithm Selection Based on Data Shape

The clustering algorithm is chosen by the AlgoRecommender skill based on:

| Condition | Preferred algorithm |
|-----------|---------------------|
| Dataset > 100 k rows | K-Means (speed) |
| Key features are multi-modal | Hierarchical / Ward |
| Strong hierarchical groupings suspected | Hierarchical / Ward |
| Sparse, high-dimensional features | Hierarchical / Ward |
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
orchestrator. The orchestrator analyses the report with Claude and routes
the pipeline back to the appropriate step.

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

The orchestrator logs all messages, passes the log to Claude when making
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
                   Clarity     F1 too low → Claude routes:
                   Gate fail      ↓              ↓
                             Recluster    Reselect or Recluster
```

---

### Step 1 — User Intent Collection (`UserInputAgent`)

| | |
|---|---|
| **Inputs** | None (interactive prompts) |
| **Outputs** | `UserIntent` (target_entity, business_purpose, dataset_path, constraints) |
| **Skills** | `collect_intent`, `validate_intent` |
| **Reports** | Always — confirms captured intent or flags ambiguity |

Questions asked:
1. "What entity are you clustering?" (customers / products / employees / other)
2. "What is the business purpose of this clustering?"
3. "Where is your dataset?" (file path or uses default)
4. "Any constraints?" (e.g. "ignore fraud transactions", "only last 12 months")

The agent validates that the answers are specific enough. If the business purpose
is vague, it asks a follow-up clarifying question before proceeding.

---

### Step 2 — Dataset Examination (`DatasetExaminerAgent`)

| | |
|---|---|
| **Inputs** | `UserIntent`, raw dataset path |
| **Outputs** | `DatasetProfile` (schema, distributions, missing rates, suggested feature groups) |
| **Skills** | `profile_schema`, `analyse_distributions`, `suggest_feature_groups` |
| **Orchestrator status** | SUCCESS / WARNING / BLOCKED |

The agent:
1. Loads the dataset and profiles column types, missing rates, cardinality
2. Analyses distribution shape (skewed, multi-modal, sparse)
3. Calls Claude with the schema + business purpose → Claude suggests which
   column groups to use for feature engineering
4. Reports findings to orchestrator

Failure modes:
- WARNING: high missing rate (> 30 %) in key columns → recommend imputation
- BLOCKED: no numeric columns, dataset empty, path not found

---

### Step 3 — Feature Engineering (`FeatureEngineerAgent`)

| | |
|---|---|
| **Inputs** | `DatasetProfile`, `UserIntent`, raw dataset |
| **Outputs** | Engineered feature `DataFrame` saved to `data/processed/` |
| **Skills** | `build_frequency_features`, `build_spend_features`, `build_recency_features`, `build_interaction_features`, `validate_feature_coverage` |
| **Orchestrator status** | SUCCESS / WARNING / BLOCKED |

The agent:
1. Receives feature group suggestions from DatasetExaminer
2. Builds behavioral features aligned with business purpose
3. Validates coverage: every suggested feature group must have ≥ 1 feature built
4. Reports feature count, group coverage, and any sparse groups

Feedback loop: orchestrator can request additional feature groups based on
business purpose (e.g. "also build recency features if not already included").

---

### Step 4 — Feature Selection (`FeatureSelectionAgent`)

| | |
|---|---|
| **Inputs** | Engineered feature DataFrame, `UserIntent`, orchestrator feedback |
| **Outputs** | Selected feature list, VIF table, PCA/AE scores |
| **Skills** | `score_pca`, `score_autoencoder`, `check_vif`, `check_pvalue`, `select_with_llm` |
| **Quality gates** | VIF < 5 all features; \|r\| < 0.85; ≥ 10 features retained |
| **Orchestrator status** | SUCCESS / WARNING / BLOCKED |

Process:
1. Score all features via PCA importance + AE reconstruction error
2. Run VIF analysis; iteratively remove highest-VIF feature until all VIF < 5
3. Run pairwise correlation check; flag or remove features with \|r\| > 0.85
4. Pass ranked + filtered list to Claude; Claude selects final subset
5. Report selected features, VIF table, and reasoning

Feedback loop: orchestrator provides text guidance that is appended to the
Claude prompt in the next iteration.

---

### Step 5 — Clustering (`ClusteringAgent`)

| | |
|---|---|
| **Inputs** | Selected features, `UserIntent`, orchestrator feedback |
| **Outputs** | Cluster labels, profiles, lineage, silhouette scores |
| **Skills** | `recommend_algorithm`, `optimize_k_silhouette`, `cluster`, `deepen_oversized` |
| **Quality gates** | Silhouette ≥ 0.25; no cluster > 40 % |
| **Orchestrator status** | SUCCESS / WARNING / BLOCKED |

Algorithm selection (`recommend_algorithm` skill):
- Uses DatasetProfile + feature stats to choose K-Means or Hierarchical
- Reasoning is included in the orchestrator report

K selection (`optimize_k_silhouette` skill):
- Tries k ∈ {3, 4, 5, 6, 7, 8, 10, 12, 15} by default
- Computes silhouette for each; picks the maximum
- Reports the full silhouette curve to orchestrator

Deepening loop (same as existing notebook 03 logic):
- Any cluster > 40 % is sub-clustered in-place
- Claude decides whether to sub-cluster or request new features

---

### Step 6 — Persona Labelling (`PersonaNamingAgent`)

| | |
|---|---|
| **Inputs** | Cluster profiles, lineage, tone setting |
| **Outputs** | Persona dict (name, tagline, description, traits, confidence) |
| **Skills** | `build_cluster_prompt`, `call_llm_naming`, `clarity_gate` |
| **Quality gates** | Avg confidence ≥ 6/10; no duplicate names |
| **Orchestrator status** | SUCCESS / WARNING / BLOCKED |

If gate fails: report to orchestrator with specific issues (which clusters
have low confidence, what makes them ambiguous). Orchestrator routes back
to clustering with targeted feedback.

---

### Step 7 — Classifier Validation (`ClassifierAgent`)

| | |
|---|---|
| **Inputs** | Feature DataFrame, cluster labels, personas |
| **Outputs** | CV F1 scores, feature importances, routing decision |
| **Skills** | `train_classifier`, `evaluate_cv`, `route_failure` |
| **Quality gates** | CV macro-F1 ≥ 0.70 |
| **Orchestrator status** | SUCCESS / WARNING / BLOCKED |

If F1 < 0.70: Claude diagnoses root cause and recommends reselect_features
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
| `outputs/feature_selection_report.json` | VIF table, PCA/AE scores |

---

## File Structure

```
agents/
  user_input.py             ← NEW  UserInputAgent
  dataset_examiner.py       ← NEW  DatasetExaminerAgent
  feature_engineer.py       ← NEW  FeatureEngineerAgent (stub + LLM-guided)
  feature_selector.py       ← ENHANCED: VIF + p-value gates
  clusterer.py              ← ENHANCED: silhouette k-opt, auto algo selection
  persona_namer.py          ← existing (minor bus integration)
  classifier.py             ← existing (minor bus integration)
  orchestrator.py           ← ENHANCED: route new agents, orchestrator bus
  state.py                  ← ENHANCED: UserIntent, DatasetProfile, new fields

skills/
  __init__.py
  vif_checker.py            ← VIF, correlation, p-value analysis
  silhouette_optimizer.py   ← Auto k selection via silhouette
  algo_recommender.py       ← Algorithm recommendation from data shape
  orchestrator_bus.py       ← Agent → orchestrator message protocol

plan.md                     ← THIS FILE (pipeline design & principles)
agent.md                    ← Agent registry: roles, interfaces, protocols
skill.md                    ← Skill registry: descriptions, owners, APIs
config.yaml                 ← Pipeline configuration (extended)
```
