# Automated Cluster Interpretation with a Multi-Agent Pipeline

> **The hard part of clustering is not the math — it's the meaning.**

---

## The Problem

Clustering appears in almost every domain of applied data science: segmenting customers by behaviour, grouping patients by symptom profile, categorising documents by topic, organising images by visual similarity, partitioning sensor readings by operational mode. The algorithms are well established — k-means, hierarchical, GMM, DBSCAN, etc. — and any of them can produce as many clusters as you ask for in seconds.

The unsolved part is what comes after. What do those clusters *mean*? What should each one be called? What makes one group different from its neighbours — not in terms of centroid coordinates, but in terms a business or scientific audience can act on? Clustering is an unsupervised task: there are no ground-truth labels, so you cannot measure accuracy. The challenge is not prediction. It is **interpretability**.

Without automation, this loop typically runs multiple times, each iteration requiring the labelling step to be redone in full. The result is a process that is slow (days per project), undocumented (diagnostic reasoning is rarely recorded), and non-reproducible (the same data produces different segments depending on who runs the analysis).

The system described here automates the entire loop — feature engineering, selection, clustering, constraint checking, contrastive labelling, and iterative diagnosis — using a **multi-agent architecture** in which a Decision Maker (any LLM API) handles the steps that require judgment. The result: a complete, named, validated cluster solution in under one hour, at under one dollar of API cost, with a full reasoning trace for every decision.

---

## The Agent Approach

The pipeline is driven by **`run_pipeline.py`**. Seven specialised agents plus a Decision Maker form a feedback loop. Every quality gate can push the pipeline backward; it only moves forward when all gates pass (or the user approves):

<img width="1097" height="592" alt="image" src="https://github.com/user-attachments/assets/97aa473b-0055-452b-bb87-448cb1d701fb" />


### What each agent does

| # | Agent | Role |
|---|-------|------|
| ⓪ | **UserInputAgent** | Prompts for clustering intent (target entity, business purpose, dataset path). |
| ① | **DatasetExaminerAgent** | Profiles the raw data — schema, missingness, distribution shape, cardinality — and asks the Decision Maker to suggest feature engineering groups aligned with the business purpose. Also emits an algorithm hint based on skewness. |
| ② | **FeatureEngineerAgent** | Builds an entity-level feature matrix from raw event-level data. The Decision Maker reads the actual column names from the dataset schema and plans which of 8 generic statistical operations to apply (group aggregation, trends, streaks, diversity, temporal patterns, etc.). No domain vocabulary is hard-coded — the LLM reasons from the data. Saves to `data/processed/`. Skipped when a pre-built parquet is provided. |
| ③ | **FeatureSelectionAgent** | Scores all features with PCA importance and autoencoder reconstruction error, runs a VIF collinearity gate, then asks the Decision Maker to pick the best subset (typically 25–55 features). The VIF threshold and a feature-focus hint are set dynamically by the Decision Maker each iteration. |
| ④ | **ClusteringAgent** | Auto-selects the best algorithm from five options (`kmeans`, `hierarchical`, `dbscan`, `gmm`, `fuzzy_cmeans`) via `algo_recommender`. Auto-selects k via silhouette score optimisation. Runs a deepening loop to split any oversized cluster (>40%). All numeric columns are log-transformed automatically if skewed (|skewness| > 2.0). The k range, algorithm, and minimum acceptable silhouette are tuned dynamically each iteration. |
| ⑤ | **PersonaNamingAgent** | Sends cluster profiles to the Decision Maker as tables of feature deviations from the global mean. The Decision Maker writes name, tagline, description, and five traits per cluster. A **Clarity Gate** (avg confidence ≥ 6.0, all names unique) must pass or the pipeline re-clusters. |
| ⑥ | **ClassifierAgent** | Asks the Decision Maker to select the best classifier (`random_forest`, `xgboost`, `gradient_boosting`, `logistic_regression`) for the data. Trains with stratified 5-fold CV. If macro-F1 < 0.70, the Decision Maker diagnoses the root cause and routes back to ③ or ④. |

### How each agent calls the Decision Maker

Every agent follows the same four-step pattern:

1. **Compute** — run sklearn / numpy / pandas to produce statistics (PCA scores, cluster profiles, silhouette scores, etc.).
2. **Format** — assemble those statistics into a structured text prompt in Python using actual column names discovered from the data.
3. **Call** — send the prompt through `OrchestratorBus` to the LLM API (any chat-completion endpoint — Claude, GPT, Gemini, etc.).
4. **Parse** — read the Decision Maker's JSON response and act on it.

The agents are Python scripts that construct precise, data-rich prompts and parse structured responses. All LLM access of agents is mediated by the Orchestrator through `OrchestratorBus`.

**Concrete example — `PersonaNamingAgent`**

The function `build_all_clusters_prompt()` dynamically assembles a prompt using features discovered from the actual data:

```
You are a behavioral analyst interpreting entity clusters.
Each cluster is described by its most distinguishing features:
  vs_avg: ratio of cluster mean to overall mean (◀ = 40%+ above; ◀◀ = 100%+ above; ▼ = 50%+ below)
  mean: the cluster's average value for that feature

CLUSTER 0  (1 234 entities, 12.3% of all entities)
Algorithm: kmeans

  ABOVE AVERAGE (strongest signals):
    count_category_food_12m                   mean=      87.4  vs_avg=2.41x ◀◀
    sum_category_travel_12m                   mean=   8200.1  vs_avg=3.18x ◀◀
    streak_category_grocery_pos               mean=       9.1  vs_avg=1.72x ◀
    …

  BELOW AVERAGE:
    event_count_all_6m                        mean=      12.3  vs_avg=0.38x ▼
    …

CRITICAL NAMING RULES — read carefully before writing any name:
1. SPECIFICITY — names must describe what the entity ACTUALLY DOES …
…

Return ONLY a valid JSON object …
```

The cluster statistics are computed at runtime from the actual feature matrix and injected into the prompt. The Decision Maker reads those numbers and returns structured JSON with `name`, `tagline`, `description`, `dominant_features`, `traits`, `confidence`.

**What each agent computes vs. what it asks the Decision Maker:**

| Agent | What Python computes | What it asks the Decision Maker |
|-------|---------------------|---------------------|
| **DatasetExaminerAgent** | Schema, missingness, skewness, cardinality | "Which feature groups should we engineer for this business purpose?" |
| **FeatureEngineerAgent** | Actual column names, entity key, timestamp, value column | "Plan which statistical operations to apply to build a rich feature matrix." |
| **FeatureSelectionAgent** | PCA + autoencoder importance scores, VIF table | "Which features best separate these personas?" |
| **ClusteringAgent** | Silhouette scores per algorithm/k, IQR spread, skewness | "Which algorithm fits this data shape? This cluster is >40% — sub-cluster or reselect features?" |
| **ClassifierAgent** | n_entities, n_features, n_classes, class balance | "Which classifier fits this data? F1 is 0.62 — diagnose root cause." |
| **PersonaNamingAgent** | Per-cluster feature means and deviations from global mean | "Name each cluster with a specific, evidence-backed persona." |

---

## How Feature Engineering Works

`FeatureEngineerAgent` knows **8 generic statistical operations**, independent of any domain:

| Builder | What it computes | Column naming pattern |
|---------|-----------------|----------------------|
| `group_aggregate` | count/sum/mean/std/max per group value × time window | `{metric}_{group_col}_{group_val}_{window}` |
| `group_trend` | change in count or sum between two windows | `trend_count_{col}_{val}`, `trend_sum_{col}_{val}` |
| `group_streak` | consecutive active periods per group value | `streak_{group_col}_{val}` |
| `overall_aggregate` | count/sum/mean/std/max over all events | `{metric}_{value_col}_{window}` |
| `frequency_recency` | event frequency, active periods, recency, gap | `event_count_{w}`, `days_since_last`, `avg_gap_days_{w}` |
| `entity_diversity` | number of unique values per column | `n_unique_{col}_{window}` |
| `temporal_patterns` | morning/evening/weekend ratios, peak hour | `pct_morning_{w}`, `pct_weekend_{w}`, `peak_hour_{w}` |
| `static_attributes` | entity-level non-temporal columns copied as-is | original column name |

The Decision Maker reads the actual schema (column names, types, sample values) and the business purpose, then plans which builders to apply to which columns — reasoning from the data, not from a hard-coded template. The same pipeline applies to transaction data, product catalogs, patient visits, sensor readings, or any other tabular event log.

---

## Dynamic Algorithm Selection

### Clustering algorithm

`AlgoRecommenderSkill` scores five algorithms on each dataset and picks the best:

| Algorithm | When chosen |
|-----------|------------|
| `kmeans` | Low skewness, large n, compact spherical clusters |
| `hierarchical` | Moderate skewness, dendrogram structure useful |
| `dbscan` | High outlier spread, irregular shapes, noise detection |
| `gmm` | Soft boundaries, overlapping groups, probabilistic membership |
| `fuzzy_cmeans` | Gradual transitions, partial membership useful |

The Decision Maker can override the recommender's choice after each iteration based on observed silhouette scores.

### Classifier

`ClassifierAgent` asks the Decision Maker to select among:

| Model | When chosen |
|-------|------------|
| `random_forest` | General default; robust to outliers and feature scale |
| `xgboost` | Tabular data with complex interactions and many features |
| `gradient_boosting` | Moderate datasets where accuracy is paramount |
| `logistic_regression` | Linearly separable, small-to-medium datasets |

---

## Dynamic Parameter Tuning

After each failed iteration the Decision Maker sees a compact history of what happened — silhouette scores, VIF removals, k-curve, feature counts — and proposes improved parameters for the next round:

| Parameter | Default | What the Decision Maker can change |
|-----------|---------|----------------------|
| `vif_threshold` | 10.0 | Raise to keep correlated-but-informative features (range 5–25) |
| `algorithm` | auto | Switch among `kmeans` / `hierarchical` / `dbscan` / `gmm` / `fuzzy_cmeans` |
| `k_range` | `[3,4,5,6,7,8,10,12,15]` | Narrow or widen the search range |
| `min_silhouette` | 0.05 | Hard-block threshold; the Decision Maker may lower it for data that genuinely resists clustering (floor 0.02) |
| `feature_focus` | *(empty)* | A short hint injected into the FeatureSelector prompt (e.g. "prioritise absolute magnitude over ratios") |

Parameters are clamped to safe ranges before use. Each iteration prints the tuning decision and the Decision Maker's one-sentence reasoning.

---

## Best-Effort Fallback

If 10 iterations complete without any result passing all gates, the pipeline does **not** just exit empty-handed. Instead it:

1. Identifies the iteration with the highest silhouette score across all attempts.
2. Runs **PersonaNamer** on that clustering with `force_proceed=True` (Clarity Gate bypassed).
3. Runs **Classifier** on the result.
4. Saves all outputs and returns `status='best_effort'`.

The console prints a `⚠ BEST-EFFORT RESULT` banner so the output is clearly flagged. This guarantees a full analysis is always delivered regardless of data difficulty.

---

## Data

The pipeline is dataset-agnostic. Point it at any tabular CSV where rows are events and columns include an entity identifier, a timestamp, and descriptive attributes. The `UserInputAgent` will ask what is being clustered and what the clustering goal is. The Decision Maker reasons from the actual schema to plan feature engineering — no domain vocabulary needs to be pre-configured.

### Demo dataset

The included demo uses the [**Fraud Detection**](https://www.kaggle.com/datasets/kartik2112/fraud-detection) dataset by Kartik Shenoy on Kaggle (`kartik2112/fraud-detection`). It contains ~1.3 million simulated credit-card transactions for ~983 cardholders, with columns for merchant, category, amount, timestamp, and demographics.

**Download** (requires a [Kaggle API token](https://www.kaggle.com/docs/api)):

```bash
pip install kaggle
kaggle datasets download -d kartik2112/fraud-detection -p data/raw --unzip
```

The pipeline uses `data/raw/fraudTrain.csv` (~335 MB) for feature engineering and clustering.

---

## How to Run

**Prerequisites**

```bash
pip install -r requirements.txt
export LLM_API_KEY="sk-ant-..."   # or add to .env
```

**Run from a raw event-level CSV**

```bash
python run_pipeline.py
# UserInputAgent will prompt for: entity being clustered, business purpose, dataset path
# FeatureEngineerAgent builds the feature matrix automatically
```

**Run from a pre-built feature parquet**

```bash
# If a feature parquet already exists, the pipeline skips FeatureEngineerAgent
# and reads it directly. No extra steps needed.
python run_pipeline.py
```

The script:

- Loads `.env` and `config.yaml`
- Auto-detects whether to run FeatureEngineerAgent (raw CSV) or skip it (parquet)
- Runs the Decision Maker loop with `max_total_iterations=10`
- After each failure, the Decision Maker proposes new VIF/k/algorithm/silhouette parameters
- At max iterations, delivers a best-effort result if no iteration fully passed
- Writes all outputs under `outputs/` and prints a full console report

---

## Configuration (`config.yaml`)

```yaml
# ── Clustering ──────────────────────────────────────────────────────
n_clusters: ~               # null = auto-select k via silhouette optimizer (recommended)
                            # Set an integer (e.g. 6) only to force a specific k

clustering_algorithm: auto  # auto | kmeans | hierarchical | dbscan | gmm | fuzzy_cmeans
                            # auto = AlgoRecommender scores all five and picks the best
                            # The Decision Maker may override this per-iteration

# ── Classifier ──────────────────────────────────────────────────────
classifier_model: auto      # auto | random_forest | xgboost | gradient_boosting | logistic_regression
                            # auto = Decision Maker selects based on data characteristics

# ── Deepening loop ──────────────────────────────────────────────────
max_cluster_size_pct: 0.40  # split any cluster larger than this share of total entities
sub_n_clusters: 3           # how many sub-clusters to create when splitting
max_depth: 2                # max splitting rounds (0 = disabled)

# ── Persona tone ────────────────────────────────────────────────────
persona_tone: easy          # easy | professional | data-driven | creative
```

**`n_clusters: ~` (null) is the default and recommended setting.** It lets the silhouette optimizer scan `[3, 4, 5, 6, 7, 8, 10, 12, 15]` and pick the best k automatically. Set an integer only when you have a specific business requirement.

**`clustering_algorithm: auto` is recommended.** The `AlgoRecommender` skill scores all five algorithms against data shape metrics (n_entities, n_features, skewness, outlier spread) and business purpose keywords, then picks the best fit. The Decision Maker can override after each iteration.

### VIF threshold

The VIF gate threshold is **not in `config.yaml`** — it is managed dynamically. It starts at `10.0` and the Decision Maker adjusts it each iteration based on how many features were removed and whether silhouette improved. You do not need to tune it manually.

---

## Silhouette Score Interpretation

Real-world ratio and frequency features typically produce lower silhouette scores than textbook examples. The pipeline's thresholds reflect this:

| Silhouette | Interpretation | Pipeline action |
|------------|---------------|----------------|
| < 0.05 | Near-random — no useful structure | Hard block → reselect features |
| 0.05 – 0.15 | Low but present (common with ratio/frequency features) | Warning, proceeds |
| 0.15 – 0.25 | Weak but usable | Warning, proceeds |
| 0.25 – 0.50 | Reasonable | Proceeds cleanly |
| ≥ 0.50 | Strong | Proceeds cleanly |

The exact hard-block threshold (`min_silhouette`) is adjusted dynamically by the Decision Maker between iterations.

---

## Outputs

After a successful (or best-effort) run:

| File | Description |
|------|-------------|
| `outputs/personas.json` | Machine-readable personas: name, tagline, traits, cluster stats, lineage. |
| `outputs/persona_summary.txt` | Human-readable persona cards with top distinguishing features. |
| `outputs/persona_metrics.csv` | One row per cluster × distinguishing feature: `mean_value`, `relative_to_avg`, signal strength. |
| `outputs/classifier_metrics.json` | CV accuracy/F1, per-class F1, top feature importances, reasoning. |
| `outputs/cluster_profiles.json` | Raw per-cluster statistics: `n_entities`, `pct_total`, `top_above_average`, `top_below_average`, `feature_means`. |
| `outputs/cluster_lineage.json` | Cluster tree: parent/child relationships from the deepening loop. |
| `outputs/silhouette_curve.json` | k vs silhouette score curve from the optimizer, best k, algorithm reasoning. |
| `outputs/pipeline_log.json` | Full structured log of every agent's status report across all iterations. |
| `outputs/agents_conversation.txt` | Full text log of every LLM prompt and response. |
| `data/processed/engineered_features.parquet` | Entity-level feature matrix built by FeatureEngineerAgent (when starting from CSV). |

---

## Skills

The agents do not have hard-coded logic for every decision. They call shared **skills** — focused Python modules — for statistical tasks, and route Decision Maker queries through `OrchestratorBus`:

| Skill | File | Used by |
|-------|------|---------|
| **OrchestratorBus** | `skills/orchestrator_bus.py` | All agents — the sole LLM gateway; logs every prompt and response |
| **VIF checker** | `skills/vif_checker.py` | FeatureSelector — multicollinearity gate |
| **Silhouette optimizer** | `skills/silhouette_optimizer.py` | Clusterer — auto k-selection |
| **Algorithm recommender** | `skills/algo_recommender.py` | Clusterer — scores 5 algorithms and recommends the best fit |

---

## Key Metrics (Example Run — Credit-Card Transaction Demo)

From a typical run on the demo dataset (~983 cardholders):

| Metric | Value |
|--------|--------|
| **Personas** | 7–10 (including sub-clusters from deepening loop) |
| **Entities** | ~983 |
| **CV F1 (macro)** | 0.85 – 0.95 |
| **Pipeline iterations** | 1–4 (with dynamic tuning) |

Detailed per-cluster metrics are in **`outputs/persona_metrics.csv`**.

---

## Why This Matters

The agent pipeline collapses the traditional loop: cluster → manually inspect statistics → write descriptions → iterate. The agents produce names, evidence, and reasoning in one automated run and save everything to version-controlled files. When someone asks "why is this entity in this cluster?", the answer is in `persona_summary.txt` and `persona_metrics.csv`.

You do not need labelled training data or a domain expert to pre-define segments — the pipeline discovers structure from whatever data you provide, names it using feature deviations the Decision Maker can read directly, validates it with a downstream classifier, and self-corrects when results are poor. The same code runs on transaction logs, product catalogs, patient records, sensor streams, or any other tabular event data.

The dynamic tuning means the pipeline adapts to your specific dataset rather than requiring manual threshold tuning across config files.

---

## Setup (Quick)

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."   # or add to .env
python run_pipeline.py
```

Open **`outputs/persona_summary.txt`** for full persona cards and **`outputs/persona_metrics.csv`** for structured metrics after the run.
