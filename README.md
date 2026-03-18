# Automated Cluster Interpretation with a Multi-Agent Pipeline

> **The hard part of clustering is not the math — it's the meaning.**

---

## The Problem

Clustering appears in almost every domain of applied data science: segmenting customers by behaviour, grouping patients by symptom profile, categorising documents by topic, organising images by visual similarity, partitioning sensor readings by operational mode. The algorithms are well established — k-means, hierarchical, GMM, DBSCAN, etc. — and any of them can produce as many clusters as you ask for in seconds.

The unsolved part is what comes after. What do those clusters *mean*? What should each one be called? What makes one group different from its neighbours — not in terms of centroid coordinates, but in terms a business or scientific audience can act on? Clustering is an unsupervised task: there are no ground-truth labels, so you cannot measure accuracy. The challenge is not prediction. It is **interpretability**.

Without automation, this loop typically runs multiple times, each iteration requiring the labelling step to be redone in full. The result is a process that is slow (days per project), undocumented (diagnostic reasoning is rarely recorded), and non-reproducible (the same data produces different segments depending on who runs the analysis).

The system described here automates the entire loop — feature engineering, selection, clustering, constraint checking, contrastive labelling, and iterative diagnosis — using a **multi-agent architecture** in which a Decision Maker LLM handles the steps that require judgment. The result: a complete, named, validated cluster solution in under one hour, at under one dollar of API cost, with a full reasoning trace for every decision.

---

## The Agent Approach

The pipeline is driven by **`run_pipeline.py`**. Seven specialised agents plus an Orchestrator form a feedback loop. Every quality gate can push the pipeline backward; it only moves forward when all gates pass (or the user approves):

```
  ┌───────────────────────────────────────────────────────────────────────┐
  │                        ORCHESTRATOR                                   │
  │  (Python coordinator · LLM decision-maker · param tuner)              │
  │                                                                       │
  │  ⓪ UserInputAgent   — collects clustering intent (once)               │
  │  ① DatasetExaminer  — profiles dataset, suggests feature groups       │
  │  ② FeatureEngineer  — builds ~232 behavioral features from CSV        │
  │         │  (skipped if a pre-built parquet is supplied)               │
  │         ▼                                                             │
  │  ③ FeatureSelector  — PCA + AE + VIF gate → orchestrator picks subset │
  │         │  ◄── orchestrator tunes VIF threshold, feature focus hint   │
  │         ▼                                                             │
  │  ④ Clusterer        — auto k-selection + algorithm + deepening        │
  │         │  ◄── orchestrator tunes k_range, algorithm, min_silhouette  │
  │         ▼                                                             │
  │  ⑤ PersonaNamer     — orchestrator names clusters · Clarity Gate      │
  │         ▼                                                             │
  │  ⑥ Classifier       — Random Forest CV · F1 ≥ 0.70 gate               │
  │         ▼                                                             │
  │  ⑦ Human Checkpoint — approve / re-cluster / reselect / quit          │
  └───────────────────────────────────────────────────────────────────────┘
```

### What each agent does

| # | Agent | Role |
|---|-------|------|
| ⓪ | **UserInputAgent** | Prompts for clustering intent (target entity, business purpose, dataset path). |
| ① | **DatasetExaminerAgent** | Profiles the raw data — schema, missingness, cardinality — and emits a suggested feature-group list and algorithm hint. |
| ② | **FeatureEngineerAgent** | Builds ~232 entity-level behavioral features from raw data rows: spend/frequency per category × time window, loyalty signals, recency, geographic mobility, temporal patterns. Saves to `data/processed/`. Skipped when a pre-built parquet is provided. |
| ③ | **FeatureSelectionAgent** | Scores all features with PCA importance and autoencoder reconstruction error, runs a VIF collinearity gate, then asks the LLM to pick the best subset (typically 25–55 features). The VIF threshold and a feature-focus hint are set dynamically by the Orchestrator each iteration. |
| ④ | **ClusteringAgent** | Auto-selects k via silhouette score optimisation and algorithm via `algo_recommender`. Runs a deepening loop to split any oversized cluster (>40%). The k range, algorithm, and minimum acceptable silhouette are tuned dynamically each iteration. |
| ⑤ | **PersonaNamingAgent** | Sends cluster profiles to the LLM, which writes name, tagline, description, and five traits per cluster. A **Clarity Gate** (avg confidence ≥ 6.0, all names unique) must pass or the pipeline re-clusters. |
| ⑥ | **ClassifierAgent** | Trains a Random Forest with stratified 5-fold CV. If macro-F1 < 0.70, the LLM diagnoses the root cause and routes back to ③ or ④. |

### How each agent calls the LLM

Every agent follows the same four-step pattern:

1. **Compute** — run sklearn / numpy / pandas to produce statistics (PCA scores, cluster profiles, SHAP values, etc.).
2. **Format** — assemble those statistics into a structured text prompt in Python.
3. **Call** — send the prompt to the LLM API (any chat-completion endpoint — Claude, GPT, Gemini, etc.).
4. **Parse** — read the LLM's JSON response and act on it.

The agents are not autonomous "give an LLM a goal and let it figure things out" agents. They are Python scripts that construct precise, data-rich prompts and parse structured responses. The `.md` design-spec files describe intent but are not loaded at runtime.

**Concrete example — `PersonaNamingAgent`**

The function `build_all_clusters_prompt()` dynamically assembles a prompt like:

```
You are a behavioral analyst interpreting entity clusters.
Each row shows: annual transactions, annual spend, avg per transaction…

CLUSTER 0  (1 234 entities, 12.3 % of total)
  food_dining   txns/yr= 120  spend/yr=$3 400  vs_avg= 1.8× ◀
  travel        txns/yr=  45  spend/yr=$8 200  vs_avg= 3.2× ◀◀
  …

CRITICAL NAMING RULES — read carefully before writing any name:
1. SPECIFICITY — names must describe what the entity ACTUALLY DOES …

Return ONLY a valid JSON object …
```

The cluster statistics are computed at runtime and injected into the prompt. The LLM reads those numbers and returns structured JSON with `name`, `tagline`, `description`, `traits`, `confidence`.

**What each agent computes vs. what it asks the LLM:**

| Agent | What Python computes | What it asks the LLM |
|-------|---------------------|---------------------|
| **FeatureSelectionAgent** | PCA + autoencoder importance scores, VIF table | "Which features best separate these personas?" |
| **ClusteringAgent** | sklearn cluster sizes, silhouette scores | "This cluster is >40 % of entities — sub-cluster or reselect features?" |
| **ClassifierAgent** | Random Forest CV F1, SHAP values | "F1 is 0.62 — diagnose root cause. Go back to feature selection or clustering?" |
| **PersonaNamingAgent** | Per-cluster statistics | "Name each cluster with a specific, evidence-backed persona." |

---

## Dynamic Parameter Tuning

After each failed iteration the Orchestrator calls the LLM with a compact history of what happened — silhouette scores, VIF removals, k-curve, feature counts — and asks it to suggest improved parameters for the next round:

| Parameter | Default | What the LLM can change |
|-----------|---------|----------------------|
| `vif_threshold` | 10.0 | Raise to keep correlated-but-informative features (range 5–25) |
| `algorithm` | auto | Switch between `kmeans` / `hierarchical` based on observed silhouette |
| `k_range` | `[3,4,5,6,7,8,10,12,15]` | Narrow or widen the search range |
| `min_silhouette` | 0.05 | Hard-block threshold; the LLM may lower it for data that genuinely resists clustering (floor 0.02) |
| `feature_focus` | *(empty)* | A short hint injected into the FeatureSelector prompt (e.g. "prioritise absolute spend over ratios") |

Parameters are clamped to safe ranges before use. Each iteration prints the tuning decision and the LLM's one-sentence reasoning.

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

The pipeline is dataset-agnostic. Point it at any tabular CSV where rows are events/transactions and columns include an entity identifier, a timestamp, and descriptive attributes. The `UserInputAgent` will ask which column is the entity key and what the clustering goal is.

### Demo dataset

The included demo uses the [**Fraud Detection**](https://www.kaggle.com/datasets/kartik2112/fraud-detection) dataset by Kartik Shenoy on Kaggle (`kartik2112/fraud-detection`). It contains ~1.3 million simulated credit-card transactions for ~983 customers, with columns for merchant, category, amount, timestamp, and demographics.

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
# Add your LLM API key
echo 'LLM_API_KEY=your-key-here' >> .env
```

**Run from raw transaction CSV**

```bash
python run_pipeline.py
# UserInputAgent will prompt: enter the path to data/raw/fraudTrain.csv
# FeatureEngineerAgent builds the feature table automatically
```

**Run from a pre-built feature parquet**

```bash
# If data/processed/customer_features.parquet exists, the pipeline skips
# FeatureEngineer and uses it directly. No extra steps needed.
python run_pipeline.py
```

The script:

- Loads `.env` and `config.yaml`
- Auto-detects whether to run FeatureEngineerAgent (raw CSV) or skip it (parquet)
- Runs the Orchestrator with `max_total_iterations=10`
- After each failure, the LLM proposes new VIF/k/algorithm/silhouette parameters
- At max iterations, delivers a best-effort result if no iteration fully passed
- Writes all outputs under `outputs/` and prints a full console report

---

## Configuration (`config.yaml`)

```yaml
# ── Clustering ──────────────────────────────────────────────────────
n_clusters: ~               # null = auto-select k via silhouette optimizer (recommended)
                            # Set an integer (e.g. 6) only to force a specific k

clustering_algorithm: kmeans  # kmeans | hierarchical
                              # The dynamic tuner may override this per-iteration

# ── Deepening loop ──────────────────────────────────────────────────
max_cluster_size_pct: 0.40  # split any cluster larger than this share of total
sub_n_clusters: 3           # how many sub-clusters to create when splitting
max_depth: 2                # max splitting rounds (0 = disabled)

# ── Persona tone ────────────────────────────────────────────────────
persona_tone: easy          # easy | professional | data-driven | creative
```

**`n_clusters: ~` (null) is the default and recommended setting.** It lets the silhouette optimizer scan `[3, 4, 5, 6, 7, 8, 10, 12, 15]` and pick the best k automatically. Set an integer only when you have a specific business requirement.

**The dynamic tuner can change `clustering_algorithm` and the k search range per-iteration**, so even if you set `kmeans` here, the LLM may suggest switching to `hierarchical` after observing the data — and back again in the next round.

### VIF threshold

The VIF gate threshold is **not in `config.yaml`** — it is managed dynamically. It starts at `10.0` and the LLM adjusts it each iteration based on how many features were removed and whether silhouette improved. You do not need to tune it manually.

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

The exact hard-block threshold (`min_silhouette`) is adjusted dynamically by the LLM between iterations.

---

## Outputs

After a successful (or best-effort) run:

| File | Description |
|------|-------------|
| `outputs/personas.json` | Machine-readable personas: name, tagline, traits, cluster stats, lineage. |
| `outputs/persona_summary.txt` | Human-readable persona cards in notebook-04 style with supporting metrics. |
| `outputs/persona_metrics.csv` | One row per cluster × category: `n_txn_12m`, `total_amt_12m`, `rel_n_txn`, signal strength, etc. |
| `outputs/classifier_metrics.json` | CV accuracy/F1, per-class F1, top-20 feature importances, reasoning. |
| `outputs/cluster_profiles.json` | Raw per-cluster statistics (category stats + overall metrics). |
| `outputs/cluster_lineage.json` | Cluster tree: parent/child relationships from the deepening loop. |
| `outputs/silhouette_curve.json` | k vs silhouette score curve from the optimizer, best k, algorithm reasoning. |
| `outputs/pipeline_log.json` | Full structured log of every agent's status report across all iterations. |
| `data/processed/engineered_features.parquet` | Entity-level feature matrix built by FeatureEngineerAgent (when starting from CSV). |

---

## Skills

The agents do not have hardcoded logic for every decision. They call shared **skills** — focused Python modules — for statistical tasks, and route LLM decisions through the Orchestrator bus:

| Skill | File | Used by |
|-------|------|---------|
| **OrchestratorBus** | `skills/orchestrator_bus.py` | All agents — the sole LLM gateway |
| **VIF checker** | `skills/vif_checker.py` | FeatureSelector — multicollinearity gate |
| **Silhouette optimizer** | `skills/silhouette_optimizer.py` | Clusterer — auto k-selection |
| **Algorithm recommender** | `skills/algo_recommender.py` | Clusterer — auto algorithm selection |

The Orchestrator loads `skill.md` and `agent.md` at startup and injects the relevant sections as system context into every LLM call, so the model always knows what capabilities are available when making routing or planning decisions.

---

## Key Metrics (Example Run — Credit-Card Transactions)

From a typical run on the demo dataset (~983 customers):

| Metric | Value |
|--------|--------|
| **Personas** | 7–10 (including sub-clusters from deepening loop) |
| **Entities** | ~983 |
| **CV F1 (macro)** | 0.85 – 0.95 |
| **Pipeline iterations** | 1–4 (with dynamic tuning) |

Detailed per-cluster metrics are in **`outputs/persona_metrics.csv`**.

---

## Why This Matters

The agent pipeline collapses the traditional loop: cluster → manually inspect statistics → write descriptions → iterate. The agents produce names, evidence, and reasoning in one automated run and save everything to version-controlled files. When someone asks "why is this entity in this cluster?", the answer is in `persona_summary.txt` and `persona_metrics.csv`. You do not need labelled training data or a domain expert to pre-define segments — the pipeline discovers structure, names it, validates it with the classifier, and self-corrects when results are poor.

The dynamic tuning means the pipeline adapts to your specific dataset rather than requiring manual threshold tuning across config files.

---

## Setup (Quick)

```bash
pip install -r requirements.txt
export LLM_API_KEY="your-key-here"   # or add to .env
python run_pipeline.py
```

Open **`outputs/persona_summary.txt`** for full persona cards and **`outputs/persona_metrics.csv`** for structured metrics after the run.
