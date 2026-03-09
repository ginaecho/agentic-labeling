# Customer Persona Discovery with Agentic Interpretation

> **The hard part of clustering is not the math — it's the meaning.**

---

## The Problem

Every bank has millions of transactions. Hidden inside them is a simple truth: customers don't all behave the same way. Some people spend heavily on travel. Some are creatures of habit at the grocery store. Some barely use their card except for one big annual purchase. These differences are real, consistent, and actionable — if you can find them.

We call these recurring behavioral archetypes **personas**.

Discovering personas from raw transactions is deceptively difficult. The bottleneck is **interpretation**: turning cluster statistics into clear names, taglines, and evidence-backed descriptions. This project uses a **multi-agent pipeline** (orchestrated in Python, with Claude as the decision-maker) to do that automatically — end-to-end, from raw transaction CSV to named, validated, classifier-verified personas.

---

## The Agent Approach

The pipeline is driven by **`run_pipeline.py`**. Seven specialised agents plus an Orchestrator form a feedback loop. Every quality gate can push the pipeline backward; it only moves forward when all gates pass (or the user approves):

```
  ┌───────────────────────────────────────────────────────────────────────┐
  │                        ORCHESTRATOR                                   │
  │  (Python coordinator · Claude decision-maker · param tuner)           │
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
| ② | **FeatureEngineerAgent** | Builds ~232 customer-level behavioral features from raw transaction rows: spend/frequency per category × time window, loyalty signals, recency, geographic mobility, temporal patterns. Saves to `data/processed/`. Skipped when a pre-built parquet is provided. |
| ③ | **FeatureSelectionAgent** | Scores all features with PCA importance and autoencoder reconstruction error, runs a VIF collinearity gate, then asks Claude to pick the best subset (typically 25–55 features). The VIF threshold and a feature-focus hint are set dynamically by the Orchestrator each iteration. |
| ④ | **ClusteringAgent** | Auto-selects k via silhouette score optimisation and algorithm via `algo_recommender`. Runs a deepening loop to split any oversized cluster (>40%). The k range, algorithm, and minimum acceptable silhouette are tuned dynamically each iteration. |
| ⑤ | **PersonaNamingAgent** | Sends cluster profiles to Claude, which writes name, tagline, description, and five traits per cluster. A **Clarity Gate** (avg confidence ≥ 6.0, all names unique) must pass or the pipeline re-clusters. |
| ⑥ | **ClassifierAgent** | Trains a Random Forest with stratified 5-fold CV. If macro-F1 < 0.70, Claude diagnoses the root cause and routes back to ③ or ④. |

---

## Dynamic Parameter Tuning

After each failed iteration the Orchestrator calls Claude with a compact history of what happened — silhouette scores, VIF removals, k-curve, feature counts — and asks it to suggest improved parameters for the next round:

| Parameter | Default | What Claude can change |
|-----------|---------|----------------------|
| `vif_threshold` | 10.0 | Raise to keep correlated-but-informative features (range 5–25) |
| `algorithm` | auto | Switch between `kmeans` / `hierarchical` based on observed silhouette |
| `k_range` | `[3,4,5,6,7,8,10,12,15]` | Narrow or widen the search range |
| `min_silhouette` | 0.05 | Hard-block threshold; Claude may lower it for data that genuinely resists clustering (floor 0.02) |
| `feature_focus` | *(empty)* | A short hint injected into the FeatureSelector prompt (e.g. "prioritise absolute spend over ratios") |

Parameters are clamped to safe ranges before use. Each iteration prints the tuning decision and Claude's one-sentence reasoning.

---

## Best-Effort Fallback

If 10 iterations complete without any result passing all gates, the pipeline does **not** just exit empty-handed. Instead it:

1. Identifies the iteration with the highest silhouette score across all attempts.
2. Runs **PersonaNamer** on that clustering with `force_proceed=True` (Clarity Gate bypassed).
3. Runs **Classifier** on the result.
4. Saves all outputs and returns `status='best_effort'`.

The console prints a `⚠ BEST-EFFORT RESULT` banner so the output is clearly flagged. This guarantees a full analysis is always delivered regardless of data difficulty.

---

## How to Run

**Prerequisites**

```bash
pip install -r requirements.txt
# Add your Anthropic API key
echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env
```

**Run from raw transaction CSV**

```bash
python run_pipeline.py
# UserInputAgent will prompt: enter the path to fraudTrain.csv
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
- After each failure, Claude proposes new VIF/k/algorithm/silhouette parameters
- At max iterations, delivers a best-effort result if no iteration fully passed
- Writes all outputs under `outputs/` and prints a full console report

---

## Configuration (`config.yaml`)

```yaml
# ── Clustering ──────────────────────────────────────────────────────
n_clusters: ~               # null = auto-select k via silhouette optimizer (recommended)
                            # Set an integer (e.g. 6) only to force a specific k

clustering_algorithm: kmeans  # kmeans | hierarchical
                              # Claude's dynamic tuner may override this per-iteration

# ── Deepening loop ──────────────────────────────────────────────────
max_cluster_size_pct: 0.40  # split any cluster larger than this share of total
sub_n_clusters: 3           # how many sub-clusters to create when splitting
max_depth: 2                # max splitting rounds (0 = disabled)

# ── Persona tone ────────────────────────────────────────────────────
persona_tone: easy          # easy | professional | data-driven | creative
```

**`n_clusters: ~` (null) is the default and recommended setting.** It lets the silhouette optimizer scan `[3, 4, 5, 6, 7, 8, 10, 12, 15]` and pick the best k automatically. Set an integer only when you have a specific business requirement.

**The dynamic tuner can change `clustering_algorithm` and the k search range per-iteration**, so even if you set `kmeans` here, Claude may suggest switching to `hierarchical` after observing the data — and back again in the next round.

### VIF threshold

The VIF gate threshold is **not in `config.yaml`** — it is managed dynamically. It starts at `10.0` and Claude adjusts it each iteration based on how many features were removed and whether silhouette improved. You do not need to tune it manually.

---

## Silhouette Score Interpretation

Banking transaction-ratio and frequency features produce lower silhouette scores than textbook examples. The pipeline's thresholds reflect this:

| Silhouette | Interpretation | Pipeline action |
|------------|---------------|----------------|
| < 0.05 | Near-random — no useful structure | Hard block → reselect features |
| 0.05 – 0.15 | Low but present (typical for ratio features) | Warning, proceeds |
| 0.15 – 0.25 | Weak but usable | Warning, proceeds |
| 0.25 – 0.50 | Reasonable | Proceeds cleanly |
| ≥ 0.50 | Strong | Proceeds cleanly |

The exact hard-block threshold (`min_silhouette`) is adjusted dynamically by Claude between iterations.

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
| `data/processed/engineered_features.parquet` | Customer-level feature matrix built by FeatureEngineerAgent (when starting from CSV). |

---

## Skills

The agents do not have hardcoded logic for every decision. They call shared **skills** — focused Python modules — for statistical tasks, and route LLM decisions through the Orchestrator bus:

| Skill | File | Used by |
|-------|------|---------|
| **OrchestratorBus** | `skills/orchestrator_bus.py` | All agents — the sole LLM gateway |
| **VIF checker** | `skills/vif_checker.py` | FeatureSelector — multicollinearity gate |
| **Silhouette optimizer** | `skills/silhouette_optimizer.py` | Clusterer — auto k-selection |
| **Algorithm recommender** | `skills/algo_recommender.py` | Clusterer — auto algorithm selection |

The Orchestrator loads `skill.md` and `agent.md` at startup and injects the relevant sections as system context into every Claude call, so Claude always knows what capabilities are available when making routing or planning decisions.

---

## Key Metrics (Example Run)

From a typical run on ~983 customers:

| Metric | Value |
|--------|--------|
| **Personas** | 7–10 (including sub-clusters from deepening loop) |
| **Customers** | ~983 |
| **CV F1 (macro)** | 0.85 – 0.95 |
| **Pipeline iterations** | 1–4 (with dynamic tuning) |

Detailed per-cluster × category metrics (txns/yr, spend/yr, loyalty, vs-dataset ratios, signal strength) are in **`outputs/persona_metrics.csv`**.

---

## Why This Matters

The agent pipeline collapses the traditional loop: cluster → manually inspect statistics → write descriptions → iterate. The agents produce names, evidence, and reasoning in one automated run and save everything to version-controlled files. When someone asks "why is this customer in this persona?", the answer is in `persona_summary.txt` and `persona_metrics.csv`. You do not need labelled training data or a domain expert to pre-define segments — the pipeline discovers structure, names it, validates it with the classifier, and self-corrects when results are poor.

The dynamic tuning means the pipeline adapts to your specific dataset rather than requiring manual threshold tuning across config files.

---

## Setup (Quick)

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."   # or add to .env
python run_pipeline.py
```

Open **`outputs/persona_summary.txt`** for full persona cards and **`outputs/persona_metrics.csv`** for structured metrics after the run.
