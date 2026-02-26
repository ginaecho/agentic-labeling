# Customer Persona Discovery with Agentic Interpretation

> **The hard part of clustering is not the math — it's the meaning.**

---

## The Problem

Every bank has millions of transactions. Hidden inside them is a simple truth: customers don't all behave the same way. Some people spend heavily on travel. Some are creatures of habit at the grocery store. Some barely use their card except for one big annual purchase. These differences are real, consistent, and actionable — if you can find them.

We call these recurring behavioral archetypes **personas**.

Discovering personas from raw transactions is deceptively difficult. The bottleneck is **interpretation**: turning cluster statistics into clear names, taglines, and evidence-backed descriptions. This project uses a **multi-agent pipeline** (orchestrated in Python, with Claude as the decision-maker) to do that automatically.

---

## The Agent Approach

The pipeline is driven by **`run_pipeline.py`**. It does not use notebooks; instead, four specialised agents plus an Orchestrator form a feedback loop:

1. **FeatureSelector** — Claude reads PCA and autoencoder importance scores and decides which features maximise persona separability (~30–60 of 108).
2. **Clusterer** — sklearn fits clusters; if any cluster exceeds a size limit (e.g. 40%), Claude decides whether to sub-cluster in place or reselect features. Outputs leaf cluster labels, profiles, and a lineage tree.
3. **PersonaNamer** — Claude writes name, tagline, description, and traits for every cluster in one structured prompt (siblings grouped so names must contrast). A **Clarity Gate** (silhouette, confidence, unique names) must pass or the pipeline loops back.
4. **Classifier** — Random Forest with cross-validation validates that cluster labels are crisp in feature space. If macro F1 &lt; 0.70, Claude diagnoses root cause and the Orchestrator can loop back to FeatureSelector or Clusterer.
5. **Human Checkpoint** — Final persona table, CV-F1, and metric proof are shown; approve to save all outputs.

The Orchestrator coordinates these agents. Every quality gate can push the pipeline backward; it only moves forward when all gates pass (or the user approves manually).

---

## How to Run the Pipeline

**Prerequisites**

- Python 3.10+
- `pip install -r requirements.txt`
- **ANTHROPIC_API_KEY** in `.env` or in the environment (used for Claude in FeatureSelector, Clusterer, PersonaNamer, Classifier)

**Run the full pipeline**

```bash
# From the project root (where run_pipeline.py and config.yaml live)
python run_pipeline.py
```

The script:

- Loads `.env` (if present) and **config.yaml**
- Expects **`data/processed/customer_features.parquet`** (948 customers × 108 features from notebook 02 or your own feature build)
- Runs the Orchestrator with `max_total_iterations=10`
- Writes **outputs/personas.json**, **outputs/persona_summary.txt**, **outputs/persona_metrics.csv**, **outputs/classifier_metrics.json**
- Prints to console: how agents interact, cluster size check, persona summary, key features, Claude token usage, and total time

**Config**

Edit **`config.yaml`** to change `n_clusters`, `clustering_algorithm`, `max_cluster_size_pct`, `sub_n_clusters`, `max_depth`, `persona_tone`, etc. Then re-run:

```bash
python run_pipeline.py
```

---

## Key Metrics (Example Run)

From a typical run on 948 customers:

| Metric | Value |
|--------|--------|
| **Personas** | 8 (including sub-clusters) |
| **Customers** | 948 |
| **CV F1 (macro)** | 0.926 |

The classifier’s cross-validated F1 (macro) measures how well the chosen features and clusters separate personas; the pipeline uses a threshold (e.g. ≥ 0.70) to decide whether to proceed or loop back.

Detailed per-cluster × category metrics (txns/yr, spend/yr, loyalty, vs-dataset ratios, signal strength) are in **`outputs/persona_metrics.csv`**.

---

## Persona Summary (Example Run)

The following is from **`outputs/persona_summary.txt`** — full persona cards in notebook-04 style plus key-metric evidence. Each cluster has a name, tagline, dominant categories, traits, description, and a “Key metrics driving this persona” block (above/below average categories and overall vs dataset).

| # | Persona | Tagline | Size |
|---|----------|---------|-----:|
| 0 | **The High-Volume All-Category Power Spender** | Shops everywhere, often, and has been doing it for over a year and a half. | 182 |
| 1 | **The Rare Big-Ticket Online Shopper** | Barely transacts, but when they do, they spend hundreds at a time online. | 11 |
| 2 | **The Light, Sporadic Across-the-Board Undershopper** | Active in every category, but at roughly a third to a half of normal levels. | 227 |
| 4 | **The High-Frequency Ghost — Active Spend Pattern, Zero Loyalty Signal** | Looks like a big spender in volume, but consecutive months are nearly zero. | 18 |
| 5 | **The Ultra-Rare Luxury Online Shopper** | Almost never uses their card, but when they do, it's for big online purchases. | 11 |
| 6 | **The Committed Commuter & Steady Household Manager** *(sub of C3)* | Gas spending near average — not above it — with a calmer, balanced profile. | 219 |
| 7 | **The High-Mileage Commuter Who Also Travels and Stays Active** *(sub of C3)* | Heavy gas use with above-average travel spend and highest travel cost-per-trip. | 133 |
| 8 | **The Highest-Frequency Everyday Spender with Strong Online Grocery Habit** *(sub of C3)* | Leads in transaction volume and above-average online grocery habit. | 147 |

**Full persona cards** (traits, description, key metrics tables) are in **`outputs/persona_summary.txt`**.  
**Structured metrics** (per cluster × category: `n_txn_12m`, `total_amt_12m`, `rel_n_txn`, `signal`, etc.) are in **`outputs/persona_metrics.csv`**.

---

## Configuration

All tunable parameters live in **`config.yaml`**, used by **`run_pipeline.py`**:

```yaml
n_clusters: 10                    # initial number of clusters
clustering_algorithm: hierarchical # hierarchical | kmeans
persona_tone: easy                 # easy | professional | data-driven | creative

max_cluster_size_pct: 0.40        # split any cluster larger than this share
sub_n_clusters: 3                 # sub-clusters when splitting
max_depth: 2                      # max rounds of splitting (0 = disabled)
```

Set `max_depth: 0` to disable the deepening loop (flat clustering only).  
Set `max_cluster_size_pct: 1.0` to effectively disable the size guard.

---

## Data and Features

The pipeline expects **`data/processed/customer_features.parquet`**: one row per customer, with ~108 behavioral features (time-windowed spend and frequency per category, loyalty signals, overall metrics). You can generate this from the notebooks **00 → 01 → 02** (data acquisition, EDA, feature engineering), or supply your own parquet with the same schema.

After the pipeline runs, you get:

- **outputs/personas.json** — machine-readable personas (name, tagline, traits, cluster_stats).
- **outputs/persona_summary.txt** — human-readable persona cards + key metrics (notebook-04 style).
- **outputs/persona_metrics.csv** — one row per cluster × category with raw and relative metrics.
- **outputs/classifier_metrics.json** — CV accuracy/F1, per-class F1, top feature importances.

---

## Why This Matters

The agent pipeline collapses the traditional loop: clustering → manual interpretation → slide deck. The agents produce names, evidence, and reasoning in one run and save them to version-controlled files. When someone asks “why is this customer in this persona?”, the answer is in **persona_summary.txt** and **persona_metrics.csv**. You don’t need labelled training data or a domain expert to pre-define segments; the pipeline discovers structure, names it, and validates it with the classifier.

---

## Setup (Quick)

```bash
pip install -r requirements.txt
# Add ANTHROPIC_API_KEY to .env or export it
python run_pipeline.py
```

For full persona cards and key metrics, open **`outputs/persona_summary.txt`** and **`outputs/persona_metrics.csv`** after the run.
