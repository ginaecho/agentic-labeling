# AI Personalization Pipeline

## Overview

This project clusters bank customers by spending behavior and assigns human-readable
**personas** using Claude. A downstream XGBoost classifier enables real-time persona
prediction for new customers from their raw transactions.

## Notebook Chain

| Step | Notebook                      | What it does                                                |
|------|-------------------------------|-------------------------------------------------------------|
| 1    | `00_data_acquisition.ipynb`   | Download raw transaction CSV from Kaggle                    |
| 2    | `01_eda.ipynb`                | Exploratory data analysis & visualizations                  |
| 3    | `02_feature_engineering.ipynb`| Build 108 behavioral features per customer                  |
| 4    | `03_clustering.ipynb`         | Cluster customers → `outputs/cluster_profiles.json`         |
| 5    | `04_llm_persona.ipynb`        | Name clusters with Claude → `outputs/personas.json`         |
| 6    | `05_classifier.ipynb`         | Train XGBoost classifier + generate LLM cluster explanations|

---

## Customizing the Pipeline

All tunable parameters live in **`config.yaml`** at the project root.
Edit that file, save it, then re-run the notebooks listed under "Re-run Strategy" below.

### Parameters

#### `n_clusters` (default: `10`)
How many customer segments to create.
- Range 4–6: broad personas, easy to action
- Range 7–12: balanced segmentation (default 10 works well)
- Range 12–15: very fine-grained, may overlap
- If clusters turn out ambiguous, the Clarity Gate in notebook 04 will warn you.

#### `clustering_algorithm` (default: `hierarchical`)
Which algorithm segments the customers:
- **`hierarchical`** — Agglomerative clustering with Ward linkage. Builds a
  dendrogram tree; excellent for discovering nested sub-groups (e.g. a "traveller"
  cluster whose internal branches reveal regional vs. cross-continent travellers).
  The Ward method minimises total within-cluster variance at each merge step.
- **`kmeans`** — K-Means. Places each customer into the group whose centroid it is
  closest to. Fast, scalable, and very interpretable for large datasets.

#### `max_cluster_size_pct` (default: `0.40`)
Trigger threshold for the deepening loop. Any cluster whose share of total customers
exceeds this value is automatically split.
- `0.40` → split if one cluster has >40% of all customers
- `1.0`  → never split (same as disabling the loop)

#### `sub_n_clusters` (default: `3`)
How many sub-clusters to create when splitting an oversized cluster.
Smaller values (2–3) keep personas manageable; larger values (4–5) produce finer splits.

#### `max_depth` (default: `2`)
Maximum number of splitting rounds.
- `0` → deepening loop disabled entirely (flat clustering only)
- `1` → split once, then stop
- `2` → split, then check again and split once more if needed (default)

#### `persona_tone` (default: `easy`)
Tone Claude uses when writing persona names, descriptions, and explanations:
- **`easy`** — Plain, everyday language. No jargon. Simple analogies.
- **`professional`** — Formal, board-ready language. Actionable insights.
- **`data-driven`** — Heavy on specific numbers and statistical ratios.
- **`creative`** — Vivid metaphors, storytelling, imaginative analogies.

---

## Re-run Strategy

| What you changed                 | Re-run from          |
|----------------------------------|----------------------|
| `n_clusters` or `clustering_algorithm` | Notebook **03** |
| `persona_tone` only              | Notebook **04**      |
| Nothing (re-generate explanations only) | Notebook **05**, section 6 |

---

## Key File Paths

| File                                              | Description                                    |
|---------------------------------------------------|------------------------------------------------|
| `config.yaml`                                     | Pipeline configuration (edit this)             |
| `data/raw/fraudTrain.csv`                         | Raw transaction data                           |
| `data/processed/customer_features.parquet`        | 108 features per customer (output of nb 02)    |
| `data/processed/customer_features_clustered.parquet` | Features + cluster labels (output of nb 03) |
| `outputs/cluster_profiles.json`                   | Per-cluster statistics + lineage (output of nb 03) |
| `outputs/cluster_lineage.json`                    | Full cluster tree (parent/child relationships) |
| `outputs/personas.json`                           | Cluster → Persona mapping (output of nb 04)    |
| `outputs/cluster_explanations.json`               | LLM explanations per cluster (output of nb 05) |
| `outputs/cluster_plots/`                          | Visualization PNGs                             |

---

## Environment

Set your Anthropic API key before running notebooks 04 and 05:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Notebooks 04 and 05 both call the `claude-sonnet-4-6` model.

---

## How the LLM Explanation Works (notebook 05)

After training the XGBoost classifier and computing SHAP values, notebook 05 calls
Claude to explain **why** each cluster received its persona name. The explanation:

1. Identifies the **top-tier metrics** — categories where this cluster is significantly
   above the dataset average (e.g. travel transactions 3.2× higher than average)
2. Identifies the **weak metrics** — categories well below average
3. References the **clustering algorithm** — e.g. for hierarchical/Ward, Claude notes
   what the dendrogram structure implies about sub-group cohesion; for K-Means, what
   the centroid position reveals
4. Cites the **XGBoost SHAP features** — the top features the classifier learned to
   use to spot this persona, grounding the explanation in the ML model's logic
5. Suggests **potential sub-segments** within the cluster

Results are saved to `outputs/cluster_explanations.json`.
