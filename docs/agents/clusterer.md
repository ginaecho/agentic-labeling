# ClusteringAgent

**File:** `agents/clusterer.py`
**Class:** `ClusteringAgent`

## Role

Selects the clustering algorithm and number of clusters data-driven (via silhouette optimisation and AlgoRecommender), fits the model, runs the deepening loop for oversized clusters, and builds generic cluster profiles. Works with any feature matrix — no domain-specific column names or category lists.

## Skills used

- [orchestrator_bus](../skills/orchestrator_bus.md) — `ask()`, `report()`
- [algo_recommender](../skills/algo_recommender.md) — `recommend_algorithm()`
- [silhouette_optimizer](../skills/silhouette_optimizer.md) — `optimize_k()`

## Inputs

- Engineered feature DataFrame
- `selected_features: list[str]`
- `UserIntent`
- `DatasetProfile` (for algorithm recommendation)
- Orchestrator feedback (tuning params: `k_range`, `algorithm`, `min_silhouette`)
- `text_artifacts: dict | None` — when present, the agent takes the **text
  branch**: skip log-transform + `StandardScaler`, L2-normalize the matrix so
  Euclidean distance becomes cosine, pass `metric='cosine'` to
  `optimize_k()` + `silhouette_score()`, and build text-mode profiles via
  `_extract_text_profiles()` (c-TF-IDF distinctive terms + centroid-nearest
  representative documents). The artifacts dict carries `raw_docs`,
  `feature_names`, `tfidf`, `tfidf_matrix`, `doc_index`, and `method`.

## Outputs

- `ClusteringResult`:
  - `action: str` — `proceed | reselect_features`
  - `cluster_labels: pd.Series`
  - `profiles: dict` — see Profile structure below
  - `lineage: dict`
  - `silhouette: float`
  - `n_leaf: int`
  - `k_scores: dict[int, float]` — silhouette for each k tried
  - `algo_name: str`, `algo_detail: str`

## Profile structure (per cluster)

**Tabular mode** — values are numeric mean ratios:

```json
{
  "n_entities": 1234,
  "pct_total": 12.3,
  "top_above_average": { "feature_name": 2.41, "...": "..." },
  "top_below_average": { "feature_name": 0.38, "...": "..." },
  "feature_means": { "feature_name": 87.4, "...": "..." },
  "feature_relative": { "feature_name": 2.41, "...": "..." },
  "lineage": { "depth": 0, "parent": null, "siblings": [], "pct_of_parent": 1.0 },
  "algorithm": "kmeans",
  "algo_detail": "KMeans(k=7)"
}
```

**Text mode** — same schema, but `top_above_average` / `feature_means` carry
**c-TF-IDF distinctive-term scores** (term → score) and the profile gains
`top_terms` + `representative_docs`. The UI, cluster chat, and cross-cluster
comparison consume the same fields and therefore work for text unchanged.

```json
{
  "n_entities": 135,
  "pct_total": 13.5,
  "top_above_average": { "car": 51.7, "cars": 24.4, "dealer": 16.5, "engine": 16.0 },
  "top_below_average": { "people": 0.83, "know": 0.80, "like": 0.78 },
  "feature_means": { "car": 51.7, "cars": 24.4, "...": "..." },
  "top_terms": ["car", "cars", "dealer", "engine", "speed", "tires", "ford"],
  "representative_docs": ["I was wondering if anyone out there could enlighten me ...", "..."],
  "modality": "text",
  "lineage": { "...": "..." },
  "algorithm": "KMeans",
  "algo_detail": "K-Means | k=5"
}
```

`top_above_average` and `top_below_average` are the 10 features with the largest positive/negative deviation from the global mean (ratio of cluster mean to global mean).

## Algorithm selection

Delegated to [algo_recommender](../skills/algo_recommender.md). Supports: `kmeans`, `hierarchical`, `dbscan`, `gmm`, `fuzzy_cmeans`. On algorithm failure, asks the LLM via OrchestratorBus for an alternative.

## K selection

Priority order:

| Priority | Source | Behaviour |
|----------|--------|-----------|
| 1 (highest) | `user_intent.n_clusters_requested` | Used directly; silhouette optimisation is **skipped** |
| 2 | `config.yaml` `n_clusters` | Used directly |
| 3 (default) | [silhouette_optimizer](../skills/silhouette_optimizer.md) | Tries k in the configured range; picks k with maximum silhouette |

Not applicable to DBSCAN (auto-determines cluster count via eps/min_samples).

## Log-transform

Any non-negative numeric column with |skewness| > 2.0 is automatically log1p-transformed before scaling. No hard-coded column names.

## Communication protocol

Reports via [orchestrator_bus](../skills/orchestrator_bus.md):

```json
{
  "agent": "Clusterer",
  "status": "success | warning | blocked",
  "what_was_done": "AlgoRecommender selected hierarchical; silhouette optimizer tried k=3..15; selected k=7 (sil=0.38); ran deepening loop",
  "what_was_not_done": "",
  "doubts": "Silhouette improvement from k=6 to k=7 is marginal (0.01 diff)",
  "issues": [],
  "metrics": { "best_k": 7, "silhouette": 0.38, "n_leaf": 9, "algo": "hierarchical" },
  "recommendation": "proceed"
}
```

## Failure modes

| Issue | Status | Recommendation |
|-------|--------|----------------|
| Silhouette < min_silhouette for all k/algorithms tried | `blocked` | `reselect_features` |
| Algorithm raises exception | tries next algorithm; asks LLM for advice | — |
| Deepening loop can't resolve oversized cluster | `warning` | `reselect_features` |
