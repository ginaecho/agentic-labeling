# ClusteringAgent

**File:** `agents/clusterer.py`  
**Class:** `ClusteringAgent`

## Role

Selects the clustering algorithm and number of clusters data-driven (via silhouette optimisation), fits the model, runs the deepening loop for oversized clusters, and builds cluster profiles.

## Skills used

- [orchestrator_bus](../skills/orchestrator_bus.md) — `report()`
- [algo_recommender](../skills/algo_recommender.md) — `recommend_algorithm()`
- [silhouette_optimizer](../skills/silhouette_optimizer.md) — `optimize_k()`

## Inputs

- Engineered feature DataFrame
- `selected_features: list[str]`
- `UserIntent`
- `DatasetProfile` (for algorithm recommendation)
- Orchestrator feedback

## Outputs

- `ClusteringResult`:
  - `action: str` (proceed | reselect_features)
  - `cluster_labels: pd.Series`
  - `profiles: dict`
  - `lineage: dict`
  - `silhouette: float`
  - `n_leaf: int`
  - `k_scores: dict[int, float]` (silhouette for each k tried)
  - `algo_used: str`

## Algorithm selection

Delegated to [algo_recommender](../skills/algo_recommender.md). Factors: `n_rows`, mean skewness, business purpose.

## K selection

Delegated to [silhouette_optimizer](../skills/silhouette_optimizer.md). Tries k in config `k_search_range`; picks k with maximum silhouette.

## Communication protocol

Reports via [orchestrator_bus](../skills/orchestrator_bus.md):

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

## Failure modes

| Issue | Status | Recommendation |
|-------|--------|----------------|
| Silhouette < 0.20 for all k | `warning` | `retry` with different algo |
| Silhouette < 0.15 after both algos | `blocked` | `reselect_features` |
| Deepening loop can't resolve oversized cluster | `warning` | `reselect_features` |
