# silhouette_optimizer — Data-Driven Cluster Count Selection

**File:** `skills/silhouette_optimizer.py`  
**Used by:** [ClusteringAgent](../agents/clusterer.md)

## Purpose

Tries a range of k values, fits the chosen clustering algorithm for each, computes the silhouette score, and returns the k that maximises it. Also computes the elbow-method inertia curve for K-Means as a secondary signal.

## API

```python
from skills.silhouette_optimizer import optimize_k, SilhouetteResult

result = optimize_k(
    X_scaled,                        # np.ndarray, preprocessed features
    algorithm="hierarchical",        # "hierarchical" | "kmeans"
    k_range=[3, 4, 5, 6, 7, 8, 10, 12, 15],
    random_state=42,
)

result.best_k          # int — k with highest silhouette
result.best_silhouette # float — silhouette at best_k
result.scores          # dict[int, float] — silhouette for each k tried
result.inertias        # dict[int, float] — inertia for each k (K-Means only)
result.reasoning       # str — human-readable summary
```

## Silhouette interpretation

| Score | Quality |
|-------|---------|
| ≥ 0.50 | Strong structure |
| 0.25 – 0.50 | Reasonable structure |
| 0.10 – 0.25 | Weak structure (consider different features) |
| < 0.10 | No meaningful structure |
