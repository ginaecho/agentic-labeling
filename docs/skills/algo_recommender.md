# algo_recommender — Clustering Algorithm Recommendation

**File:** `skills/algo_recommender.py`  
**Used by:** [ClusteringAgent](../agents/clusterer.md)

## Purpose

Analyses dataset size, feature distribution shape, and business purpose to recommend the most appropriate clustering algorithm.

## API

```python
from skills.algo_recommender import recommend_algorithm, AlgoRecommendation

rec = recommend_algorithm(
    n_rows=10000,
    n_features=45,
    feature_skewness={"travel_spend": 3.2, "grocery_spend": 1.1, ...},
    dataset_profile=profile,   # DatasetProfile from DatasetExaminerAgent
    user_intent=intent,        # UserIntent
)

rec.algorithm    # "hierarchical" | "kmeans"
rec.reasoning    # str — explanation of the choice
rec.confidence   # float 0–1 — how confident the recommendation is
```

## Decision rules

| Condition | Recommendation |
|-----------|----------------|
| `n_rows > 100_000` | K-Means (speed) |
| Mean feature skewness > 2.0 | Hierarchical (robust to skew after log-transform) |
| Business purpose mentions "segments within segments" | Hierarchical |
| Fewer than 5 candidate k values have silhouette > 0.25 | K-Means (try different shape) |
| Default | Hierarchical / Ward |
