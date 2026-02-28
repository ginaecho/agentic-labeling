# FeatureSelectionAgent

**File:** `agents/feature_selector.py`  
**Class:** `FeatureSelectionAgent`

## Role

Scores all engineered features using PCA importance and autoencoder reconstruction error, then applies VIF and correlation gates, and finally asks Claude to select the optimal subset for clustering.

## Skills used

- [orchestrator_bus](../skills/orchestrator_bus.md) — `report()`
- [vif_checker](../skills/vif_checker.md) — `compute_vif()`, `remove_high_vif()`, `flag_high_correlation()`

**Inline:** PCA importance scoring, autoencoder reconstruction-error scoring (see [Skills index](../skills/README.md)).

## Inputs

- Engineered feature DataFrame
- `UserIntent`
- Orchestrator feedback (free-text)

## Outputs

- `FeatureSelectionResult`:
  - `selected_features: list[str]`
  - `n_features: int`
  - `pca_scores: dict[str, float]`
  - `ae_scores: dict[str, float]`
  - `vif_table: dict[str, float]`
  - `reasoning: str`

## Quality gates (in order)

1. VIF < 5 for all features (iterative removal) — via [vif_checker](../skills/vif_checker.md)
2. \|r\| < 0.85 between any two features
3. ≥ 10 features survive; ≤ 80 features retained

## Communication protocol

Reports via [orchestrator_bus](../skills/orchestrator_bus.md):

```json
{
  "agent": "FeatureSelector",
  "status": "success | warning | blocked",
  "what_was_done": "Scored 108 features, removed 22 via VIF gate, Claude selected 45",
  "what_was_not_done": "Did not apply p-value gate (sufficient features after VIF)",
  "doubts": "Several travel features have borderline VIF (~4.8)",
  "issues": [],
  "metrics": { "n_input": 108, "n_after_vif": 86, "n_selected": 45, "max_vif": 4.2 },
  "recommendation": "proceed"
}
```

## Failure modes

| Issue | Status | Recommendation |
|-------|--------|----------------|
| < 10 features survive VIF gate | `blocked` | `retry` (relax to VIF < 8) or `escalate` |
| Claude returns invalid JSON | `warning` | `retry` |
