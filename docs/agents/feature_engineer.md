# FeatureEngineerAgent

**File:** `agents/feature_engineer.py`  
**Class:** `FeatureEngineerAgent`

## Role

Builds a rich feature matrix from raw data, guided by the `DatasetProfile` and business purpose. Uses Claude to decide which transformations to apply.

## Skills used

- [orchestrator_bus](../skills/orchestrator_bus.md) — `report()`

**Inline / internal:** build_frequency_features, build_spend_features, build_recency_features, build_interaction_features (see [Skills index — FeatureEngineerAgent](../skills/README.md#inline-skills)).

## Inputs

- Raw DataFrame
- `DatasetProfile`
- `UserIntent`
- Orchestrator feedback (free-text, from previous iteration if any)

## Outputs

- Engineered feature `DataFrame` (persisted to `data/processed/`)
- `FeatureEngineeringResult` dataclass with feature count and group coverage

## Communication protocol

Reports via [orchestrator_bus](../skills/orchestrator_bus.md):

```json
{
  "agent": "FeatureEngineer",
  "status": "success | warning | blocked",
  "what_was_done": "Built 108 features across 6 behavioral groups",
  "what_was_not_done": "Could not build loyalty/tenure features (no signup date column)",
  "doubts": "Recency features may overlap heavily with frequency features",
  "issues": [],
  "metrics": { "n_features": 108, "n_groups": 6, "missing_groups": ["loyalty"] },
  "recommendation": "proceed"
}
```

## Failure modes

| Issue | Status | Recommendation |
|-------|--------|----------------|
| Required columns missing | `warning` | `proceed` (fewer feature groups) |
| Fewer than 20 features built | `blocked` | `escalate` |
| All features are binary/constant | `blocked` | `escalate` |
