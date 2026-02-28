# DatasetExaminerAgent

**File:** `agents/dataset_examiner.py`  
**Class:** `DatasetExaminerAgent`

## Role

Profiles the raw dataset and identifies feature engineering opportunities aligned with the stated business purpose. Calls Claude with the schema + business purpose to get suggested feature groups.

## Skills used

- [orchestrator_bus](../skills/orchestrator_bus.md) — `report()`

## Inputs

- `user_intent: UserIntent`
- Raw DataFrame (loaded from `user_intent.dataset_path`)

## Outputs

- `DatasetProfile` dataclass:
  - `n_rows: int`, `n_cols: int`
  - `column_types: dict[str, str]`
  - `missing_rates: dict[str, float]`
  - `distribution_summary: dict[str, dict]` (skewness, kurtosis, min/max/mean)
  - `suggested_feature_groups: list[str]` (from Claude)
  - `warnings: list[str]`

## Communication protocol

Reports via [orchestrator_bus](../skills/orchestrator_bus.md):

```json
{
  "agent": "DatasetExaminer",
  "status": "success | warning | blocked",
  "what_was_done": "Profiled schema, analysed distributions, called Claude for feature group suggestions",
  "what_was_not_done": "Did not load data subsets for validation",
  "doubts": "Column 'merchant_category' has 300+ unique values — may need grouping",
  "issues": ["Column 'age' missing in 15% of rows"],
  "metrics": { "n_rows": 10000, "n_cols": 25, "n_suggested_groups": 5 },
  "recommendation": "proceed"
}
```

## Failure modes

| Issue | Status | Recommendation |
|-------|--------|----------------|
| Dataset not found | `blocked` | `escalate` |
| No numeric columns | `blocked` | `escalate` |
| > 30% missing in key columns | `warning` | `proceed` (with imputation note) |
| All columns constant | `blocked` | `escalate` |
