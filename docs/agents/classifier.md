# ClassifierAgent

**File:** `agents/classifier.py`  
**Class:** `ClassifierAgent`

## Role

Treats persona labels as pseudo ground truth, trains a Random Forest classifier, evaluates cluster separability via stratified CV, and routes the pipeline back to feature selection or clustering if performance is poor.

## Skills used

- [orchestrator_bus](../skills/orchestrator_bus.md) — `report()`

**Inline:** `train_classifier` — RandomForestClassifier, stratified k-fold CV, accuracy/F1 reporting (see [Skills index](../skills/README.md)).

## Inputs

- Feature DataFrame
- `cluster_labels: pd.Series`
- `personas: dict`
- `UserIntent`
- History + feedback

## Outputs

- `ClassifierResult`:
  - `action: str` (proceed | reselect_features | recluster)
  - `cv_accuracy`, `cv_f1_macro`, `cv_f1_weighted: float`
  - `per_class_f1: dict[str, float]`
  - `feature_importances: dict[str, float]`
  - `reasoning: str`

## Quality gate

- CV macro-F1 ≥ 0.70 → proceed
- Below threshold → Claude diagnoses and routes (reselect_features or recluster)

## Communication protocol

Reports via [orchestrator_bus](../skills/orchestrator_bus.md):

```json
{
  "agent": "Classifier",
  "status": "success | warning | blocked",
  "what_was_done": "Trained RF, 5-fold CV, computed feature importances",
  "what_was_not_done": "Did not compute SHAP values (not in current skill set)",
  "doubts": "Persona 'Moderate All-Rounder' is borderline (F1=0.65)",
  "issues": [],
  "metrics": { "cv_f1_macro": 0.82, "cv_accuracy": 0.85, "n_classes": 9 },
  "recommendation": "proceed"
}
```
