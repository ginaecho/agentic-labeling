# FeatureSelectionAgent

**File:** `agents/feature_selector.py`
**Class:** `FeatureSelectionAgent`

## Role

Scores all engineered features using PCA importance and autoencoder reconstruction error, then applies a VIF collinearity gate and a correlation gate, and finally asks the LLM (via OrchestratorBus) to select the optimal subset for clustering.

## Skills used

- [orchestrator_bus](../skills/orchestrator_bus.md) — `ask()`, `report()`
- [vif_checker](../skills/vif_checker.md) — `compute_vif()`, `remove_high_vif()`, `flag_high_correlation()`

**Inline:** PCA importance scoring, autoencoder reconstruction-error scoring (see [Skills index](../skills/README.md)).

## Inputs

- Engineered feature DataFrame
- `UserIntent`
- Orchestrator feedback (free-text)
- `vif_threshold` override (set dynamically by the Orchestrator per iteration)
- `feature_focus` hint (injected into the LLM prompt by the Orchestrator)
- `modality: 'tabular' | 'text'` — defaults to tabular. When `'text'`, the
  agent **short-circuits**: PCA / autoencoder / VIF are skipped (embeddings
  from `TextPreparerAgent` are already compact + decorrelated), every
  embedding column is kept, no LLM call is made, and the same
  `FeatureSelectionResult` shape is returned so the loop history stays valid.

## Outputs

- `FeatureSelectionResult`:
  - `selected_features: list[str]`
  - `n_features: int`
  - `pca_scores: dict[str, float]`
  - `ae_scores: dict[str, float]`
  - `vif_table: dict[str, float]`
  - `removed_by_vif: list[str]`
  - `reasoning: str`

## Pipeline (in order)

1. Log-transform any skewed column (|skewness| > 2.0) — auto-detected from data, no hard-coded column names
2. Scale to zero-mean, unit-variance
3. PCA importance score (weighted squared loadings)
4. Autoencoder reconstruction error score
5. Combined score = 0.5 × PCA + 0.5 × AE
6. **VIF gate** — iteratively remove highest-VIF feature until all VIF < threshold (default 10.0, adjusted by Orchestrator)
7. **Correlation gate** — flag pairs with |r| > 0.85
8. LLM selects final subset from ranked, VIF-filtered list

## Quality gates

| Gate | Default threshold | Who controls |
|------|-------------------|--------------|
| VIF | < 10.0 | Orchestrator — adjusted dynamically per iteration |
| Pairwise correlation | \|r\| < 0.85 | Fixed |
| Minimum features surviving | ≥ 10 | Fixed |
| Typical selected range | 25–55 | LLM decision |

## Communication protocol

Reports via [orchestrator_bus](../skills/orchestrator_bus.md):

```json
{
  "agent": "FeatureSelector",
  "status": "success | warning | blocked",
  "what_was_done": "Scored 108 features, removed 22 via VIF gate, LLM selected 45",
  "what_was_not_done": "Did not apply p-value gate (sufficient features after VIF)",
  "doubts": "",
  "issues": [],
  "metrics": { "n_input_features": 108, "n_after_vif": 86, "n_selected": 45, "max_vif_remaining": 9.1 },
  "recommendation": "proceed"
}
```

## Failure modes

| Issue | Status | Recommendation |
|-------|--------|----------------|
| < 10 features survive VIF gate | `blocked` | `escalate` — Orchestrator should relax VIF threshold |
| LLM returns invalid JSON | `warning` | `retry` |
