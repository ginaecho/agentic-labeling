# PersonaNamingAgent

**File:** `agents/persona_namer.py`
**Class:** `PersonaNamingAgent`

## Role

Sends cluster profiles to the LLM (via OrchestratorBus) to generate human-readable persona names, taglines, descriptions, and traits. Applies the Clarity Gate to validate output quality before proceeding. Works with any domain — cluster profiles are generic (feature deviations from mean), not domain-specific.

## Skills used

- [orchestrator_bus](../skills/orchestrator_bus.md) — `ask()`, `report()`

## Inputs

- `profiles: dict` — from ClusteringAgent; each entry has:
  - `n_entities: int`
  - `pct_total: float`
  - `top_above_average: dict[str, float]` — top features where cluster is above global mean (or **c-TF-IDF distinctive terms** in text mode)
  - `top_below_average: dict[str, float]` — top features where cluster is below global mean (or **absent-here terms** in text mode)
  - `feature_means: dict[str, float]` — mean value per feature
  - `lineage: dict` — depth, parent, siblings
  - `modality: str` *(text mode only)* — `"text"`
  - `top_terms: list[str]` *(text mode only)* — ordered distinctive terms
  - `representative_docs: list[str]` *(text mode only)* — documents nearest the cluster centroid

When `modality == 'text'` (or `representative_docs` is present), the
internal `_format_cluster_block()` renders a text-aware prompt with
`DISTINCTIVE TERMS`, `ABSENT-HERE TERMS`, and `REPRESENTATIVE DOCUMENTS`
sections instead of the tabular ABOVE/BELOW-AVERAGE numeric layout. The
output schema (name/tagline/description/traits/confidence) is unchanged,
so the Clarity Gate, UI, and classifier path stay compatible.
- `lineage: dict`
- `tone: str` — one of `easy | professional | data-driven | creative`
- `user_intent: UserIntent` — used to extract `must_have_clusters` for the LLM prompt and Clarity Gate
- Orchestrator feedback (free-text)

## Outputs

- `NamingResult`:
  - `personas: dict` — cid → `{name, tagline, description, dominant_features, traits, confidence}`
  - `passed: bool`
  - `avg_confidence: float`
  - `issues: list[str]`

## Must-have cluster constraint

If `user_intent.must_have_clusters` is non-empty (e.g. `['traveller', 'VIP']`):
1. A **MANDATORY CLUSTER REQUIREMENT** section is appended to the LLM prompt listing the required types
2. The Clarity Gate checks that every required type appears (case-insensitive substring match, both hyphenated and space-separated variants) in at least one persona name or description
3. Any missing required types are added to `issues` and trigger `action='recluster'`

## Clarity Gate thresholds

1. Avg LLM confidence ≥ 6/10
2. No duplicate persona names across all clusters
3. All `must_have_clusters` types covered in persona names/descriptions (if any were specified)

## Communication protocol

Reports via [orchestrator_bus](../skills/orchestrator_bus.md):

```json
{
  "agent": "PersonaNamer",
  "status": "success | warning | blocked",
  "what_was_done": "Named 9 clusters using LLM (tone='easy'); Clarity Gate PASSED; avg confidence=7.2",
  "what_was_not_done": "Did not validate description text references specific numbers",
  "doubts": "",
  "issues": [],
  "metrics": {
    "n_clusters": 9,
    "avg_confidence": 7.2,
    "gate_passed": true,
    "names_unique": true,
    "must_have_clusters": ["traveller", "VIP"]
  },
  "recommendation": "proceed"
}
```

## Failure modes

| Issue | Status | Recommendation |
|-------|--------|----------------|
| Avg confidence < 6.0 | `warning` or `blocked` | `retry` — recluster |
| Duplicate persona names | `blocked` | `retry` — recluster |
| Must-have cluster type not found in any persona | `blocked` | `retry` — recluster |
| LLM response not valid JSON | returns `recluster` action | Orchestrator retries |
