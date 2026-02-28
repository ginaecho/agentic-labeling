# PersonaNamingAgent

**File:** `agents/persona_namer.py`  
**Class:** `PersonaNamingAgent`

## Role

Sends cluster profiles to Claude to generate human-readable persona names, taglines, descriptions, and traits. Applies the Clarity Gate to validate output quality before proceeding.

## Skills used

- [orchestrator_bus](../skills/orchestrator_bus.md) — `report()`

**Inline:** `clarity_gate` — checks avg_confidence ≥ 6, unique names, description references numbers (see [Skills index](../skills/README.md)).

## Inputs

- `profiles: dict` (from ClusteringAgent)
- `lineage: dict`
- `tone: str`
- `UserIntent`
- Orchestrator feedback

## Outputs

- `NamingResult`:
  - `personas: dict` (cid → name, tagline, description, traits, confidence)
  - `passed: bool`
  - `avg_confidence: float`
  - `issues: list[str]`

## Clarity Gate thresholds

- Avg confidence ≥ 6/10
- No duplicate persona names
- Every persona description references ≥ 2 quantitative signals (checked by regex)

## Communication protocol

Reports via [orchestrator_bus](../skills/orchestrator_bus.md):

```json
{
  "agent": "PersonaNamer",
  "status": "success | warning | blocked",
  "what_was_done": "Named 9 clusters; avg confidence=7.2; Clarity Gate passed",
  "what_was_not_done": "",
  "doubts": "Cluster 4 and 7 are similar; confidence 6/10 for both",
  "issues": [],
  "metrics": { "n_clusters": 9, "avg_confidence": 7.2, "gate_passed": true },
  "recommendation": "proceed"
}
```
