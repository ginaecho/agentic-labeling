# UserInputAgent

**File:** `agents/user_input.py`
**Class:** `UserInputAgent`

## Role

Collects and validates the user's clustering intent before any computation. This is the entry point of the pipeline.

## Skills used

None (pure interactive I/O).

## Inputs

- None (interactive terminal)

## Outputs

- `UserIntent` dataclass:
  - `target_entity: str` — what is being clustered
  - `business_purpose: str` — why we are clustering
  - `dataset_path: str` — path to raw data file
  - `constraints: str` — optional free-text constraints
  - `n_clusters_requested: int | None` — if set, ClusteringAgent uses this exact k and skips silhouette optimisation
  - `must_have_clusters: list[str]` — cluster labels that MUST appear in the final personas (e.g. `['traveller', 'VIP']`); enforced by PersonaNamingAgent Clarity Gate

## Questions asked (interactive)

| # | Question | Field set | Required? |
|---|----------|-----------|-----------|
| 1 | "What entity are you clustering?" | `target_entity` | Yes |
| 2 | "What is the business purpose?" | `business_purpose` | Yes (follow-up if < 20 chars) |
| 3 | "Dataset path?" | `dataset_path` | No (defaults to config) |
| 4 | "Any constraints?" | `constraints` | No |
| 5 | "How many clusters? (Enter = data-driven)" | `n_clusters_requested` | No |
| 6 | "Must any specific types appear as clusters?" | `must_have_clusters` | No |

## Communication protocol

Reports via [orchestrator_bus](../skills/orchestrator_bus.md):

```json
{
  "agent": "UserInput",
  "status": "success | blocked",
  "what_was_done": "Collected target entity and business purpose from user",
  "what_was_not_done": "Did not validate that the dataset file actually exists",
  "doubts": "Business purpose may still be too vague",
  "issues": [],
  "metrics": {
    "target_entity": "customers",
    "purpose_length": 72,
    "has_constraints": false,
    "n_clusters_requested": 5,
    "must_have_clusters": ["traveller", "VIP"]
  },
  "recommendation": "proceed"
}
```

## Retry behaviour

- If user gives a vague business purpose (< 20 chars), agent asks one clarifying follow-up question before proceeding.
- If running non-interactively (EOFError), agent uses sensible defaults from `config.yaml` and reports `doubts="running with defaults"`.
- `n_clusters_requested` of 1 or 0 is rejected; data-driven selection is used instead.
