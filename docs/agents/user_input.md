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

## Communication protocol

Reports via [orchestrator_bus](../skills/orchestrator_bus.md):

```json
{
  "agent": "UserInput",
  "status": "success | blocked",
  "what_was_done": "Collected target entity and business purpose from user",
  "what_was_not_done": "",
  "doubts": "Business purpose may still be too vague",
  "issues": [],
  "metrics": {},
  "recommendation": "proceed"
}
```

## Retry behaviour

- If user gives a vague business purpose (< 20 chars), agent asks one clarifying follow-up question before proceeding.
- If running non-interactively (EOFError), agent uses sensible defaults from `config.yaml` and reports `doubts="running with defaults"`.
