# orchestrator_bus — Agent → Orchestrator Communication

**File:** `skills/orchestrator_bus.py`  
**Used by:** All agents

## Purpose

Provides a shared message bus so every agent can report structured status messages to the orchestrator without tight coupling.

## API

```python
from skills.orchestrator_bus import OrchestratorMessage, OrchestratorBus

msg = OrchestratorMessage(
    agent="FeatureSelector",
    iteration=2,
    status="success",           # "success" | "warning" | "blocked" | "failure"
    what_was_done="...",
    what_was_not_done="...",
    doubts="...",
    issues=[],
    metrics={"n_features": 45},
    recommendation="proceed",   # "proceed" | "retry" | "escalate"
    context={},                 # arbitrary agent-specific payload
)

bus = OrchestratorBus()
bus.report(msg)
log = bus.get_log()             # list[OrchestratorMessage]
bus.save_log("outputs/pipeline_log.json")
```

## Status meanings

| Status | When to use |
|--------|-------------|
| `success` | Agent completed its task with no issues |
| `warning` | Agent completed, but with caveats (e.g. low silhouette, sparse features) |
| `blocked` | Agent cannot proceed; orchestrator must reroute |
| `failure` | Unexpected exception; pipeline should halt or retry |

## Recommendation meanings

| Recommendation | When to use |
|----------------|-------------|
| `proceed` | Orchestrator should move to the next agent |
| `retry` | Orchestrator should re-run this agent with adjusted params |
| `escalate` | Trigger human checkpoint immediately |

## See also

- [Orchestrator](../agents/orchestrator.md) — consumes the log and routes on `recommendation`
