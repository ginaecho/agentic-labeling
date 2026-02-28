# Orchestrator

**File:** `agents/orchestrator.py`  
**Class:** `Orchestrator`

## Role

Central coordinator. Owns the pipeline state, routes feedback between agents, maintains the orchestrator message log, and calls Claude to diagnose complex failures. Presents a human checkpoint when the pipeline converges or exhausts its retry budget.

## Skills used

None (orchestrator consumes messages from [orchestrator_bus](../skills/orchestrator_bus.md) and coordinates agents).

## Inputs

- `config: dict` (from `config.yaml`)
- `user_intent: UserIntent`
- `features_path: str`

## Outputs

- `dict` with keys `status`, `personas`, `run_history`, `timing`, `claude_usage`

## Responsibilities

1. Receive `OrchestratorMessage` from every agent via the message bus
2. Log all messages to `pipeline_log` (saved to `outputs/pipeline_log.json`)
3. Use Claude to analyse failure reports and decide routing
4. Enforce per-loop retry budgets
5. Present human checkpoint with full pipeline log summary

## Routing decisions (Claude-assisted)

| Agent reports | Orchestrator considers |
|---------------|-------------------------|
| FeatureSelector BLOCKED | → route to FeatureEngineer (more features needed) |
| Clusterer WARNING (low silhouette) | → try different k or algorithm |
| Clusterer BLOCKED | → route to FeatureSelector |
| PersonaNamer BLOCKED | → route to Clusterer |
| Classifier BLOCKED | → route to FeatureSelector or Clusterer |
| Any `recommendation=escalate` | → trigger human checkpoint immediately |
