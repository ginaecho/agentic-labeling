# Adaptive-Learning Convergence Experiment

Replace the human-in-the-loop reviewer with three **judge agents** and run
the pipeline back-to-back for 3 hours. The question is falsifiable:

> Do the named clusters stabilise (higher ARI vs the previous run) as the
> feedback log grows, while silhouette / F1 hold or improve and max-VIF
> stays low?

## Architecture

```
3-hour loop ─────────────────────────────────────────────────
│
│  run_pipeline.py --bypass --max-iterations 6
│       ↓ outputs/{personas.json, classifier_metrics.json,
│              cluster_labels.csv, cluster_profiles.json}
│
│  3 Judge Agents critique in sequence (each Claude Sonnet 4.6):
│    • Statistical+Analytical  — where does this conclusion come from?
│                                which columns/data points prove it?
│    • Business+Explainability — is this human-readable? simpler name?
│                                over-complex is always bad.
│    • Domain                  — does the name reflect what data shows?
│                                why? compared to others? show proof.
│       ↓ experiments/runs/run_NNN/critiques.jsonl
│
│  Arbiter (Claude Sonnet 4.6) — one accept/reject per critique:
│    accept → append rule to outputs/user_feedback_log.jsonl
│             (the LIVE feedback log; next pipeline run reads it)
│    reject → append rebuttal to experiments/runs/learning_ledger.jsonl
│
│  Diff vs previous run → experiments/runs/run_history.jsonl
│    ARI · name-Jaccard · size-L1 · F1-delta
│
│  Stop when ARI ≥ 0.90 for 3 consecutive runs OR 3-hour wall clock.
└──────────────────────────────────────────────────────────────
```

## Run it

```bash
# Foreground (you'll see live logs):
python -m experiments.convergence_loop

# Detached — survives terminal close, on macOS:
experiments/run_detached.sh

# Custom budget / dataset intent:
experiments/run_detached.sh \
    --max-hours 6 \
    --intent-target "credit card customers" \
    --intent-purpose "discover fraud-risk personas for screening"

# Reset feedback log so the experiment starts from a clean state:
experiments/run_detached.sh --reset-feedback
```

## Monitor

```bash
tail -f experiments/runs/run.log              # full stdout
cat experiments/runs/run_history.jsonl | jq   # per-run metrics, one line each
cat experiments/runs/learning_ledger.jsonl | jq   # every critique + arbiter decision
```

## Stop

```bash
kill "$(cat experiments/runs/current.pid)"
```

## What lands on disk

```
experiments/runs/
├── run.log                          # combined stdout of the loop
├── current.pid                      # PID while running (deleted on exit)
├── run_history.jsonl                # one line per run — the headline timeline
├── learning_ledger.jsonl            # every critique + arbiter decision + reasoning
└── run_NNN/
    ├── pipeline.log                 # one inner pipeline's stdout
    ├── critiques.jsonl              # the 3 judges' raw output for that run
    └── outputs/                     # FROZEN SNAPSHOT of outputs/ at end of run
        ├── personas.json
        ├── classifier_metrics.json
        ├── cluster_profiles.json
        ├── cluster_labels.csv
        ├── pipeline_log.json
        ├── agents_conversation.txt
        ├── user_feedback_log_before_judges.jsonl
        └── user_feedback_log_after_judges.jsonl
```

The two `user_feedback_log_*` snapshots let you diff what the judges
contributed during each run — that is the "what did the pipeline learn"
trace, as a literal file diff.

## Cost / wall-time

- Inner run with 6 iterations: ~12–20 minutes, ~$0.80–$1.20 of API.
- Judges + arbiter add ~$0.30–$0.50 per run.
- 3-hour budget ≈ 8–12 outer runs ≈ **$10–$18 of API** per experiment.

If accept-rate stays >90% or <10% across two consecutive runs, the
loop prints a warning — that usually means the arbiter prompt needs
a sharper accept/reject rubric.

## Environment

Same as the main pipeline:
- `LLM_API_KEY` (or `ANTHROPIC_API_KEY`) — required.
- `JUDGE_MODEL` / `ARBITER_MODEL` — optional, default `claude-sonnet-4-6`.
