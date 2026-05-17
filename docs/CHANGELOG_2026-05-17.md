# Changelog — 2026-05-17

Branch: `feat/experiments` (local) → pushed as `feat/experiments-judges` (ruleset blocked direct update of `feat/experiments`).
Commits (oldest → newest):

| sha       | title                                                                        |
|-----------|------------------------------------------------------------------------------|
| `5c81667` | feat(experiments): 3-judge convergence loop for adaptive learning            |
| `1461aa9` | feat(experiments): make judges fully blind + stateless                       |
| `a6e7254` | test(experiments): runtime proof that judges are blind to each other         |
| `64c6a43` | feat(experiments): input asymmetry — judges see different evidence            |
| `ceb4002` | feat(experiments): live judges→arbiter→feedback test + bypass-mode EOF fix   |
| `eb36443` | test(experiments): runtime proof PersonaNamer consumes judge-generated rules |
| `bdcdc16` | feat(experiments): rule hygiene — dedup, convergence review, human fallback  |

A single day of design + build + verify that turned "adaptive learning is a slogan in the README" into "the full loop runs, the judges produce substantive critiques, every invariant is enforced at runtime, and rules end up inside the next PersonaNamer prompt — provable in 5 seconds for $0."

---

## 1. The experiment frame

The earlier code already had `outputs/user_feedback_log.jsonl` + `build_preferences_block()` (the UI's adaptive memory), but no way to test whether agents actually *learn* from those rules across runs without a human in the loop. The day's question:

> Replace the human reviewer with 3 judge agents. Run the pipeline back-to-back for 3 hours. Does the system converge (ARI ≥ 0.90 across 3 consecutive runs) as the feedback log grows? If yes, which rules were load-bearing? If not, what should a human review next?

That's a falsifiable test of the "adaptive" claim.

---

## 2. Three blind judges with asymmetric input views

The cheap version is three different prompts over the same evidence — theater. The version we built gives each judge **genuinely different evidence**, so disagreements come from different epistemic bases, not different lenses on the same data:

| Judge | Sees | Doesn't see |
|---|---|---|
| **Statistical+Analytical** | deviation ratios + sizes + per-group F1, with persona names **replaced by anonymous "Group A / Group B / ..."** | persona names, descriptions, raw rows |
| **Business+Explainability** | persona name + one-line description | sizes, features, ratios, F1, dataset subject — **zero digits** |
| **Domain** | names + ratios + **3 raw record samples per cluster** + dataset subject | (everything; only judge with row-level data) |

A round-trip mapping translates the statistical judge's "Group A" critique back to the real cluster id so the arbiter sees a unified id space.

### Runtime invariants (test_blindness.py — ~1 second, mocked LLM, 12 assertions)

1. Exactly 3 API calls per run
2. Each call: 1 user message, 0 assistant turns
3. 3 distinct system prompts
4. 3 distinct user prompts
5. No judge's prompt contains another's rubric
6. Each per-judge packet's unique marker appears in exactly one prompt
7. No prior-critique JSON fields in any prompt
8. **Statistical judge prompt contains no persona names**
9. **Business judge prompt contains zero digit characters**
10. **Only the domain judge prompt contains raw row samples ("row#")**
11. Statistical packet uses anonymized "Group A/B/…" labels
12. Alias round-trip maps "Group A" / "A" back to real cluster ids

---

## 3. Arbiter + rule schema

Single-turn accept/reject per critique. Accepted rules land in `outputs/user_feedback_log.jsonl` with strict provenance:

```json
{
  "id": "fb_xxxx",
  "priority": "medium",
  "source": "judge:statistical",
  "provenance": "agent",
  "judge_severity": "high",
  "convergence_verdict": "unreviewed",
  "rule": "..."
}
```

Important: **all agent rules land at MEDIUM regardless of the judge's stated severity** — they are unverified hypotheses. Promotion to HIGH happens only when the post-convergence Decision Maker review marks them load-bearing, or a human explicitly endorses them.

Live API test (`test_judges_live.py`) on a 5-cluster synthetic personas.json (drawn from a real prior-run structure):

- 11 critiques in parallel (~5 s wall time, ~$0.40 API)
- **Statistical + business judges INDEPENDENTLY converged on the same near-duplicate-cluster defect** — different evidence, same finding. That is the multi-view validation working.
- Statistical judge also caught a real data error: `city_pop=0.42x` listed in the "above-avg" block when 0.42x is below the 1.00x mean.
- Arbiter accepted 6/11 (55%) — correctly rejecting duplicates and a methodologically flawed row-level-outlier-vs-cluster-stat critique.

---

## 4. Consumption-side proof — does PersonaNamer actually read this?

This was the question the smoke run failed to answer (fraud pipeline got stuck in an oversized-cluster loop; wholesale-customers hit the bypass-mode entity-column input prompt). Built a self-contained runtime test that:

1. Writes 6 judge-generated rules into the feedback log (backed up + restored at end via `try/finally`).
2. Instantiates the **real** `PersonaNamingAgent` class.
3. Wraps `OrchestratorBus.ask()` to capture the prompt.
4. Calls `PersonaNamingAgent.run(profiles, lineage, ...)`.
5. Asserts: prompt contains "USER PREFERENCES" header, every one of the 6 rules' distinctive text, and the prefs block is **prepended** before the cluster-data section.

`[PersonaNamer] Injected 11 lines of user preferences from prior UI feedback.` — proof line in the agent's own stdout, not a mock.

---

## 5. Rule hygiene — dedup, convergence review, human fallback

User push: "11 lines injected — call an LLM to clean it up, only keep the unique ones."

Built four-part rule management:

### `experiments/dedup_prefs.py` — LLM compaction before injection

- Reads active feedback entries (NOT MUTATED).
- LLM merges near-duplicates, drops stale rules (rules targeting a cluster name that no longer exists).
- Cached by SHA-256 of input entries — second PersonaNamer call against the same log is free.
- Per-decision rationale written to `experiments/runs/rule_compaction_ledger.jsonl`.
- Graceful fallback to raw entries if the dedup LLM fails — pipeline never blocks on this.

Wired via `EXPERIMENT_DEDUP_PREFS=1` env var set by the convergence loop in the inner pipeline's environment. Live UI runs (env unset) keep the original verbatim path → **zero behaviour change for non-experiment.**

### `experiments/convergence_review.py` — Decision Maker on converge

Fires when `converged=True` (ARI ≥ 0.90 × 3 runs). Reads:
- final converged clustering (`personas.json`)
- run-by-run metric trajectory (`run_history.jsonl`)
- all active agent-generated rules

Asks Decision Maker LLM: *"These rules were active when the system converged. Which were load-bearing?"*

| verdict | action |
|---|---|
| `useful` | promote to HIGH priority |
| `neutral` | keep MEDIUM |
| `noise`   | mark `active=false` (silently retired) |

Verdicts logged to `experiments/runs/convergence_review_ledger.jsonl`.

### `experiments/human_review.py` — no-convergence fallback

Fires when 3-hour budget exhausts without ARI streak. Writes `experiments/runs/human_review.md`:

- Rules grouped by source judge so the reviewer can compare critiques from the same lens.
- Per-rule: text, target, current priority, judge_severity, ARI / F1 trajectory while it was active.
- Per-rule checkboxes: `[ ] KEEP` / `[ ] REVISE` / `[ ] DROP`, plus a `\`\`\`revised\`\`\`` code-fence the user can edit.

Companion `apply` mode: `python -m experiments.human_review apply experiments/runs/human_review.md` re-imports the edited markdown, stamping kept/revised rules `provenance='human'` `priority='high'`.

### Convergence loop branches at exit

```python
if converged:
    review_after_convergence()       # promote/demote based on what was load-bearing
else:
    write_review_md()                # hand to human
```

---

## 6. Verification status — what is/isn't proven

**Proven end-to-end at runtime** (no full pipeline needed):

```
judges → arbiter → feedback_store.append() → user_feedback_log.jsonl
   → build_preferences_block() / build_deduplicated_preferences_block()
   → PersonaNamer prompt → LLM
```

12 blindness/asymmetry invariants + the consumption test all pass in ~6 seconds for $0.

**Not yet proven**: a full 3-hour live convergence run on fraud data. Two pipeline-side blockers surfaced today:

- Fraud data: at k=3 the clusterer keeps producing a 92%-oversized cluster C2; the LLM-routing for sub-clustering vs reselect features loops without progress. After 25 minutes of CPU burn with no log output, we killed it.
- Wholesale-customers (small, fast alternative): the pipeline assumes transactional rows that need to be aggregated into entity-level features. With a pre-aggregated dataset there is no natural entity-ID column; the bypass fallback picked "Channel" (2 distinct values) → FeatureEngineer correctly rejected the resulting 2-row × 49-col matrix.

These are pre-existing pipeline issues, not experiment scaffolding bugs. The experiment scaffolding is verified in isolation; the convergence run is unblocked the moment one of those bugs is addressed.

---

## 7. Side fixes

- `agents/feature_engineer.py:_resolve_col` now catches `EOFError` from `input()` so non-interactive bypass / detached / piped runs fall through to the caller's default (`df.columns[0]`) instead of crashing.
- `agents/orchestrator.py:save_outputs` now writes `outputs/cluster_labels.csv` so the inter-run ARI computation has a real label vector to read.
- `run_pipeline.py` gained `--bypass`, `--max-iterations`, `--intent-target`, `--intent-purpose` flags so the convergence loop can drive headless runs.

---

## 8. Files added (all under `experiments/` unless noted)

```
experiments/
├── __init__.py
├── README.md                                  — full design + run/monitor/stop
├── judges.py                                  — 3 blind asymmetric judges
├── arbiter.py                                 — accept/reject + medium-priority/agent-provenance
├── diff_personas.py                           — ARI + name-Jaccard + size-L1 + F1-delta
├── convergence_loop.py                        — outer loop with 3-h budget + dedup + branch
├── dedup_prefs.py                             — LLM compaction (cached, ledgered)
├── convergence_review.py                      — Decision Maker post-convergence
├── human_review.py                            — write + apply for the human fallback
├── run_detached.sh                            — macOS-safe nohup launcher
├── test_blindness.py                          — 12 blindness/asymmetry invariants (mocked, ~1 s)
├── test_judges_live.py                        — full live chain on synthetic personas (~$0.40)
└── test_persona_namer_consumes_feedback.py    — proves prefs reach PersonaNamer prompt (~5 s, $0)

agents/
├── feature_engineer.py                        — EOFError fallback in bypass mode
├── orchestrator.py                            — saves cluster_labels.csv
└── persona_namer.py                           — env-gated dedup branch

run_pipeline.py                                 — new CLI flags
```

---

## 9. What's worth doing next

1. **Fix the pipeline-side blockers** so a real 3-hour convergence run can execute:
   - Make bypass mode handle the entity-column for pre-aggregated datasets (or document the requirement).
   - Diagnose the k=3 / 92%-oversized-cluster loop on fraud data — likely a sub-clustering deepening-loop termination issue.
2. **Compare convergence on different datasets**: once the above is fixed, run the experiment on 2–3 datasets from `data/raw/` and chart ARI / F1 over runs. That's the publishable claim.
3. **Open PR from `feat/experiments-judges` into `main`** so this work lands. (Direct update of `feat/experiments` is blocked by a repo ruleset; that's why today's push went to a new branch.)
