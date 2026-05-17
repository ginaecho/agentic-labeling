"""Decision Maker review when the system converges.

Trigger: the convergence loop ends with status='converged' (ARI ≥ 0.90
for STABILITY_WINDOW consecutive runs).

What this does: reads the full feedback log + run_history.jsonl and
asks the Decision Maker LLM to judge each rule's contribution to
convergence:

  useful  → rule was active during the converging runs and points at
            an actual stable property of the final clusters → promote
            to 'high' priority for future runs on this dataset.
  neutral → rule was active but its target / scope no longer obviously
            applies → keep at 'medium', flag for human review later.
  noise   → rule was active but contradicts the converged result →
            mark active=false so it stops influencing future runs.

The verdict + new priority is written back to outputs/user_feedback_log.jsonl
via feedback_store; the original entry's metadata is preserved
(judge_severity, source, provenance) so the audit trail stays intact.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import time
from typing import Optional

import anthropic

REVIEW_MODEL = os.environ.get('REVIEW_MODEL', 'claude-sonnet-4-6')
RUNS_DIR = pathlib.Path('experiments/runs')


_REVIEW_SYSTEM = """You are the Decision Maker for a self-improving
clustering pipeline. The system just CONVERGED — three consecutive runs
produced nearly identical clusterings (ARI ≥ 0.90). Now you must judge
which of the accumulated rules genuinely contributed to that convergence
and which were noise.

Inputs:
  • The current set of active rules (each tagged with source judge,
    target cluster, priority, and judge_severity).
  • The final converged clustering's persona names + key metrics.
  • The run-by-run timeline showing when each metric stabilised.

For every rule, decide ONE of:
  - "useful": rule directly explains a property of the final converged
    clustering, or its prescription clearly improved a metric trajectory.
    These get promoted to HIGH priority for future runs on this dataset.
  - "neutral": rule was reasonable but the converged result neither
    confirms nor contradicts it. Keep at medium priority. Flag for human.
  - "noise": rule contradicts the converged result (e.g. recommended a
    rename that the converged system did not adopt, or targeted a
    cluster that no longer exists). Deactivate it.

OUTPUT — strict JSON only:
{
  "verdicts": [
    {"id": "fb_xxxx", "verdict": "useful" | "neutral" | "noise",
     "reasoning": "<2 sentences explaining the verdict>"}
  ]
}

Every input rule id must appear exactly once.
"""


def _load_run_history() -> list[dict]:
    p = RUNS_DIR / 'run_history.jsonl'
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _load_final_personas() -> dict:
    """Read the most recent per-run snapshot's personas.json."""
    runs = sorted(d for d in RUNS_DIR.glob('run_*') if d.is_dir())
    if not runs:
        return {}
    last = runs[-1] / 'outputs' / 'personas.json'
    return json.loads(last.read_text()) if last.exists() else {}


def _build_review_prompt(rules: list[dict], history: list[dict],
                          personas: dict) -> str:
    lines = ['CONVERGED CLUSTERING — final persona names + sizes:']
    for cid, d in personas.items():
        p = d.get('persona', {}) or {}
        s = d.get('cluster_stats', {})
        n = s.get('n_entities', s.get('n_customers', 0))
        pct = s.get('pct_total', s.get('pct_of_total', 0) * 100)
        lines.append(f'  Cluster {cid}: "{p.get("name", "?")}"  '
                     f'n={n} ({pct:.1f}%)')

    lines.append('')
    lines.append('RUN-BY-RUN METRIC TIMELINE (showing how convergence happened):')
    for row in history[-10:]:
        m = row.get('metrics', {})
        d = row.get('diff_vs_prev', {})
        lines.append(
            f'  run {row.get("run")}: '
            f'F1={m.get("cv_f1_macro")}  '
            f'n_clusters={m.get("n_clusters")}  '
            f'ARI vs prev={d.get("ari")}  '
            f'accepted_rules_this_run={row.get("arbiter", {}).get("n_accepted")}'
        )

    lines.append('')
    lines.append('ACTIVE RULES TO REVIEW:')
    for r in rules:
        text = (r.get('rule') or r.get('hint') or '').strip()
        target = (r.get('target_cluster_name') or r.get('target_cluster_id')
                   or 'GLOBAL')
        lines.append('')
        lines.append(f'  id={r.get("id")}  source={r.get("source", "?")}  '
                     f'judge_severity={r.get("judge_severity", "?")}  '
                     f'target={target}')
        lines.append(f'    {text}')
    return '\n'.join(lines)


def _parse_verdicts(text: str) -> list[dict]:
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
        v = obj.get('verdicts', [])
        return v if isinstance(v, list) else []
    except json.JSONDecodeError:
        return []


def review_after_convergence() -> dict:
    """Score every active agent-generated rule. Update priorities / active
    flags via feedback_store. Returns a summary dict."""
    from ui import feedback_store

    active = [e for e in feedback_store.read_all() if e.get('active', True)
              and e.get('provenance') == 'agent']
    if not active:
        print('  [conv-review] no agent-generated rules to review.')
        return {'reviewed': 0}

    history = _load_run_history()
    personas = _load_final_personas()
    if not personas:
        print('  [conv-review] no persona snapshot found — skipping review.')
        return {'reviewed': 0}

    prompt = _build_review_prompt(active, history, personas)
    print(f'  [conv-review] asking Decision Maker about {len(active)} rules…')
    t0 = time.perf_counter()
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=REVIEW_MODEL,
            max_tokens=3500,
            system=_REVIEW_SYSTEM,
            messages=[{'role': 'user', 'content': prompt}],
        )
        verdicts = _parse_verdicts(resp.content[0].text)
    except Exception as e:
        print(f'  [conv-review] LLM call failed: {e}')
        return {'reviewed': 0, 'error': str(e)}

    counts = {'useful': 0, 'neutral': 0, 'noise': 0}
    review_ledger = RUNS_DIR / 'convergence_review_ledger.jsonl'
    review_ledger.parent.mkdir(parents=True, exist_ok=True)
    with review_ledger.open('a', encoding='utf-8') as ledger:
        for v in verdicts:
            rule_id = v.get('id')
            verdict = v.get('verdict')
            reasoning = v.get('reasoning', '')
            if not rule_id or verdict not in ('useful', 'neutral', 'noise'):
                continue
            counts[verdict] += 1

            if verdict == 'useful':
                feedback_store.update_priority(rule_id, 'high')
            elif verdict == 'noise':
                feedback_store.set_active(rule_id, False)
            # neutral: leave priority + active as-is

            ledger.write(json.dumps({
                'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'rule_id': rule_id,
                'verdict': verdict,
                'reasoning': reasoning,
            }, ensure_ascii=False) + '\n')

    elapsed = time.perf_counter() - t0
    print(f'  [conv-review] {counts}  ({elapsed:.1f}s)')
    return {
        'reviewed': sum(counts.values()),
        'useful': counts['useful'],
        'neutral': counts['neutral'],
        'noise': counts['noise'],
    }


if __name__ == '__main__':
    sys.exit(0 if review_after_convergence() else 1)
