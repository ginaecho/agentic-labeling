"""Arbiter — single-turn accept/reject on each judge critique.

Accepted critiques are written into the same outputs/user_feedback_log.jsonl
that the live UI uses, so the *next* pipeline run will pick them up via
ui.feedback_store.build_preferences_block(). Rejected critiques are
written to a learning_ledger.jsonl with the arbiter's rebuttal so we
can audit what the system chose to ignore and why.

The arbiter is intentionally simple: one LLM call per critique. Multi-turn
debate blows up the token budget without converging.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import time
from dataclasses import dataclass, asdict
from typing import Optional

import anthropic

from experiments.judges import Critique
from ui import feedback_store

ARBITER_MODEL = os.environ.get('ARBITER_MODEL', 'claude-sonnet-4-6')


@dataclass
class ArbiterDecision:
    accepted: bool
    reasoning: str
    feedback_id: Optional[str] = None    # if accepted, the fb_xxxx id we appended
    rule_text: Optional[str] = None      # the literal rule that was logged

    def to_dict(self) -> dict:
        return asdict(self)


_ARBITER_SYSTEM = """You are the Arbiter for a self-improving clustering pipeline.

Three judge agents critique each run's named clusters from three angles
(statistical, business, domain). Your job is to decide, for ONE critique
at a time, whether the pipeline should LEARN from it (accept) or whether
it should be politely rejected with a rebuttal.

Accept criteria (any one is enough):
  - The critique points to concrete evidence (a metric, a feature, a name
    clash) that the pipeline can act on next run.
  - The suggestion is specific, achievable, and likely to improve at least
    one of: silhouette, F1, VIF, or human readability of names.
  - The critique surfaces a real contradiction between the persona's claim
    and the underlying feature data.

Reject criteria (any one is enough):
  - Vague aesthetic preference with no actionable change.
  - The critique contradicts strong existing evidence in the run.
  - The suggested change would harm a quantitative metric without a
    compensating gain elsewhere.
  - Duplicate of guidance the pipeline already has in its preferences.

OUTPUT FORMAT — strict JSON only:
{
  "accepted": true | false,
  "reasoning": "<2-3 sentences explaining the decision>",
  "rule_text": "<if accepted: the imperative rule to add to the feedback log, e.g. 'Rename cluster 0 to High-Value Travelers because its travel-spend features are 2.4x average and city_pop is 1.8x'. Empty string if rejected.>"
}
"""


def _arbiter_user_prompt(critique: Critique, evidence: str,
                          existing_rules: str) -> str:
    parts = [
        "RUN EVIDENCE (the data the critique is reacting to):",
        evidence,
        "",
        "EXISTING ACCEPTED RULES (so you can spot duplicates):",
        existing_rules or "(none yet)",
        "",
        "CRITIQUE TO ARBITRATE:",
        f"  Judge: {critique.judge}",
        f"  Target cluster: {critique.target_cluster_id or 'GLOBAL'} "
        f"({critique.target_cluster_name or ''})",
        f"  Severity: {critique.severity}",
        f"  Issue: {critique.issue}",
        f"  Suggestion: {critique.suggestion}",
        f"  Evidence cited: {critique.evidence}",
        "",
        "Decide accept or reject. Return JSON only.",
    ]
    return '\n'.join(parts)


def _parse_decision(text: str) -> dict:
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        return {'accepted': False, 'reasoning': '(arbiter returned non-JSON)',
                'rule_text': ''}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {'accepted': False, 'reasoning': '(arbiter JSON parse failed)',
                'rule_text': ''}


def _existing_rules_block() -> str:
    """Compact dump of currently-active feedback rules."""
    try:
        entries = [e for e in feedback_store.read_all() if e.get('active', True)]
    except Exception:
        return ''
    lines = []
    for e in entries[-15:]:
        rule = e.get('rule') or e.get('hint') or ''
        if rule:
            target = e.get('target_cluster_name') or e.get('target_cluster_id') or 'global'
            lines.append(f'  - [{target}] {rule}')
    return '\n'.join(lines)


def _append_to_feedback_log(critique: Critique, rule_text: str) -> str:
    """Append accepted rule to outputs/user_feedback_log.jsonl.
    Returns the new entry's id.

    All agent-authored rules land at MEDIUM priority regardless of the
    judge's stated severity — they are unverified hypotheses. Promotion
    to 'high' happens only when:
      - convergence_review marks the rule as load-bearing for convergence
      - a human review explicitly endorses it (provenance becomes 'human')
    """
    is_global = critique.target_cluster_id is None
    entry: dict = {
        'type': 'global_rule' if is_global else 'naming_hint',
        'priority': 'medium',
        'source': f'judge:{critique.judge}',
        'provenance': 'agent',
        'judge_severity': critique.severity,    # preserved for later review
        'convergence_verdict': 'unreviewed',
    }
    if is_global:
        entry['rule'] = rule_text
    else:
        entry['target_cluster_id'] = critique.target_cluster_id
        entry['target_cluster_name'] = critique.target_cluster_name
        entry['hint'] = rule_text
    appended = feedback_store.append(entry)
    return appended.get('id', '')


def arbitrate(critiques: list[Critique], evidence: str,
              ledger_path: pathlib.Path) -> dict:
    """Decide on every critique. Writes accepted rules to user_feedback_log,
    writes the full transcript (accepts + rejects) to learning_ledger.jsonl.

    Returns counters for the run: {n_accepted, n_rejected, accept_rate}.
    """
    client = anthropic.Anthropic()
    n_accept = n_reject = 0
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    with ledger_path.open('a', encoding='utf-8') as ledger:
        for c in critiques:
            existing = _existing_rules_block()
            user_prompt = _arbiter_user_prompt(c, evidence, existing)
            t0 = time.perf_counter()
            try:
                resp = client.messages.create(
                    model=ARBITER_MODEL,
                    max_tokens=800,
                    system=_ARBITER_SYSTEM,
                    messages=[{'role': 'user', 'content': user_prompt}],
                )
                parsed = _parse_decision(resp.content[0].text)
            except Exception as e:
                parsed = {'accepted': False,
                          'reasoning': f'(arbiter call failed: {e})',
                          'rule_text': ''}

            accepted = bool(parsed.get('accepted'))
            reasoning = str(parsed.get('reasoning', '')).strip()
            rule_text = str(parsed.get('rule_text', '')).strip()

            decision = ArbiterDecision(
                accepted=accepted,
                reasoning=reasoning,
                rule_text=rule_text if accepted else None,
            )
            if accepted and rule_text:
                try:
                    decision.feedback_id = _append_to_feedback_log(c, rule_text)
                    n_accept += 1
                except Exception as e:
                    decision.accepted = False
                    decision.reasoning += f' (log-append failed: {e})'
                    n_reject += 1
            else:
                n_reject += 1

            ledger.write(json.dumps({
                'critique': c.to_dict(),
                'decision': decision.to_dict(),
                'elapsed_s': round(time.perf_counter() - t0, 2),
            }, ensure_ascii=False) + '\n')

    total = n_accept + n_reject
    accept_rate = (n_accept / total) if total else 0.0
    print(f'  [arbiter] accepted={n_accept}  rejected={n_reject}  '
          f'rate={accept_rate:.0%}')
    return {
        'n_accepted': n_accept,
        'n_rejected': n_reject,
        'accept_rate': round(accept_rate, 3),
    }
