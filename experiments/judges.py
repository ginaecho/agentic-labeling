"""Three judge agents that critique a pipeline run's named clusters.

Each judge reads the same artifacts (personas.json + classifier_metrics.json
+ cluster_profiles.json + a brief feature_means summary) but evaluates them
through a distinct lens. Critiques are returned as structured Critique
records; the arbiter then decides which to accept as adaptive feedback.

Judges call their own Anthropic client so they do not pollute the
pipeline's LLM-usage accounting.
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

JUDGE_MODEL = os.environ.get('JUDGE_MODEL', 'claude-sonnet-4-6')


@dataclass
class Critique:
    judge: str                  # 'statistical' | 'business' | 'domain'
    target_cluster_id: Optional[str]   # None = global, else cluster id (str)
    target_cluster_name: Optional[str]
    severity: str               # 'high' | 'medium' | 'low'
    issue: str                  # what is wrong
    suggestion: str             # what to do instead (concrete, actionable)
    evidence: str               # which features / metrics support the critique

    def to_dict(self) -> dict:
        return asdict(self)


# ── Prompts ────────────────────────────────────────────────────────────────────

_RUBRIC_TAIL = """

OUTPUT FORMAT — strict JSON only, no prose before or after:
{
  "critiques": [
    {
      "target_cluster_id": "0" | null,
      "target_cluster_name": "...",
      "severity": "high" | "medium" | "low",
      "issue": "<one sentence>",
      "suggestion": "<one concrete action — rename / merge / split / change a feature / etc.>",
      "evidence": "<which metric or feature backs this — be specific>"
    }
  ]
}

Rules:
- Return at MOST 4 critiques per judge. Quality over quantity.
- Use target_cluster_id=null only for cross-cluster/global issues.
- Suggestion must be actionable, not "consider rethinking" — say what to do.
- If the run looks good, return {"critiques": []}.
"""

_STAT_SYSTEM = (
    "You are the STATISTICAL+ANALYTICAL JUDGE for a clustering pipeline. "
    "Your job is to interrogate every persona's evidence base: where does "
    "this conclusion come from? Which columns and which data points prove it? "
    "If a persona's name claims something the feature evidence does not support, "
    "or if the classifier F1 contradicts the persona structure, you must flag it."
    + _RUBRIC_TAIL
)

_BIZ_SYSTEM = (
    "You are the BUSINESS+EXPLAINABILITY JUDGE for a clustering pipeline. "
    "Your job is to check that every persona name is human-readable, directly "
    "describes what the cluster IS, and could be explained to a non-technical "
    "stakeholder in one sentence. Over-complex, jargon-y, or vague names are "
    "always bad — propose simpler, more direct names. Two personas with names "
    "that overlap in meaning are bad — propose a merge or a sharper distinction."
    + _RUBRIC_TAIL
)

_DOMAIN_SYSTEM = (
    "You are the DOMAIN JUDGE for a clustering pipeline. Given the dataset "
    "context (what the data is ABOUT — fraud detection, transactions, customer "
    "behaviour), your job is to verify each cluster name truly reflects what "
    "the underlying data shows. Compare each cluster against the OTHERS — does "
    "the name capture what makes it distinct from its neighbours? Demand proof: "
    "if the name is 'High-Spending Travelers', the travel-spend features must "
    "actually be elevated. If not, propose a name backed by the actual top features."
    + _RUBRIC_TAIL
)

_JUDGES = {
    'statistical': _STAT_SYSTEM,
    'business':    _BIZ_SYSTEM,
    'domain':      _DOMAIN_SYSTEM,
}


# ── Evidence packet shared across judges ──────────────────────────────────────

def _build_evidence_packet(run_dir: pathlib.Path) -> str:
    """Compact textual digest of one run's outputs that all three judges see."""
    out = run_dir / 'outputs'
    personas = json.loads((out / 'personas.json').read_text()) \
        if (out / 'personas.json').exists() else {}
    clf = json.loads((out / 'classifier_metrics.json').read_text()) \
        if (out / 'classifier_metrics.json').exists() else {}

    lines = []
    lines.append('DATASET CONTEXT: customer-level features engineered from '
                 'credit-card transaction history (fraud detection corpus).')
    lines.append('')
    lines.append(f'OVERALL: n_clusters={len(personas)}  '
                 f"cv_f1_macro={clf.get('cv_f1_macro', 0):.3f}  "
                 f"cv_accuracy={clf.get('cv_accuracy', 0):.3f}")
    lines.append('')
    lines.append('PER-CLUSTER EVIDENCE:')
    for cid, d in personas.items():
        stats = d.get('cluster_stats', {})
        p = d.get('persona', {}) or {}
        n = stats.get('n_entities', stats.get('n_customers', 0))
        pct = stats.get('pct_total', stats.get('pct_of_total', 0) * 100)
        top_above = stats.get('top_above_average', {}) or {}
        top_below = stats.get('top_below_average', {}) or {}
        f1 = (clf.get('per_class_f1') or {}).get(p.get('name', ''), None)

        lines.append('')
        lines.append(f'  Cluster {cid}: name="{p.get("name", "?")}"  '
                     f'tagline="{p.get("tagline", "")}"  '
                     f'n={n} ({pct:.1f}%)  '
                     f'cv_f1={f1:.3f}' if f1 is not None
                     else f'  Cluster {cid}: name="{p.get("name", "?")}"  '
                          f'tagline="{p.get("tagline", "")}"  '
                          f'n={n} ({pct:.1f}%)  cv_f1=n/a')
        if top_above:
            top5 = sorted(top_above.items(), key=lambda x: -x[1])[:5]
            lines.append('    above-avg: ' +
                         ', '.join(f'{k}={v:.2f}x' for k, v in top5))
        if top_below:
            bot3 = sorted(top_below.items(), key=lambda x: x[1])[:3]
            lines.append('    below-avg: ' +
                         ', '.join(f'{k}={v:.2f}x' for k, v in bot3))
        traits = p.get('traits') or []
        if traits:
            lines.append('    traits: ' + ' | '.join(traits[:3]))
    return '\n'.join(lines)


# ── Single-judge call ─────────────────────────────────────────────────────────

def _call_judge(client: anthropic.Anthropic, name: str,
                system: str, evidence: str) -> list[Critique]:
    user_prompt = (
        "Here is the run you are evaluating. Apply your judging rubric and "
        "return critiques in the JSON format your system prompt specified.\n\n"
        f"{evidence}"
    )
    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=2000,
        system=system,
        messages=[{'role': 'user', 'content': user_prompt}],
    )
    text = resp.content[0].text
    parsed = _parse_critiques(text)
    out = []
    for c in parsed:
        try:
            out.append(Critique(
                judge=name,
                target_cluster_id=(str(c['target_cluster_id'])
                                    if c.get('target_cluster_id') is not None
                                    else None),
                target_cluster_name=c.get('target_cluster_name'),
                severity=c.get('severity', 'medium'),
                issue=str(c.get('issue', '')).strip(),
                suggestion=str(c.get('suggestion', '')).strip(),
                evidence=str(c.get('evidence', '')).strip(),
            ))
        except Exception:
            continue
    return out


def _parse_critiques(text: str) -> list[dict]:
    """Extract the critiques list from the model's JSON response.
    Tolerates leading/trailing prose by grabbing the first {...} block."""
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
        c = obj.get('critiques', [])
        return c if isinstance(c, list) else []
    except json.JSONDecodeError:
        return []


# ── Public entry point ────────────────────────────────────────────────────────

def critique_run(run_dir: pathlib.Path) -> list[Critique]:
    """Run all three judges against one pipeline run. Returns flat list."""
    client = anthropic.Anthropic()
    evidence = _build_evidence_packet(run_dir)
    all_critiques: list[Critique] = []
    for name, system in _JUDGES.items():
        t0 = time.perf_counter()
        try:
            cs = _call_judge(client, name, system, evidence)
            print(f'  [judge:{name}] returned {len(cs)} critiques '
                  f'({time.perf_counter() - t0:.1f}s)')
            all_critiques.extend(cs)
        except Exception as e:
            print(f'  [judge:{name}] FAILED: {e}')
    return all_critiques
