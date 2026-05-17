"""Three BLIND judges with ASYMMETRIC INPUT VIEWS of the same clustering.

Design constraints (deliberate, do not weaken):

  • Stateless. Each judge call is a fresh API call with ONLY a system
    prompt + one user message. No conversation history.

  • Blind to each other. The three judges run in parallel threads and
    never see each other's critiques.

  • No pipeline knowledge. Prompts avoid "pipeline", "iteration",
    "agent", "orchestrator", "feedback", "adaptive". A judge sees
    only the artifact it is asked to evaluate.

  • INPUT ASYMMETRY (the structural source of disagreement):

      Statistical Judge — sees ONLY deviation-ratio tables, group sizes,
        and F1, with persona names REPLACED by anonymous labels
        ("Group A", "Group B", ...). Cannot anchor judgment on the
        name; can only reason about whether the numbers cohere.

      Business Judge — sees ONLY the persona names + one-line
        descriptions. No sizes, no features, no metrics. Pure
        readability evaluation: would a non-technical reader
        understand this name?

      Domain Judge — sees names + deviation ratios + a small RAW
        RECORD SAMPLE (3 random members per group, top-6 features
        each). Has a different epistemic base — actual data points —
        not just summary statistics.

    Now the three judges disagree because they have DIFFERENT EVIDENCE,
    not just different prompts over the same input. A statistical
    objection cannot be rebutted with "but the name sounds good", and
    a domain objection grounded in raw records cannot be hand-waved by
    a name's plausibility.

  • Anonymization round-trip. The statistical judge returns critiques
    keyed on "Group A" / "Group B"; critique_run() maps those back to
    the real cluster_ids before returning, so the arbiter sees a
    consistent id space.
"""
from __future__ import annotations

import concurrent.futures as _cf
import json
import os
import pathlib
import random
import re
import string
import time
from dataclasses import dataclass, asdict
from typing import Optional

import anthropic
import pandas as pd

JUDGE_MODEL = os.environ.get('JUDGE_MODEL', 'claude-sonnet-4-6')


@dataclass
class Critique:
    judge: str                  # 'statistical' | 'business' | 'domain'
    target_cluster_id: Optional[str]
    target_cluster_name: Optional[str]
    severity: str
    issue: str
    suggestion: str
    evidence: str

    def to_dict(self) -> dict:
        return asdict(self)


# ── System prompts ────────────────────────────────────────────────────────────

_RUBRIC_TAIL = """

OUTPUT FORMAT — strict JSON only, no prose before or after:
{
  "critiques": [
    {
      "target_cluster_id": "<id-as-shown-in-the-input>" | null,
      "target_cluster_name": "<copy from input or null>",
      "severity": "high" | "medium" | "low",
      "issue": "<one sentence>",
      "suggestion": "<one concrete action: rename / merge / split / specific change>",
      "evidence": "<which metric or feature value backs this — be specific>"
    }
  ]
}

Rules:
- Return at MOST 4 critiques. Quality over quantity.
- target_cluster_id=null means a global / cross-group issue.
- Every suggestion must be ACTIONABLE — say what to do, not "consider".
- Use the EXACT identifier shown in the input for target_cluster_id.
- If nothing is wrong, return {"critiques": []}.
"""

_STAT_SYSTEM = (
    "You are an analyst evaluating a set of anonymous groupings of data points. "
    "You see ONLY deviation-ratio tables (per-group feature mean divided by the "
    "overall mean), group sizes, and one accuracy-style metric per group. "
    "You do NOT see any human-assigned names or descriptions for the groups — "
    "this is deliberate. Judge only what the numbers say. "
    "Your rubric: "
    "(a) Are any two groups' numerical profiles nearly identical? (redundancy) "
    "(b) Does any group have a profile that fails to distinguish it from the "
    "    others on any feature? (weak separation) "
    "(c) Are there features where one group is implausibly extreme (e.g. 5x+) "
    "    while having very low classifier F1? (the cluster may be noise) "
    "Use the EXACT group identifier shown (e.g. 'Group A') as target_cluster_id."
    + _RUBRIC_TAIL
)

_BIZ_SYSTEM = (
    "You are evaluating named labels for groupings, purely for human readability. "
    "You see ONLY the labels (names) and a one-line description for each. "
    "You do NOT see sizes, features, metrics, or any numerical evidence — this "
    "is deliberate. Judge ONLY whether the names are clear. "
    "Your rubric: "
    "(a) Could a non-technical reader understand each name in one sentence? "
    "(b) Are any names vague, jargon-heavy, or over-complex? Propose simpler "
    "    alternatives — over-complex is always bad. "
    "(c) Do any two names overlap in meaning? Propose either a merge or a "
    "    sharper distinction. "
    "(d) Does any one-line description fail to clarify what the name means? "
    "Use the persona name as both target_cluster_id and target_cluster_name."
    + _RUBRIC_TAIL
)

_DOMAIN_SYSTEM = (
    "You are evaluating whether a set of named groupings actually reflects "
    "what the underlying data shows. You see the names, the deviation-ratio "
    "tables, AND a small SAMPLE OF RAW RECORDS from each group (a few real "
    "members, with their top-feature values). "
    "Your rubric: "
    "(a) For each group, do the raw record samples plus the deviation ratios "
    "    actually support the name? Cite specific values. "
    "(b) Compare each group to its neighbours — does the name capture what "
    "    makes it distinct, or could it equally describe another group? "
    "(c) Are there raw records that look out of place for their group's name? "
    "    (potential mislabeled outliers) "
    "Use the cluster id shown (e.g. '0', '1') as target_cluster_id."
    + _RUBRIC_TAIL
)


# ── Anonymization helpers (statistical judge only) ────────────────────────────

def _build_cluster_alias(personas: dict) -> dict[str, str]:
    """Map real cluster_id -> 'Group A' / 'Group B' / ... so the statistical
    judge cannot anchor on the human-assigned name."""
    sorted_ids = sorted(personas.keys(), key=lambda x: (len(str(x)), str(x)))
    letters = string.ascii_uppercase + string.ascii_lowercase
    return {str(cid): f'Group {letters[i % len(letters)]}'
            for i, cid in enumerate(sorted_ids)}


def _reverse_alias(alias: dict[str, str]) -> dict[str, str]:
    """Both 'Group A' and bare 'A' map back to the real cluster id, so we
    can recover regardless of how the judge formats its reference."""
    rev = {}
    for cid, label in alias.items():
        rev[label] = cid                   # 'Group A' -> '0'
        rev[label.split()[-1]] = cid       # 'A' -> '0'
    return rev


# ── Per-judge evidence packets ────────────────────────────────────────────────

def _load_artifacts(run_dir: pathlib.Path) -> tuple[dict, dict]:
    out = run_dir / 'outputs'
    personas = json.loads((out / 'personas.json').read_text()) \
        if (out / 'personas.json').exists() else {}
    clf = json.loads((out / 'classifier_metrics.json').read_text()) \
        if (out / 'classifier_metrics.json').exists() else {}
    return personas, clf


def _load_cluster_labels(run_dir: pathlib.Path) -> Optional[pd.Series]:
    p = run_dir / 'outputs' / 'cluster_labels.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df.set_index('row_index')['cluster_id']


def _packet_statistical(personas: dict, clf: dict,
                         alias: dict[str, str]) -> str:
    """Deviation ratios + sizes + F1 only. Persona names are REPLACED by
    'Group A' / 'Group B' / ... so the judge cannot anchor on the name."""
    lines = [
        'ANONYMOUS GROUPS WITH DEVIATION RATIOS AND PER-GROUP METRICS',
        '(Each ratio is per-feature group-mean divided by overall-mean; '
        '1.00 = at average, 2.00 = double, 0.50 = half.)',
    ]
    per_class = clf.get('per_class_f1') or {}
    for cid, d in personas.items():
        s = d.get('cluster_stats', {})
        p = d.get('persona', {}) or {}
        # F1 is keyed by persona name in the artifact — we look it up here
        # and emit only the number, never the name.
        f1 = per_class.get(p.get('name', ''))
        n = s.get('n_entities', s.get('n_customers', 0))
        pct = s.get('pct_total', s.get('pct_of_total', 0) * 100)
        f1_str = f'{f1:.3f}' if isinstance(f1, (int, float)) else 'n/a'
        gid = alias[str(cid)]
        lines.append('')
        lines.append(f'  {gid}: n={n} ({pct:.1f}%)  cv_f1={f1_str}')
        top_above = s.get('top_above_average', {}) or {}
        top_below = s.get('top_below_average', {}) or {}
        if top_above:
            top = sorted(top_above.items(), key=lambda x: -x[1])[:8]
            lines.append('    above-avg: ' +
                         ', '.join(f'{k}={v:.2f}x' for k, v in top))
        if top_below:
            bot = sorted(top_below.items(), key=lambda x: x[1])[:5]
            lines.append('    below-avg: ' +
                         ', '.join(f'{k}={v:.2f}x' for k, v in bot))
    return '\n'.join(lines)


def _packet_business(personas: dict) -> str:
    """Names + one-line descriptions ONLY. No numbers, no features, no sizes.
    The business judge must judge the name on its own merit, with at most
    a one-line description as the contextual hint."""
    lines = ['NAMED LABELS FOR EVALUATION (name + one-line description):']
    for cid, d in personas.items():
        p = d.get('persona', {}) or {}
        name = p.get('name', '?')
        # Collapse description to a single sentence / 200 chars max
        desc = (p.get('description') or p.get('tagline') or '').strip()
        desc = desc.replace('\n', ' ')
        if len(desc) > 200:
            desc = desc[:197].rsplit(' ', 1)[0] + '…'
        lines.append('')
        lines.append(f'  "{name}"')
        if desc:
            lines.append(f'    — {desc}')
    return '\n'.join(lines)


def _sample_raw_records(features_df: pd.DataFrame,
                         cluster_labels: pd.Series,
                         cid: str,
                         top_features: dict[str, float],
                         n_samples: int = 3,
                         n_features: int = 6) -> list[str]:
    """Return N formatted real-record samples for a given cluster id.

    Picks the top N most-distinguishing features (highest above-avg ratio)
    so the judge sees what was supposedly load-bearing for the persona,
    not random columns.
    """
    if features_df is None or cluster_labels is None:
        return []
    try:
        # cluster_labels values may be int or str — match the cid type.
        sample_type = type(cluster_labels.iloc[0]) if len(cluster_labels) else str
        cid_typed = sample_type(cid)
    except Exception:
        cid_typed = cid
    matching = cluster_labels[cluster_labels == cid_typed].index.tolist()
    if not matching:
        return []
    feat_names = [f for f in list(top_features.keys())[:n_features]
                   if f in features_df.columns]
    if not feat_names:
        return []
    n = min(n_samples, len(matching))
    sample_idx = random.sample(matching, n) if len(matching) >= n else matching
    out = []
    for idx in sample_idx:
        if idx < 0 or idx >= len(features_df):
            continue
        row = features_df.iloc[idx]
        cells = []
        for f in feat_names:
            v = row[f]
            try:
                cells.append(f'{f}={float(v):.3g}')
            except (TypeError, ValueError):
                cells.append(f'{f}={v}')
        out.append(f'row#{idx}: ' + ' | '.join(cells))
    return out


def _packet_domain(personas: dict,
                    dataset_description: str,
                    features_df: Optional[pd.DataFrame],
                    cluster_labels: Optional[pd.Series]) -> str:
    """Names + deviation ratios + raw record samples + dataset subject.
    Only the domain judge sees raw rows — that is its unique evidence
    base, the thing the other two judges do not have."""
    lines = []
    if dataset_description.strip():
        lines.append(f'DATASET SUBJECT: {dataset_description.strip()}')
        lines.append('')
    lines.append('NAMED GROUPINGS — DEVIATION RATIOS + RAW RECORD SAMPLES:')
    for cid, d in personas.items():
        s = d.get('cluster_stats', {})
        p = d.get('persona', {}) or {}
        name = p.get('name', '?')
        n = s.get('n_entities', s.get('n_customers', 0))
        pct = s.get('pct_total', s.get('pct_of_total', 0) * 100)
        lines.append('')
        lines.append(f'  Cluster {cid}: name="{name}"  '
                     f'tagline="{p.get("tagline", "")}"  '
                     f'n={n} ({pct:.1f}%)')
        top_above = s.get('top_above_average', {}) or {}
        if top_above:
            top = sorted(top_above.items(), key=lambda x: -x[1])[:6]
            lines.append('    strongly above average: ' +
                         ', '.join(f'{k} ({v:.2f}x)' for k, v in top))
        traits = p.get('traits') or []
        if traits:
            lines.append('    claimed traits: ' + ' | '.join(traits[:3]))
        samples = _sample_raw_records(features_df, cluster_labels, cid, top_above)
        if samples:
            lines.append('    raw member samples (3 randomly drawn rows, '
                         'shown for top-6 distinguishing features):')
            for s_line in samples:
                lines.append('      ' + s_line)
    return '\n'.join(lines)


# ── Single-judge call (fresh client, fresh context) ───────────────────────────

def _call_judge(name: str, system: str, evidence: str) -> list[Critique]:
    client = anthropic.Anthropic()
    user_prompt = (
        "Here is the artifact you are evaluating. Apply your rubric and "
        "return findings in the JSON format your system prompt specified.\n\n"
        f"{evidence}"
    )
    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=2000,
        system=system,
        messages=[{'role': 'user', 'content': user_prompt}],
    )
    parsed = _parse_critiques(resp.content[0].text)
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

def critique_run(run_dir: pathlib.Path,
                  dataset_description: str = '',
                  features_df: Optional[pd.DataFrame] = None) -> list[Critique]:
    """Run all three judges IN PARALLEL with ASYMMETRIC views of the run.

    Parameters
    ----------
    run_dir : pathlib.Path
        Snapshot directory containing outputs/personas.json,
        classifier_metrics.json, cluster_labels.csv.
    dataset_description : str
        Short factual sentence about what the data is about (e.g. "credit
        card transactions"). Shown ONLY to the domain judge.
    features_df : pd.DataFrame or None
        The engineered features matrix (rows aligned with cluster_labels'
        row_index). Used by the domain judge to sample real records.
        If None, the domain judge falls back to ratio-only evidence.

    Returns
    -------
    list[Critique]
        Statistical-judge critiques have their anonymous 'Group A' /
        'Group B' targets translated back to real cluster_ids before
        being returned, so the arbiter sees a consistent id space.
    """
    personas, clf = _load_artifacts(run_dir)
    cluster_labels = _load_cluster_labels(run_dir)
    alias = _build_cluster_alias(personas)

    jobs = [
        ('statistical', _STAT_SYSTEM,
         _packet_statistical(personas, clf, alias)),
        ('business',    _BIZ_SYSTEM,
         _packet_business(personas)),
        ('domain',      _DOMAIN_SYSTEM,
         _packet_domain(personas, dataset_description, features_df, cluster_labels)),
    ]

    all_critiques: list[Critique] = []
    with _cf.ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(_call_judge, name, system, evidence): name
                   for name, system, evidence in jobs}
        for fut in _cf.as_completed(futures):
            name = futures[fut]
            try:
                cs = fut.result()
                print(f'  [judge:{name}] returned {len(cs)} critiques')
                all_critiques.extend(cs)
            except Exception as e:
                print(f'  [judge:{name}] FAILED: {e}')

    # Translate statistical-judge anonymous labels back to real cluster_ids.
    rev = _reverse_alias(alias)
    name_to_cid = {(d.get('persona') or {}).get('name'): str(cid)
                    for cid, d in personas.items()}
    for c in all_critiques:
        if c.target_cluster_id is None:
            continue
        t = c.target_cluster_id.strip()
        if c.judge == 'statistical' and t in rev:
            c.target_cluster_id = rev[t]
            c.target_cluster_name = (personas.get(rev[t], {})
                                       .get('persona', {}) or {}).get('name')
        elif c.judge == 'business' and t in name_to_cid:
            # Business judge keys on the persona name; map name → cid.
            c.target_cluster_name = t
            c.target_cluster_id = name_to_cid[t]

    return all_critiques


def build_arbiter_evidence(run_dir: pathlib.Path,
                            dataset_description: str = '',
                            features_df: Optional[pd.DataFrame] = None) -> str:
    """Neutral combined evidence packet for the arbiter only (which is
    allowed to know what the pipeline is doing)."""
    personas, clf = _load_artifacts(run_dir)
    alias = _build_cluster_alias(personas)
    cluster_labels = _load_cluster_labels(run_dir)
    parts = []
    if dataset_description.strip():
        parts.append(f'DATASET SUBJECT: {dataset_description.strip()}')
        parts.append('')
    parts.append(_packet_statistical(personas, clf, alias))
    parts.append('')
    parts.append('NAMED VERSION:')
    for cid, d in personas.items():
        p = d.get('persona', {}) or {}
        parts.append(f'  Cluster {cid} ({alias[str(cid)]}): "{p.get("name", "?")}" '
                     f'— {p.get("tagline", "")}')
    return '\n'.join(parts)
