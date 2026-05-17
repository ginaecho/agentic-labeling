"""Human-review fallback path — used when the loop ends WITHOUT convergence.

Two entry points:

  python -m experiments.human_review write
    → Generates experiments/runs/human_review.md from the current
      feedback log + run_history.jsonl. The file groups rules by source
      judge and shows each rule's text, target, the per-run metric
      trajectory it was active in, and a checkbox for the human's verdict.

  python -m experiments.human_review apply experiments/runs/human_review.md
    → Reads the (edited) markdown back and updates the feedback log:
        - rules marked [x] KEEP   → provenance='human', priority='high'
        - rules marked [x] REVISE → text replaced from the markdown
                                     body; provenance='human', priority='high'
        - rules marked [x] DROP   → active=false
        - rules with no checkbox  → unchanged

Markdown is the medium because it is the most-edited file format the
user already lives in — checkboxes work in any editor (and on GitHub).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Optional

RUNS_DIR = pathlib.Path('experiments/runs')
REVIEW_PATH = RUNS_DIR / 'human_review.md'

# Markdown markers we read back on apply.
_RULE_RE = re.compile(
    r'^### Rule\s+`(?P<id>fb_[A-Za-z0-9]+)`\s*$',
    re.MULTILINE,
)
_VERDICT_RE = re.compile(
    r'^- \[(?P<x>[ xX])\]\s+(?P<verdict>KEEP|REVISE|DROP)\b',
    re.MULTILINE,
)
_REVISED_BLOCK_RE = re.compile(
    r'```revised\n(?P<body>.*?)\n```',
    re.DOTALL,
)


def _load_run_history() -> list[dict]:
    p = RUNS_DIR / 'run_history.jsonl'
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def write_review_md() -> pathlib.Path:
    """Generate the human review markdown."""
    from ui import feedback_store

    rules = [e for e in feedback_store.read_all() if e.get('active', True)
             and e.get('provenance') == 'agent']
    history = _load_run_history()

    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# Human Review — Adaptive-Learning Rules (No Convergence)',
        '',
        f'The convergence loop completed **without** reaching '
        f'ARI ≥ 0.90 for 3 consecutive runs.',
        f'{len(rules)} agent-generated rules accumulated. '
        f'Please review each below.',
        '',
        '## Instructions',
        '',
        '- For each rule, **check ONE box**: KEEP, REVISE, or DROP.',
        '- For REVISE, replace the text inside the ` ```revised ` block.',
        '- Leave a rule unchecked to keep it untouched.',
        '- When done, save this file and run:',
        '  ```bash',
        '  python -m experiments.human_review apply '
        f'{REVIEW_PATH}',
        '  ```',
        '',
        '## Run-by-run metric trajectory',
        '',
        '| run | F1   | n_clusters | ARI vs prev | rules accepted |',
        '|----:|-----:|-----------:|-----------:|---------------:|',
    ]
    for row in history:
        m = row.get('metrics') or {}
        d = row.get('diff_vs_prev') or {}
        lines.append(
            f'| {row.get("run")} | {m.get("cv_f1_macro")} | '
            f'{m.get("n_clusters")} | {d.get("ari")} | '
            f'{row.get("arbiter", {}).get("n_accepted")} |'
        )
    lines.append('')

    # Group rules by source judge so the reviewer can compare critiques
    # from the same lens side by side.
    by_source: dict = {}
    for r in rules:
        by_source.setdefault(r.get('source', 'unknown'), []).append(r)

    for source, group in sorted(by_source.items()):
        lines.append(f'## Source: `{source}`  ({len(group)} rule(s))')
        lines.append('')
        for r in group:
            text = (r.get('rule') or r.get('hint') or '').strip()
            target = (r.get('target_cluster_name')
                       or r.get('target_cluster_id') or 'GLOBAL')
            lines.extend([
                f'### Rule `{r.get("id")}`',
                f'- target: **{target}**',
                f'- judge_severity: {r.get("judge_severity", "?")}',
                f'- current priority: {r.get("priority", "medium")}',
                '',
                f'> {text}',
                '',
                '- [ ] KEEP  (mark as human-endorsed, promote to HIGH priority)',
                '- [ ] REVISE  (use the text from the ```revised``` block below; '
                'becomes human-endorsed HIGH)',
                '- [ ] DROP  (mark inactive; pipeline will ignore it)',
                '',
                '```revised',
                text,
                '```',
                '',
                '---',
                '',
            ])

    REVIEW_PATH.write_text('\n'.join(lines), encoding='utf-8')
    print(f'  [human-review] wrote {REVIEW_PATH}  ({len(rules)} rules)')
    return REVIEW_PATH


def apply_review_md(path: pathlib.Path) -> dict:
    """Read the (edited) markdown back; update the feedback log accordingly."""
    from ui import feedback_store

    if not path.exists():
        print(f'  [human-review] {path} not found')
        return {'updated': 0}

    text = path.read_text(encoding='utf-8')
    counts = {'kept': 0, 'revised': 0, 'dropped': 0, 'untouched': 0}

    # Split into per-rule sections by the `### Rule ...` headers.
    headers = list(_RULE_RE.finditer(text))
    for i, m in enumerate(headers):
        rule_id = m.group('id')
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        section = text[start:end]

        verdict_matches = [
            (vm.group('x').strip().lower() == 'x', vm.group('verdict'))
            for vm in _VERDICT_RE.finditer(section)
        ]
        checked = [v for ticked, v in verdict_matches if ticked]
        if not checked:
            counts['untouched'] += 1
            continue
        verdict = checked[0]   # first checked box wins if user ticked multiple

        if verdict == 'KEEP':
            feedback_store.update_priority(rule_id, 'high')
            _patch_entry(rule_id, {'provenance': 'human',
                                    'human_verdict': 'keep'})
            counts['kept'] += 1
        elif verdict == 'REVISE':
            body_match = _REVISED_BLOCK_RE.search(section)
            new_text = (body_match.group('body').strip()
                         if body_match else None)
            patch = {'provenance': 'human', 'human_verdict': 'revise',
                     'priority': 'high'}
            if new_text:
                # Updates whichever field carried the rule text.
                _patch_entry(rule_id, patch, new_rule_text=new_text)
            else:
                _patch_entry(rule_id, patch)
            counts['revised'] += 1
        elif verdict == 'DROP':
            feedback_store.set_active(rule_id, False)
            _patch_entry(rule_id, {'human_verdict': 'drop'})
            counts['dropped'] += 1

    print(f'  [human-review] applied: kept={counts["kept"]}  '
          f'revised={counts["revised"]}  dropped={counts["dropped"]}  '
          f'untouched={counts["untouched"]}')
    return counts


def _patch_entry(rule_id: str, fields: dict,
                  new_rule_text: Optional[str] = None) -> None:
    """Mutate a single feedback entry in place on disk."""
    from ui import feedback_store

    def transform(e: dict) -> dict:
        if e.get('id') != rule_id:
            return e
        new = {**e, **fields}
        if new_rule_text is not None:
            if 'rule' in e:
                new['rule'] = new_rule_text
            elif 'hint' in e:
                new['hint'] = new_rule_text
        return new

    feedback_store._rewrite(transform)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('write', 'apply'))
    ap.add_argument('path', nargs='?', default=str(REVIEW_PATH))
    args = ap.parse_args()
    if args.mode == 'write':
        write_review_md()
    else:
        apply_review_md(pathlib.Path(args.path))
    return 0


if __name__ == '__main__':
    sys.exit(main())
