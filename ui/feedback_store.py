"""Persistent user-feedback store for adaptive learning.

Every change a user makes in the web UI is appended as a JSON line to
outputs/user_feedback_log.jsonl. The Decision Maker (PersonaNamingAgent
and any future agent) calls `build_preferences_block()` to surface
recent high-priority feedback as a preamble in its prompts, so the
agent system genuinely remembers and adapts to user preferences.

Entry schema:
{
  "id":                "fb_<8hex>",
  "date":              ISO-8601 UTC timestamp,
  "type":              "manual_override" | "naming_hint" | "merge" | "global_rule",
  "target_cluster_id": "0" | null,
  "target_cluster_name": "City Food Lovers" | null,
  "before":            {...},        # for manual_override
  "after":             {...},        # for manual_override
  "hint":              "...",        # for naming_hint
  "merged_ids":        ["1","3"],    # for merge
  "rule":              "...",        # for global_rule
  "priority":          "high" | "medium" | "low",
  "active":            true
}
"""
from __future__ import annotations

import json
import pathlib
import uuid
from datetime import datetime, timezone
from typing import Iterable

LOG_PATH = pathlib.Path('outputs/user_feedback_log.jsonl')

VALID_TYPES = {'manual_override', 'naming_hint', 'merge', 'global_rule'}
VALID_PRIORITY = ('high', 'medium', 'low')


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _new_id() -> str:
    return 'fb_' + uuid.uuid4().hex[:8]


def append(entry: dict) -> dict:
    if entry.get('type') not in VALID_TYPES:
        raise ValueError(f"Invalid feedback type: {entry.get('type')!r}")
    entry.setdefault('id', _new_id())
    entry.setdefault('date', _now_iso())
    entry.setdefault('priority', 'medium')
    entry.setdefault('active', True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    return entry


def read_all() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    out = []
    for line in LOG_PATH.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _rewrite(transform) -> bool:
    entries = read_all()
    if not entries:
        return False
    new_entries = [transform(e) for e in entries]
    if new_entries == entries:
        return False
    LOG_PATH.write_text(
        '\n'.join(json.dumps(e, ensure_ascii=False) for e in new_entries) + '\n',
        encoding='utf-8',
    )
    return True


def update_priority(fb_id: str, priority: str) -> bool:
    if priority not in VALID_PRIORITY:
        raise ValueError(f"Invalid priority: {priority!r}")
    return _rewrite(lambda e: {**e, 'priority': priority} if e.get('id') == fb_id else e)


def set_active(fb_id: str, active: bool) -> bool:
    return _rewrite(lambda e: {**e, 'active': bool(active)} if e.get('id') == fb_id else e)


def delete(fb_id: str) -> bool:
    """Hard-delete a feedback entry by id. Returns True if anything changed."""
    entries = read_all()
    kept = [e for e in entries if e.get('id') != fb_id]
    if len(kept) == len(entries):
        return False
    if kept:
        LOG_PATH.write_text(
            '\n'.join(json.dumps(e, ensure_ascii=False) for e in kept) + '\n',
            encoding='utf-8',
        )
    else:
        LOG_PATH.write_text('', encoding='utf-8')
    return True


def build_preferences_block(
    types: Iterable[str] | None = None,
    limit: int = 25,
) -> str:
    """Format active feedback as a textual block to prepend to LLM prompts.

    Returns '' when the log is empty / nothing relevant is active.
    Entries are sorted high-priority first, then most-recent first.
    """
    entries = [e for e in read_all() if e.get('active', True)]
    if types is not None:
        types = set(types)
        entries = [e for e in entries if e.get('type') in types]
    pri_order = {'high': 0, 'medium': 1, 'low': 2}
    entries.sort(key=lambda e: (
        pri_order.get(e.get('priority', 'medium'), 1),
        # Most recent first within the same priority bucket
        -_date_sort_key(e.get('date', '')),
    ))
    entries = entries[:limit]
    if not entries:
        return ''

    lines = [
        '================================================================',
        '  USER PREFERENCES — learned from prior UI feedback',
        '  Respect these unless they directly contradict data evidence.',
        '================================================================',
    ]
    for e in entries:
        date = (e.get('date') or '?')[:10]
        pri = e.get('priority', 'medium').upper()
        t = e.get('type', '?')
        target = e.get('target_cluster_name') or e.get('target_cluster_id') or 'global'
        if t == 'manual_override':
            before = e.get('before', {}) or {}
            after = e.get('after', {}) or {}
            for k in after:
                if before.get(k) != after.get(k):
                    lines.append(
                        f'  - [{pri}] {date} | cluster "{target}": user set '
                        f'{k} = "{_short(after.get(k))}" '
                        f'(was "{_short(before.get(k))}")'
                    )
        elif t == 'naming_hint':
            hint = (e.get('hint') or '').strip()
            lines.append(
                f'  - [{pri}] {date} | cluster "{target}": user requested — "{hint}"'
            )
        elif t == 'merge':
            merged = e.get('merged_ids') or []
            lines.append(
                f'  - [{pri}] {date} | user merged clusters {merged} into "{target}"'
            )
        elif t == 'global_rule':
            rule = (e.get('rule') or '').strip()
            lines.append(f'  - [{pri}] {date} | GLOBAL RULE: "{rule}"')
    lines.append('================================================================')
    return '\n'.join(lines) + '\n'


def _short(v, n: int = 80) -> str:
    s = str(v).replace('\n', ' ').strip()
    return s if len(s) <= n else s[: n - 1] + '…'


def _date_sort_key(iso: str) -> float:
    try:
        return datetime.fromisoformat(iso.replace('Z', '+00:00')).timestamp()
    except Exception:
        return 0.0
