"""LLM compaction of the preferences block before PersonaNamer injection.

Why: as judges generate rules across runs, the feedback log accumulates
near-duplicates ("merge clusters C and D" from the statistical judge AND
the domain judge, with slightly different wording) and stale guidance
(rules that targeted a cluster that no longer exists). Injecting all 11
lines verbatim wastes prompt budget and dilutes the strong rules.

What this does:
  • Reads the currently-active feedback entries (NOT MUTATED).
  • Asks an LLM: "Merge near-duplicates, drop redundancies, keep the
    strongest version of each distinct idea."
  • Returns a deduplicated list of entries + a per-rule rationale
    (which were kept, which were merged into which, which were dropped
    and why) written to experiments/runs/rule_compaction_ledger.jsonl.
  • Caches by SHA-256 of the input entries so two PersonaNamer calls
    against the same feedback log share one LLM dedup result.

The feedback log on disk is intentionally NOT mutated — compaction is a
read-time view, so the audit trail of every original rule is preserved.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import time
from typing import Optional

import anthropic

DEDUP_MODEL = os.environ.get('DEDUP_MODEL', 'claude-haiku-4-5-20251001')

# In-process cache: {entries_hash: (deduped_entries, kept_ids)}
_CACHE: dict[str, tuple[list[dict], set[str]]] = {}

LEDGER_PATH = pathlib.Path('experiments/runs/rule_compaction_ledger.jsonl')


_DEDUP_SYSTEM = """You are compacting a list of rules learned by automated
judges across many runs of a clustering pipeline. Your job:

1. MERGE near-duplicates — when two or more rules express substantively the
   same idea (even with different wording, different cluster targets, or
   different evidence), keep the CLEAREST single version and mark the
   others as merged into it.

2. DROP redundancies — when a rule is fully implied by another, drop the
   weaker one.

3. DROP stale rules — when a rule targets a specific cluster name or id
   that no longer appears among the current cluster set, drop it.

4. KEEP all distinct ideas — never merge rules that point in different
   directions or address different defects.

OUTPUT: strict JSON only, no prose:
{
  "decisions": [
    {"id": "fb_xxxx", "action": "keep", "rationale": "..."},
    {"id": "fb_yyyy", "action": "merge_into", "into": "fb_xxxx", "rationale": "..."},
    {"id": "fb_zzzz", "action": "drop", "rationale": "..."}
  ]
}

Every input rule id must appear exactly once. Be conservative — if in
doubt, KEEP. Only merge when the idea is genuinely the same.
"""


def _hash_entries(entries: list[dict]) -> str:
    """Stable content hash over the rule text + ids."""
    payload = json.dumps(
        [(e.get('id'), e.get('rule') or e.get('hint') or '',
          e.get('target_cluster_id'), e.get('target_cluster_name'))
         for e in entries],
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


def _build_dedup_prompt(entries: list[dict],
                         current_cluster_names: Optional[list[str]] = None) -> str:
    lines = []
    if current_cluster_names:
        lines.append('CURRENT CLUSTER NAMES (rules targeting names not in '
                     'this list may be stale):')
        for n in current_cluster_names:
            lines.append(f'  - "{n}"')
        lines.append('')
    lines.append('RULES TO COMPACT:')
    for e in entries:
        text = (e.get('rule') or e.get('hint') or '').strip()
        target = e.get('target_cluster_name') or e.get('target_cluster_id') or 'GLOBAL'
        source = e.get('source', 'unknown')
        priority = e.get('priority', 'medium').upper()
        lines.append('')
        lines.append(f'  id={e.get("id")}  source={source}  '
                     f'priority={priority}  target={target}')
        lines.append(f'    {text}')
    return '\n'.join(lines)


def _parse_decisions(text: str) -> list[dict]:
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
        d = obj.get('decisions', [])
        return d if isinstance(d, list) else []
    except json.JSONDecodeError:
        return []


def _write_ledger(entries: list[dict], decisions: list[dict],
                   entries_hash: str) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'entries_hash': entries_hash,
        'n_input': len(entries),
        'n_kept': sum(1 for d in decisions if d.get('action') == 'keep'),
        'n_merged': sum(1 for d in decisions if d.get('action') == 'merge_into'),
        'n_dropped': sum(1 for d in decisions if d.get('action') == 'drop'),
        'decisions': decisions,
    }
    with LEDGER_PATH.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def deduplicated_entries(
    entries: list[dict],
    current_cluster_names: Optional[list[str]] = None,
    use_cache: bool = True,
) -> list[dict]:
    """Return a compacted view of `entries`. Does NOT mutate disk state.

    Stale-aware: pass current_cluster_names so rules targeting a cluster
    that no longer exists can be dropped as stale.

    Cached: subsequent calls with the same entries return the cached
    decision without paying for a second LLM call.
    """
    if not entries:
        return []

    h = _hash_entries(entries)
    if use_cache and h in _CACHE:
        deduped, _ = _CACHE[h]
        return deduped

    user_prompt = _build_dedup_prompt(entries, current_cluster_names)
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=DEDUP_MODEL,
            max_tokens=2500,
            system=_DEDUP_SYSTEM,
            messages=[{'role': 'user', 'content': user_prompt}],
        )
        decisions = _parse_decisions(resp.content[0].text)
    except Exception as e:
        # If the dedup call fails, fall back to the raw entries.
        print(f'  [dedup_prefs] LLM call failed ({e}) — using raw entries.')
        return entries

    kept_ids = {d['id'] for d in decisions
                if d.get('action') == 'keep' and d.get('id')}
    deduped = [e for e in entries if e.get('id') in kept_ids]
    if not deduped:
        # Pathological output — keep raw to avoid breaking the pipeline.
        print('  [dedup_prefs] LLM returned no kept rules — falling back '
              'to raw entries.')
        return entries

    _write_ledger(entries, decisions, h)
    if use_cache:
        _CACHE[h] = (deduped, kept_ids)

    print(f'  [dedup_prefs] {len(entries)} → {len(deduped)} rules '
          f'(merged={sum(1 for d in decisions if d.get("action") == "merge_into")}  '
          f'dropped={sum(1 for d in decisions if d.get("action") == "drop")})')
    return deduped


def build_deduplicated_preferences_block(
    current_cluster_names: Optional[list[str]] = None,
    use_cache: bool = True,
) -> str:
    """Drop-in replacement for ui.feedback_store.build_preferences_block()
    that runs LLM compaction first. Returns the formatted block string.

    The compacted entries are formatted using the same template as the
    live UI's build_preferences_block, so PersonaNamer sees a familiar
    shape — just shorter and de-duplicated.
    """
    from ui import feedback_store

    active = [e for e in feedback_store.read_all() if e.get('active', True)]
    if not active:
        return ''

    deduped = deduplicated_entries(active, current_cluster_names, use_cache)

    # Use feedback_store's own formatting on the filtered subset by
    # temporarily filtering its read_all via monkeypatch — cleanest way
    # to reuse the canonical formatter without copy-pasting it.
    from unittest.mock import patch
    deduped_ids = {e.get('id') for e in deduped}
    original_read_all = feedback_store.read_all
    with patch.object(feedback_store, 'read_all',
                       lambda: [e for e in original_read_all()
                                if e.get('id') in deduped_ids]):
        block = feedback_store.build_preferences_block()
    return block
