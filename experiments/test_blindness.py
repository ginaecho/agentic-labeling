"""Runtime proof that the three judges are blind to each other AND see
genuinely different evidence (input asymmetry).

Monkey-patches anthropic.Anthropic to capture every call critique_run
makes, then asserts:

  1. Exactly 3 API calls are made (one per judge).
  2. Every call has exactly one user message and zero assistant turns.
  3. The three system prompts are pairwise distinct.
  4. The three user prompts are pairwise distinct.
  5. No judge's user prompt contains another judge's rubric.
  6. INPUT ASYMMETRY — each packet's unique structural marker appears
     in exactly one prompt.
  7. No judge prompt contains prior-critique JSON fields.

  + ASYMMETRY-SPECIFIC INVARIANTS:

  A. Statistical judge sees NO persona names (cannot anchor on names).
  B. Business judge sees NO numbers — no above/below-avg ratios, no
     'cv_f1=', no 'n=' size annotation, no 'pct'.
  C. Domain judge is the ONLY one to receive raw record samples
     (lines starting with 'row#' from the sampler).
  D. Statistical judge uses anonymous 'Group A' identifiers, and
     the round-trip mapping correctly translates those back to real
     cluster_ids when the judge's response uses them.

Run:  python -m experiments.test_blindness
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


def _build_fake_run(tmp: pathlib.Path) -> tuple[pathlib.Path, pd.DataFrame]:
    run = tmp / 'run_001'
    out = run / 'outputs'
    out.mkdir(parents=True)
    personas = {
        '0': {'persona': {'name': 'Casual Shoppers',
                          'tagline': 'low frequency low spend',
                          'description': 'shop occasionally',
                          'traits': ['weekly', 'small basket']},
              'cluster_stats': {'n_entities': 200, 'pct_total': 40.0,
                                'top_above_average': {'visits_30d': 1.8,
                                                       'small_basket': 1.6},
                                'top_below_average': {'avg_spend': 0.3}}},
        '1': {'persona': {'name': 'High-Value Travelers',
                          'tagline': 'spend on travel',
                          'description': 'frequent travel transactions',
                          'traits': ['airfare', 'hotels']},
              'cluster_stats': {'n_entities': 100, 'pct_total': 20.0,
                                'top_above_average': {'travel_spend': 3.1,
                                                       'hotel_count': 2.5},
                                'top_below_average': {'grocery_spend': 0.4}}},
    }
    (out / 'personas.json').write_text(json.dumps(personas))
    (out / 'classifier_metrics.json').write_text(json.dumps({
        'cv_f1_macro': 0.72,
        'per_class_f1': {'Casual Shoppers': 0.7, 'High-Value Travelers': 0.74},
    }))
    # cluster_labels.csv + features_df so the domain judge has raw rows
    n = 300
    labels = [0] * 200 + [1] * 100
    (out / 'cluster_labels.csv').write_text(
        'row_index,cluster_id\n' +
        '\n'.join(f'{i},{labels[i]}' for i in range(n)))
    features_df = pd.DataFrame({
        'visits_30d':    [3, 4, 5] * 100,
        'small_basket':  [1, 2, 1] * 100,
        'avg_spend':     [10, 20, 30] * 100,
        'travel_spend':  [100, 200, 50] * 100,
        'hotel_count':   [1, 0, 2] * 100,
        'grocery_spend': [40, 50, 60] * 100,
    })
    return run, features_df


_FAKE_REPLY = '{"critiques":[]}'


def _make_fake_anthropic(captured: list):
    class _FakeClient:
        def __init__(self, *a, **kw):
            self.messages = self
        def create(self, *, model, max_tokens, system, messages):
            captured.append({'system': system, 'messages': messages})
            return SimpleNamespace(
                content=[SimpleNamespace(text=_FAKE_REPLY)],
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            )
    return _FakeClient


def main() -> int:
    captured: list[dict] = []
    FakeClient = _make_fake_anthropic(captured)

    with patch('anthropic.Anthropic', FakeClient):
        from experiments import judges
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, features_df = _build_fake_run(pathlib.Path(tmp))
            judges.critique_run(run_dir,
                                 dataset_description='test corpus',
                                 features_df=features_df)

    print(f'Captured {len(captured)} API call(s)')

    # ── Core blindness invariants ───────────────────────────────────────────
    assert len(captured) == 3, f'expected 3 calls, got {len(captured)}'
    print('  [1] exactly 3 calls made: OK')

    for i, call in enumerate(captured):
        msgs = call['messages']
        assert len(msgs) == 1, f'call {i}: expected 1 message, got {len(msgs)}'
        assert msgs[0]['role'] == 'user', \
            f'call {i}: expected user role, got {msgs[0]["role"]}'
    print('  [2] every call has exactly 1 user message, 0 assistant turns: OK')

    systems = [c['system'] for c in captured]
    assert len(set(systems)) == 3, 'judge system prompts must be distinct'
    print('  [3] three system prompts are pairwise distinct: OK')

    user_prompts = [c['messages'][0]['content'] for c in captured]
    assert len(set(user_prompts)) == 3, 'judge user prompts must be distinct'
    print('  [4] three user prompts are pairwise distinct: OK')

    rubric_fps = [s[:300] for s in systems]
    for i, up in enumerate(user_prompts):
        for j, fp in enumerate(rubric_fps):
            if i == j:
                continue
            assert fp not in up, \
                f"judge #{i}'s prompt contains judge #{j}'s rubric"
    print("  [5] no judge prompt contains another judge's rubric: OK")

    # Unique structural markers per packet:
    unique_markers = {
        'statistical': 'ANONYMOUS GROUPS WITH DEVIATION RATIOS',
        'business':    'NAMED LABELS FOR EVALUATION',
        'domain':      'DEVIATION RATIOS + RAW RECORD SAMPLES',
    }
    for marker_name, marker in unique_markers.items():
        hits = [up for up in user_prompts if marker in up]
        assert len(hits) == 1, \
            f'{marker_name} marker {marker!r} appears in {len(hits)} prompts'
    print('  [6] each packet has a unique structural marker '
          '(input asymmetry intact): OK')

    critique_input_markers = ('"target_cluster_id"', '"suggestion"',
                              '"severity"', '"issue"', '"judge":')
    for i, up in enumerate(user_prompts):
        for marker in critique_input_markers:
            assert marker not in up, \
                f'judge #{i} prompt contains {marker!r} (prior-critique leak)'
    print('  [7] no prior-critique JSON fields in any prompt: OK')

    # ── Asymmetry-specific invariants ───────────────────────────────────────
    # Identify each judge by its system prompt rubric.
    def _find(rubric_substr):
        for idx, s in enumerate(systems):
            if rubric_substr in s:
                return idx
        raise AssertionError(f'no judge with rubric {rubric_substr!r}')

    stat_idx = _find('anonymous groupings')
    biz_idx = _find('purely for human readability')
    dom_idx = _find('SAMPLE OF RAW RECORDS')

    # A. Statistical sees NO persona names
    stat_p = user_prompts[stat_idx]
    for name in ('Casual Shoppers', 'High-Value Travelers'):
        assert name not in stat_p, \
            f'statistical judge SAW persona name "{name}" — anchoring risk'
    print('  [A] statistical judge sees no persona names: OK')

    # B. Business sees NO numbers / no metric markers
    biz_p = user_prompts[biz_idx]
    for forbidden in ('above-avg:', 'below-avg:', 'cv_f1=', 'n=200',
                      '40.0%', '1.80x', '3.10x'):
        assert forbidden not in biz_p, \
            f'business judge prompt leaked quantitative marker {forbidden!r}'
    # Stricter: the only digits permitted are inside the JSON rubric tail
    # (not present in user prompt) — count digits in the business packet.
    biz_digits = sum(ch.isdigit() for ch in biz_p)
    assert biz_digits == 0, \
        f'business judge prompt contains {biz_digits} digit chars — ' \
        f'should be zero'
    print('  [B] business judge prompt is digit-free '
          '(no numeric evidence): OK')

    # C. Domain is the ONLY one with raw record samples
    dom_p = user_prompts[dom_idx]
    assert 'row#' in dom_p, 'domain judge did not receive raw record samples'
    for other_idx in (stat_idx, biz_idx):
        assert 'row#' not in user_prompts[other_idx], \
            f'non-domain judge #{other_idx} received raw record samples'
    print('  [C] only the domain judge receives raw record samples: OK')

    # D. Statistical uses anonymized "Group A/B/..." labels
    assert 'Group A' in stat_p and 'Group B' in stat_p, \
        'statistical packet missing anonymized group labels'
    print('  [D] statistical packet uses anonymized "Group X" labels: OK')

    # D-roundtrip: simulate a statistical-judge critique referring to
    # "Group A" and verify it maps back to real cluster_id "0".
    from experiments.judges import _build_cluster_alias, _reverse_alias
    fake_personas = {'0': {}, '1': {}}
    alias = _build_cluster_alias(fake_personas)
    rev = _reverse_alias(alias)
    assert rev[alias['0']] == '0' and rev[alias['1']] == '1'
    assert rev['A'] == '0' and rev['B'] == '1'
    print('  [D] alias round-trip maps "Group A" / "A" back to real ids: OK')

    print('\nALL BLINDNESS + ASYMMETRY INVARIANTS HOLD AT RUNTIME ✓')
    return 0


if __name__ == '__main__':
    sys.exit(main())
