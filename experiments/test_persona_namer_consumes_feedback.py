"""Runtime proof that PersonaNamingAgent CONSUMES rules from the feedback log.

What's been proved so far:
  - judges generate substantive critiques (test_judges_live.py — live API)
  - arbiter writes accepted rules into outputs/user_feedback_log.jsonl
  - build_preferences_block() formats those rules as a text block

What's been MISSING: a runtime demonstration that the next PersonaNamer
invocation actually places that block inside its LLM prompt.

This test closes that gap WITHOUT running the whole pipeline (which has
unrelated bugs around oversized clusters and pre-aggregated datasets):

  1. Backs up the real feedback log; replaces it with a controlled set
     of judge-generated rules (the same 6 the arbiter accepted in
     test_judges_live.py).
  2. Instantiates the REAL PersonaNamingAgent class.
  3. Wraps OrchestratorBus.ask() to CAPTURE the prompt it receives
     (returning a stub JSON response so PersonaNamer's downstream
     parsing doesn't crash).
  4. Calls PersonaNamer.run() with a small synthetic profiles+lineage.
  5. Asserts the captured prompt:
       - contains the 'USER PREFERENCES' header
       - contains specific rule text from the judge-generated rules
       - has the prefs block PREPENDED (before the cluster-data table)
  6. Restores the real feedback log.

Run:  .venv/bin/python -m experiments.test_persona_namer_consumes_feedback
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import tempfile
import time
from unittest.mock import patch

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# The 6 judge-generated rules our test_judges_live.py produced and the
# arbiter accepted (same wording as the live run, condensed for the test).
JUDGE_RULES = [
    {
        'type': 'global_rule', 'priority': 'high',
        'rule': "Merge near-duplicate clusters whose deviation ratios all "
                "differ by less than 0.20x and whose cv_f1 scores are within "
                "0.05 of each other — rerun clustering with k reduced by one.",
        'source': 'judge:statistical',
    },
    {
        'type': 'naming_hint', 'priority': 'high',
        'target_cluster_id': '1',
        'target_cluster_name': 'Dormant Premium Lifestyle and Luxury Shopping Vertical Spenders',
        'hint': "Treat any cluster with n<50 AND cv_f1<0.70 AND extreme "
                "deviation ratios as an outlier stratum rather than a "
                "behavioral segment — apply isolation forest before clustering.",
        'source': 'judge:statistical',
    },
    {
        'type': 'naming_hint', 'priority': 'medium',
        'target_cluster_id': '4',
        'target_cluster_name': 'Frequent Male Travellers',
        'hint': "Audit feature-ratio reporting for any cluster whose "
                "'above-avg' block contains values below 1.00x — move them "
                "to 'below-avg'.",
        'source': 'judge:statistical',
    },
    {
        'type': 'naming_hint', 'priority': 'medium',
        'target_cluster_id': '4',
        'target_cluster_name': 'Frequent Male Travellers',
        'hint': "Rename clusters whose top features are gas/transport (not "
                "travel) to 'Active Commuters' rather than 'Travellers' — "
                "'Travellers' implies leisure travel and is misleading when "
                "sum_travel_all is below average.",
        'source': 'judge:business',
    },
    {
        'type': 'naming_hint', 'priority': 'medium',
        'target_cluster_id': '0',
        'target_cluster_name': 'Grocery-Obsessed Female Bulk Buyers',
        'hint': "Avoid informal or pejorative words ('obsessed', 'addicts', "
                "'crazy') in cluster names — use 'High-Frequency' or "
                "'Heavy-Spend' instead for stakeholder-facing reports.",
        'source': 'judge:business',
    },
    {
        'type': 'naming_hint', 'priority': 'medium',
        'target_cluster_id': '1',
        'target_cluster_name': 'Dormant Premium Lifestyle and Luxury Shopping Vertical Spenders',
        'hint': "Do not use 'Premium' or 'Luxury' as cluster labels unless "
                "absolute spend exceeds the 75th percentile — relative "
                "deviation ratios alone do not justify these labels.",
        'source': 'judge:domain',
    },
]


# Minimal cluster_profiles + lineage so PersonaNamer can build its prompt.
# The actual content doesn't matter — we only care about what PROMPT
# PersonaNamer sends to the LLM, which is the test target.
PROFILES = {
    '0': {
        'n_entities': 256, 'pct_total': 26.0,
        'top_above_average': {'sum_grocery_net_12m': 1.82, 'gender_F': 1.95},
        'top_below_average': {'sum_travel_all': 0.28},
        'feature_means': {'sum_grocery_net_12m': 3216, 'gender_F': 0.93,
                          'sum_travel_all': 12.5},
        'overall': {},
        'lineage': {'parent': None, 'children': [], 'depth': 0},
    },
    '4': {
        'n_entities': 207, 'pct_total': 21.1,
        'top_above_average': {'sum_gas_transport_all': 2.20,
                              'count_gas_transport_6m': 2.05},
        'top_below_average': {'sum_shopping_net_all': 0.52,
                              'sum_travel_all': 0.38},
        'feature_means': {'sum_gas_transport_all': 1200,
                          'sum_shopping_net_all': 800,
                          'sum_travel_all': 35},
        'overall': {},
        'lineage': {'parent': None, 'children': [], 'depth': 0},
    },
}
LINEAGE = {}

# Stub LLM response — valid JSON the PersonaNamer parser will accept.
STUB_RESPONSE = json.dumps({
    '0': {
        'name': 'Stub Cluster Zero',
        'tagline': 'placeholder',
        'description': 'placeholder description for cluster 0.',
        'dominant_categories': [],
        'traits': ['trait1', 'trait2', 'trait3'],
        'confidence': 9,
    },
    '4': {
        'name': 'Stub Cluster Four',
        'tagline': 'placeholder',
        'description': 'placeholder description for cluster 4.',
        'dominant_categories': [],
        'traits': ['trait1', 'trait2', 'trait3'],
        'confidence': 9,
    },
})


def main() -> int:
    print('=' * 72)
    print('CONSUMPTION-SIDE PROOF — does PersonaNamer inject the prefs block?')
    print('=' * 72)

    # ── 1. Set up controlled feedback log ───────────────────────────────────
    log_path = pathlib.Path('outputs/user_feedback_log.jsonl')
    backup = pathlib.Path(f'outputs/.feedback_log_backup_{int(time.time())}.jsonl')
    if log_path.exists():
        shutil.copy2(log_path, backup)
        print(f'  [1] backed up real log → {backup}')

    try:
        # Use the real feedback_store.append() to write our rules so they
        # land with proper id/date/active fields — exactly like the live system.
        log_path.write_text('')   # start clean for the test
        from ui import feedback_store
        for r in JUDGE_RULES:
            feedback_store.append(r)
        print(f'  [1] wrote {len(JUDGE_RULES)} judge-generated rules to '
              f'{log_path}')

        # ── 2. Mock the bus.ask() to capture the prompt ─────────────────────
        captured: dict = {}

        from skills.orchestrator_bus import OrchestratorBus
        from agents.persona_namer import PersonaNamingAgent

        bus = OrchestratorBus()

        def _capture_ask(*, agent, purpose, prompt, max_tokens=None, **kw):
            captured['agent'] = agent
            captured['purpose'] = purpose
            captured['prompt'] = prompt
            captured['max_tokens'] = max_tokens
            return STUB_RESPONSE

        bus.ask = _capture_ask
        print('  [2] OrchestratorBus.ask() wrapped to capture prompt')

        # ── 3. Run PersonaNamer ─────────────────────────────────────────────
        agent = PersonaNamingAgent(bus)
        print('  [3] calling PersonaNamingAgent.run(profiles, lineage)…')
        result = agent.run(
            profiles=PROFILES, lineage=LINEAGE, tone='easy',
            iteration=1, force_proceed=True,
        )
        print(f'  [3] PersonaNamer.run() returned action={result.action!r}, '
              f'passed={result.passed}')

        # ── 4. Inspect captured prompt ──────────────────────────────────────
        prompt = captured.get('prompt') or ''
        print('\n──── CAPTURED PROMPT INSPECTION ────')
        print(f'  Agent: {captured.get("agent")}')
        print(f'  Purpose: {captured.get("purpose")}')
        print(f'  Prompt length: {len(prompt)} chars')

        assert 'USER PREFERENCES' in prompt, \
            'FAIL — prefs block header not in prompt'
        print('  [✓] prompt contains "USER PREFERENCES" header')

        # Each judge-generated rule's distinctive phrase must appear:
        signature_phrases = [
            ('global_rule: merge near-duplicate clusters', 'less than 0.20x'),
            ('outlier stratum hint', 'isolation forest'),
            ('audit feature ratios', "below 1.00x"),
            ('commuters rename', 'Active Commuters'),
            ('no pejorative names', 'pejorative'),
            ('no Premium without 75th pct', '75th percentile'),
        ]
        for label, phrase in signature_phrases:
            assert phrase in prompt, f'FAIL — rule "{label}" missing ({phrase!r})'
            print(f'  [✓] prompt contains rule: {label}')

        # The prefs block must appear BEFORE the cluster-data table.
        prefs_pos = prompt.find('USER PREFERENCES')
        cluster_pos = max(prompt.find('Cluster 0'),
                           prompt.find('Cluster ID'),
                           prompt.find('cluster_id'),
                           prompt.find('Cluster '))
        assert prefs_pos >= 0 and (cluster_pos < 0 or prefs_pos < cluster_pos), \
            f'FAIL — prefs block not before cluster section ' \
            f'(prefs={prefs_pos}, cluster={cluster_pos})'
        print('  [✓] prefs block appears BEFORE cluster data in prompt')

        # ── 5. Show what the LLM would actually receive (preview) ───────────
        print('\n──── FIRST 1800 CHARS OF THE PROMPT SENT TO LLM ────')
        print(prompt[:1800])
        print('  …')

        print('\n' + '=' * 72)
        print('PROOF: PersonaNamer reads the feedback log and prepends the')
        print('       preferences block to its LLM prompt on every call.')
        print('       The 6 judge-generated rules are all inside the prompt.')
        print('=' * 72)
        return 0
    finally:
        # ── 6. Restore the original feedback log ───────────────────────────
        if backup.exists():
            shutil.copy2(backup, log_path)
            backup.unlink()
            print(f'\n  [6] restored real feedback log from backup '
                  f'({log_path}: '
                  f'{len(log_path.read_text().splitlines())} entries)')
        elif log_path.exists():
            log_path.unlink()
            print('\n  [6] removed test-only log (no backup existed)')


if __name__ == '__main__':
    sys.exit(main())
