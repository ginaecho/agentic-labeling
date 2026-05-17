"""Live end-to-end test of the judge → arbiter → feedback → next-run path.

Uses a static, hand-built personas.json (derived from the structure of a
real prior run) so we exercise:

  1. critique_run() — three blind judges with asymmetric views, real
     Anthropic API calls.
  2. arbitrate() — accept/reject each critique with rebuttal logging.
  3. feedback_store — accepted rules appended to a TEMP feedback log
     (your real outputs/user_feedback_log.jsonl is left untouched).
  4. build_preferences_block() — shows the EXACT block of text that
     PersonaNamingAgent would inject into its prompt on the next run.

Cost: ~$0.30–$0.50 of API for 3 judges + up to N arbiter calls.

Run:
  .venv/bin/python -m experiments.test_judges_live
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import shutil
import sys
import tempfile
from unittest.mock import patch

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── Hand-built realistic personas.json ────────────────────────────────────────
# Drawn from outputs/pipeline_run_20260517_101048.txt — the structure
# matches what save_outputs() writes. 5 clusters chosen to give each judge
# something to chew on:
#   - cluster 0: clean, business-judge bait (good name, simple)
#   - cluster 1: business-judge bait (over-complex name)
#   - cluster 2: statistical-judge bait (near-duplicate of cluster 3)
#   - cluster 3: statistical-judge bait (the near-duplicate)
#   - cluster 4: domain-judge bait (name doesn't match the top features)

PERSONAS = {
    "0": {
        "cluster_stats": {
            "n_entities": 256, "pct_total": 26.0,
            "top_above_average": {
                "sum_grocery_net_12m": 1.82,
                "sum_grocery_net_all": 1.81,
                "count_grocery_net_12m": 1.73,
                "count_grocery_pos_6m": 1.65,
                "pct_count_grocery_pos_all": 1.55,
                "gender_F": 1.95,
            },
            "top_below_average": {
                "sum_travel_all": 0.28,
                "max_amt_all":    0.41,
                "days_since_last": 0.12,
            },
        },
        "persona": {
            "name": "Grocery-Obsessed Female Bulk Buyers",
            "tagline": "Filling the trolley — and then some",
            "description": "256 customers who spend nearly 2x the average on "
                            "groceries, almost entirely female, with very high "
                            "shop frequency and recent activity.",
            "traits": ["very high grocery frequency",
                       "overwhelmingly female",
                       "active recently"],
        },
    },
    "1": {
        "cluster_stats": {
            "n_entities": 27, "pct_total": 2.7,
            "top_above_average": {
                "sum_shopping_net_all": 2.4,
                "sum_misc_pos_all":     2.1,
                "mean_amt_all":         1.9,
                "max_amt_all":          1.8,
                "days_since_last":      3.1,
            },
            "top_below_average": {
                "count_grocery_pos_6m": 0.18,
                "count_food_dining_6m": 0.22,
            },
        },
        "persona": {
            "name": "Dormant Premium Lifestyle and Luxury Shopping Vertical Spenders",
            "tagline": "Previously high-amplitude discretionary spend across "
                        "shopping and lifestyle categories with prolonged "
                        "absence from transactional engagement",
            "description": "A small segment of 27 customers historically "
                            "characterised by elevated discretionary spending "
                            "across shopping and lifestyle categories, now "
                            "exhibiting attenuated transactional cadence.",
            "traits": ["historically elevated discretionary spend",
                       "current attenuated engagement",
                       "lifestyle-category focus"],
        },
    },
    "2": {
        "cluster_stats": {
            "n_entities": 142, "pct_total": 14.4,
            "top_above_average": {
                "sum_grocery_pos_all": 1.45,
                "count_grocery_pos_6m": 1.40,
                "gender_M": 1.6,
                "city_pop": 1.8,
                "mean_grocery_net_all": 1.35,
            },
            "top_below_average": {
                "sum_travel_all": 0.42,
                "max_amt_all":   0.55,
            },
        },
        "persona": {
            "name": "Urban Male Grocery Regulars",
            "tagline": "City guys who keep the fridge stocked",
            "description": "142 customers, mostly male and urban, with "
                            "consistently above-average grocery spend.",
            "traits": ["urban", "predominantly male",
                       "steady grocery spending"],
        },
    },
    "3": {
        "cluster_stats": {
            "n_entities": 138, "pct_total": 14.0,
            "top_above_average": {
                "sum_grocery_pos_all": 1.43,
                "count_grocery_pos_6m": 1.38,
                "gender_M": 1.5,
                "city_pop": 1.6,
                "mean_grocery_net_all": 1.32,
            },
            "top_below_average": {
                "sum_travel_all": 0.46,
                "max_amt_all":   0.58,
            },
        },
        "persona": {
            "name": "Metropolitan Gentlemen Grocery Patrons",
            "tagline": "Big-city men with a steady grocery habit",
            "description": "138 city-dwelling male customers with elevated "
                            "grocery spend.",
            "traits": ["urban", "male", "regular grocery activity"],
        },
    },
    "4": {
        "cluster_stats": {
            "n_entities": 207, "pct_total": 21.1,
            "top_above_average": {
                "sum_gas_transport_all": 2.20,
                "count_gas_transport_6m": 2.05,
                "pct_count_gas_transport_all": 1.95,
                "city_pop": 0.42,
                "count_food_dining_all": 1.15,
            },
            "top_below_average": {
                "sum_shopping_net_all": 0.52,
                "sum_travel_all":       0.38,
            },
        },
        "persona": {
            "name": "Frequent Male Travellers",
            "tagline": "Always on the road for business",
            "description": "207 customers with elevated activity in transport "
                            "and food categories.",
            "traits": ["frequent transport spend",
                       "moderate food spend",
                       "low shopping spend"],
        },
    },
}

CLASSIFIER_METRICS = {
    "cv_accuracy": 0.85,
    "cv_f1_macro": 0.79,
    "cv_f1_weighted": 0.84,
    "per_class_f1": {
        "Grocery-Obsessed Female Bulk Buyers": 0.91,
        "Dormant Premium Lifestyle and Luxury Shopping Vertical Spenders": 0.62,
        "Urban Male Grocery Regulars": 0.71,
        "Metropolitan Gentlemen Grocery Patrons": 0.68,
        "Frequent Male Travellers": 0.84,
    },
    "top20_features": {},
    "reasoning": "5-fold CV on the 5-cluster solution.",
}


def _build_run_dir(tmp: pathlib.Path) -> pathlib.Path:
    run = tmp / 'run_static'
    out = run / 'outputs'
    out.mkdir(parents=True)
    (out / 'personas.json').write_text(json.dumps(PERSONAS, indent=2))
    (out / 'classifier_metrics.json').write_text(json.dumps(CLASSIFIER_METRICS, indent=2))

    # Synth cluster_labels.csv proportional to cluster sizes — domain judge
    # samples raw rows from features_df aligned with this.
    rows = []
    idx = 0
    for cid, d in PERSONAS.items():
        n = d['cluster_stats']['n_entities']
        for _ in range(n):
            rows.append((idx, int(cid)))
            idx += 1
    df = pd.DataFrame(rows, columns=['row_index', 'cluster_id'])
    df.to_csv(out / 'cluster_labels.csv', index=False)
    return run


def _build_synth_features_df(n_rows: int) -> pd.DataFrame:
    """Tiny features matrix so the domain judge can sample real rows."""
    import numpy as np
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'sum_grocery_net_12m':   rng.uniform(500, 4000, n_rows),
        'count_grocery_pos_6m':  rng.integers(0, 60, n_rows),
        'sum_gas_transport_all': rng.uniform(50, 800,  n_rows),
        'sum_shopping_net_all':  rng.uniform(0, 1500, n_rows),
        'sum_travel_all':        rng.uniform(0, 600,  n_rows),
        'mean_amt_all':          rng.uniform(20, 120, n_rows),
        'max_amt_all':           rng.uniform(50, 800, n_rows),
        'city_pop':              rng.integers(500, 200000, n_rows),
        'gender_F':              rng.integers(0, 2, n_rows),
        'gender_M':              rng.integers(0, 2, n_rows),
        'days_since_last':       rng.integers(0, 90, n_rows),
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--persist', action='store_true',
                    help='Write accepted rules to the REAL '
                         'outputs/user_feedback_log.jsonl '
                         '(default: isolated temp file).')
    args = ap.parse_args()

    if not (os.environ.get('LLM_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')):
        print('ERROR: no API key found in env. Source .env first.')
        return 2

    print('=' * 70)
    print('LIVE TEST — judges + arbiter + feedback chain')
    print('=' * 70)

    tmp = pathlib.Path(tempfile.mkdtemp(prefix='judges-live-'))
    run_dir = _build_run_dir(tmp)
    n_rows = sum(d['cluster_stats']['n_entities'] for d in PERSONAS.values())
    features_df = _build_synth_features_df(n_rows)
    print(f'  run_dir: {run_dir}')
    print(f'  n_personas: {len(PERSONAS)}   features rows: {len(features_df)}')

    if args.persist:
        # Use the REAL feedback log so the next pipeline run picks up
        # our rules via build_preferences_block().
        log_ctx = contextlib.nullcontext()
        print('  feedback log: outputs/user_feedback_log.jsonl '
              '(PERSIST mode — real log will be appended to)')
    else:
        temp_feedback = tmp / 'user_feedback_log.jsonl'
        log_ctx = patch('ui.feedback_store.LOG_PATH', temp_feedback)
        print(f'  feedback log (temp, isolated): {temp_feedback}')

    with log_ctx:
        from experiments.judges import critique_run, build_arbiter_evidence
        from experiments.arbiter import arbitrate
        from ui import feedback_store

        # ── Step 1: judges ────────────────────────────────────────────────────
        print('\n──── STEP 1: three blind judges fire in parallel ────')
        critiques = critique_run(
            run_dir,
            dataset_description='customer-level features engineered from credit-card '
                                 'transaction history (fraud-detection corpus).',
            features_df=features_df,
        )
        print(f'\nTotal critiques: {len(critiques)}')
        for i, c in enumerate(critiques, 1):
            print(f'\n  [{i}] judge={c.judge}  severity={c.severity}  '
                  f'target=cluster {c.target_cluster_id} ({c.target_cluster_name or "—"})')
            print(f'      issue:      {c.issue}')
            print(f'      suggestion: {c.suggestion}')
            print(f'      evidence:   {c.evidence}')

        # ── Step 2: arbiter ───────────────────────────────────────────────────
        print('\n──── STEP 2: arbiter decides accept/reject for each ────')
        ledger = tmp / 'learning_ledger.jsonl'
        evidence = build_arbiter_evidence(
            run_dir,
            dataset_description='credit-card transaction history',
            features_df=features_df,
        )
        counts = arbitrate(critiques, evidence, ledger)
        print(f'\nResult: accepted={counts["n_accepted"]}  '
              f'rejected={counts["n_rejected"]}  '
              f'accept_rate={counts["accept_rate"]:.0%}')

        # ── Step 3: show what was written to the (temp) feedback log ──────────
        print('\n──── STEP 3: rules written to the feedback log ────')
        appended = feedback_store.read_all()
        if not appended:
            print('  (nothing accepted — log is empty)')
        for e in appended:
            print(f'  • [{e.get("priority", "?").upper()}] '
                  f'{e.get("type")} → '
                  f'target={e.get("target_cluster_name") or e.get("target_cluster_id") or "global"}')
            print(f'    {e.get("rule") or e.get("hint") or "(no text)"}')

        # ── Step 4: what the next PersonaNamer call would see ────────────────
        print('\n──── STEP 4: preferences block injected into PersonaNamer next run ────')
        block = feedback_store.build_preferences_block()
        if not block:
            print('  (empty — no high/medium priority rules)')
        else:
            print(block)

        # ── Step 5: where it gets injected (file:line) ───────────────────────
        print('──── STEP 5: which agent reads this (file:line references) ────')
        import subprocess as _sp
        try:
            grep = _sp.run(['grep', '-n', 'build_preferences_block\\|feedback_store',
                             'agents/persona_namer.py', 'agents/orchestrator.py'],
                            capture_output=True, text=True)
            print(grep.stdout.strip() or '  (no references found)')
        except Exception as e:
            print(f'  (grep failed: {e})')

        # ── Step 6: full ledger (accepts + rebuttals) ────────────────────────
        print('\n──── STEP 6: full ledger (accepts AND rebuttals) ────')
        for line in ledger.read_text().splitlines():
            obj = json.loads(line)
            c = obj['critique']; d = obj['decision']
            tag = 'ACCEPTED' if d['accepted'] else 'REJECTED'
            print(f'\n  {tag} — judge={c["judge"]}  '
                  f'target=cluster {c["target_cluster_id"]}')
            print(f'    critique:  {c["issue"]}')
            print(f'    arbiter:   {d["reasoning"]}')

    print('\n' + '=' * 70)
    print(f'DONE — artifacts in {tmp}')
    if args.persist:
        print(f'  (rules were APPENDED to the real outputs/user_feedback_log.jsonl)')
    else:
        print(f'  (the real outputs/user_feedback_log.jsonl was NOT modified)')
    print('=' * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
