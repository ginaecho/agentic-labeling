"""Outer convergence loop.

For up to N hours (default 3), repeatedly:
  1. Run run_pipeline.py --bypass --max-iterations 6 against a per-run output dir.
  2. Three judge agents critique the resulting named clusters.
  3. The arbiter decides which critiques to accept; accepted ones are
     appended to outputs/user_feedback_log.jsonl so the *next* pipeline
     run reads them via ui.feedback_store.build_preferences_block().
  4. Compute ARI / name-Jaccard / size-L1 / F1-delta vs the previous run
     and append a row to experiments/runs/run_history.jsonl.
  5. Stop early if ARI >= STABILITY_THRESHOLD for STABILITY_WINDOW
     consecutive runs.

Per-run artifacts are stored under experiments/runs/run_NNN/ as a
*snapshot* of outputs/ at the end of each run, so the timeline is
auditable end to end.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / 'experiments' / 'runs'
OUTPUTS_DIR = ROOT / 'outputs'

STABILITY_THRESHOLD = 0.90
STABILITY_WINDOW = 3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _next_run_number() -> int:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in RUNS_DIR.iterdir()
                       if p.is_dir() and p.name.startswith('run_'))
    if not existing:
        return 1
    last = existing[-1].name.split('_')[-1]
    try:
        return int(last) + 1
    except ValueError:
        return len(existing) + 1


def _snapshot_outputs(run_dir: pathlib.Path) -> None:
    """Copy the key outputs/ artifacts into run_dir/outputs/ so each run
    is fully auditable later."""
    dest = run_dir / 'outputs'
    dest.mkdir(parents=True, exist_ok=True)
    for name in ('personas.json', 'classifier_metrics.json',
                  'cluster_profiles.json', 'cluster_lineage.json',
                  'cluster_labels.csv', 'silhouette_curve.json',
                  'pipeline_log.json', 'agents_conversation.txt',
                  'pca_iterations.json'):
        src = OUTPUTS_DIR / name
        if src.exists():
            shutil.copy2(src, dest / name)
    fb = OUTPUTS_DIR / 'user_feedback_log.jsonl'
    if fb.exists():
        shutil.copy2(fb, dest / 'user_feedback_log_before_judges.jsonl')


def _snapshot_feedback_after(run_dir: pathlib.Path) -> None:
    fb = OUTPUTS_DIR / 'user_feedback_log.jsonl'
    if fb.exists():
        shutil.copy2(fb, run_dir / 'outputs' / 'user_feedback_log_after_judges.jsonl')


def _run_pipeline(max_iterations: int, intent_target: str,
                   intent_purpose: str, run_log: pathlib.Path) -> bool:
    """Invoke run_pipeline.py headless. Returns True on exit-code 0."""
    cmd = [
        sys.executable, str(ROOT / 'run_pipeline.py'),
        '--bypass',
        '--max-iterations', str(max_iterations),
        '--intent-target', intent_target,
        '--intent-purpose', intent_purpose,
    ]
    print(f'  [loop] launching: {" ".join(cmd)}')
    t0 = time.perf_counter()
    # Inner pipeline opts into LLM rule compaction at injection time
    # via this env var. PersonaNamer reads it and routes through
    # experiments.dedup_prefs instead of the raw feedback_store.
    env = {**os.environ, 'EXPERIMENT_DEDUP_PREFS': '1'}
    with run_log.open('w', encoding='utf-8') as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                cwd=str(ROOT), env=env)
    elapsed = time.perf_counter() - t0
    print(f'  [loop] pipeline exit={proc.returncode}  wall={elapsed/60:.1f} min')
    return proc.returncode == 0


def _read_run_metrics(run_dir: pathlib.Path) -> dict:
    """Pull the headline metrics out of one run's outputs."""
    out = run_dir / 'outputs'
    metrics: dict = {'cv_f1_macro': None, 'silhouette': None,
                      'max_vif': None, 'n_clusters': None}
    p = out / 'classifier_metrics.json'
    if p.exists():
        d = json.loads(p.read_text())
        metrics['cv_f1_macro'] = d.get('cv_f1_macro')
    # silhouette curve has the best k's score; cluster_profiles holds size info
    pers = out / 'personas.json'
    if pers.exists():
        metrics['n_clusters'] = len(json.loads(pers.read_text()))
    return metrics


def main() -> int:
    # Load .env so the inner subprocess (run_pipeline.py) inherits the
    # API key via os.environ.copy() in _run_pipeline().
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-hours', type=float, default=3.0,
                    help='Wall-clock budget for the outer loop (default 3)')
    ap.add_argument('--max-iterations', type=int, default=6,
                    help='Inner pipeline iterations per run (default 6)')
    ap.add_argument('--intent-target', type=str, default='customers')
    ap.add_argument('--intent-purpose', type=str,
                    default='discover spending personas for marketing')
    ap.add_argument('--dataset-description', type=str,
                    default='customer-level features engineered from credit-card '
                            'transaction history (fraud-detection corpus). '
                            'Each row is one customer; columns describe spending '
                            'amounts, category breakdowns, and recency.',
                    help='Short factual description of WHAT the data is about. '
                         'Shown ONLY to the domain judge.')
    ap.add_argument('--features-path', type=str, default=None,
                    help='Engineered features parquet/CSV used for raw-record '
                         'sampling by the domain judge. Auto-detected if omitted.')
    ap.add_argument('--reset-feedback', action='store_true',
                    help='Wipe outputs/user_feedback_log.jsonl before starting')
    args = ap.parse_args()

    # Auto-detect features file if not provided. The domain judge samples
    # raw rows from this; the other two judges never see it.
    features_path = args.features_path
    if features_path is None:
        for cand in ('data/processed/engineered_features.parquet',
                     'data/processed/customer_features.parquet',
                     'data/raw/fraudTrain.csv'):
            if (ROOT / cand).exists():
                features_path = str(ROOT / cand)
                break
    features_df = None
    if features_path:
        try:
            import pandas as pd
            features_df = (pd.read_parquet(features_path)
                            if features_path.endswith('.parquet')
                            else pd.read_csv(features_path))
            print(f'  [loop] loaded features for domain-judge sampling: '
                  f'{features_path}  shape={features_df.shape}')
        except Exception as e:
            print(f'  [loop] could NOT load features ({e}) — domain judge '
                  f'will fall back to ratio-only evidence.')

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Lazy imports — only after RUNS_DIR exists and only inside main so
    # the module is importable without anthropic / ui deps for tests.
    from experiments.judges import critique_run, build_arbiter_evidence
    from experiments.arbiter import arbitrate
    from experiments.diff_personas import diff

    if args.reset_feedback:
        fb = OUTPUTS_DIR / 'user_feedback_log.jsonl'
        if fb.exists():
            fb.unlink()
            print('  [loop] wiped outputs/user_feedback_log.jsonl')

    history_path = RUNS_DIR / 'run_history.jsonl'
    pid_path = RUNS_DIR / 'current.pid'
    pid_path.write_text(str(os.getpid()))

    deadline = time.time() + args.max_hours * 3600
    stable_streak = 0
    converged = False
    prev_run_dir: pathlib.Path | None = None
    run_idx = _next_run_number() - 1   # _run_one increments

    print('=' * 65)
    print(f'CONVERGENCE LOOP  budget={args.max_hours}h  '
          f'inner_iters={args.max_iterations}  pid={os.getpid()}')
    print(f'  stop when ARI >= {STABILITY_THRESHOLD} for '
          f'{STABILITY_WINDOW} consecutive runs, or wall-clock exhausted.')
    print('=' * 65)

    while time.time() < deadline:
        run_idx += 1
        run_dir = RUNS_DIR / f'run_{run_idx:03d}'
        run_dir.mkdir(parents=True, exist_ok=True)
        run_log = run_dir / 'pipeline.log'
        print(f'\n──── RUN {run_idx:03d}  ({_now()}) ────')

        ok = _run_pipeline(args.max_iterations, args.intent_target,
                            args.intent_purpose, run_log)
        if not ok:
            print(f'  [loop] pipeline failed — see {run_log}; aborting outer loop.')
            break

        _snapshot_outputs(run_dir)

        print('  [loop] running 3 blind judges in parallel (asymmetric views)…')
        critiques = critique_run(run_dir, args.dataset_description, features_df)
        evidence = build_arbiter_evidence(run_dir, args.dataset_description,
                                           features_df)
        with (run_dir / 'critiques.jsonl').open('w', encoding='utf-8') as f:
            for c in critiques:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + '\n')

        ledger_path = RUNS_DIR / 'learning_ledger.jsonl'
        arb_counts = arbitrate(critiques, evidence, ledger_path)
        _snapshot_feedback_after(run_dir)

        metrics = _read_run_metrics(run_dir)
        diff_metrics = diff(prev_run_dir, run_dir) if prev_run_dir else {
            'ari': None, 'name_jaccard': None, 'size_l1': None,
            'f1_delta': None, 'n_personas_prev': None,
            'n_personas_curr': metrics.get('n_clusters'),
        }

        row = {
            'run': run_idx,
            'ts': _now(),
            'metrics': metrics,
            'diff_vs_prev': diff_metrics,
            'judges': {'n_critiques': len(critiques)},
            'arbiter': arb_counts,
        }
        with history_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f'  [loop] ARI={diff_metrics.get("ari")}  '
              f'F1={metrics.get("cv_f1_macro")}  '
              f'n={metrics.get("n_clusters")}')

        ari = diff_metrics.get('ari')
        if ari is not None and ari >= STABILITY_THRESHOLD:
            stable_streak += 1
            print(f'  [loop] stable_streak={stable_streak}/{STABILITY_WINDOW}')
            if stable_streak >= STABILITY_WINDOW:
                print(f'\n[loop] CONVERGED — ARI >= {STABILITY_THRESHOLD} '
                      f'for {STABILITY_WINDOW} runs. Stopping.')
                converged = True
                break
        else:
            if stable_streak:
                print(f'  [loop] stable_streak reset (was {stable_streak})')
            stable_streak = 0

        prev_run_dir = run_dir
        if time.time() >= deadline:
            print('\n[loop] wall-clock budget exhausted.')
            break

    # ── Rule-review branch ────────────────────────────────────────────────
    # Converged → Decision Maker promotes/demotes rules based on what
    # was load-bearing. Not converged → write human_review.md for the
    # user to inspect and re-import via experiments.human_review apply.
    if converged:
        print(f'\n[loop] convergence reached — running Decision Maker review…')
        try:
            from experiments.convergence_review import review_after_convergence
            review_after_convergence()
        except Exception as e:
            print(f'  [loop] convergence_review failed: {e}')
    else:
        print(f'\n[loop] no convergence — writing human_review.md…')
        try:
            from experiments.human_review import write_review_md
            md_path = write_review_md()
            print(f'  [loop] edit {md_path}, then run:')
            print(f'         python -m experiments.human_review apply {md_path}')
        except Exception as e:
            print(f'  [loop] human_review write failed: {e}')

    if pid_path.exists():
        pid_path.unlink()
    print(f'\n[loop] done. {run_idx} run(s). converged={converged}. '
          f'history → {history_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
