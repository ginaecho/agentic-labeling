"""
run_pipeline.py

Loads .env, runs the 4-agent pipeline, then saves and prints:

  outputs/persona_summary.txt   — full notebook-04-style persona cards
                                  + per-persona key-metric evidence
  outputs/persona_metrics.csv   — one row per cluster × category with
                                  all raw + relative metrics

Console report sections:
  HOW AGENTS INTERACT          — architecture diagram + persona-labelling prose
  CLUSTER SIZE CHECK           — n + % per cluster, 40% guard confirmation
  PERSONA SUMMARY              — notebook-04 format: name / tagline /
                                  dominant / traits + KEY METRICS block
  KEY FEATURES                 — top-15 RF importance + CV scores
  LLM AGENT USAGE              — tokens (in/out) + time per agent per call
  AGENTS & TIME                — wall-clock timing per agent
  TOTAL TIME                   — pipeline total
"""
import atexit
import csv
import io
import os
import pathlib
import sys
import textwrap
import time
from datetime import datetime


class _Tee:
    """Write to both the original stream and a log file."""
    def __init__(self, stream, logfile):
        self._stream = stream
        self._logfile = logfile
    def write(self, data):
        self._stream.write(data)
        if self._logfile and not self._logfile.closed:
            self._logfile.write(data)
    def flush(self):
        self._stream.flush()
        if self._logfile and not self._logfile.closed:
            self._logfile.flush()
    def __getattr__(self, name):
        return getattr(self._stream, name)


_log_file = None
_orig_stdout = _orig_stderr = None


def _close_log():
    global _log_file, _orig_stdout, _orig_stderr
    if _log_file is not None and not _log_file.closed:
        _log_file.flush()
        _log_file.close()
    if _orig_stdout is not None:
        sys.stdout = _orig_stdout
    if _orig_stderr is not None:
        sys.stderr = _orig_stderr


# ── Load .env ─────────────────────────────────────────────────────────────────
env_path = pathlib.Path(__file__).parent / '.env'
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, _, v = line.partition('=')
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

if not os.environ.get('ANTHROPIC_API_KEY'):
    sys.exit('ERROR: ANTHROPIC_API_KEY not found in .env or environment.')

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import json
import yaml
from agents.orchestrator import Orchestrator

# ── Tee all terminal output to a log file ─────────────────────────────────────
_root = pathlib.Path(__file__).resolve().parent
_out_dir = _root / 'outputs'
_out_dir.mkdir(parents=True, exist_ok=True)
_log_path = _out_dir / f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
_log_file = _log_path.open('w', encoding='utf-8')
_orig_stdout = sys.stdout
_orig_stderr = sys.stderr
sys.stdout = _Tee(_orig_stdout, _log_file)
sys.stderr = _Tee(_orig_stderr, _log_file)
atexit.register(_close_log)
print(f"[run_pipeline] Logging full output to {_log_path}")

with open(_root / 'config.yaml') as f:
    config = yaml.safe_load(f)

# ── Auto-approve at human checkpoint ──────────────────────────────────────────
# We do NOT approve at the first passing iteration. Instead we ask for a few
# more recluster rounds so the orchestrator's state.best_* tracker can compare
# multiple full {naming, F1} candidates and pick the actual winner. Path A in
# orchestrator (max_iterations_reached) saves state.best_naming_result, which
# is now ranked by F1 (see state.update_best). The approve path also saves the
# all-time best, so even if we DO approve mid-loop the winning iteration's
# personas reach outputs/personas.json (and the Named Clusters tab).
import agents.orchestrator as _orch_mod
from agents.state import HumanDecision

# Approve only after at least this many passing iterations have been collected.
# Each subsequent recluster also re-runs _ask_parameter_tuning so the next
# iteration tries a different algorithm/k for diversity.
_MIN_PASSING_BEFORE_APPROVE = 3
_passing_count = {'n': 0}

_orig_chk = _orch_mod.human_checkpoint
def _auto_approve(personas, cr, clf, bus):
    _orig_chk(personas, cr, clf, bus)
    _passing_count['n'] += 1
    n = _passing_count['n']
    if n < _MIN_PASSING_BEFORE_APPROVE:
        print(f'\n[Auto-approve] Passing iteration #{n} — continuing to collect '
              f'more candidates before picking the winner ({_MIN_PASSING_BEFORE_APPROVE} target).')
        return HumanDecision(
            action='recluster',
            feedback=(
                f'Exploration round {n}/{_MIN_PASSING_BEFORE_APPROVE}: keep iterating '
                'to find a higher-F1 cluster set. Try a different algorithm or k.'
            ),
        )
    print(f'\n[Auto-approve] Collected {n} passing iterations — selecting best by F1.')
    return HumanDecision(action='approve')
_orch_mod.human_checkpoint = _auto_approve

# ── Resolve features path ──────────────────────────────────────────────────────
# Priority: CLI argument > config.yaml dataset_path > default raw CSV > parquet
import argparse
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--data', type=str, default=None,
                     help='Path to input CSV/parquet (overrides config.yaml)')
_parser.add_argument('--no-ui', action='store_true',
                     help='Skip launching the interactive UI (headless mode)')
_parser.add_argument('--ui-port', type=int, default=5057,
                     help='Port for the interactive UI (default 5057)')
_parser.add_argument('--max-iterations', type=int, default=10,
                     help='Maximum inner pipeline iterations (default 10)')
_parser.add_argument('--bypass', action='store_true',
                     help='No prompts. Synthesize user intent from --intent-* '
                          'flags and auto-approve. Implies --no-ui.')
_parser.add_argument('--intent-target', type=str, default='customers',
                     help='Bypass: target entity for clustering')
_parser.add_argument('--intent-purpose', type=str,
                     default='discover spending personas for marketing',
                     help='Bypass: business purpose')
_args, _ = _parser.parse_known_args()
if _args.bypass:
    _args.no_ui = True

# ── Launch the interactive UI in a background thread ─────────────────────────
# The pipeline keeps running in the foreground; the UI streams live agent
# activity over Server-Sent Events and auto-switches to the cluster grid
# when the pipeline completes. Use --no-ui for headless runs.
if not _args.no_ui:
    import threading
    import webbrowser

    def _launch_ui_background(host: str, port: int) -> None:
        try:
            from ui.app import app as _ui_app
        except Exception as _exc:  # noqa: BLE001
            print(f'[run_pipeline] Could not import UI ({_exc}). Continuing headless.')
            return

        def _serve():
            try:
                _ui_app.run(host=host, port=port, debug=False,
                            use_reloader=False, threaded=True)
            except Exception as _serve_exc:  # noqa: BLE001
                print(f'[run_pipeline] UI server stopped: {_serve_exc}')

        t = threading.Thread(target=_serve, name='ui-server', daemon=True)
        t.start()

        url = f'http://{host}:{port}/'
        print(f'[run_pipeline] Launching live UI at {url} (use --no-ui to disable)')

        def _open_when_ready():
            time.sleep(1.2)
            try:
                webbrowser.open(url)
            except Exception:  # noqa: BLE001
                pass

        threading.Thread(target=_open_when_ready, name='ui-opener',
                         daemon=True).start()

    _launch_ui_background('127.0.0.1', _args.ui_port)

_config_data_path = config.get('dataset_path')
_raw_csv = pathlib.Path('data/raw/fraudTrain.csv')
_parquet = pathlib.Path('data/processed/customer_features.parquet')

if _args.data:
    _default_features_path = _args.data
    print(f'[run_pipeline] Using CLI-specified data: {_args.data}')
elif _config_data_path:
    _default_features_path = _config_data_path
    print(f'[run_pipeline] Using config.yaml dataset_path: {_config_data_path}')
elif _raw_csv.exists():
    _default_features_path = str(_raw_csv)
    print(f'[run_pipeline] Using raw CSV for fresh feature engineering: {_raw_csv}')
elif _parquet.exists():
    _default_features_path = str(_parquet)
    print(f'[run_pipeline] Raw CSV not found — falling back to pre-built parquet: {_parquet}')
else:
    _default_features_path = str(_raw_csv)   # orchestrator will surface the error

# ── Run pipeline ──────────────────────────────────────────────────────────────
orchestrator = Orchestrator(config)

_bypass_intent = None
if _args.bypass:
    from agents.state import UserIntent
    _bypass_intent = UserIntent(
        target_entity=_args.intent_target,
        business_purpose=_args.intent_purpose,
        dataset_path=_default_features_path,
        constraints='',
    )
    print(f'[run_pipeline] BYPASS mode — synthesized intent: '
          f'target={_args.intent_target!r}  purpose={_args.intent_purpose!r}')

# Abort/restart loop: the UI can POST /api/abort to write
# outputs/pipeline_abort.json. The orchestrator picks it up at an iteration
# boundary and returns status='aborted'. We then wait for the user to submit
# a new intent (pending_intent.json) and re-run from scratch — orchestrator
# config is restored from baseline so per-run overrides don't leak.
def _wait_for_new_intent() -> None:
    """Block until outputs/pending_intent.json appears (UI re-submits)."""
    p = pathlib.Path('outputs/pending_intent.json')
    print('[run_pipeline] Waiting for a new intent from the UI '
          '(submit via the intent form)... press Ctrl-C to give up.')
    while not p.exists():
        time.sleep(0.5)
    print('[run_pipeline] New intent received — restarting the pipeline.\n')


while True:
    # After an abort the user submitted a fresh intent via the UI; switch off
    # bypass for the restarted run so UserInputAgent reads it from disk.
    _skip_input = bool(_args.bypass) and _bypass_intent is not None
    _intent = _bypass_intent

    result = orchestrator.run(
        features_path=_default_features_path,
        max_total_iterations=_args.max_iterations,
        skip_user_input=_skip_input,
        user_intent=_intent,
    )

    _status = result.get('status')

    # ── Recoverable terminations: keep the UI server alive and let the user
    # submit a fresh intent. Anything that's NOT an explicit user-driven exit
    # ("quit") falls through to the restart wait.
    if _status == 'aborted':
        if result.get('restart', True):
            _wait_for_new_intent()
            _bypass_intent = None   # restart goes through normal intent flow
            continue
        print('\n[run_pipeline] Pipeline aborted; restart=false — exiting.')
        sys.exit(0)

    if _status == 'quit':
        # User explicitly typed 'quit' at the human checkpoint — exit.
        print('\n[run_pipeline] Pipeline ended with status=quit — exiting.')
        sys.exit(0)

    if _status == 'blocked':
        print(f"\n[run_pipeline] Pipeline blocked (a precondition failed — "
              f"see log above). Submit a new intent in the UI to try again.")
        _wait_for_new_intent()
        _bypass_intent = None
        continue

    # Sanity check: if the orchestrator returned a success-ish status but the
    # outputs aren't on disk, treat it as a soft failure and wait for a new
    # intent rather than killing the UI.
    if not pathlib.Path('outputs/personas.json').exists():
        print('\n[run_pipeline] Expected outputs/personas.json after a '
              f'{_status!r} run, but the file is missing.')
        print('[run_pipeline] Submit a new intent in the UI to try again.')
        _wait_for_new_intent()
        _bypass_intent = None
        continue

    break

# ── Banner for best-effort fallback ───────────────────────────────────────────
if result.get('status') == 'best_effort':
    sil = result.get('silhouette', 0)
    print(f'\n{"⚠" * 65}')
    print(f'  BEST-EFFORT RESULT (max iterations reached)')
    print(f'  No iteration passed all quality gates. Using the best clustering')
    print(f'  found (silhouette={sil:.4f}). Persona names are delivered with')
    print(f'  force_proceed=True — confidence scores may be lower than usual.')
    print(f'{"⚠" * 65}\n')

# ── Load saved outputs ────────────────────────────────────────────────────────
# personas.json existence was verified inside the restart loop above; here
# we trust it.
_personas_path = pathlib.Path('outputs/personas.json')
personas_data      = json.loads(_personas_path.read_text())
clf_metrics        = json.loads(pathlib.Path('outputs/classifier_metrics.json').read_text()) \
                     if pathlib.Path('outputs/classifier_metrics.json').exists() else {}

result_clf         = result.get('classifier') or {}
timing             = result.get('timing') or {}
llm_usage          = result.get('llm_usage') or {}
top_feats          = clf_metrics.get('top20_features', {})

MAX_PCT  = config.get('max_cluster_size_pct', 0.40)
W        = 72

# ══════════════════════════════════════════════════════════════════════════════
# Helper: build the full text for one cluster (notebook-04 style + key metrics)
# ══════════════════════════════════════════════════════════════════════════════
def cluster_card(cid: str, data: dict, clf_per_class: dict,
                 all_avg: dict, n_clusters: int) -> str:
    """Return the full text block for one cluster."""
    stats = data['cluster_stats']
    p     = data['persona']
    lin   = stats.get('lineage', {})

    # Support both new profile keys (n_entities, pct_total) and old (n_customers, pct_of_total)
    n_cust = stats.get('n_entities', stats.get('n_customers', 0))
    pct    = stats.get('pct_total', stats.get('pct_of_total', 0) * 100) / 100  # normalise to 0-1

    cv_f1   = clf_per_class.get(p.get('name', ''), None)
    cv_str  = f'  CV-F1={cv_f1:.3f}' if cv_f1 is not None else ''
    sub_str = f'  (sub-cluster of C{lin.get("parent")})' if lin.get('parent') else ''

    lines = []
    lines.append('')
    lines.append(f'Cluster {cid}{sub_str}  ({n_cust} entities, {pct:.1%} of total){cv_str}')
    lines.append(f'  Persona  : {p.get("name", "?")}')
    lines.append(f'  Tagline  : {p.get("tagline", "?")}')
    lines.append(f'  Dominant : {p.get("dominant_categories", p.get("top_features", []))}')
    lines.append('  Traits:')
    for t in p.get('traits', []):
        for i, wrapped in enumerate(textwrap.wrap(t, width=W - 6)):
            prefix = '    • ' if i == 0 else '      '
            lines.append(prefix + wrapped)

    lines.append('')
    lines.append('  Description:')
    for wrapped in textwrap.wrap(p.get('description', ''), width=W - 4):
        lines.append('    ' + wrapped)

    # ── Key metrics block (generic — works for any feature matrix) ────────────
    lines.append('')
    lines.append('  Key features driving this persona:')

    top_above = stats.get('top_above_average', {})
    top_below = stats.get('top_below_average', {})
    feat_means = stats.get('feature_means', {})

    if top_above:
        lines.append('  ▲ Strongly above-average features:')
        lines.append(f'    {"Feature":<40} {"vs avg":>7}  {"mean value":>12}')
        lines.append('    ' + '─' * 62)
        for feat, ratio in sorted(top_above.items(), key=lambda x: -x[1])[:8]:
            flag = '◀◀' if ratio >= 2.0 else '◀ '
            mean_val = feat_means.get(feat, 0)
            lines.append(
                f'  {flag} {feat:<40} {ratio:>6.2f}x  {mean_val:>12.4g}'
            )

    if top_below:
        lines.append('  ▼ Notably below-average features:')
        lines.append(f'    {"Feature":<40} {"vs avg":>7}  {"mean value":>12}')
        lines.append('    ' + '─' * 62)
        for feat, ratio in sorted(top_below.items(), key=lambda x: x[1])[:5]:
            mean_val = feat_means.get(feat, 0)
            lines.append(
                f'    ▼ {feat:<40} {ratio:>6.2f}x  {mean_val:>12.4g}'
            )

    return '\n'.join(lines)

# ══════════════════════════════════════════════════════════════════════════════
# Compute weighted all-cluster averages for comparison
# ══════════════════════════════════════════════════════════════════════════════
def _n_entities(d):
    s = d['cluster_stats']
    return s.get('n_entities', s.get('n_customers', 0))

total_customers = sum(_n_entities(d) for d in personas_data.values())

# Weighted average of any feature across all clusters (uses generic feature_means)
def _wavg(field):
    total = 0.0
    for d in personas_data.values():
        s   = d['cluster_stats']
        n   = _n_entities(d)
        val = s.get('feature_means', {}).get(field,
              s.get('overall', {}).get(field, 0))
        total += val * n
    return total / total_customers if total_customers else 0

all_avg = {}  # populated on demand via _wavg; no fixed field names needed

# ══════════════════════════════════════════════════════════════════════════════
# Build the full persona-summary text (notebook-04 format)
# ══════════════════════════════════════════════════════════════════════════════
sep  = '=' * 65
sep2 = '-' * 65

txt_lines = [sep, 'PERSONA SUMMARY', sep]
txt_lines.append(f'  {len(personas_data)} personas  |  {total_customers} customers  '
                 f'|  silhouette={result.get("run_history", [{}])[-1].get("silhouette", "n/a")}  '
                 f'|  CV F1 (macro)={result_clf.get("cv_f1_macro", "n/a")}')
txt_lines.append(sep)

for cid in sorted(personas_data.keys(), key=lambda x: int(x)):
    txt_lines.append(sep2)
    txt_lines.append(
        cluster_card(cid, personas_data[cid],
                     result_clf.get('per_class_f1', {}),
                     all_avg, len(personas_data))
    )

txt_lines.append('')
txt_lines.append(sep)
full_txt = '\n'.join(txt_lines)

# ══════════════════════════════════════════════════════════════════════════════
# Save persona_summary.txt
# ══════════════════════════════════════════════════════════════════════════════
pathlib.Path('outputs').mkdir(exist_ok=True)
txt_path = pathlib.Path('outputs/persona_summary.txt')
txt_path.write_text(full_txt, encoding='utf-8')

# ══════════════════════════════════════════════════════════════════════════════
# Build and save persona_metrics.csv
# ══════════════════════════════════════════════════════════════════════════════
csv_path = pathlib.Path('outputs/persona_metrics.csv')
# Generic CSV: one row per cluster × feature (top distinguishing features only)
FIELDNAMES = [
    'cluster_id', 'persona_name', 'n_entities', 'pct_of_total',
    'feature', 'mean_value', 'relative_to_avg', 'signal',
]
csv_rows = []
for cid in sorted(personas_data.keys(), key=lambda x: int(x)):
    stats  = personas_data[cid]['cluster_stats']
    pname  = personas_data[cid]['persona'].get('name', f'Cluster {cid}')
    n_ent  = _n_entities(personas_data[cid])
    s_data = stats.get('cluster_stats', stats)  # handle either nesting style
    pct    = s_data.get('pct_total', s_data.get('pct_of_total', 0) * 100) / 100
    feat_means   = stats.get('feature_means', {})
    feat_relative = stats.get('feature_relative', {})
    # Combine above + below average features
    top_above = stats.get('top_above_average', {})
    top_below = stats.get('top_below_average', {})
    all_feat = {**top_above, **top_below}
    for feat, rel in all_feat.items():
        if rel >= 2.0:       sig = 'STRONG_ABOVE_2x'
        elif rel >= 1.4:     sig = 'ABOVE_1.4x'
        elif rel >= 1.0:     sig = 'MODERATE_ABOVE'
        elif rel >= 0.6:     sig = 'NEUTRAL'
        else:                sig = 'BELOW'
        csv_rows.append({
            'cluster_id':      cid,
            'persona_name':    pname,
            'n_entities':      n_ent,
            'pct_of_total':    round(pct, 4),
            'feature':         feat,
            'mean_value':      round(feat_means.get(feat, 0), 6),
            'relative_to_avg': round(rel, 4),
            'signal':          sig,
        })

with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(csv_rows)

# ══════════════════════════════════════════════════════════════════════════════
# Console report
# ══════════════════════════════════════════════════════════════════════════════
def header(title):
    print('\n' + '═' * W)
    print(f'  {title}')
    print('═' * W)

print('\n\n' + '█' * W)
print('  MULTI-AGENT PERSONA DISCOVERY — FINAL REPORT')
print('█' * W)

# ── HOW AGENTS INTERACT ───────────────────────────────────────────────────────
header('HOW THE AGENTS INTERACT')
_max_pct_str   = f'{config.get("max_cluster_size_pct", 0.40):.0%}'
_sub_k_str     = str(config.get('sub_n_clusters', 3))
_max_depth_str = str(config.get('max_depth', 2))
print(f"""
  Four specialised agents + one Orchestrator form a feedback loop.
  Every agent's quality gate can push the pipeline backward; it only
  moves forward when ALL gates pass (or the user approves manually).

  ┌─────────────────────────────────────────────────────────────────┐
  │                        ORCHESTRATOR                             │
  │  (Python coordinator + LLM decision-maker)                      │
  │                                                                 │
  │  ① FeatureSelector ──────────────────────────────────────────► │
  │    Role: LLM reads PCA + AE importance scores and               │
  │    decides which features maximise persona separability.        │
  │    Output: selected_features list (~30–60 features)             │
  │         │                                                       │
  │         ▼   ◄── loops here if ② or ④ request reselection      │
  │  ② Clusterer ────────────────────────────────────────────────► │
  │    Role: sklearn fits clusters; if a cluster > {_max_pct_str}   │
  │    LLM decides: sub-cluster in-place OR reselect features.     │
  │    Output: leaf cluster labels + profiles + lineage tree        │
  │         │                                                       │
  │         ▼   ◄── loops here if ③ Clarity Gate fails or ④ says  │
  │  ③ PersonaNamer ─────────────────────────────────────────────► │
  │    Role: LLM writes name/tagline/description/traits for         │
  │    every cluster in one structured prompt (siblings grouped     │
  │    together so names must contrast).                            │
  │    Clarity Gate: sil ≥ 0.15 · avg_conf ≥ 6.0 · unique names   │
  │    Output: personas dict                                        │
  │         │                                                       │
  │         ▼   ◄── loops here if LLM diagnoses poor clusters      │
  │  ④ Classifier ───────────────────────────────────────────────► │
  │    Role: classifier CV validates crispness; if macro-F1         │
  │    < 0.70 LLM diagnoses root cause → route to ① or ②.         │
  │    Output: CV metrics + per-class F1 + feature importances      │
  │         │                                                       │
  │         ▼                                                       │
  │  ⑤ Human Checkpoint (persona table + CV-F1 + metric proof)    │
  │    Approve → save all outputs.  Otherwise loop back.           │
  └─────────────────────────────────────────────────────────────────┘

  HOW PERSONAS ARE LABELLED
  ─────────────────────────
  ② Clusterer groups the {total_customers} customers into k leaf segments by
    behavioural distance in feature space.

  ③ PersonaNamer builds a structured feature summary (top above-average
    and below-average features, relative ratios vs dataset average)
    for every cluster and sends it to the LLM.  The LLM reads the
    numbers and writes a name, tagline, description, and traits
    that are GROUNDED IN THOSE SPECIFIC METRICS — not invented.
    Sub-clusters are shown alongside their siblings so each name
    must explain HOW this group differs from its neighbours.

  ④ Classifier proves the labelling is correct: it trains a
    classifier to predict cluster membership from the same features.
    Near-perfect CV-F1 means the clusters are genuinely crisp in
    feature space — the personas are real segments.

  40% GUARD  (config: max_cluster_size_pct={_max_pct_str}, sub_n_clusters={_sub_k_str}, max_depth={_max_depth_str})
  ─────────────────────────────────────────────────────────────────
  After initial clustering, any cluster holding > {_max_pct_str} of entities
  is automatically split into {_sub_k_str} sub-clusters.  The LLM is consulted
  if the split decision is ambiguous.  The Cluster Size table below
  proves all final clusters respect the limit.
""")

# ── CLUSTER SIZE CHECK ────────────────────────────────────────────────────────
header('CLUSTER SIZE CHECK  (40% guard)')
print(f'\n  Threshold   : > {MAX_PCT:.0%} triggers automatic split')
print(f'  Total cust  : {total_customers}\n')
print(f'  {"C":<4} {"Persona":<46} {"N":>5}  {"Pct":>6}  {"Status":<12}  Distribution')
print('  ' + '─' * 80)
any_violation = False
for cid in sorted(personas_data.keys(), key=lambda x: int(x)):
    d   = personas_data[cid]
    n   = _n_entities(d)
    s   = d['cluster_stats']
    pct = s.get('pct_total', s.get('pct_of_total', 0) * 100) / 100
    nm  = d['persona'].get('name', '')[:45]
    ok  = pct <= MAX_PCT
    if not ok:
        any_violation = True
    flag = '✓ OK' if ok else '✗ EXCEEDS'
    bar  = '█' * max(1, int(pct * 40))
    print(f'  C{cid:<3} {nm:<46} {n:>5}  {pct:>5.1%}  {flag:<12}  {bar}')
print()
if any_violation:
    print('  ⚠  Violation detected — check max_depth in config.yaml.')
else:
    print('  ✓  All clusters within limit. Deepening loop enforced correctly.')

# ── PERSONA SUMMARY (notebook-04 format + key metrics) ───────────────────────
header('PERSONA SUMMARY  (notebook-04 format + supporting metrics)')
print(full_txt)

# ── KEY FEATURES ──────────────────────────────────────────────────────────────
header('KEY FEATURES THAT SEPARATE THE PERSONAS')
if top_feats:
    max_imp = max(top_feats.values())
    print(f'\n  {"Rank":<5} {"Feature":<40} {"Importance":>11}  Relative weight')
    print('  ' + '─' * 66)
    for rank, (feat, imp) in enumerate(list(top_feats.items())[:15], 1):
        bar = '█' * max(1, int(imp / max_imp * 22))
        print(f'  {rank:<5} {feat:<40} {imp:>9.5f}   {bar}')

if result_clf:
    print(f'\n  Classifier CV summary:')
    print(f'    Accuracy (CV)     : {result_clf.get("cv_accuracy", 0):.4f}')
    print(f'    F1 macro  (CV)    : {result_clf.get("cv_f1_macro", 0):.4f}  (threshold ≥ 0.70 to proceed)')
    print(f'    F1 weighted (CV)  : {result_clf.get("cv_f1_weighted", 0):.4f}')
    print()
    print(f'  {"Persona":<50}  {"CV-F1":>6}  Separation quality')
    print('  ' + '─' * 72)
    for name, score in sorted(result_clf.get('per_class_f1', {}).items(), key=lambda x: -x[1]):
        bar = '█' * int(score * 22)
        print(f'  {name:<50}  {score:>5.3f}   {bar}')

# ── LLM AGENT USAGE ───────────────────────────────────────────────────────────
header('LLM AGENT USAGE  (role: orchestrator + python-script agent)')
print("""
  Every LLM call serves as an autonomous decision-maker embedded
  inside a Python agent.  The Orchestrator coordinates these calls;
  each call is one "reasoning task" within the pipeline.
""")

by_agent = llm_usage.get('by_agent', {})
ORDER    = ['Orchestrator', 'FeatureSelector', 'Clusterer', 'PersonaNamer', 'Classifier']
ROLES    = {
    'Orchestrator':    'Parameter tuning between iterations',
    'FeatureSelector': 'Feature-subset decision',
    'Clusterer':       'Oversized-cluster routing',
    'PersonaNamer':    'Cluster naming + description',
    'Classifier':      'Low-F1 root-cause routing',
}
AGENT_LABELS = {
    'Orchestrator':    '⚙ Orchestrator',
    'FeatureSelector': '① FeatureSelector',
    'Clusterer':       '② Clusterer',
    'PersonaNamer':    '③ PersonaNamer',
    'Classifier':      '④ Classifier',
}

print(f'  {"Agent":<22}  {"Role":<30}  {"Calls":>5}  {"In tok":>8}  {"Out tok":>8}  {"API time":>9}')
print('  ' + '─' * 88)
total_in = total_out = total_api_t = 0
for name in ORDER:
    info   = by_agent.get(name, {'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'time_s': 0.0, 'detail': []})
    calls  = info['calls']
    in_tok = info['input_tokens']
    out_tok= info['output_tokens']
    api_t  = info['time_s']
    total_in  += in_tok
    total_out += out_tok
    total_api_t += api_t
    role   = ROLES.get(name, '')
    label  = AGENT_LABELS.get(name, name)
    print(f'  {label:<22}  {role:<30}  {calls:>5}  {in_tok:>8,}  {out_tok:>8,}  {api_t:>8.1f}s')
    for i, d in enumerate(info.get('detail', []), 1):
        print(f'  {"":22}  {"":30}  call {i}: in={d["input_tokens"]:,} out={d["output_tokens"]:,} time={d["time_s"]}s')

print('  ' + '─' * 88)
print(f'  {"TOTAL":<22}  {"":30}  {llm_usage.get("total_calls",0):>5}  '
      f'{total_in:>8,}  {total_out:>8,}  {total_api_t:>8.1f}s')
print()
print(f'  Input tokens  → prompt context sent to the LLM')
print(f'  Output tokens → LLM generated responses')
# Rough cost estimate (claude-sonnet-4-6: $3/M in, $15/M out as approx)
cost_est = (total_in / 1_000_000) * 3.0 + (total_out / 1_000_000) * 15.0
print(f'  Approx cost   : ~${cost_est:.4f}  '
      f'(Sonnet ~$3/M in · ~$15/M out, illustrative)')

# ── AGENTS & WALL-CLOCK TIME ──────────────────────────────────────────────────
header('AGENTS & WALL-CLOCK TIME PER AGENT')
if timing:
    total_s = timing.get('total_s', 0)
    agents  = timing.get('agents', {})
    print(f'\n  {"Agent":<24}  {"Calls":>5}  {"Total":>8}  {"Avg/call":>9}  {"Share":>7}  Per-call breakdown')
    print('  ' + '─' * 76)
    for name in ORDER:
        info   = agents.get(name, {'calls': 0, 'total_s': 0.0, 'per_call_s': []})
        if not info['calls']:
            continue
        calls  = info['calls']
        tot    = info['total_s']
        avg    = round(tot / calls, 1) if calls else 0.0
        pct    = round(tot / total_s * 100, 1) if total_s > 0 else 0.0
        breakdown = '  '.join(f'{r}s' for r in info['per_call_s']) or '—'
        print(f'  {AGENT_LABELS[name]:<24}  {calls:>5}  {tot:>7.1f}s  {avg:>8.1f}s  {pct:>6.1f}%  [{breakdown}]')
    print('  ' + '─' * 76)

# ── TOTAL TIME ────────────────────────────────────────────────────────────────
header('TOTAL PIPELINE TIME')
if timing:
    total_s = timing.get('total_s', 0)
    m, s_r  = divmod(int(total_s), 60)
    n_iters = max((e.get('iteration', 1) for e in result.get('run_history', [{}])), default=1)
    print(f'\n  {total_s:.1f} seconds  ({m}m {s_r}s)  across {n_iters} pipeline iteration(s)')
    agents = timing.get('agents', {})
    llm_t  = sum(agents.get(a, {}).get('total_s', 0) for a in ['FeatureSelector', 'PersonaNamer'])
    clus_t = agents.get('Clusterer', {}).get('total_s', 0)
    clf_t  = agents.get('Classifier', {}).get('total_s', 0)
    other  = max(0, total_s - llm_t - clus_t - clf_t)
    print()
    for label, val in [
        ('LLM API calls  (①+③)',  llm_t),
        ('sklearn clustering  (②)',   clus_t),
        ('Random Forest CV    (④)',   clf_t),
        ('Orchestration / I/O',        other),
    ]:
        bar = '█' * max(1, int(val / total_s * 30)) if total_s > 0 else ''
        print(f'    {label:<28}  {val:>6.1f}s  {val/total_s*100:>5.1f}%  {bar}')

print('\n\n' + '█' * W)
print(f'  Outputs saved to outputs/')
print(f'    persona_summary.txt   — full persona cards (notebook-04 format)')
print(f'    persona_metrics.csv   — {len(csv_rows)} rows  ({len(personas_data)} clusters × 14 categories)')
print(f'    personas.json         — machine-readable personas')
print(f'    classifier_metrics.json — CV scores + feature importances')
print(f'    pipeline_run_*.txt     — full terminal log from this run')
print(f'    agents_conversation.txt — agent status messages + full LLM prompts/responses')
print('█' * W)

if _args.no_ui:
    print('\n  Fine-tune these personas interactively in the browser:')
    print('      python -m ui.launch')
else:
    print(f'\n  The interactive UI is still running at http://127.0.0.1:{_args.ui_port}/')
    print('  Edit personas, regenerate names, merge clusters, or add global rules.')
print('  (edits + suggestions are logged to outputs/user_feedback_log.jsonl')
print('   and replayed to the Decision Maker on every subsequent run.)\n')

if not _args.no_ui:
    print('  Submit a new intent in the UI to start another run, '
          'or Ctrl-C to exit.')
    _intent_path = pathlib.Path('outputs/pending_intent.json')
    _abort_path = pathlib.Path('outputs/pipeline_abort.json')
    # Ignore the intent file that was just consumed by this run — only react
    # to one that appears AFTER this completion message.
    _baseline_intent_mtime = (
        _intent_path.stat().st_mtime if _intent_path.exists() else 0.0
    )
    try:
        while True:
            time.sleep(1.0)
            # New intent submitted → re-exec for a fully-fresh restart.
            if _intent_path.exists() and _intent_path.stat().st_mtime > _baseline_intent_mtime:
                print('\n[run_pipeline] New intent detected — restarting the pipeline '
                      'in a fresh process.')
                os.execv(sys.executable, [sys.executable] + sys.argv)
            # Abort signal arrived → consume it and also restart.
            if _abort_path.exists():
                try:
                    _abort_path.unlink(missing_ok=True)
                except OSError:
                    pass
                print('\n[run_pipeline] Abort signal received post-completion — '
                      'waiting for a new intent.')
                _wait_for_new_intent()
                os.execv(sys.executable, [sys.executable] + sys.argv)
    except KeyboardInterrupt:
        print('\n[run_pipeline] Shutting down UI. Bye!')
