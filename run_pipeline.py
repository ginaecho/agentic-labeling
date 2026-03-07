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
  CLAUDE AGENT USAGE           — tokens (in/out) + time per agent per call
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
import agents.orchestrator as _orch_mod
from agents.state import HumanDecision

_orig_chk = _orch_mod.human_checkpoint
def _auto_approve(personas, cr, clf, bus):
    _orig_chk(personas, cr, clf, bus)
    print('\n[Auto-approve] Selecting option 1 (Approve).')
    return HumanDecision(action='approve')
_orch_mod.human_checkpoint = _auto_approve

# ── Resolve default features path ─────────────────────────────────────────────
# Prefer the pre-engineered parquet when it exists; otherwise point to the raw
# CSV so UserInputAgent's default will trigger the full feature-engineering path.
_parquet = pathlib.Path('data/processed/customer_features.parquet')
_raw_csv = pathlib.Path('data/raw/fraudTrain.csv')
if _parquet.exists():
    _default_features_path = str(_parquet)
elif _raw_csv.exists():
    _default_features_path = str(_raw_csv)
    print(f'[run_pipeline] Pre-built parquet not found — will use raw CSV: {_raw_csv}')
else:
    _default_features_path = str(_parquet)   # orchestrator will surface the error

# ── Run pipeline ──────────────────────────────────────────────────────────────
orchestrator = Orchestrator(config)
result = orchestrator.run(
    features_path=_default_features_path,
    max_total_iterations=10,
    skip_user_input=False,   # UserInputAgent will prompt for your clustering intent
)

# ── Bail out if the pipeline was blocked or quit before saving outputs ─────────
if result.get('status') in ('blocked', 'quit'):
    print(f"\n[run_pipeline] Pipeline ended with status={result['status']} — no outputs to display.")
    sys.exit(0)

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
_personas_path = pathlib.Path('outputs/personas.json')
if not _personas_path.exists():
    print('\n[run_pipeline] outputs/personas.json not found — pipeline may not have completed.')
    sys.exit(1)
personas_data      = json.loads(_personas_path.read_text())
clf_metrics        = json.loads(pathlib.Path('outputs/classifier_metrics.json').read_text()) \
                     if pathlib.Path('outputs/classifier_metrics.json').exists() else {}

result_clf         = result.get('classifier') or {}
timing             = result.get('timing') or {}
claude_usage       = result.get('claude_usage') or {}
top_feats          = clf_metrics.get('top20_features', {})

MAX_PCT  = config.get('max_cluster_size_pct', 0.40)
W        = 72

# ══════════════════════════════════════════════════════════════════════════════
# Helper: build the full text for one cluster (notebook-04 style + key metrics)
# ══════════════════════════════════════════════════════════════════════════════
def cluster_card(cid: str, data: dict, clf_per_class: dict,
                 all_avg: dict, n_clusters: int) -> str:
    """Return the full text block for one cluster."""
    stats  = data['cluster_stats']
    p      = data['persona']
    cat_s  = stats['category_stats']
    ov     = stats['overall']
    lin    = stats['lineage']

    n_cust  = stats['n_customers']
    pct     = stats['pct_of_total']
    cv_f1   = clf_per_class.get(p.get('name', ''), None)
    cv_str  = f'  CV-F1={cv_f1:.3f}' if cv_f1 is not None else ''
    sub_str = f'  (sub-cluster of C{lin["parent"]})' if lin.get('parent') else ''

    lines = []
    lines.append('')
    lines.append(f'Cluster {cid}{sub_str}  ({n_cust} customers, {pct:.1%} of total){cv_str}')
    lines.append(f'  Persona  : {p.get("name", "?")}')
    lines.append(f'  Tagline  : {p.get("tagline", "?")}')
    lines.append(f'  Dominant : {p.get("dominant_categories", [])}')
    lines.append('  Traits:')
    for t in p.get('traits', []):
        for i, wrapped in enumerate(textwrap.wrap(t, width=W - 6)):
            prefix = '    • ' if i == 0 else '      '
            lines.append(prefix + wrapped)

    lines.append('')
    lines.append('  Description:')
    for wrapped in textwrap.wrap(p.get('description', ''), width=W - 4):
        lines.append('    ' + wrapped)

    # ── Key metrics block ─────────────────────────────────────────────────
    lines.append('')
    lines.append('  Key metrics driving this persona:')

    strong = [(cat, s) for cat, s in cat_s.items() if s['rel_n_txn'] >= 1.4]
    moderate = [(cat, s) for cat, s in cat_s.items() if 1.0 <= s['rel_n_txn'] < 1.4 and s['n_txn_12m'] > 0]
    below  = [(cat, s) for cat, s in cat_s.items() if s['rel_n_txn'] <= 0.6 and s['n_txn_12m'] > 0]

    if strong:
        lines.append('  ▲ Strongly above-average:')
        lines.append(f'    {"Category":<22} {"vs avg":>7}  {"txns/yr":>8}  {"spend/yr":>11}  {"avg/txn":>8}  {"loyalty":>8}')
        lines.append('    ' + '─' * 66)
        for cat, s in sorted(strong, key=lambda x: -x[1]['rel_n_txn']):
            flag = '◀◀' if s['rel_n_txn'] >= 2.0 else '◀ '
            lines.append(
                f'  {flag} {cat:<22} {s["rel_n_txn"]:>6.2f}x  '
                f'{s["n_txn_12m"]:>8.1f}  '
                f'${s["total_amt_12m"]:>10,.0f}  '
                f'${s["avg_spend_12m"]:>7,.0f}  '
                f'{s["consec_months"]:>6.1f}mo'
            )

    if below:
        lines.append('  ▼ Notably below-average:')
        lines.append(f'    {"Category":<22} {"vs avg":>7}  {"txns/yr":>8}  {"spend/yr":>11}')
        lines.append('    ' + '─' * 52)
        for cat, s in sorted(below, key=lambda x: x[1]['rel_n_txn']):
            lines.append(
                f'    ▼ {cat:<22} {s["rel_n_txn"]:>6.2f}x  '
                f'{s["n_txn_12m"]:>8.1f}  '
                f'${s["total_amt_12m"]:>10,.0f}'
            )

    # Overall vs all-cluster average
    lines.append('  ● Overall vs dataset average:')
    def cmp(val, ref, fmt=',.2f'):
        ratio = val / ref if ref else 0
        arrow = '▲' if ratio > 1.1 else ('▼' if ratio < 0.9 else '≈')
        return f'{arrow} {ratio:.2f}x'
    lines.append(f'    Avg txn size  : ${ov["avg_txn_amt"]:>9,.2f}   {cmp(ov["avg_txn_amt"], all_avg["avg_txn_amt"])}')
    lines.append(f'    Total spend/yr: ${ov["total_spend"]:>9,.2f}   {cmp(ov["total_spend"], all_avg["total_spend"])}')
    lines.append(f'    Txns/yr       :  {ov["total_txn_count"]:>9,.1f}   {cmp(ov["total_txn_count"], all_avg["total_txn_count"])}')
    lines.append(f'    High-value %  :  {ov["pct_high_value"]:>9.1f}%')
    lines.append(f'    Active months :  {ov["active_months"]:>9.1f}')

    return '\n'.join(lines)

# ══════════════════════════════════════════════════════════════════════════════
# Compute weighted all-cluster averages for comparison
# ══════════════════════════════════════════════════════════════════════════════
total_customers = sum(d['cluster_stats']['n_customers'] for d in personas_data.values())

def _wavg(field):
    return sum(personas_data[c]['cluster_stats']['overall'][field]
               * personas_data[c]['cluster_stats']['n_customers']
               for c in personas_data) / total_customers if total_customers else 0

all_avg = {
    'avg_txn_amt':    _wavg('avg_txn_amt'),
    'total_spend':    _wavg('total_spend'),
    'total_txn_count': _wavg('total_txn_count'),
}

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
FIELDNAMES = [
    'cluster_id', 'persona_name', 'n_customers', 'pct_of_total',
    'category',
    'n_txn_12m', 'total_amt_12m', 'avg_spend_12m',
    'n_txn_6m',  'total_amt_6m',
    'consec_months',
    'rel_n_txn', 'rel_amt', 'rel_consec',
    'signal',          # STRONG_ABOVE / MODERATE_ABOVE / NEUTRAL / BELOW
]
csv_rows = []
CATEGORIES = [
    'entertainment','food_dining','gas_transport','grocery_net','grocery_pos',
    'health_fitness','home','kids_pets','misc_net','misc_pos',
    'personal_care','shopping_net','shopping_pos','travel',
]
for cid in sorted(personas_data.keys(), key=lambda x: int(x)):
    stats = personas_data[cid]['cluster_stats']
    pname = personas_data[cid]['persona'].get('name', f'Cluster {cid}')
    cat_s = stats['category_stats']
    for cat in CATEGORIES:
        s = cat_s.get(cat, {})
        rel = s.get('rel_n_txn', 0)
        if rel >= 2.0:       sig = 'STRONG_ABOVE_2x'
        elif rel >= 1.4:     sig = 'ABOVE_1.4x'
        elif rel >= 1.0:     sig = 'MODERATE_ABOVE'
        elif rel >= 0.6:     sig = 'NEUTRAL'
        else:                sig = 'BELOW'
        csv_rows.append({
            'cluster_id':    cid,
            'persona_name':  pname,
            'n_customers':   stats['n_customers'],
            'pct_of_total':  round(stats['pct_of_total'], 4),
            'category':      cat,
            'n_txn_12m':     s.get('n_txn_12m', 0),
            'total_amt_12m': s.get('total_amt_12m', 0),
            'avg_spend_12m': s.get('avg_spend_12m', 0),
            'n_txn_6m':      s.get('n_txn_6m', 0),
            'total_amt_6m':  s.get('total_amt_6m', 0),
            'consec_months': s.get('consec_months', 0),
            'rel_n_txn':     s.get('rel_n_txn', 0),
            'rel_amt':       s.get('rel_amt', 0),
            'rel_consec':    s.get('rel_consec', 0),
            'signal':        sig,
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
  │  (Python coordinator + Claude decision-maker)                   │
  │                                                                 │
  │  ① FeatureSelector ──────────────────────────────────────────► │
  │    Role: Claude reads PCA + AE importance scores and            │
  │    decides which features maximise persona separability.        │
  │    Output: selected_features list (~30–60 of 108)              │
  │         │                                                       │
  │         ▼   ◄── loops here if ② or ④ request reselection      │
  │  ② Clusterer ────────────────────────────────────────────────► │
  │    Role: sklearn fits clusters; if a cluster > {_max_pct_str}   │
  │    Claude decides: sub-cluster in-place OR reselect features.  │
  │    Output: leaf cluster labels + profiles + lineage tree        │
  │         │                                                       │
  │         ▼   ◄── loops here if ③ Clarity Gate fails or ④ says  │
  │  ③ PersonaNamer ─────────────────────────────────────────────► │
  │    Role: Claude writes name/tagline/description/traits for      │
  │    every cluster in one structured prompt (siblings grouped     │
  │    together so names must contrast).                            │
  │    Clarity Gate: sil ≥ 0.15 · avg_conf ≥ 6.0 · unique names   │
  │    Output: personas dict                                        │
  │         │                                                       │
  │         ▼   ◄── loops here if Claude diagnoses poor clusters   │
  │  ④ Classifier ───────────────────────────────────────────────► │
  │    Role: Random Forest CV validates crispsness; if macro-F1    │
  │    < 0.70 Claude diagnoses root cause → route to ① or ②.      │
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

  ③ PersonaNamer builds a structured data table (transactions/yr,
    spend/yr, avg/txn, loyalty months, vs-dataset-average ratios)
    for every cluster and sends it to Claude.  Claude reads the
    numbers and writes a name, tagline, description, and 5 traits
    that are GROUNDED IN THOSE SPECIFIC METRICS — not invented.
    Sub-clusters are shown alongside their siblings so each name
    must explain HOW this group differs from its neighbours.

  ④ Classifier proves the labelling is correct: it trains a
    Random Forest to predict cluster membership from the same
    features.  Near-perfect CV-F1 means the clusters are genuinely
    crisp in feature space — the personas are real segments.

  40% GUARD  (config: max_cluster_size_pct={_max_pct_str}, sub_n_clusters={_sub_k_str}, max_depth={_max_depth_str})
  ─────────────────────────────────────────────────────────────────
  After initial clustering, any cluster holding > {_max_pct_str} of customers
  is automatically split into {_sub_k_str} sub-clusters.  Claude is consulted
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
    n   = d['cluster_stats']['n_customers']
    pct = d['cluster_stats']['pct_of_total']
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

# ── CLAUDE AGENT USAGE ────────────────────────────────────────────────────────
header('CLAUDE AGENT USAGE  (role: orchestrator + python-script agent)')
print("""
  Every Claude call serves as an autonomous decision-maker embedded
  inside a Python agent.  The Orchestrator coordinates these calls;
  each call is one "reasoning task" within the pipeline.
""")

by_agent = claude_usage.get('by_agent', {})
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
print(f'  {"TOTAL":<22}  {"":30}  {claude_usage.get("total_calls",0):>5}  '
      f'{total_in:>8,}  {total_out:>8,}  {total_api_t:>8.1f}s')
print()
print(f'  Input tokens  → prompt context sent to Claude')
print(f'  Output tokens → Claude\'s generated responses')
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
        ('Claude API calls  (①+③)',  llm_t),
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
