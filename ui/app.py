"""Flask backend for the interactive cluster fine-tuning UI.

Endpoints
---------
GET   /                              dashboard HTML
GET   /api/state                     personas + profiles + global summary
PUT   /api/personas/<cid>            save edited fields (manual_override)
POST  /api/personas/<cid>/regenerate re-call Decision Maker for this cluster
POST  /api/clusters/merge            merge >=2 clusters, name with Decision Maker
GET   /api/feedback                  list every feedback entry
PATCH /api/feedback/<fb_id>          update priority / active flag
POST  /api/feedback/global           add a global_rule
GET   /api/preferences-preview       show the text that gets prepended to prompts
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import threading
import time

# Allow running as `python ui/app.py` or `python -m ui.app`
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flask import Flask, Response, jsonify, request, send_from_directory

from ui import feedback_store as fb
# `make_persona_agent` is imported lazily inside the endpoints that need it,
# so the Flask app can boot without numpy/anthropic installed (read/edit/
# feedback-log flows work even in a stripped-down environment).

OUT = _ROOT / 'outputs'
PERSONAS_PATH = OUT / 'personas.json'
PROFILES_PATH = OUT / 'cluster_profiles.json'
LINEAGE_PATH = OUT / 'cluster_lineage.json'
CLF_METRICS_PATH = OUT / 'classifier_metrics.json'
EVENTS_PATH = OUT / 'pipeline_events.jsonl'

app = Flask(
    __name__,
    static_folder=str(pathlib.Path(__file__).parent / 'static'),
    template_folder=str(pathlib.Path(__file__).parent / 'templates'),
)
# Allow large dataset uploads (default Flask cap is too small for typical CSVs)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024   # 1 GB

UPLOAD_DIR = _ROOT / 'data' / 'uploads'
_ALLOWED_SUFFIXES = {'.csv', '.tsv', '.parquet'}


def _safe_filename(name: str) -> str:
    """Strip directories and risky chars from an uploaded filename."""
    base = pathlib.Path(name).name
    cleaned = ''.join(c for c in base if c.isalnum() or c in ('.', '-', '_'))
    return cleaned or 'upload.dat'


# ── File helpers ──────────────────────────────────────────────────────────────

def _load_json(path: pathlib.Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def _save_json(path: pathlib.Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def _load_personas() -> dict:
    return _load_json(PERSONAS_PATH, {})


def _load_profiles() -> dict:
    p = _load_json(PROFILES_PATH, None)
    if p is not None:
        return p
    # Fall back: profiles are embedded as cluster_stats inside personas.json
    return {cid: d.get('cluster_stats', {}) for cid, d in _load_personas().items()}


def _sorted_cluster_ids(personas: dict) -> list[str]:
    """Cluster ids in numeric order when possible, else lexical."""
    return sorted(
        personas.keys(),
        key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)),
    )


def _clusters_overview_lines(personas: dict, max_feats: int = 6) -> list[str]:
    """One compact block per cluster — name, size, and the features it is
    stronger / weaker in. This roster lets the LLM reason ACROSS clusters
    (compare, contrast, explain why two clusters that share a high-level trait
    still diverge) instead of being boxed into a single cluster's numbers."""
    lines: list[str] = []
    for ocid in _sorted_cluster_ids(personas):
        c = personas[ocid]
        per = c.get('persona', {})
        st = c.get('cluster_stats', {})
        fm = st.get('feature_means', {}) or {}
        above = list((st.get('top_above_average') or {}).items())[:max_feats]
        below = list((st.get('top_below_average') or {}).items())[:max_feats]
        above_s = ", ".join(
            f"{f} ({round(r, 2)}x, mean={round(fm.get(f, 0), 3)})" for f, r in above
        ) or "—"
        below_s = ", ".join(
            f"{f} ({round(r, 2)}x, mean={round(fm.get(f, 0), 3)})" for f, r in below
        ) or "—"
        try:
            pct = float(st.get('pct_total', 0) or 0)
        except (TypeError, ValueError):
            pct = 0.0
        lines.append(
            f"Cluster {ocid} — \"{per.get('name', '?')}\": {per.get('tagline', '')}\n"
            f"    size={st.get('n_entities', '?')} ({pct:.1f}%), "
            f"confidence={per.get('confidence', '?')}/10\n"
            f"    stronger in: {above_s}\n"
            f"    weaker in: {below_s}"
        )
    return lines


# ── Cluster math (for merge) ──────────────────────────────────────────────────

def _n_entities(profile: dict) -> int:
    return int(profile.get('n_entities', profile.get('n_customers', 0)) or 0)


def _compute_global_means(profiles: dict) -> dict[str, float]:
    """Weighted feature means across all clusters — the denominator for ratios."""
    totals: dict[str, float] = {}
    total_n = 0
    for prof in profiles.values():
        n = _n_entities(prof)
        if n <= 0:
            continue
        total_n += n
        for f, v in (prof.get('feature_means') or {}).items():
            try:
                totals[f] = totals.get(f, 0.0) + float(v) * n
            except (TypeError, ValueError):
                continue
    return {f: (s / total_n) for f, s in totals.items()} if total_n else {}


def _merge_profiles(ids: list[str], profiles: dict, all_profiles: dict) -> dict:
    """Build a merged cluster_stats dict by weighted aggregation of feature means."""
    selected = [profiles[i] for i in ids if i in profiles]
    if not selected:
        raise ValueError('No matching clusters to merge')

    ns = [_n_entities(p) for p in selected]
    n_total = sum(ns)
    if n_total <= 0:
        raise ValueError('Selected clusters have zero entities')

    feature_means: dict[str, float] = {}
    feature_keys = set()
    for p in selected:
        feature_keys.update((p.get('feature_means') or {}).keys())
    for f in feature_keys:
        s = 0.0
        weight = 0
        for p, n in zip(selected, ns):
            v = (p.get('feature_means') or {}).get(f)
            if v is None:
                continue
            try:
                s += float(v) * n
                weight += n
            except (TypeError, ValueError):
                continue
        if weight > 0:
            feature_means[f] = round(s / weight, 6)

    global_means = _compute_global_means(all_profiles)

    ratios: dict[str, float] = {}
    for f, mean_val in feature_means.items():
        g = global_means.get(f)
        if g is None or abs(g) < 1e-12:
            continue
        ratios[f] = round(mean_val / g, 4)

    above = dict(sorted(
        ((f, r) for f, r in ratios.items() if r > 1.0),
        key=lambda x: -x[1],
    )[:10])
    below = dict(sorted(
        ((f, r) for f, r in ratios.items() if 0 < r < 1.0),
        key=lambda x: x[1],
    )[:10])

    total_pct = sum(float(p.get('pct_total', p.get('pct_of_total', 0)) or 0)
                    for p in selected)

    merged = {
        'n_entities': n_total,
        'pct_total': round(total_pct, 2),
        'top_above_average': above,
        'top_below_average': below,
        'feature_means': feature_means,
        'algo_detail': 'merged_by_user',
        'lineage': {'parent': None, 'merged_from': list(ids)},
    }
    return merged


def _next_cluster_id(personas: dict) -> str:
    used = set(personas.keys())
    i = 0
    while str(i) in used:
        i += 1
    return str(i)


# ── Persona naming via Decision Maker ─────────────────────────────────────────

def _name_one(profile: dict, hint: str, prior_name: str | None = None) -> dict:
    """Call PersonaNamingAgent for a single cluster profile, return persona dict."""
    from ui.llm_bridge import make_persona_agent
    agent, _ = make_persona_agent()
    cid = '0'  # the agent only cares about keys within profiles/lineage
    profiles = {cid: profile}
    lineage = {cid: profile.get('lineage', {'parent': None})}

    feedback_text = ''
    if hint:
        feedback_text = f'USER REQUEST for this cluster: {hint}'
    if prior_name:
        feedback_text = (feedback_text + '\n' if feedback_text else '') + (
            f'Previous name was: "{prior_name}". The user wants a different '
            f'name — do not return the same one unless absolutely forced by the data.'
        )

    result = agent.run(
        profiles=profiles,
        lineage=lineage,
        tone='easy',
        feedback=feedback_text,
        iteration=1,
        force_proceed=True,
    )
    if not result.personas or cid not in result.personas:
        raise RuntimeError(
            f'PersonaNamer returned no persona for the cluster. '
            f'Issues: {result.issues}'
        )
    return result.personas[cid]


# ── Routes: static + state ────────────────────────────────────────────────────

@app.get('/')
def index():
    return send_from_directory(app.template_folder, 'index.html')


@app.get('/api/state')
def get_state():
    personas = _load_personas()
    profiles = _load_profiles()
    clf = _load_json(CLF_METRICS_PATH, {})
    lineage = _load_json(LINEAGE_PATH, {})

    total = sum(_n_entities(d.get('cluster_stats', {})) for d in personas.values())
    return jsonify({
        'personas': personas,
        'profiles': profiles,
        'lineage': lineage,
        'classifier_metrics': clf,
        'summary': {
            'n_clusters': len(personas),
            'total_entities': total,
            'cv_f1_macro': clf.get('cv_f1_macro'),
        },
    })


# ── Routes: edits + regenerate ────────────────────────────────────────────────

_EDITABLE_FIELDS = {'name', 'tagline', 'description', 'traits',
                    'dominant_features', 'confidence'}


@app.put('/api/personas/<cid>')
def edit_persona(cid: str):
    personas = _load_personas()
    if cid not in personas:
        return jsonify({'error': f'Cluster {cid} not found'}), 404
    payload = request.get_json(force=True) or {}
    edits = {k: v for k, v in payload.items() if k in _EDITABLE_FIELDS}
    if not edits:
        return jsonify({'error': 'No editable fields in payload'}), 400

    before = {k: personas[cid]['persona'].get(k) for k in edits}
    personas[cid]['persona'].update(edits)
    _save_json(PERSONAS_PATH, personas)

    priority = payload.get('priority', 'high')
    entry = fb.append({
        'type': 'manual_override',
        'target_cluster_id': cid,
        'target_cluster_name': personas[cid]['persona'].get('name'),
        'before': before,
        'after': edits,
        'priority': priority,
    })
    return jsonify({'persona': personas[cid]['persona'], 'feedback': entry})


@app.post('/api/personas/<cid>/regenerate')
def regenerate_persona(cid: str):
    personas = _load_personas()
    profiles = _load_profiles()
    if cid not in personas or cid not in profiles:
        return jsonify({'error': f'Cluster {cid} not found'}), 404

    payload = request.get_json(force=True) or {}
    hint = (payload.get('hint') or '').strip()
    if not hint:
        return jsonify({'error': 'hint is required'}), 400

    prior_name = personas[cid]['persona'].get('name')

    try:
        new_persona = _name_one(profiles[cid], hint, prior_name=prior_name)
    except Exception as exc:
        return jsonify({'error': f'LLM call failed: {exc}'}), 500

    personas[cid]['persona'] = new_persona
    _save_json(PERSONAS_PATH, personas)

    entry = fb.append({
        'type': 'naming_hint',
        'target_cluster_id': cid,
        'target_cluster_name': new_persona.get('name'),
        'hint': hint,
        'before': {'name': prior_name},
        'after': {'name': new_persona.get('name')},
        'priority': payload.get('priority', 'high'),
    })
    return jsonify({'persona': new_persona, 'feedback': entry})


# ── Routes: merge ─────────────────────────────────────────────────────────────

@app.post('/api/clusters/merge')
def merge_clusters():
    payload = request.get_json(force=True) or {}
    ids = [str(c) for c in (payload.get('cluster_ids') or [])]
    hint = (payload.get('hint') or '').strip()
    if len(ids) < 2:
        return jsonify({'error': 'Need at least 2 cluster_ids to merge'}), 400

    personas = _load_personas()
    profiles = _load_profiles()
    missing = [c for c in ids if c not in personas or c not in profiles]
    if missing:
        return jsonify({'error': f'Unknown cluster ids: {missing}'}), 404

    try:
        merged_profile = _merge_profiles(ids, profiles, profiles)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    prior_names = [personas[c]['persona'].get('name', f'C{c}') for c in ids]
    merge_hint = (
        f'This cluster was formed by merging the following clusters: '
        f'{", ".join(repr(n) for n in prior_names)}. '
        f'Name it to capture what unifies them.'
    )
    if hint:
        merge_hint += f' Additional user guidance: {hint}'

    try:
        new_persona = _name_one(merged_profile, merge_hint)
    except Exception as exc:
        return jsonify({'error': f'LLM call failed: {exc}'}), 500

    new_cid = _next_cluster_id(personas)
    personas[new_cid] = {
        'cluster_stats': merged_profile,
        'persona': new_persona,
    }
    profiles[new_cid] = merged_profile

    for c in ids:
        personas.pop(c, None)
        profiles.pop(c, None)

    _save_json(PERSONAS_PATH, personas)
    _save_json(PROFILES_PATH, profiles)

    entry = fb.append({
        'type': 'merge',
        'target_cluster_id': new_cid,
        'target_cluster_name': new_persona.get('name'),
        'merged_ids': ids,
        'merged_names': prior_names,
        'hint': hint,
        'priority': payload.get('priority', 'high'),
    })
    return jsonify({
        'new_cluster_id': new_cid,
        'persona': new_persona,
        'stats': merged_profile,
        'feedback': entry,
    })


# ── Routes: feedback log + global rules ───────────────────────────────────────

@app.get('/api/feedback')
def list_feedback():
    return jsonify({'entries': fb.read_all()})


@app.patch('/api/feedback/<fb_id>')
def update_feedback(fb_id: str):
    payload = request.get_json(force=True) or {}
    changed = False
    if 'priority' in payload:
        changed = fb.update_priority(fb_id, payload['priority']) or changed
    if 'active' in payload:
        changed = fb.set_active(fb_id, bool(payload['active'])) or changed
    if not changed:
        return jsonify({'error': 'No matching entry or no fields changed'}), 404
    return jsonify({'ok': True})


@app.delete('/api/feedback/<fb_id>')
def delete_feedback(fb_id: str):
    if not fb.delete(fb_id):
        return jsonify({'error': 'No matching entry'}), 404
    return jsonify({'ok': True})


@app.post('/api/feedback/global')
def add_global_rule():
    payload = request.get_json(force=True) or {}
    rule = (payload.get('rule') or '').strip()
    if not rule:
        return jsonify({'error': 'rule is required'}), 400
    entry = fb.append({
        'type': 'global_rule',
        'rule': rule,
        'priority': payload.get('priority', 'high'),
    })
    return jsonify({'feedback': entry})


@app.get('/api/preferences-preview')
def preferences_preview():
    return jsonify({'text': fb.build_preferences_block()})


# ── Routes: live pipeline event stream ────────────────────────────────────────

def _read_events() -> list[dict]:
    """Return all events from outputs/pipeline_events.jsonl (empty if absent)."""
    if not EVENTS_PATH.exists():
        return []
    out: list[dict] = []
    for line in EVENTS_PATH.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _derive_status(events: list[dict]) -> dict:
    """Reduce the events log into a coarse pipeline status snapshot."""
    run_id = None
    last_ts = None
    current_agent = None
    iteration = None
    pipeline_running = False
    last_complete = None
    for e in events:
        last_ts = e.get('ts') or last_ts
        ev = e.get('event')
        if ev in ('run_started', 'pipeline_started'):
            run_id = e.get('run_id') or run_id
            pipeline_running = True
            last_complete = None
        elif ev == 'iteration_started':
            iteration = e.get('iteration')
        elif ev == 'agent_report':
            current_agent = e.get('agent') or current_agent
        elif ev == 'pipeline_complete':
            pipeline_running = False
            last_complete = e
    return {
        'pipeline_running': pipeline_running,
        'run_id': run_id,
        'current_agent': current_agent,
        'iteration': iteration,
        'last_event_ts': last_ts,
        'last_complete': last_complete,
        'has_personas': PERSONAS_PATH.exists(),
    }


@app.get('/api/status')
def get_status():
    return jsonify(_derive_status(_read_events()))


@app.get('/api/capabilities')
def get_capabilities():
    """Feature flags so the frontend can detect an outdated UI server."""
    return jsonify({
        'cluster_comparison': True,
        'cluster_chat': True,
        'explain': True,
    })


@app.get('/api/events')
def get_events():
    return jsonify({'events': _read_events()})


@app.delete('/api/upload-preview')
def clear_upload_preview():
    """Remove the saved upload preview so the Evidence tab cleans up."""
    p = OUT / 'last_upload_preview.json'
    try:
        p.unlink(missing_ok=True)
    except OSError:
        pass
    return jsonify({'ok': True})


@app.post('/api/upload')
def upload_dataset():
    """Save an uploaded CSV/parquet to data/uploads/ and return its path."""
    if 'file' not in request.files:
        return jsonify({'error': "No file part in request (expected field 'file')"}), 400
    f = request.files['file']
    if not f or not f.filename:
        return jsonify({'error': 'No file selected'}), 400

    name = _safe_filename(f.filename)
    suffix = pathlib.Path(name).suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        return jsonify({
            'error': f'Unsupported file type: {suffix or "(none)"}. Allowed: '
                     + ', '.join(sorted(_ALLOWED_SUFFIXES)),
        }), 400

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / name
    # If a file with the same name already exists, suffix with a counter
    if dest.exists():
        stem = dest.stem
        i = 1
        while dest.exists():
            dest = UPLOAD_DIR / f'{stem}_{i}{suffix}'
            i += 1

    f.save(str(dest))
    size = dest.stat().st_size
    rel = dest.relative_to(_ROOT)

    preview = _peek_file(dest)

    # Persist so /api/evidence can return it (Evidence tab uses the full preview)
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'last_upload_preview.json').write_text(
            json.dumps({
                'path': str(rel), 'name': dest.name, 'size': size, 'preview': preview,
            }, ensure_ascii=False),
            encoding='utf-8',
        )
    except OSError:
        pass

    return jsonify({
        'path': str(rel),
        'name': dest.name,
        'size': size,
        'size_human': f'{size / (1024*1024):.1f} MB' if size >= 1024*1024
                       else f'{size / 1024:.1f} KB',
        'preview': preview,
    })


def _peek_file(path: pathlib.Path) -> dict | None:
    """Return shape + columns + first 10 rows + per-column stats.

    Per-column stats (dtype, missing%, skewness) provide the *evidence* behind
    any later DatasetExaminer warning — the UI shows them as bars so the user
    can verify the agent's claim instead of just trusting it.

    Best-effort: large CSVs sample up to 50k rows for the stats; the row count
    comes from a streamed line count of the full file.
    """
    try:
        import pandas as pd
    except Exception:
        return None
    suffix = path.suffix.lower()
    try:
        if suffix == '.parquet':
            df = pd.read_parquet(path)
            n_rows = len(df)
            stats_df = df.sample(min(len(df), 50_000), random_state=42) if len(df) > 50_000 else df
            sample = df.head(10)
        elif suffix in ('.csv', '.tsv'):
            sep = '\t' if suffix == '.tsv' else ','
            sample = pd.read_csv(path, sep=sep, nrows=10, low_memory=False)
            stats_df = pd.read_csv(path, sep=sep, nrows=50_000, low_memory=False)
            try:
                with path.open('r', encoding='utf-8', errors='ignore') as fh:
                    n_rows = max(0, sum(1 for _ in fh) - 1)
            except Exception:
                n_rows = None
        else:
            return None
    except Exception as exc:  # noqa: BLE001
        return {'error': f'preview failed: {exc}'}

    cols = [str(c) for c in sample.columns]
    rows = []
    for _, r in sample.iterrows():
        rows.append([
            ('' if pd.isna(v) else str(v))[:80]
            for v in r.tolist()
        ])

    # Per-column statistics for the warning-evidence chart
    try:
        import numpy as np
    except Exception:
        np = None
    col_stats = []
    for c in stats_df.columns:
        s = stats_df[c]
        dtype = str(s.dtype)
        missing_pct = round(float(s.isna().mean() * 100), 2)
        is_numeric = bool(getattr(s, 'dtype', None) is not None and
                          str(s.dtype) not in ('object', 'string')) and \
                     (s.dtype.kind in 'biufc' if hasattr(s.dtype, 'kind') else False)
        skew = None
        histogram = None
        stats = None
        if is_numeric:
            try:
                v = s.dropna()
                if len(v) > 2 and v.std() > 0:
                    skew = round(float(v.skew()), 4)
                    # 30-bin histogram — the visual proof of the skew
                    if np is not None and len(v) >= 10:
                        arr = v.to_numpy(dtype=float, na_value=float('nan'))
                        arr = arr[~np.isnan(arr)]
                        if arr.size:
                            counts, edges = np.histogram(arr, bins=30)
                            histogram = {
                                'counts': counts.tolist(),
                                'edges': [round(float(e), 4) for e in edges.tolist()],
                            }
                            stats = {
                                'min': round(float(arr.min()), 4),
                                'max': round(float(arr.max()), 4),
                                'mean': round(float(arr.mean()), 4),
                                'median': round(float(np.median(arr)), 4),
                                'std': round(float(arr.std()), 4),
                            }
            except Exception:
                pass
        col_stats.append({
            'name': str(c),
            'dtype': dtype,
            'numeric': bool(is_numeric),
            'missing_pct': missing_pct,
            'skew': skew,
            'histogram': histogram,
            'stats': stats,
        })

    return {
        'n_rows': n_rows,
        'n_cols': len(cols),
        'columns': cols,
        'rows': rows,
        'col_stats': col_stats,
        'stats_sample_size': len(stats_df),
    }


MODE_PATH = OUT / 'pipeline_mode.json'
PENDING_DECISION_PATH = OUT / 'pending_decision.json'


@app.get('/api/mode')
def get_mode():
    mode = 'bypass'
    if MODE_PATH.exists():
        try:
            data = json.loads(MODE_PATH.read_text(encoding='utf-8'))
            if data.get('mode') == 'interactive':
                mode = 'interactive'
        except Exception:
            pass
    return jsonify({'mode': mode})


@app.post('/api/mode')
def set_mode():
    payload = request.get_json(force=True) or {}
    mode = payload.get('mode')
    if mode not in ('interactive', 'bypass'):
        return jsonify({'error': "mode must be 'interactive' or 'bypass'"}), 400
    OUT.mkdir(parents=True, exist_ok=True)
    MODE_PATH.write_text(json.dumps({'mode': mode}, indent=2), encoding='utf-8')
    return jsonify({'mode': mode})


@app.post('/api/case-recall-decision')
def submit_case_recall_decision():
    """Save the user's Reuse / Modify / Ignore decision for a case-memory recall.

    The paused orchestrator polls outputs/pending_case_recall.json (see
    Orchestrator._wait_for_case_recall_decision) and applies the decision
    before starting iteration 1.
    """
    payload = request.get_json(force=True) or {}
    decision = str(payload.get('decision') or '').strip().lower()
    if decision not in ('reuse', 'modify', 'ignore'):
        return jsonify({'error': "decision must be 'reuse' | 'modify' | 'ignore'"}), 400
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'pending_case_recall.json').write_text(
        json.dumps({'decision': decision}, ensure_ascii=False),
        encoding='utf-8',
    )
    return jsonify({'ok': True, 'decision': decision})


@app.post('/api/silhouette-target')
def submit_silhouette_target():
    """Save the user's new silhouette_target so the paused orchestrator picks it up."""
    payload = request.get_json(force=True) or {}
    try:
        t = float(payload.get('target'))
    except (TypeError, ValueError):
        return jsonify({'error': 'target must be a number'}), 400
    if not (0.05 <= t <= 1.0):
        return jsonify({'error': 'target must be between 0.05 and 1.0'}), 400
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'pending_target_change.json').write_text(
        json.dumps({'target': t}, ensure_ascii=False),
        encoding='utf-8',
    )
    return jsonify({'ok': True, 'target': t})


@app.post('/api/decision')
def submit_decision():
    """Save the user's mid-pipeline decision so the paused bus can consume it."""
    payload = request.get_json(force=True) or {}
    response = (payload.get('response') or '').strip()
    action = payload.get('action') or 'apply'
    priority = payload.get('priority') or 'high'
    if not response and action != 'ignore':
        return jsonify({'error': 'response is required (or set action=ignore)'}), 400
    OUT.mkdir(parents=True, exist_ok=True)
    PENDING_DECISION_PATH.write_text(
        json.dumps({'response': response, 'action': action, 'priority': priority,
                    'agent': payload.get('agent')}, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    return jsonify({'ok': True})


@app.post('/api/human-checkpoint')
def submit_human_checkpoint():
    """Save the user's Approve / Re-cluster / Reselect / Quit decision so the
    paused orchestrator can resume.

    Body: {"action": "approve" | "recluster" | "reselect_features" | "quit",
           "feedback": "<optional reason string>"}.
    The orchestrator polls outputs/pending_human_checkpoint.json (see
    agents/orchestrator.py:_collect_human_decision) and consumes the file.
    """
    payload = request.get_json(force=True) or {}
    action = str(payload.get('action') or '').strip().lower()
    feedback = str(payload.get('feedback') or '').strip()
    if action not in ('approve', 'recluster', 'reselect_features', 'quit'):
        return jsonify({'error': "action must be 'approve' | 'recluster' | "
                                 "'reselect_features' | 'quit'"}), 400
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'pending_human_checkpoint.json').write_text(
        json.dumps({'action': action, 'feedback': feedback}, ensure_ascii=False),
        encoding='utf-8',
    )
    return jsonify({'ok': True, 'action': action})


@app.post('/api/threshold-decision')
def submit_threshold_decision():
    """Save a structured choice for a paused threshold-decision point.

    Body: {"decision_id": "<id>", "chosen_key": "<one of the option keys>"}.
    The paused agent / orchestrator polls outputs/pending_threshold_decision.json
    (see skills/user_decisions.py:ask_user_decision) and applies the chosen
    option. Separate from /api/decision, which handles the older free-form
    interactive-mode warning flow.
    """
    payload = request.get_json(force=True) or {}
    decision_id = (payload.get('decision_id') or '').strip()
    chosen_key = (payload.get('chosen_key') or '').strip()
    if not decision_id:
        return jsonify({'error': 'decision_id is required'}), 400
    if not chosen_key:
        return jsonify({'error': 'chosen_key is required'}), 400
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'pending_threshold_decision.json').write_text(
        json.dumps({'decision_id': decision_id, 'chosen_key': chosen_key},
                   ensure_ascii=False),
        encoding='utf-8',
    )
    return jsonify({'ok': True, 'decision_id': decision_id, 'chosen_key': chosen_key})


@app.post('/api/intent')
def submit_intent():
    """Save the UI-submitted intent so UserInputAgent can pick it up."""
    payload = request.get_json(force=True) or {}
    target = (payload.get('target_entity') or '').strip()
    purpose = (payload.get('business_purpose') or '').strip()
    if not target:
        return jsonify({'error': 'target_entity is required'}), 400
    if len(purpose) < 5:
        return jsonify({'error': 'business_purpose is required (a sentence or two)'}), 400

    max_iters_raw = payload.get('max_total_iterations')
    try:
        max_iters = int(max_iters_raw) if max_iters_raw not in (None, '', 'null') else None
        if max_iters is not None:
            max_iters = max(1, min(max_iters, 50))
    except (TypeError, ValueError):
        max_iters = None

    cleaned = {
        'target_entity': target,
        'business_purpose': purpose,
        'dataset_path': (payload.get('dataset_path') or '').strip(),
        'constraints': (payload.get('constraints') or '').strip(),
        'n_clusters_requested': payload.get('n_clusters_requested'),
        'must_have_clusters': payload.get('must_have_clusters') or [],
        'max_total_iterations': max_iters,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'pending_intent.json').write_text(
        json.dumps(cleaned, indent=2, ensure_ascii=False), encoding='utf-8',
    )
    return jsonify({'ok': True, 'intent': cleaned})


@app.post('/api/control-gates')
def submit_control_gates():
    """Save the user's control-gate choices so the orchestrator can consume them.

    Body: {"max_cluster_size_pct": 0.35, "sub_n_clusters": 3, "max_depth": 2}
    """
    payload = request.get_json(force=True) or {}
    try:
        max_pct = float(payload.get('max_cluster_size_pct', 0.40))
        sub_k = int(payload.get('sub_n_clusters', 3))
        depth = int(payload.get('max_depth', 2))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid numeric values'}), 400

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'pending_control_gates.json').write_text(
        json.dumps({
            'max_cluster_size_pct': max_pct,
            'sub_n_clusters': sub_k,
            'max_depth': depth,
        }, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    return jsonify({
        'ok': True,
        'max_cluster_size_pct': max_pct,
        'sub_n_clusters': sub_k,
        'max_depth': depth,
    })


@app.post('/api/column-resolution')
def submit_column_resolution():
    """Save the user's column-resolution choices so the orchestrator can consume them.

    Body: {"entity_id": "customer_id", "timestamp": "Date", "amount": "amount", "category": "category"}
    """
    payload = request.get_json(force=True) or {}
    cleaned = {
        'entity_id': (payload.get('entity_id') or '').strip() or None,
        'timestamp': (payload.get('timestamp') or '').strip() or None,
        'amount': (payload.get('amount') or '').strip() or None,
        'category': (payload.get('category') or '').strip() or None,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'pending_column_resolution.json').write_text(
        json.dumps(cleaned, indent=2, ensure_ascii=False), encoding='utf-8',
    )
    return jsonify({'ok': True, 'columns': cleaned})


@app.post('/api/abort')
def abort_pipeline():
    """Write an abort flag the Orchestrator picks up at the next iteration
    boundary. The current iteration finishes (no half-baked state); the run
    returns with status='aborted'. run_pipeline.py then waits for a new
    pending_intent.json before starting again — so the user can immediately
    submit fresh intent in the UI for a clean restart.

    Body: {"reason": "...", "restart": true|false}
    """
    payload = request.get_json(silent=True) or {}
    reason = (payload.get('reason') or 'user_abort').strip() or 'user_abort'
    restart = payload.get('restart')
    restart = True if restart is None else bool(restart)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'pipeline_abort.json').write_text(
        json.dumps({
            'reason': reason,
            'restart': restart,
            'requested_at': time.time(),
        }, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    return jsonify({'ok': True, 'reason': reason, 'restart': restart})


@app.get('/api/evidence')
def get_evidence():
    """Aggregated evidence for the Evidence tab: dataset preview, silhouette
    curve, cluster sizes, per-class F1, etc. Best-effort; missing files just
    omit their section."""
    out: dict = {}

    # Silhouette / k-curve
    sil = _load_json(OUT / 'silhouette_curve.json', None)
    if sil:
        out['silhouette_curve'] = sil

    # Cluster sizes from profiles
    profiles = _load_json(PROFILES_PATH, None)
    if profiles:
        sizes = []
        for cid, p in profiles.items():
            n = int(p.get('n_entities', p.get('n_customers', 0)) or 0)
            pct = float(p.get('pct_total', (p.get('pct_of_total', 0) or 0) * 100) or 0)
            sizes.append({'cluster_id': cid, 'n': n, 'pct': pct,
                          'algo_detail': p.get('algo_detail') or ''})
        out['cluster_sizes'] = sizes

    # Classifier metrics
    clf = _load_json(CLF_METRICS_PATH, None)
    if clf:
        out['classifier'] = {
            'cv_f1_macro': clf.get('cv_f1_macro'),
            'cv_accuracy': clf.get('cv_accuracy'),
            'cv_f1_weighted': clf.get('cv_f1_weighted'),
            'per_class_f1': clf.get('per_class_f1', {}),
            'top20_features': clf.get('top20_features', {}),
        }

    # Lineage tree (parent/child relationships from deepening)
    lineage = _load_json(LINEAGE_PATH, None)
    if lineage:
        out['lineage'] = lineage

    # Most recent uploaded dataset preview
    upload = _load_json(OUT / 'last_upload_preview.json', None)
    if upload:
        out['upload_preview'] = upload

    # Dataset preview AFTER FeatureEngineer / FeatureSelector (pre-modelling).
    # Written by the orchestrator each time those agents run.
    pre_modelling = _load_json(OUT / 'pre_modelling_preview.json', None)
    if pre_modelling:
        out['pre_modelling_preview'] = pre_modelling

    # Per-iteration PCA projections (data points + cluster labels)
    pca = _load_json(OUT / 'pca_iterations.json', None)
    if pca:
        out['pca_iterations'] = pca

    return jsonify(out)


@app.post('/api/cluster-chat')
def cluster_chat():
    """Multi-turn chat with the LLM about a single named cluster.

    Stateless: the client maintains conversation history and sends it on every
    call. Each LLM call lands in the 'naming' cost ledger, separate from the
    pipeline + evidence ledgers.
    """
    payload = request.get_json(force=True) or {}
    cid = str(payload.get('cluster_id', '')).strip()
    message = (payload.get('message') or '').strip()
    history = payload.get('history') or []
    mode = payload.get('mode') or 'discuss'   # 'discuss' or 'conclude'
    if not cid or not message:
        return jsonify({'error': 'cluster_id and message are required'}), 400

    personas = _load_personas()
    if cid not in personas:
        return jsonify({'error': f'Cluster {cid} not found'}), 404
    cluster = personas[cid]
    persona = cluster.get('persona', {})
    stats = cluster.get('cluster_stats', {})

    # System prompt grounds the LLM in the cluster's actual numbers
    top_above = stats.get('top_above_average', {})
    top_below = stats.get('top_below_average', {})
    feat_means = stats.get('feature_means', {})
    above_lines = [f"  {f}: {round(r, 2)}x avg (mean={round(feat_means.get(f, 0), 4)})"
                   for f, r in list(top_above.items())[:15]]
    below_lines = [f"  {f}: {round(r, 2)}x avg (mean={round(feat_means.get(f, 0), 4)})"
                   for f, r in list(top_below.items())[:10]]

    # Build the cross-cluster roster FIRST (before the focus block) so the LLM
    # cannot miss it. Prior versions buried it under FOCUS and the model would
    # claim "I only have data for cluster X" even though every cluster was in
    # the prompt. Putting it up top + restating in the rules section fixes that.
    other_lines = _clusters_overview_lines(personas)
    cluster_ids = _sorted_cluster_ids(personas)
    n_clusters = len(personas)
    ids_csv = ", ".join(cluster_ids)

    sys_lines = [
        f"You are a data analyst for a customer-segmentation pipeline. The run",
        f"produced {n_clusters} clusters: {ids_csv}. You have the full profile",
        f"(name, size, top features above/below average) for EVERY cluster — see",
        f"the roster below. The user's current focus is cluster {cid}, but you",
        f"can and must reason across clusters when asked.",
        f"",
        f"RULES",
        f"  1. NEVER say 'I don't have data for cluster N' or 'cluster N's profile",
        f"     hasn't been shared' for any cluster id in the roster below. All",
        f"     {n_clusters} cluster profiles are in this prompt.",
        f"  2. When asked WHY a label was chosen, cite specific feature names +",
        f"     values from the cluster's STRONGER/WEAKER lists.",
        f"  3. When asked to compare/contrast two clusters, look up BOTH in the",
        f"     roster and quote the specific features and ratios that diverge.",
        f"  4. If a trait was inferred indirectly (proxy feature), say so — don't",
        f"     fabricate evidence.",
        f"",
        f"─" * 48,
        f"ALL {n_clusters} CLUSTERS IN THIS RUN — full reference data",
        f"(use these numbers verbatim; do not guess any value)",
        f"",
        *other_lines,
        f"",
        f"─" * 48,
        f"FOCUS — Cluster {cid} — \"{persona.get('name', '?')}\"",
        f"  Tagline: {persona.get('tagline', '')}",
        f"  Description: {persona.get('description', '')}",
        f"  Traits: {persona.get('traits', [])}",
        f"  Confidence: {persona.get('confidence', '?')}/10",
        f"  Size: {stats.get('n_entities', '?')} entities ({stats.get('pct_total', 0):.1f}% of total)",
        f"",
        f"Cluster {cid} features STRONGER than overall average (extended):",
        *above_lines,
        f"",
        f"Cluster {cid} features WEAKER than overall average (extended):",
        *below_lines,
    ]

    if mode == 'conclude':
        sys_lines += [
            "",
            "The user is wrapping up. Based on the conversation so far, summarise",
            "the conclusion in 2 sentences AND propose ONE of these actions:",
            "  rename: a new name (give the new name)",
            "  merge: suggest which cluster id to merge with",
            "  keep: keep the current name",
            "  recluster: re-run clustering with specific guidance",
            "Return ONLY a JSON object: {\"summary\": \"...\", \"action\": \"rename|merge|keep|recluster\", \"new_name\": \"...\", \"merge_with\": \"<cid>\", \"reason\": \"...\"}",
        ]

    system_prompt = "\n".join(sys_lines)

    # Build message array (history + new user message)
    messages = []
    for m in history:
        role = m.get('role')
        content = (m.get('content') or '').strip()
        if role in ('user', 'assistant') and content:
            messages.append({'role': role, 'content': content})
    messages.append({'role': 'user', 'content': message})

    from ui.llm_bridge import _load_env_file
    _load_env_file()
    if not os.environ.get('ANTHROPIC_API_KEY'):
        return jsonify({'error': 'ANTHROPIC_API_KEY not set'}), 500

    import anthropic, time as _time
    client = anthropic.Anthropic()

    # Emit started event into the live event log (separate naming ledger)
    events_path = OUT / 'pipeline_events.jsonl'
    def _emit(event, **payload):
        if not events_path.exists():
            return
        from datetime import datetime, timezone
        rec = {'event': event,
               'ts': datetime.now(timezone.utc).isoformat(timespec='seconds'),
               **payload}
        try:
            with events_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
        except OSError:
            pass

    agent_label = f'ClusterChat_C{cid}'
    purpose = f'discuss cluster {cid} ({persona.get("name", "?")[:30]})'
    _emit('llm_call_started', agent=agent_label, purpose=purpose,
          prompt_chars=len(system_prompt) + len(message),
          prompt=f'[SYSTEM]\n{system_prompt}\n\n[USER]\n{message}',
          category='naming')

    t0 = _time.perf_counter()
    try:
        resp = client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=800,
            system=system_prompt,
            messages=messages,
        )
    except Exception as exc:
        return jsonify({'error': f'LLM call failed: {exc}'}), 500
    elapsed = round(_time.perf_counter() - t0, 2)
    reply = resp.content[0].text

    _emit('llm_call_finished', agent=agent_label, purpose=purpose,
          input_tokens=resp.usage.input_tokens,
          output_tokens=resp.usage.output_tokens,
          time_s=elapsed, response=reply, category='naming')

    out = {'reply': reply,
           'tokens': {'in': resp.usage.input_tokens, 'out': resp.usage.output_tokens,
                      'time_s': elapsed}}

    # On 'conclude' we try to parse the JSON proposal
    if mode == 'conclude':
        try:
            text = reply.strip()
            if '```' in text:
                for part in text.split('```'):
                    p = part.strip()
                    if p.startswith('json'):
                        p = p[4:].strip()
                    if p.startswith('{'):
                        text = p
                        break
            out['proposal'] = json.loads(text)
        except Exception:
            out['proposal'] = None
    return jsonify(out)


# ── /api/cluster-comparison cache ─────────────────────────────────────────────
# Cross-cluster contrasting analysis for the Data & Evidence tab. Cached by the
# set of clusters + focus so repeated opens / multiple browser windows don't
# re-bill the evidence ledger for an identical request.
_COMPARE_CACHE: dict = {}
_COMPARE_CACHE_TTL = 600.0   # seconds
_COMPARE_CACHE_LOCK = threading.Lock()


@app.post('/api/cluster-comparison')
def cluster_comparison():
    """LLM-driven CROSS-CLUSTER contrasting analysis across every named cluster.

    Lazy / on-demand (the Data & Evidence tab fires it on a button click) so it
    never auto-bills. Cost lands on the 'evidence' ledger, separate from the
    pipeline + naming ledgers.
    """
    payload = request.get_json(silent=True) or {}
    focus = (payload.get('focus') or '').strip()

    personas = _load_personas()
    if len(personas) < 2:
        return jsonify({'error': 'Need at least 2 named clusters to compare'}), 400

    # Cache key: cluster ids + names + the optional focus. A rename / merge
    # changes the key so the analysis refreshes; identical re-opens reuse it.
    sig = "|".join(
        f"{cid}:{personas[cid].get('persona', {}).get('name', '')}"
        for cid in _sorted_cluster_ids(personas)
    ) + f"||focus={focus}"
    now = time.time()
    with _COMPARE_CACHE_LOCK:
        hit = _COMPARE_CACHE.get(sig)
        if hit and (now - hit['ts']) < _COMPARE_CACHE_TTL:
            cached = dict(hit['response'])
            cached['_cached'] = True
            return jsonify(cached)

    roster = "\n".join(_clusters_overview_lines(personas, max_feats=8))
    focus_line = (
        f"\nThe user wants the comparison focused on: {focus}\n" if focus else ""
    )
    prompt = f"""You are a data analyst comparing the customer segments produced by a
clustering pipeline. Below is EVERY named cluster, its size, and the features it
is stronger / weaker in versus the overall average.

{roster}
{focus_line}
Write a CROSS-CLUSTER COMPARISON / contrasting analysis (not one-cluster-at-a-time
descriptions). Structure it as short markdown sections / bullets:

1. **Separating axes** — the 1-2 dimensions that best pull the clusters apart.
2. **Contrasts** — pairs of clusters that look alike on one trait but diverge on
   another (e.g. "C0 and C3 are both high-X, but C0 leans <feature> while C3
   leans <feature>"). Always cite the specific feature names + ratios.
3. **Overlap / merge candidates** — any clusters that look redundant.
4. **Takeaway** — one or two sentences on how the segmentation hangs together.

Cite cluster ids and feature names from the data above. Do NOT invent features
that are not listed."""

    from ui.llm_bridge import _load_env_file
    _load_env_file()
    if not os.environ.get('ANTHROPIC_API_KEY'):
        return jsonify({'error': 'ANTHROPIC_API_KEY not set'}), 500

    try:
        from ui.llm_bridge import make_persona_agent
        _, bus = make_persona_agent()
        raw = bus.ask(agent='ClusterComparison',
                      purpose='cross-cluster contrasting analysis',
                      prompt=prompt, max_tokens=1300, category='evidence')
    except Exception as exc:  # noqa: BLE001
        return jsonify({'error': f'LLM call failed: {exc}'}), 500

    result = {'comparison': raw.strip(), 'n_clusters': len(personas)}
    with _COMPARE_CACHE_LOCK:
        _COMPARE_CACHE[sig] = {'response': result, 'ts': now}
    return jsonify(result)


# ── /api/explain dedup cache ──────────────────────────────────────────────────
# When 2+ browser windows are open (recording mode runs 3+ Chromium contexts),
# each independently subscribes to SSE and fires its own POST /api/explain for
# every warning. The LLM call is identical across windows, so we cache by
# (agent, issue) for a short TTL and return the cached response — saving 2–3×
# token spend on the evidence ledger without changing UX.
_EXPLAIN_CACHE: dict = {}        # key -> {"response": dict, "ts": float}
_EXPLAIN_CACHE_TTL = 120.0       # seconds — short enough that re-fires after
                                 # genuine state change get a fresh answer
_EXPLAIN_CACHE_LOCK = threading.Lock()


@app.post('/api/explain')
def explain_warning():
    """LLM call (category='evidence') that explains a warning and recommends
    which existing visual to look at. Cost is tracked separately so it doesn't
    inflate the main pipeline's spend."""
    payload = request.get_json(force=True) or {}
    agent = (payload.get('agent') or 'agent').strip()
    issue = (payload.get('issue') or '').strip()
    iteration = payload.get('iteration')
    evidence = payload.get('evidence') or {}   # numbers backing the warning
    if not issue:
        return jsonify({'error': 'issue is required'}), 400

    # Cache hit? Return immediately without billing another LLM call.
    cache_key = f"{agent}::{issue}::{iteration or ''}"
    now = time.time()
    with _EXPLAIN_CACHE_LOCK:
        hit = _EXPLAIN_CACHE.get(cache_key)
        if hit and (now - hit['ts']) < _EXPLAIN_CACHE_TTL:
            cached = dict(hit['response'])
            cached['_cached'] = True   # so the UI/log can tell it was deduped
            return jsonify(cached)

    # In bypass mode the user sees this as the AGENT'S DECISION (not advice).
    # The LLM must respond as if the action has already been taken / queued —
    # active voice, present/past tense, no "we recommend" or "you should".
    _iter_note = f" (iteration {iteration})" if iteration is not None else ""
    ev_text = json.dumps(evidence, indent=2, ensure_ascii=False)[:1500]
    prompt = f"""You speak FOR the multi-agent pipeline. A warning fired and the
pipeline is in BYPASS mode — that means the decision has already been made and
the pipeline is continuing. Your job is to tell the user, in active voice,
what the agents JUST DECIDED to do about this warning.

Use phrasing like: "The pipeline will / chose to / is applying / is keeping /
is ignoring". DO NOT use "we recommend", "you should", "consider", "you may want
to". Be concrete and short.

CRITICAL: Always reference the exact iteration number ({iteration or 'unknown'}) and
the exact cluster numbers / values from the warning text. Do NOT invent different
numbers. If the warning says "Cluster 4 has 50.9%", your explanation must say
"Cluster 4 has 50.9%" — never "Cluster 0 has 92.4%".

Warning from: {agent}{_iter_note}
Warning text: "{issue}"

Supporting evidence (numbers the agent saw):
{ev_text}

Return ONLY a valid JSON object — no markdown fences, no extra text:
{{
  "decision": "<the agent's actual decision in 1-2 sentences, active voice>",
  "reasoning": "<one sentence on why this is the right call given the evidence>",
  "visual_to_check": "<one sentence pointing to a specific chart already in the Data & evidence tab where the user can verify the warning was real>"
}}"""

    # Spin up a lightweight bus + handler — same shape llm_bridge.py uses
    try:
        from ui.llm_bridge import make_persona_agent  # reuses the env loader
        # We don't actually use the agent; just borrow its bus+handler scaffolding
        _, bus = make_persona_agent()
        raw = bus.ask(agent='EvidenceExplainer', purpose=f'explain warning from {agent}',
                      prompt=prompt, max_tokens=400, category='evidence')
    except Exception as exc:  # noqa: BLE001
        return jsonify({'error': f'LLM call failed: {exc}'}), 500

    # Best-effort JSON extraction
    text = raw.strip()
    if '```' in text:
        for part in text.split('```'):
            p = part.strip()
            if p.startswith('json'):
                p = p[4:].strip()
            if p.startswith('{'):
                text = p
                break
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {'explanation': raw.strip(), 'visual_to_check': ''}

    # Populate the dedup cache so simultaneous repeat calls (other browser
    # windows / quick re-fires) reuse this response instead of re-billing.
    with _EXPLAIN_CACHE_LOCK:
        _EXPLAIN_CACHE[cache_key] = {'response': parsed, 'ts': now}
    return jsonify(parsed)


@app.get('/api/outputs-files')
def list_outputs_files():
    """List every file in outputs/ with size + modified time for the Evidence tab."""
    files = []
    if OUT.exists():
        for p in sorted(OUT.iterdir(), key=lambda x: x.name.lower()):
            if p.is_file():
                try:
                    st = p.stat()
                    files.append({
                        'name': p.name,
                        'size': st.st_size,
                        'size_human': (f'{st.st_size/(1024*1024):.1f} MB' if st.st_size >= 1024*1024
                                       else f'{st.st_size/1024:.1f} KB' if st.st_size >= 1024
                                       else f'{st.st_size} B'),
                        'mtime': int(st.st_mtime),
                    })
                except OSError:
                    pass
    return jsonify({'files': files})


@app.get('/api/outputs-file/<path:name>')
def get_output_file(name: str):
    """Serve a single file from outputs/ (read-only, sandboxed to outputs)."""
    # Reject path traversal
    if '..' in name or name.startswith('/'):
        return jsonify({'error': 'invalid path'}), 400
    target = OUT / name
    try:
        target = target.resolve()
        if not str(target).startswith(str(OUT.resolve())):
            return jsonify({'error': 'invalid path'}), 400
    except OSError:
        return jsonify({'error': 'invalid path'}), 400
    if not target.exists() or not target.is_file():
        return jsonify({'error': 'not found'}), 404
    return send_from_directory(str(OUT), name)


@app.get('/api/events/stream')
def stream_events():
    """Server-Sent Events: replays the existing log, then tails new lines."""
    def _gen():
        # 1. Replay everything that's already there so a late subscriber
        #    gets the full picture without an extra /api/events round-trip.
        try:
            if EVENTS_PATH.exists():
                with EVENTS_PATH.open('r', encoding='utf-8') as f:
                    for line in f:
                        line = line.rstrip('\n')
                        if line:
                            yield f'data: {line}\n\n'
                    pos = f.tell()
            else:
                pos = 0
        except OSError:
            pos = 0

        # 2. Tail loop. The file may be deleted/truncated when a fresh
        #    pipeline starts — we detect that by checking the size.
        last_heartbeat = time.time()
        while True:
            try:
                if not EVENTS_PATH.exists():
                    pos = 0
                else:
                    size = EVENTS_PATH.stat().st_size
                    if size < pos:
                        # File was truncated (new run started). Reset.
                        pos = 0
                    if size > pos:
                        with EVENTS_PATH.open('r', encoding='utf-8') as f:
                            f.seek(pos)
                            for line in f:
                                line = line.rstrip('\n')
                                if line:
                                    yield f'data: {line}\n\n'
                            pos = f.tell()
                # Heartbeat every ~15s to keep the connection alive.
                if time.time() - last_heartbeat > 15:
                    yield ': keep-alive\n\n'
                    last_heartbeat = time.time()
            except (OSError, GeneratorExit):
                break
            time.sleep(0.4)

    return Response(_gen(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no'})


# ── Main ──────────────────────────────────────────────────────────────────────

def main(host: str = '127.0.0.1', port: int = 5057) -> None:
    if not PERSONAS_PATH.exists():
        print(f'[ui] {PERSONAS_PATH} not found yet — serving live pipeline view only.')
    print(f'[ui] Serving cluster fine-tuning UI at http://{host}:{port}')
    print(f'[ui] Personas:  {PERSONAS_PATH}')
    print(f'[ui] Feedback log: {fb.LOG_PATH}')
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


if __name__ == '__main__':
    main()
