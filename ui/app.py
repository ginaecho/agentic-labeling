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
import pathlib
import sys

# Allow running as `python ui/app.py` or `python -m ui.app`
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flask import Flask, jsonify, request, send_from_directory

from ui import feedback_store as fb
# `make_persona_agent` is imported lazily inside the endpoints that need it,
# so the Flask app can boot without numpy/anthropic installed (read/edit/
# feedback-log flows work even in a stripped-down environment).

OUT = _ROOT / 'outputs'
PERSONAS_PATH = OUT / 'personas.json'
PROFILES_PATH = OUT / 'cluster_profiles.json'
LINEAGE_PATH = OUT / 'cluster_lineage.json'
CLF_METRICS_PATH = OUT / 'classifier_metrics.json'

app = Flask(
    __name__,
    static_folder=str(pathlib.Path(__file__).parent / 'static'),
    template_folder=str(pathlib.Path(__file__).parent / 'templates'),
)


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


# ── Main ──────────────────────────────────────────────────────────────────────

def main(host: str = '127.0.0.1', port: int = 5057) -> None:
    if not PERSONAS_PATH.exists():
        print(f'[ui] {PERSONAS_PATH} not found. Run run_pipeline.py first.')
        sys.exit(1)
    print(f'[ui] Serving cluster fine-tuning UI at http://{host}:{port}')
    print(f'[ui] Personas:  {PERSONAS_PATH}')
    print(f'[ui] Feedback log: {fb.LOG_PATH}')
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == '__main__':
    main()
