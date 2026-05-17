"""Compute stability metrics between two pipeline runs.

ARI on cluster assignments is the load-bearing signal: it asks
"are the same rows grouped together as last time?", which is what
'stable' means operationally — independent of how the personas
were renamed.
"""
from __future__ import annotations

import json
import pathlib
from typing import Optional

import pandas as pd
from sklearn.metrics import adjusted_rand_score


def _load_labels(run_dir: pathlib.Path) -> Optional[pd.Series]:
    p = run_dir / 'outputs' / 'cluster_labels.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df.set_index('row_index')['cluster_id']


def _load_personas(run_dir: pathlib.Path) -> dict:
    p = run_dir / 'outputs' / 'personas.json'
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _load_classifier(run_dir: pathlib.Path) -> dict:
    p = run_dir / 'outputs' / 'classifier_metrics.json'
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _persona_names(personas: dict) -> set[str]:
    return {(d.get('persona') or {}).get('name', '') for d in personas.values()
            if (d.get('persona') or {}).get('name')}


def _size_distribution(personas: dict) -> dict[str, float]:
    out = {}
    for cid, d in personas.items():
        s = d.get('cluster_stats', {})
        pct = s.get('pct_total', s.get('pct_of_total', 0) * 100)
        out[str(cid)] = float(pct) / 100.0
    return out


def diff(prev_dir: pathlib.Path, curr_dir: pathlib.Path) -> dict:
    """Compare two pipeline runs. Returns a flat dict of stability metrics.

    All metrics are 0 when prev_dir is None or missing artifacts — the
    first run of an experiment has nothing to compare against.
    """
    prev_personas = _load_personas(prev_dir)
    curr_personas = _load_personas(curr_dir)
    prev_labels = _load_labels(prev_dir)
    curr_labels = _load_labels(curr_dir)
    prev_clf = _load_classifier(prev_dir)
    curr_clf = _load_classifier(curr_dir)

    out: dict = {
        'ari': None,
        'name_jaccard': None,
        'size_l1': None,
        'n_personas_prev': len(prev_personas) or None,
        'n_personas_curr': len(curr_personas) or None,
        'f1_delta': None,
    }

    if prev_labels is not None and curr_labels is not None and \
            len(prev_labels) == len(curr_labels):
        out['ari'] = float(adjusted_rand_score(prev_labels.values,
                                                curr_labels.values))

    if prev_personas and curr_personas:
        a, b = _persona_names(prev_personas), _persona_names(curr_personas)
        union = a | b
        out['name_jaccard'] = (len(a & b) / len(union)) if union else None

        prev_sizes = _size_distribution(prev_personas)
        curr_sizes = _size_distribution(curr_personas)
        keys = set(prev_sizes) | set(curr_sizes)
        out['size_l1'] = sum(abs(prev_sizes.get(k, 0.0) - curr_sizes.get(k, 0.0))
                              for k in keys)

    if prev_clf and curr_clf:
        out['f1_delta'] = float(curr_clf.get('cv_f1_macro', 0.0) -
                                  prev_clf.get('cv_f1_macro', 0.0))

    return out
