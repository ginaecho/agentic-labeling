"""
case_memory
===========

Per-case persistent memory for the Decision Maker (Orchestrator).

Each successful run is stored as a "case" capturing the dataset fingerprint,
the user's business intent, the winning strategy (algorithm, k, vif, features),
the outcome metrics, and lessons learned. On a future run we try to match the
current dataset+intent to a stored case and surface the recalled strategy as a
HINT to the parameter-tuning LLM — never a hard override.

Matching has two tiers:

  exact   — same column-name set (Jaccard ≥ 0.9) AND n_rows within ±10%
            → "Same dataset, same problem — reuse the winning recipe."

  similar — column-name semantic overlap OR business-purpose text similarity
            ≥ a looser threshold
            → "Different dataset/goal — recall as inspiration only; warn
               agents that this is NOT the same case."

Storage: outputs/case_memory.json (created on first save).

Lightweight by design — uses only stdlib (difflib) so no extra deps.
"""
from __future__ import annotations

import json
import pathlib
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from difflib import SequenceMatcher
from typing import Iterable

DEFAULT_STORE_PATH = pathlib.Path('outputs/case_memory.json')

# Match thresholds
EXACT_COL_JACCARD = 0.9
EXACT_ROW_TOL = 0.10          # ±10% on n_rows
SIMILAR_COL_JACCARD = 0.4
SIMILAR_PURPOSE_RATIO = 0.6


# ─────────────────────────── data containers ───────────────────────────

@dataclass
class CaseRecall:
    """A matched case the orchestrator should consider, plus the match
    classification ('exact' or 'similar')."""
    match_type: str            # 'exact' | 'similar'
    case: dict
    column_jaccard: float
    purpose_ratio: float
    row_count_delta: float     # |this_rows - case_rows| / case_rows
    notes: str                 # short human-readable reason


# ─────────────────────────── helpers ───────────────────────────

def _normalise(s: str) -> str:
    return ''.join(c.lower() for c in str(s) if c.isalnum())


def _column_set(columns: Iterable[str]) -> set[str]:
    """Normalise column names for fuzzy matching — strip case + punctuation
    so 'CO(GT)' and 'co_gt' match."""
    return {_normalise(c) for c in columns if _normalise(c)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _purpose_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _load_store(path: pathlib.Path) -> dict:
    if not path.exists():
        return {'version': 1, 'cases': []}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {'version': 1, 'cases': []}


def _save_store(store: dict, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding='utf-8')


# ─────────────────────────── public API ───────────────────────────

def find_case(
    *,
    dataset_name: str,
    columns: Iterable[str],
    n_rows: int,
    business_purpose: str,
    target_entity: str = '',
    n_clusters_requested: int | None = None,
    store_path: pathlib.Path | str = DEFAULT_STORE_PATH,
) -> CaseRecall | None:
    """Look up the best matching case for the current run.

    Returns CaseRecall(match_type='exact'|'similar') or None.
    Among ties, prefers exact > similar, then higher column Jaccard.
    """
    store = _load_store(pathlib.Path(store_path))
    cases = store.get('cases', [])
    if not cases:
        return None

    cur_cols = _column_set(columns)
    best: CaseRecall | None = None

    for case in cases:
        cc_cols = _column_set(case.get('dataset', {}).get('columns', []))
        cc_rows = int(case.get('dataset', {}).get('n_rows', 0) or 0)
        cc_purpose = case.get('intent', {}).get('business_purpose', '')

        col_j = _jaccard(cur_cols, cc_cols)
        row_d = (abs(n_rows - cc_rows) / cc_rows) if cc_rows > 0 else float('inf')
        pur_r = _purpose_ratio(business_purpose, cc_purpose)

        # Exact match: same shape + same columns
        if col_j >= EXACT_COL_JACCARD and row_d <= EXACT_ROW_TOL:
            note = (
                f"Same dataset fingerprint: column Jaccard={col_j:.2f}, "
                f"row count Δ={row_d:.1%}. Business purpose similarity={pur_r:.2f}."
            )
            recall = CaseRecall('exact', case, col_j, pur_r, row_d, note)
            if best is None or best.match_type != 'exact' or col_j > best.column_jaccard:
                best = recall
            continue

        # Similar: looser column overlap OR strong purpose match
        if col_j >= SIMILAR_COL_JACCARD or pur_r >= SIMILAR_PURPOSE_RATIO:
            note = (
                f"Different dataset — column Jaccard={col_j:.2f}, "
                f"purpose similarity={pur_r:.2f}. "
                "Treat the recalled strategy as inspiration, not a recipe."
            )
            recall = CaseRecall('similar', case, col_j, pur_r, row_d, note)
            if best is None:
                best = recall
            elif best.match_type == 'similar':
                # break ties by combined score
                cur_score = col_j + pur_r
                old_score = best.column_jaccard + best.purpose_ratio
                if cur_score > old_score:
                    best = recall

    return best


def build_hint_block(recall: CaseRecall) -> str:
    """Render a recall as a hint paragraph the tuning LLM can read.

    For 'similar' matches the wording explicitly cautions the model that this
    is NOT the same case and the strategy is only a starting point.
    """
    case = recall.case
    ds = case.get('dataset', {})
    intent = case.get('intent', {})
    strat = case.get('winning_strategy', {})
    out = case.get('outcome', {})
    lessons = case.get('lessons', []) or []
    lessons_str = '\n'.join(f'    - {x}' for x in lessons[:6]) if lessons else '    (none recorded)'

    header = (
        "── Prior experience: EXACT-MATCH CASE ──"
        if recall.match_type == 'exact' else
        "── Prior experience: SIMILAR (NOT same) CASE — inspiration only ──"
    )

    caveat = (
        "This is a different dataset and/or business goal. The recalled "
        "strategy below succeeded on a related-but-not-identical problem. "
        "Use it as a source of ideas — do NOT assume the same parameters "
        "will work here. Justify any reuse against the current data.\n"
        if recall.match_type == 'similar' else
        "This appears to be the same dataset and goal as a prior successful "
        "run. The recipe below converged before — strongly consider reusing "
        "it (or starting from it) unless current evidence contradicts it.\n"
    )

    body = f"""{header}
{caveat}
  Match details: {recall.notes}
  Prior dataset : {ds.get('name','?')}  ({ds.get('n_rows','?')} rows × {ds.get('n_cols','?')} cols)
  Prior purpose : {intent.get('business_purpose','')[:200]}
  Winning recipe:
    algorithm        = {strat.get('algorithm','?')}
    k                = {strat.get('k','?')}
    vif_threshold    = {strat.get('vif_threshold','?')}
    min_silhouette   = {strat.get('min_silhouette','?')}
    min_cluster_size = {strat.get('min_cluster_size','?')}
    n_features_kept  = {strat.get('n_features_kept','?')}
    converged at iteration {strat.get('iteration','?')} / {strat.get('total_iterations','?')}
  Outcome: silhouette={out.get('silhouette','?')}  cv_f1_macro={out.get('cv_f1_macro','?')}  n_leaf={out.get('n_leaf_clusters','?')}
  Lessons learned:
{lessons_str}
─────────────────────────────────────────────
"""
    return body


def save_case(
    *,
    dataset_name: str,
    dataset_path: str,
    columns: list[str],
    n_rows: int,
    n_cols: int,
    business_purpose: str,
    target_entity: str,
    n_clusters_requested: int | None,
    winning_strategy: dict,
    outcome: dict,
    lessons: list[str],
    store_path: pathlib.Path | str = DEFAULT_STORE_PATH,
) -> str:
    """Append (or update) a case in the store and return its case_id.

    If a case with the same exact-match fingerprint already exists, it is
    REPLACED rather than duplicated — we always keep the most recent winner
    for a given dataset+goal.
    """
    path = pathlib.Path(store_path)
    store = _load_store(path)
    cases = store.get('cases', [])

    new_case = {
        'case_id': str(uuid.uuid4()),
        'saved_at': datetime.now().isoformat(timespec='seconds'),
        'dataset': {
            'name': dataset_name,
            'path': dataset_path,
            'n_rows': int(n_rows),
            'n_cols': int(n_cols),
            'columns': list(columns),
        },
        'intent': {
            'business_purpose': business_purpose or '',
            'target_entity': target_entity or '',
            'n_clusters_requested': n_clusters_requested,
        },
        'winning_strategy': winning_strategy,
        'outcome': outcome,
        'lessons': lessons or [],
    }

    # If an exact-match case for this dataset+purpose already exists, replace it.
    cur_cols = _column_set(columns)
    cur_purpose = (business_purpose or '').lower().strip()
    replaced = False
    for i, c in enumerate(cases):
        cc_cols = _column_set(c.get('dataset', {}).get('columns', []))
        cc_rows = int(c.get('dataset', {}).get('n_rows', 0) or 0)
        cc_purpose = (c.get('intent', {}).get('business_purpose', '') or '').lower().strip()
        same_shape = (
            _jaccard(cur_cols, cc_cols) >= EXACT_COL_JACCARD
            and (abs(n_rows - cc_rows) / max(cc_rows, 1)) <= EXACT_ROW_TOL
        )
        same_purpose = (
            cur_purpose == cc_purpose
            or _purpose_ratio(cur_purpose, cc_purpose) >= 0.85
        )
        if same_shape and same_purpose:
            cases[i] = new_case
            replaced = True
            break

    if not replaced:
        cases.append(new_case)

    store['cases'] = cases
    _save_store(store, path)
    return new_case['case_id']
