"""
ClusteringAgent

Contract: docs/agents/clusterer.md. Skills: docs/skills/orchestrator_bus.md,
docs/skills/algo_recommender.md, docs/skills/silhouette_optimizer.md.

Fits the configured clustering algorithm on a selected feature subset,
runs the deepening loop (same logic as notebook 03), and asks Claude
whether to sub-cluster or request new features when a cluster is too large.

Enhancements over original:
  - Uses skills.algo_recommender to auto-select algorithm from data shape
  - Uses skills.silhouette_optimizer to auto-select k data-driven
  - Reports structured status to OrchestratorBus
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from agents.state import ClusteringResult
from agents.user_input import UserIntent
from agents.dataset_examiner import DatasetProfile
from skills.silhouette_optimizer import optimize_k
from skills.algo_recommender import recommend_algorithm
from skills.orchestrator_bus import OrchestratorBus, OrchestratorMessage

# ── Constants (mirrored from notebook 03) ─────────────────────────────────────
CATEGORIES = [
    'entertainment', 'food_dining', 'gas_transport', 'grocery_net', 'grocery_pos',
    'health_fitness', 'home', 'kids_pets', 'misc_net', 'misc_pos',
    'personal_care', 'shopping_net', 'shopping_pos', 'travel',
]
WINDOWS = [6, 12]

DEFAULT_K_SEARCH_RANGE = [3, 4, 5, 6, 7, 8, 10, 12, 15]


def _build_log_cols(columns) -> list[str]:
    candidates = (
        [f'n_txn_{cat}_{w}m'     for cat in CATEGORIES for w in WINDOWS]
        + [f'amt_{cat}_{w}m'     for cat in CATEGORIES for w in WINDOWS]
        + [f'avg_spend_{cat}_{w}m' for cat in CATEGORIES for w in WINDOWS]
        + ['total_txn_count', 'total_spend', 'avg_txn_amt', 'std_txn_amt',
           'max_txn_amt', 'n_unique_merchants', 'avg_days_between_txn']
    )
    return [c for c in candidates if c in columns]


def _fit_model(algorithm: str, n_clusters: int, X_scaled: np.ndarray):
    if algorithm == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=15, max_iter=500)
        labels = model.fit_predict(X_scaled)
        algo_name = 'KMeans'
        algo_detail = f'K-Means  |  k={n_clusters}  |  init=k-means++  |  n_init=15'
    elif algorithm == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = model.fit_predict(X_scaled)
        algo_name = 'AgglomerativeClustering'
        algo_detail = f'Hierarchical (Ward linkage)  |  k={n_clusters}'
    else:
        raise ValueError(
            f'Unknown clustering_algorithm: {algorithm!r}. '
            'Valid options: "kmeans", "hierarchical".'
        )
    return labels, algo_name, algo_detail


def _extract_profiles(features_df: pd.DataFrame, cluster_labels: pd.Series,
                      cluster_lineage: dict, X_scaled: np.ndarray,
                      algo_name: str, algo_detail: str) -> dict:
    """
    Extract per-cluster profiles (mirrors notebook 03 cell 6f316d51 exactly).
    """
    leaf_ids = sorted([c for c, v in cluster_lineage.items() if 'split_into' not in v])
    n_total = len(features_df)

    global_means = {
        cat: {
            'n_txn_12m':    features_df[f'n_txn_{cat}_12m'].mean()    if f'n_txn_{cat}_12m'    in features_df.columns else 0,
            'total_amt_12m': features_df[f'amt_{cat}_12m'].mean()     if f'amt_{cat}_12m'      in features_df.columns else 0,
            'avg_spend_12m': features_df[f'avg_spend_{cat}_12m'].mean() if f'avg_spend_{cat}_12m' in features_df.columns else 0,
            'consec_months': features_df[f'consec_months_{cat}'].mean() if f'consec_months_{cat}' in features_df.columns else 0,
        }
        for cat in CATEGORIES
    }

    profiles = {}
    for c in leaf_ids:
        grp = features_df[cluster_labels == c]
        lin = cluster_lineage[c]

        category_stats = {}
        for cat in CATEGORIES:
            n12   = grp[f'n_txn_{cat}_12m'].mean()   if f'n_txn_{cat}_12m'   in grp.columns else 0
            a12   = grp[f'amt_{cat}_12m'].mean()      if f'amt_{cat}_12m'     in grp.columns else 0
            avg12 = grp[f'avg_spend_{cat}_12m'].mean() if f'avg_spend_{cat}_12m' in grp.columns else 0
            n6    = grp[f'n_txn_{cat}_6m'].mean()    if f'n_txn_{cat}_6m'    in grp.columns else 0
            a6    = grp[f'amt_{cat}_6m'].mean()       if f'amt_{cat}_6m'      in grp.columns else 0
            cm    = grp[f'consec_months_{cat}'].mean() if f'consec_months_{cat}' in grp.columns else 0

            gm = global_means[cat]
            rel_n   = round(n12  / gm['n_txn_12m'],    2) if gm['n_txn_12m']    > 0 else 0
            rel_amt = round(a12  / gm['total_amt_12m'], 2) if gm['total_amt_12m'] > 0 else 0
            rel_cm  = round(cm   / gm['consec_months'], 2) if gm['consec_months'] > 0 else 0

            category_stats[cat] = {
                'n_txn_12m':    round(n12, 1),
                'total_amt_12m': round(a12, 2),
                'avg_spend_12m': round(avg12, 2),
                'n_txn_6m':     round(n6, 1),
                'total_amt_6m': round(a6, 2),
                'consec_months': round(cm, 1),
                'rel_n_txn':    rel_n,
                'rel_amt':      rel_amt,
                'rel_consec':   rel_cm,
            }

        category_stats_sorted = dict(
            sorted(category_stats.items(), key=lambda x: -x[1]['rel_n_txn'])
        )

        overall = {}
        for col in ['avg_txn_amt', 'std_txn_amt', 'max_txn_amt', 'pct_high_value',
                    'total_spend', 'total_txn_count', 'active_months',
                    'n_unique_categories', 'n_unique_merchants', 'avg_days_between_txn']:
            if col in grp.columns:
                val = grp[col].mean()
                overall[col] = round(val * 100, 1) if col == 'pct_high_value' else round(val, 2)
            else:
                overall[col] = 0

        profiles[str(c)] = {
            'cluster_id':           c,
            'n_customers':          len(grp),
            'pct_of_total':         round(len(grp) / n_total, 3),
            'clustering_algorithm': algo_name,
            'algorithm_detail':     algo_detail,
            'lineage': {
                'depth':          lin['depth'],
                'parent':         lin['parent'],
                'siblings':       lin['siblings'],
                'pct_of_parent':  lin['pct_of_parent'],
                'is_sub_cluster': lin['parent'] is not None,
            },
            'category_stats': category_stats_sorted,
            'overall':        overall,
        }

    return profiles


class ClusteringAgent:
    """
    Clusters customers on the selected features and runs the deepening loop.

    New in this version:
    - Auto-selects algorithm via skills.algo_recommender (unless overridden in config)
    - Auto-selects k via skills.silhouette_optimizer (unless n_clusters set in config)
    - Reports structured status to OrchestratorBus
    """

    def __init__(
        self,
        config: dict,
        bus: OrchestratorBus,
    ):
        # ClusteringAgent owns its ML skills (silhouette optimisation, sklearn
        # clustering, deepening loop). LLM routing decisions go via the bus.
        self.config = config
        self.bus = bus

    def run(
        self,
        features_df: pd.DataFrame,
        selected_features: list[str],
        user_intent: UserIntent | None = None,
        dataset_profile: DatasetProfile | None = None,
        history: list[ClusteringResult] = None,
        feedback: str = '',
        iteration: int = 1,
    ) -> ClusteringResult:
        """
        Parameters
        ----------
        features_df : pd.DataFrame
        selected_features : list[str]
        user_intent : UserIntent | None
        dataset_profile : DatasetProfile | None
        history : list[ClusteringResult]
        feedback : str
        iteration : int
        """
        if history is None:
            history = []

        print(f'\n[Clusterer] Iteration {iteration}')
        if feedback:
            print(f'  Feedback: {feedback}')

        max_pct   = float(self.config.get('max_cluster_size_pct', 0.40))
        sub_k     = int(self.config.get('sub_n_clusters', 3))
        max_depth = int(self.config.get('max_depth', 2))

        # ── Step 1: Preprocess on selected features only ──────────────────────
        sel = [f for f in selected_features if f in features_df.columns]
        if not sel:
            sel = list(features_df.select_dtypes(include=[np.number]).columns)

        log_cols = _build_log_cols(sel)
        X = features_df[sel].copy()
        for col in log_cols:
            X[col] = np.log1p(X[col])

        X = X.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ── Step 2: Algorithm selection ───────────────────────────────────────
        algo_override = str(self.config.get('clustering_algorithm', '')).lower()
        algo_reasoning = ""

        if algo_override in ('kmeans', 'hierarchical'):
            algorithm = algo_override
            algo_reasoning = f"Algorithm fixed by config: {algorithm}"
            print(f'  Algorithm: {algorithm} (from config)')
        else:
            # Auto-select based on data shape
            skewness_map = {col: float(X[col].skew()) for col in X.columns}
            rec = recommend_algorithm(
                n_rows=len(features_df),
                n_features=len(sel),
                feature_skewness=skewness_map,
                business_purpose=user_intent.business_purpose if user_intent else "",
                verbose=True,
            )
            algorithm = rec.algorithm
            algo_reasoning = rec.reasoning
            print(f'  Algorithm auto-selected: {algorithm}  (confidence={rec.confidence:.2f})')

        # ── Step 3: K selection ───────────────────────────────────────────────
        n_clusters_override = self.config.get('n_clusters', None)
        k_scores: dict[int, float] = {}

        if n_clusters_override and isinstance(n_clusters_override, int) and n_clusters_override > 0:
            n_clusters = n_clusters_override
            print(f'  k: {n_clusters} (from config)')
        else:
            k_range = self.config.get('k_search_range', DEFAULT_K_SEARCH_RANGE)
            sil_result = optimize_k(
                X_scaled,
                algorithm=algorithm,
                k_range=k_range,
                verbose=True,
            )
            n_clusters = sil_result.best_k
            k_scores = sil_result.scores
            print(f'  k auto-selected: {n_clusters}  (silhouette={sil_result.best_silhouette:.4f})')

            if sil_result.warning:
                print(f'  WARNING: {sil_result.warning}')

        # ── Step 4: Initial clustering ────────────────────────────────────────
        cluster_labels_arr, algo_name, algo_detail = _fit_model(algorithm, n_clusters, X_scaled)
        n_total = len(features_df)

        work_df = features_df.copy()
        work_df['cluster'] = cluster_labels_arr
        next_id = int(work_df['cluster'].max()) + 1

        # Initialise lineage for top-level clusters
        cluster_lineage: dict = {}
        for c in sorted(work_df['cluster'].unique()):
            c = int(c)
            cluster_lineage[c] = {
                'parent':        None,
                'depth':         0,
                'siblings':      [],
                'pct_of_total':  round((work_df['cluster'] == c).sum() / n_total, 3),
                'pct_of_parent': 1.0,
            }

        top_level = [c for c, v in cluster_lineage.items() if v['parent'] is None]
        for c in top_level:
            cluster_lineage[c]['siblings'] = [x for x in top_level if x != c]

        sil = silhouette_score(X_scaled, work_df['cluster'])
        print(f'  Initial clustering: {n_clusters} clusters  |  silhouette={sil:.4f}')

        # ── Step 5: Deepening loop ─────────────────────────────────────────────
        if max_depth > 0:
            for _round in range(1, max_depth + 1):
                oversized = sorted([
                    int(c) for c in work_df['cluster'].unique()
                    if (work_df['cluster'] == c).sum() / n_total > max_pct
                    and 'split_into' not in cluster_lineage.get(int(c), {})
                ])
                if not oversized:
                    print(f'  Round {_round}: no oversized cluster. Deepening done.')
                    break

                print(f'  Round {_round}: oversized clusters → {oversized}')
                for _parent in oversized:
                    _mask = work_df['cluster'] == _parent
                    _n = int(_mask.sum())
                    _pct = _n / n_total

                    if _n < sub_k * 5:
                        print(f'    C{_parent} ({_n} cust): too small to sub-split — skipping')
                        continue

                    top3_cats = self._get_top3_categories(features_df[_mask])
                    history_summary = self._summarise_history(history)

                    decision = self._ask_oversized_routing(
                        cluster_id=_parent,
                        pct=_pct,
                        n_customers=_n,
                        n_total=n_total,
                        top3_cats=top3_cats,
                        history_summary=history_summary,
                        feedback=feedback,
                    )

                    if decision['action'] == 'reselect_features':
                        print(f'    Claude recommends re-selecting features: {decision["reasoning"]}')
                        if self.bus:
                            self.bus.report(OrchestratorMessage(
                                agent="Clusterer",
                                iteration=iteration,
                                status="blocked",
                                what_was_done=f"Clustered with k={n_clusters}, ran deepening loop",
                                what_was_not_done=f"Could not resolve oversized cluster C{_parent}",
                                doubts="Oversized cluster may reflect feature redundancy",
                                issues=[f"Cluster {_parent} has {_pct:.1%} of data (>{max_pct:.0%} threshold)"],
                                metrics={"silhouette": round(sil, 4), "n_clusters": n_clusters},
                                recommendation="retry",
                                context={"claude_decision": decision},
                            ))
                        return ClusteringResult(
                            action='reselect_features',
                            cluster_labels=None,
                            profiles=None,
                            lineage=None,
                            silhouette=None,
                            n_leaf=None,
                            reasoning=decision['reasoning'],
                            iteration=iteration,
                            algo_name=algo_name,
                            algo_detail=algo_detail,
                            k_scores=k_scores,
                            algo_reasoning=algo_reasoning,
                        )

                    print(f'    Claude recommends sub-clustering C{_parent}.')
                    _X_sub = X_scaled[_mask.values]
                    sub_labels_arr, _, _ = _fit_model(algorithm, sub_k, _X_sub)
                    _new_ids = list(range(next_id, next_id + sub_k))
                    next_id += sub_k

                    _lmap = {i: _new_ids[i] for i in range(sub_k)}
                    work_df.loc[_mask, 'cluster'] = [_lmap[l] for l in sub_labels_arr]

                    cluster_lineage[_parent]['split_into'] = _new_ids
                    for _nid in _new_ids:
                        _n_nid = int((work_df['cluster'] == _nid).sum())
                        cluster_lineage[_nid] = {
                            'parent':        _parent,
                            'depth':         _round,
                            'siblings':      [x for x in _new_ids if x != _nid],
                            'pct_of_total':  round(_n_nid / n_total, 3),
                            'pct_of_parent': round(_n_nid / _n, 3),
                        }

                    _sizes = ', '.join(
                        f'{_nid}({int((work_df["cluster"] == _nid).sum())} cust)'
                        for _nid in _new_ids
                    )
                    print(f'    Split C{_parent} ({_n} cust) → {_sizes}')

        # ── Step 6: Final leaf info ────────────────────────────────────────────
        leaf_ids = sorted([c for c, v in cluster_lineage.items() if 'split_into' not in v])
        n_leaf = len(leaf_ids)
        sil = silhouette_score(X_scaled, work_df['cluster'])

        print(f'  Final: {n_leaf} leaf clusters  |  silhouette={sil:.4f}')

        # ── Step 7: Build profiles ─────────────────────────────────────────────
        profiles = _extract_profiles(
            features_df=work_df.drop(columns=['cluster']),
            cluster_labels=work_df['cluster'],
            cluster_lineage=cluster_lineage,
            X_scaled=X_scaled,
            algo_name=algo_name,
            algo_detail=algo_detail,
        )

        # ── Step 8: Report to orchestrator ─────────────────────────────────────
        # Per docs/agents/clusterer.md: silhouette < 0.15 → blocked, reselect_features
        if sil < 0.15:
            if self.bus:
                self.bus.report(OrchestratorMessage(
                    agent="Clusterer",
                    iteration=iteration,
                    status="blocked",
                    what_was_done=(
                        f"Used {algo_name}, k={n_clusters}; "
                        f"silhouette={sil:.4f} (no meaningful structure)."
                    ),
                    what_was_not_done="Clusters do not separate well — recommend reselecting features.",
                    doubts="",
                    issues=[f"Silhouette={sil:.3f} < 0.15 — no meaningful cluster structure."],
                    metrics={
                        "algorithm": algo_name,
                        "k_selected": n_clusters,
                        "silhouette": round(sil, 4),
                        "k_scores": {str(k): v for k, v in k_scores.items()},
                    },
                    recommendation="retry",
                    context={"action": "reselect_features", "algo_reasoning": algo_reasoning, "k_scores": k_scores},
                ))
            return ClusteringResult(
                action='reselect_features',
                cluster_labels=None,
                profiles=None,
                lineage=None,
                silhouette=sil,
                n_leaf=None,
                reasoning="Silhouette < 0.15 — no meaningful structure. See docs/agents/clusterer.md.",
                iteration=iteration,
                algo_name=algo_name,
                algo_detail=algo_detail,
                k_scores=k_scores,
                algo_reasoning=algo_reasoning,
            )

        sil_quality = (
            "strong" if sil >= 0.50 else
            "reasonable" if sil >= 0.25 else
            "weak"
        )
        status = "success" if sil >= 0.25 else "warning"
        issues = []
        if sil < 0.25:
            issues.append(
                f"Silhouette={sil:.3f} < 0.25 — clusters may overlap. "
                "Consider different k or algorithm."
            )

        if self.bus:
            self.bus.report(OrchestratorMessage(
                agent="Clusterer",
                iteration=iteration,
                status=status,
                what_was_done=(
                    f"Used {algo_name} with k={n_clusters} "
                    f"(auto-selected via silhouette). "
                    f"Ran deepening loop → {n_leaf} leaf clusters. "
                    f"Silhouette={sil:.4f} ({sil_quality})."
                ),
                what_was_not_done=(
                    "Did not try DBSCAN or GMM (not in current skill set)."
                ),
                doubts=(
                    f"Silhouette improved {min(k_scores.values(), default=0):.3f}→{sil:.3f} "
                    f"across k range — marginal gain."
                    if k_scores else ""
                ),
                issues=issues,
                metrics={
                    "algorithm": algo_name,
                    "k_selected": n_clusters,
                    "n_leaf_clusters": n_leaf,
                    "silhouette": round(sil, 4),
                    "k_scores": {str(k): v for k, v in k_scores.items()},
                },
                recommendation="proceed" if not issues else "retry",
                context={
                    "algo_reasoning": algo_reasoning,
                    "k_scores": k_scores,
                },
            ))

        return ClusteringResult(
            action='proceed',
            cluster_labels=work_df['cluster'],
            profiles=profiles,
            lineage=cluster_lineage,
            silhouette=sil,
            n_leaf=n_leaf,
            reasoning='Clustering completed successfully.',
            iteration=iteration,
            algo_name=algo_name,
            algo_detail=algo_detail,
            k_scores=k_scores,
            algo_reasoning=algo_reasoning,
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_top3_categories(self, grp: pd.DataFrame) -> list[str]:
        cat_counts = {}
        for cat in CATEGORIES:
            col = f'n_txn_{cat}_12m'
            if col in grp.columns:
                cat_counts[cat] = grp[col].mean()
        return sorted(cat_counts, key=lambda c: -cat_counts[c])[:3]

    def _summarise_history(self, history: list[ClusteringResult]) -> str:
        if not history:
            return 'No previous attempts.'
        lines = []
        for cr in history:
            lines.append(
                f'  Iteration {cr.iteration}: {cr.action}  '
                f'sil={cr.silhouette}  n_leaf={cr.n_leaf}  reason={cr.reasoning}'
            )
        return '\n'.join(lines)

    def _ask_oversized_routing(
        self,
        cluster_id: int,
        pct: float,
        n_customers: int,
        n_total: int,
        top3_cats: list[str],
        history_summary: str,
        feedback: str,
    ) -> dict:
        """
        ClusteringAgent has detected an oversized cluster. It reports the facts
        to the Orchestrator and asks for a routing decision: sub-cluster in-place
        OR go back to feature selection.
        """
        feedback_section = f'\nUser feedback: {feedback}\n' if feedback else ''
        prompt = f"""You are the orchestrator of a customer clustering pipeline.

The ClusteringAgent reports: Cluster {cluster_id} contains {pct:.1%} of {n_total}
customers ({n_customers} customers), exceeding the 40% threshold for a single persona.
Top spending categories in this cluster: {', '.join(top3_cats)}.

History of clustering attempts:
{history_summary}
{feedback_section}
Decide ONE of:
  (a) sub-cluster — split this cluster further using the current features
  (b) reselect_features — the current features can't separate this cluster;
      send the pipeline back to feature selection

Return ONLY a valid JSON object (no markdown, no extra text):
{{"action": "subcluster" or "reselect_features", "reasoning": "1-2 sentences"}}"""

        raw = self.bus.ask(
            agent="Clusterer",
            purpose=f"route oversized cluster C{cluster_id} ({pct:.0%}): sub-cluster or reselect features",
            prompt=prompt,
            max_tokens=256,
        ).strip()
        if '```' in raw:
            for part in raw.split('```'):
                p = part.strip()
                if p.startswith('json'):
                    p = p[4:].strip()
                if p.startswith('{'):
                    raw = p
                    break
        result = json.loads(raw)
        return result
