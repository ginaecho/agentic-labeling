"""
ClusteringAgent

Contract: docs/agents/clusterer.md. Skills: docs/skills/orchestrator_bus.md,
docs/skills/algo_recommender.md, docs/skills/silhouette_optimizer.md.

Fits the configured clustering algorithm on a selected feature subset,
runs the deepening loop (same logic as notebook 03), and asks the LLM
whether to sub-cluster or request new features when a cluster is too large.

Enhancements over original:
  - Uses skills.algo_recommender to auto-select algorithm from data shape
  - Uses skills.silhouette_optimizer to auto-select k data-driven
  - Supports kmeans, hierarchical, dbscan, gmm, fuzzy_cmeans
  - Reports structured status to OrchestratorBus
  - Category discovery from feature column names (no hard-coded CATEGORIES)
  - Dynamic log-column detection by column name pattern
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

# ── Constants ──────────────────────────────────────────────────────────────────
WINDOWS = [6, 12]

DEFAULT_K_SEARCH_RANGE = [3, 4, 5, 6, 7, 8, 10, 12, 15]


def _detect_log_cols(df, skewness_threshold: float = 2.0) -> list[str]:
    """Return non-negative numeric columns whose |skewness| exceeds the threshold."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return []
    # Deduplicate column names (keep first occurrence) to avoid ambiguous comparisons
    numeric = numeric.loc[:, ~numeric.columns.duplicated()]
    non_neg = [col for col in numeric.columns if float(numeric[col].min()) >= 0]
    if not non_neg:
        return []
    skews = numeric[non_neg].skew().abs()
    return list(skews[skews > skewness_threshold].index)


def _discover_groups(df: pd.DataFrame, entity_col: str) -> dict[str, list[str]]:
    """
    Discover potential grouping columns (low-cardinality categoricals) and their values.
    Returns {col_name: [val1, val2, ...]} for each discovered grouping column.
    No hard-coded column names or patterns.
    """
    groups: dict[str, list[str]] = {}
    for col in df.columns:
        if col == entity_col or col.startswith("_"):
            continue
        # Low-cardinality string/category columns are likely grouping columns
        if df[col].dtype == object or str(df[col].dtype) == "category":
            n_unique = df[col].nunique()
            if 2 <= n_unique <= 200:
                groups[col] = sorted(df[col].dropna().unique().tolist())
    return groups


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
    elif algorithm == 'dbscan':
        from sklearn.cluster import DBSCAN
        min_samples = max(5, X_scaled.shape[1] // 10)
        model = DBSCAN(eps=0.5, min_samples=min_samples)
        raw_labels = model.fit_predict(X_scaled)
        # Remap -1 (noise) to a new positive cluster ID
        max_label = int(raw_labels.max()) if raw_labels.max() >= 0 else -1
        labels = raw_labels.copy()
        noise_mask = labels == -1
        if noise_mask.any():
            labels[noise_mask] = max_label + 1
        algo_name = 'DBSCAN'
        algo_detail = f'DBSCAN  |  eps=0.5  |  min_samples={min_samples}'
    elif algorithm == 'gmm':
        from sklearn.mixture import GaussianMixture
        model = GaussianMixture(n_components=n_clusters, random_state=42, n_init=5)
        labels = model.fit_predict(X_scaled)
        algo_name = 'GaussianMixture'
        algo_detail = f'GMM  |  n_components={n_clusters}  |  covariance=full'
    elif algorithm == 'fuzzy_cmeans':
        try:
            import skfuzzy as fuzz
            cntr, u, *_ = fuzz.cluster.cmeans(
                X_scaled.T, n_clusters, 2, error=0.005, maxiter=1000
            )
            labels = np.argmax(u, axis=0)
            algo_name = 'FuzzyCMeans'
            algo_detail = f'Fuzzy C-Means  |  c={n_clusters}  |  m=2'
        except ImportError:
            # Fallback to GMM when skfuzzy is not installed
            from sklearn.mixture import GaussianMixture
            model = GaussianMixture(n_components=n_clusters, random_state=42)
            labels = model.fit_predict(X_scaled)
            algo_name = 'GaussianMixture (fuzzy_cmeans fallback)'
            algo_detail = f'GMM fallback (skfuzzy not installed)  |  n_components={n_clusters}'
    else:
        raise ValueError(
            f'Unknown clustering_algorithm: {algorithm!r}. '
            'Valid options: "kmeans", "hierarchical", "dbscan", "gmm", "fuzzy_cmeans".'
        )
    return labels, algo_name, algo_detail


def _extract_profiles(features_df: pd.DataFrame, cluster_labels: pd.Series,
                      cluster_lineage: dict, X_scaled: np.ndarray,
                      algo_name: str, algo_detail: str) -> dict:
    """
    Extract per-cluster profiles generically from whatever columns are present.
    No hard-coded column names, prefixes, or CATEGORIES list.
    Works with any feature matrix.
    """
    leaf_ids = sorted([c for c, v in cluster_lineage.items()
                       if 'split_into' not in v and 'merged_into' not in v])
    n_total = len(features_df)
    numeric_cols = list(features_df.select_dtypes(include=[np.number]).columns)

    # Global means for relative comparison
    global_means = features_df[numeric_cols].mean()

    profiles = {}
    for c in leaf_ids:
        mask = cluster_labels == c
        grp = features_df[mask]
        lin = cluster_lineage[c]
        n_cluster = len(grp)

        # Per-column cluster mean and relative deviation from global mean
        cluster_means = grp[numeric_cols].mean()
        # Relative = cluster_mean / global_mean (avoid div by zero)
        relative = (cluster_means / global_means.replace(0, np.nan)).fillna(1.0)

        # Top distinguishing features (most above or below average)
        top_above = relative.nlargest(10).to_dict()
        top_below = relative.nsmallest(10).to_dict()

        # All feature means as a flat dict (rounded)
        all_means = {col: round(float(cluster_means[col]), 4) for col in numeric_cols}
        all_relative = {col: round(float(relative[col]), 3) for col in numeric_cols}

        profiles[str(c)] = {
            "n_entities": n_cluster,
            "pct_total": round(n_cluster / n_total * 100, 1),
            "top_above_average": {k: round(v, 2) for k, v in top_above.items()},
            "top_below_average": {k: round(v, 2) for k, v in top_below.items()},
            "feature_means": all_means,
            "feature_relative": all_relative,
            "lineage": {
                "depth":         lin.get("depth", 0),
                "parent":        lin.get("parent"),
                "siblings":      lin.get("siblings", []),
                "pct_of_parent": lin.get("pct_of_parent", 100.0),
                "is_sub_cluster": lin.get("is_sub_cluster", False),
            },
            "algorithm": algo_name,
            "algo_detail": algo_detail,
        }

    return profiles


class ClusteringAgent:
    """
    Clusters entities on the selected features and runs the deepening loop.

    New in this version:
    - Auto-selects algorithm via skills.algo_recommender (unless overridden in config)
    - Auto-selects k via skills.silhouette_optimizer (unless n_clusters set in config)
    - Supports kmeans, hierarchical, dbscan, gmm, fuzzy_cmeans
    - Falls back to alternative algorithm if chosen one fails
    - Reports structured status to OrchestratorBus
    """

    def __init__(
        self,
        config: dict,
        bus: OrchestratorBus,
    ):
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
        config_override: dict | None = None,
        min_silhouette: float | None = None,
        silhouette_target: float | None = None,
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

        # Merge per-iteration config overrides over base config
        cfg = {**self.config, **(config_override or {})}
        _min_sil = min_silhouette if min_silhouette is not None else 0.05
        # silhouette_target is the orchestrator-level pass bar (dynamic — starts
        # at config.silhouette_target=0.5, relaxes -0.1 after each 3 consecutive
        # misses). The clusterer uses it to decide success vs warning so the agent
        # report aligns with what the orchestrator will do next.
        _target = silhouette_target if silhouette_target is not None \
            else float(cfg.get('silhouette_target', 0.5))

        print(f'\n[Clusterer] Iteration {iteration}')
        if feedback:
            print(f'  Feedback: {feedback}')
        if config_override:
            print(f'  Config overrides: {config_override}')

        max_pct   = float(cfg.get('max_cluster_size_pct', 0.40))
        sub_k     = int(cfg.get('sub_n_clusters', 3))
        max_depth = int(cfg.get('max_depth', 2))

        # ── Step 1: Preprocess on selected features only ──────────────────────
        sel = [f for f in selected_features if f in features_df.columns]
        if not sel:
            sel = list(features_df.select_dtypes(include=[np.number]).columns)

        X = features_df[sel].copy()
        X = X.loc[:, ~X.columns.duplicated()]  # guard against duplicate column names
        log_cols = _detect_log_cols(X)
        for col in log_cols:
            X[col] = np.log1p(X[col])

        X = X.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ── Step 2: Algorithm selection ───────────────────────────────────────
        valid_algos = ('kmeans', 'hierarchical', 'dbscan', 'gmm', 'fuzzy_cmeans')
        algo_override = str(cfg.get('clustering_algorithm', '')).lower()
        algo_reasoning = ""

        if algo_override in valid_algos:
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
        # Priority: user_intent.n_clusters_requested > config.n_clusters > silhouette auto-select
        n_clusters_override = cfg.get('n_clusters', None)
        n_clusters_user = (
            user_intent.n_clusters_requested
            if user_intent and getattr(user_intent, 'n_clusters_requested', None)
            else None
        )
        k_scores: dict[int, float] = {}

        if n_clusters_user and isinstance(n_clusters_user, int) and n_clusters_user >= 2:
            n_clusters = n_clusters_user
            print(f'  k: {n_clusters} (requested by user — skipping silhouette optimisation)')
        elif n_clusters_override and isinstance(n_clusters_override, int) and n_clusters_override > 0:
            n_clusters = n_clusters_override
            print(f'  k: {n_clusters} (from config)')
        else:
            k_range = cfg.get('k_search_range', DEFAULT_K_SEARCH_RANGE)
            # DBSCAN doesn't use k; use a default for silhouette evaluation
            if algorithm == 'dbscan':
                n_clusters = 5  # default; DBSCAN will auto-determine actual count
                print(f'  DBSCAN: skipping k optimisation (auto-detects cluster count)')
            else:
                sil_result = optimize_k(
                    X_scaled,
                    algorithm=algorithm if algorithm in ('kmeans', 'hierarchical') else 'kmeans',
                    k_range=k_range,
                    verbose=True,
                )
                n_clusters = sil_result.best_k
                k_scores = sil_result.scores
                print(f'  k auto-selected: {n_clusters}  (silhouette={sil_result.best_silhouette:.4f})')

                if sil_result.warning:
                    print(f'  WARNING: {sil_result.warning}')

        # ── Step 4: Initial clustering (with fallback on failure) ─────────────
        attempted_algos: list[str] = []
        cluster_labels_arr = None
        algo_name = ''
        algo_detail = ''

        while True:
            try:
                cluster_labels_arr, algo_name, algo_detail = _fit_model(
                    algorithm, n_clusters, X_scaled
                )
                break  # success
            except Exception as exc:
                attempted_algos.append(algorithm)
                print(f'  [Clusterer] Algorithm {algorithm!r} failed: {exc}')

                # Ask LLM for an alternative algorithm (avoid infinite loop)
                remaining = [a for a in valid_algos if a not in attempted_algos]
                if not remaining or not self.bus:
                    # Hard fallback
                    algorithm = 'kmeans' if 'kmeans' not in attempted_algos else 'hierarchical'
                    print(f'  [Clusterer] Falling back to {algorithm} (no LLM available)')
                    try:
                        cluster_labels_arr, algo_name, algo_detail = _fit_model(
                            algorithm, n_clusters, X_scaled
                        )
                    except Exception as exc2:
                        raise RuntimeError(
                            f"Clusterer: all attempted algorithms failed. "
                            f"Attempted: {attempted_algos}. Last error: {exc2}"
                        ) from exc2
                    break

                alt_prompt = (
                    f"The clustering algorithm '{algorithm}' failed with error: {exc}\n"
                    f"Attempted so far: {attempted_algos}\n"
                    f"Remaining options: {remaining}\n"
                    f"Dataset: n_rows={len(features_df)}, n_features={len(sel)}\n\n"
                    "Choose the best alternative algorithm from the remaining options.\n"
                    "Return ONLY a valid JSON object: "
                    '{"algorithm": "<choice>", "reasoning": "<1 sentence>"}'
                )
                try:
                    raw = self.bus.ask(
                        agent="Clusterer",
                        purpose="select alternative algorithm after failure",
                        prompt=alt_prompt,
                        max_tokens=128,
                    ).strip()
                    if '```' in raw:
                        for part in raw.split('```'):
                            p = part.strip()
                            if p.startswith('json'):
                                p = p[4:].strip()
                            if p.startswith('{'):
                                raw = p
                                break
                    resp = json.loads(raw)
                    alt = resp.get('algorithm', remaining[0])
                    if alt not in remaining:
                        alt = remaining[0]
                    print(f'  [Clusterer] LLM suggests alternative: {alt}  ({resp.get("reasoning", "")})')
                    algorithm = alt
                except Exception:
                    algorithm = remaining[0]
                    print(f'  [Clusterer] Defaulting to alternative: {algorithm}')

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
                'pct_total':      round((work_df['cluster'] == c).sum() / n_total, 3),
                'pct_of_parent': 1.0,
            }

        top_level = [c for c, v in cluster_lineage.items() if v['parent'] is None]
        for c in top_level:
            cluster_lineage[c]['siblings'] = [x for x in top_level if x != c]

        sil = silhouette_score(X_scaled, work_df['cluster'])
        print(f'  Initial clustering: {len(top_level)} clusters  |  silhouette={sil:.4f}')

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
                        n_entities=_n,
                        n_total=n_total,
                        top3_cats=top3_cats,
                        history_summary=history_summary,
                        feedback=feedback,
                    )

                    if decision['action'] == 'reselect_features':
                        print(f'    LLM recommends re-selecting features: {decision["reasoning"]}')
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
                                context={"llm_decision": decision},
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

                    print(f'    LLM recommends sub-clustering C{_parent}.')
                    _X_sub = X_scaled[_mask.values]
                    sub_labels_arr, _, _ = _fit_model(algorithm, sub_k, _X_sub)
                    _new_ids = list(range(next_id, next_id + sub_k))
                    next_id += sub_k

                    _lmap = {i: _new_ids[i] for i in range(sub_k)}
                    work_df.loc[_mask, 'cluster'] = np.array(
                        [_lmap[l] for l in sub_labels_arr], dtype=work_df['cluster'].dtype
                    )

                    cluster_lineage[_parent]['split_into'] = _new_ids
                    for _nid in _new_ids:
                        _n_nid = int((work_df['cluster'] == _nid).sum())
                        cluster_lineage[_nid] = {
                            'parent':        _parent,
                            'depth':         _round,
                            'siblings':      [x for x in _new_ids if x != _nid],
                            'pct_total':      round(_n_nid / n_total, 3),
                            'pct_of_parent': round(_n_nid / _n, 3),
                        }

                    _sizes = ', '.join(
                        f'{_nid}({int((work_df["cluster"] == _nid).sum())} cust)'
                        for _nid in _new_ids
                    )
                    print(f'    Split C{_parent} ({_n} cust) → {_sizes}')

        # ── Step 5.5: Merge tiny / singleton leaf clusters ───────────────────
        # Diagnoses the "F1=0 on n=1 cluster" failure mode: a leaf cluster
        # with fewer than min_cluster_size members is unpredictable under
        # stratified CV (per-class F1 collapses to 0; XGBoost may crash when
        # the singleton class is dropped from a fold). Merge each tiny leaf
        # into the nearest non-tiny leaf by centroid distance in X_scaled.
        min_cluster_size = int(cfg.get('min_cluster_size', 5))
        singleton_merges: list[dict] = []
        if min_cluster_size > 1:
            while True:
                leaf_ids_now = [c for c, v in cluster_lineage.items()
                                if 'split_into' not in v and 'merged_into' not in v]
                sizes = {c: int((work_df['cluster'] == c).sum()) for c in leaf_ids_now}
                tiny = sorted([c for c, n in sizes.items() if n < min_cluster_size],
                              key=lambda c: sizes[c])
                if not tiny:
                    break
                # Compute leaf centroids
                centroids = {}
                for c in leaf_ids_now:
                    mask = (work_df['cluster'] == c).values
                    if mask.any():
                        centroids[c] = X_scaled[mask].mean(axis=0)
                # Pick targets (must be >= min_cluster_size). Fallback: any non-tiny
                # leaf; ultimate fallback: the largest leaf even if it's also tiny.
                candidates = [c for c in leaf_ids_now if sizes[c] >= min_cluster_size]
                if not candidates:
                    candidates = sorted(leaf_ids_now, key=lambda c: -sizes[c])[:1]
                if len(candidates) == 1 and candidates[0] in tiny:
                    # All clusters are tiny — bail out to avoid infinite loop
                    break

                for src in tiny:
                    if src not in centroids:
                        continue
                    avail = [t for t in candidates if t != src]
                    if not avail:
                        break
                    # Nearest by Euclidean distance between centroids
                    dists = {t: float(np.linalg.norm(centroids[src] - centroids[t]))
                             for t in avail}
                    tgt = min(dists, key=dists.get)
                    n_src = sizes[src]
                    work_df.loc[work_df['cluster'] == src, 'cluster'] = tgt
                    cluster_lineage[src]['merged_into'] = tgt
                    cluster_lineage[src]['merge_reason'] = (
                        f'n={n_src} < min_cluster_size={min_cluster_size}'
                    )
                    singleton_merges.append({
                        'from': src, 'to': tgt, 'n_moved': n_src,
                        'distance': round(dists[tgt], 3),
                    })
                    print(
                        f'  [Clusterer] Merged tiny cluster C{src} (n={n_src}) → '
                        f'C{tgt} (nearest by centroid, dist={dists[tgt]:.2f}). '
                        f'Reason: n < min_cluster_size={min_cluster_size}.'
                    )
                    # Update centroid of target (weighted) so subsequent merges
                    # see the new center.
                    tgt_n = int((work_df['cluster'] == tgt).sum())
                    if tgt in centroids and tgt_n > 0:
                        centroids[tgt] = X_scaled[(work_df['cluster'] == tgt).values].mean(axis=0)
                # One pass per while-iteration; re-check sizes in next loop
        if singleton_merges:
            print(f'  [Clusterer] Singleton-merge step: {len(singleton_merges)} cluster(s) merged.')

        # ── Step 6: Final leaf info ────────────────────────────────────────────
        leaf_ids = sorted([c for c, v in cluster_lineage.items()
                           if 'split_into' not in v and 'merged_into' not in v])
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
        if sil < _min_sil:
            if self.bus:
                self.bus.report(OrchestratorMessage(
                    agent="Clusterer",
                    iteration=iteration,
                    status="blocked",
                    what_was_done=(
                        f"Used {algo_name}, k={n_clusters}; "
                        f"silhouette={sil:.4f} (near-random — no useful structure)."
                    ),
                    what_was_not_done="Clusters do not separate well — recommend reselecting features.",
                    doubts="",
                    issues=[f"Silhouette={sil:.3f} < {_min_sil} — near-random cluster structure."],
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
                reasoning=f"Silhouette < {_min_sil} — near-random structure. See docs/agents/clusterer.md.",
                iteration=iteration,
                algo_name=algo_name,
                algo_detail=algo_detail,
                k_scores=k_scores,
                algo_reasoning=algo_reasoning,
            )

        sil_quality = (
            "strong" if sil >= 0.50 else
            "reasonable" if sil >= 0.25 else
            "weak but usable" if sil >= 0.10 else
            "low (typical for transaction ratio features)"
        )
        # Success/warning is decided against the DYNAMIC orchestrator target,
        # not a hardcoded threshold — so the chip in the UI matches the actual
        # pass bar the orchestrator will enforce next.
        status = "success" if sil >= _target else "warning"
        issues = []
        if sil < _target:
            issues.append(
                f"Silhouette={sil:.3f} < target {_target:.2f} — orchestrator will "
                "reselect features (or escalate after 3 consecutive misses)."
            )
        elif sil < 0.25:
            issues.append(
                f"Silhouette={sil:.3f} < 0.25 — meets dynamic target {_target:.2f} "
                "but clusters may still overlap; consider different k or algorithm."
            )

        # Build doubt + merge diagnosis
        doubt_parts = []
        if k_scores:
            doubt_parts.append(
                f"Silhouette improved {min(k_scores.values(), default=0):.3f}→{sil:.3f} "
                f"across k range — marginal gain."
            )
        if singleton_merges:
            merge_summary = ", ".join(
                f"C{m['from']}(n={m['n_moved']})→C{m['to']}" for m in singleton_merges
            )
            doubt_parts.append(
                f"Diagnosed {len(singleton_merges)} tiny cluster(s) "
                f"(n<{min_cluster_size}) — root cause of per-class F1=0 / "
                f"singleton-class CV failure. Merged into nearest leaf: {merge_summary}."
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
                    + (f" Merged {len(singleton_merges)} tiny cluster(s) "
                       f"(n<{min_cluster_size}) into nearest leaf."
                       if singleton_merges else "")
                ),
                what_was_not_done=(
                    "Did not try all algorithm variants simultaneously."
                ),
                doubts=" ".join(doubt_parts),
                issues=issues,
                metrics={
                    "algorithm": algo_name,
                    "k_selected": n_clusters,
                    "n_leaf_clusters": n_leaf,
                    "silhouette": round(sil, 4),
                    "silhouette_target": round(_target, 3),
                    "k_scores": {str(k): v for k, v in k_scores.items()},
                    "singleton_merges": len(singleton_merges),
                    "min_cluster_size": min_cluster_size,
                },
                recommendation="proceed" if not issues else "retry",
                context={
                    "algo_reasoning": algo_reasoning,
                    "k_scores": k_scores,
                    "singleton_merges": singleton_merges,
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
        """Return top 3 numeric columns by mean value — used to describe oversized clusters."""
        numeric_cols = list(grp.select_dtypes(include=[np.number]).columns)
        if not numeric_cols:
            return []
        col_means = grp[numeric_cols].mean()
        return list(col_means.nlargest(3).index)

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
        n_entities: int,
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
        prompt = f"""You are the orchestrator of an entity clustering pipeline.

The ClusteringAgent reports: Cluster {cluster_id} contains {pct:.1%} of {n_total}
entities ({n_entities} entities), exceeding the size threshold for a single persona.
Top categories in this cluster: {', '.join(top3_cats) if top3_cats else 'unknown'}.

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
