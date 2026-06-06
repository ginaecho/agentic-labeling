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
from sklearn.preprocessing import StandardScaler, normalize

from agents.state import ClusteringResult
from agents.user_input import UserIntent
from agents.dataset_examiner import DatasetProfile
from skills.silhouette_optimizer import optimize_k
from skills.algo_recommender import recommend_algorithm
from skills.automl_candidate_search import (
    candidate_search_to_dict,
    search_clustering_candidates,
)
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


def _fit_model(algorithm: str, n_clusters: int, X_scaled: np.ndarray,
               text_artifacts: dict | None = None, bus=None,
               business_purpose: str = ''):
    """Fit the chosen clustering algorithm and return (labels, name, detail).

    Text-specific algorithms (`lda`, `nmf`, `llm_cluster`) need the raw TF-IDF
    matrix / raw documents rather than the L2-normalised embeddings, so they
    consume `text_artifacts` instead of `X_scaled`. Falling through to the
    geometric algos uses X_scaled unchanged.
    """
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
    elif algorithm in ('lda', 'nmf'):
        # Topic-model algorithms: each document is a mixture of K topics; hard
        # cluster = argmax of the doc-topic distribution. Both need the
        # non-negative TF-IDF matrix (not L2-normalised SVD embeddings which
        # contain negatives).
        from scipy.sparse import issparse
        tfidf = (text_artifacts or {}).get('tfidf_matrix')
        if tfidf is None:
            raise RuntimeError(
                f"Algorithm {algorithm!r} needs text_artifacts['tfidf_matrix']; "
                "TextPreparer must have run and exposed it."
            )
            # ^ caller catches this and falls back via _fit_model retry loop.
        # n_docs sanity check — text_artifacts['tfidf_matrix'] is built on the
        # cleaned corpus; if rows don't line up with X_scaled the caller will
        # have already errored upstream, but guard here just in case.
        if tfidf.shape[0] != X_scaled.shape[0]:
            raise RuntimeError(
                f"tfidf rows ({tfidf.shape[0]}) != embedding rows ({X_scaled.shape[0]}); "
                "cannot fit topic model."
            )
        if algorithm == 'lda':
            from sklearn.decomposition import LatentDirichletAllocation
            model = LatentDirichletAllocation(
                n_components=n_clusters, random_state=42,
                learning_method='batch', max_iter=20,
            )
            doc_topic = model.fit_transform(tfidf)
            labels = np.argmax(doc_topic, axis=1)
            algo_name = 'LatentDirichletAllocation'
            algo_detail = (f'LDA  |  n_topics={n_clusters}  |  '
                           f'learning=batch  |  vocab={tfidf.shape[1]}')
        else:  # nmf
            from sklearn.decomposition import NMF
            # NMF is sensitive to init on TF-IDF; nndsvd gives a deterministic
            # warm start that's better than random for sparse non-negative data.
            model = NMF(
                n_components=n_clusters, random_state=42,
                init='nndsvd', max_iter=300,
            )
            doc_topic = model.fit_transform(tfidf if not issparse(tfidf) else tfidf)
            labels = np.argmax(doc_topic, axis=1)
            algo_name = 'NMF'
            algo_detail = (f'NMF  |  n_topics={n_clusters}  |  init=nndsvd  |  '
                           f'vocab={tfidf.shape[1]}')
    elif algorithm == 'llm_cluster':
        # LLM-as-clusterer: rescue path when geometric methods + topic models
        # all fail. Sends representative docs to Claude and asks for cluster
        # assignment. ONLY safe for small corpora (the prompt grows linearly
        # with n_docs) — the caller gates this on len(docs) ≤ 200.
        labels, algo_name, algo_detail = _fit_llm_cluster(
            n_clusters=n_clusters,
            text_artifacts=text_artifacts or {},
            bus=bus,
            business_purpose=business_purpose,
        )
    else:
        raise ValueError(
            f'Unknown clustering_algorithm: {algorithm!r}. '
            'Valid options: "kmeans", "hierarchical", "dbscan", "gmm", '
            '"fuzzy_cmeans", "lda", "nmf", "llm_cluster".'
        )
    return labels, algo_name, algo_detail


def _fit_llm_cluster(n_clusters: int, text_artifacts: dict, bus,
                     business_purpose: str = '') -> tuple[np.ndarray, str, str]:
    """Ask the LLM to group documents into n_clusters categories.

    Returns hard cluster labels, the algorithm name, and a short detail string.
    The prompt includes every document numbered; the LLM returns a JSON list
    of cluster ids (0..k-1) one per document. This is the most expensive
    clustering path — gate it on small corpora (< ~200 docs) at the call site.
    """
    raw_docs = list(text_artifacts.get('raw_docs') or [])
    n_docs = len(raw_docs)
    if n_docs == 0:
        raise RuntimeError("llm_cluster: text_artifacts['raw_docs'] is empty.")
    if bus is None:
        raise RuntimeError("llm_cluster needs an LLM bus to make Claude calls.")

    # Truncate per-doc so the total prompt stays manageable for very long bodies.
    _MAX_CHARS = 600
    numbered = '\n'.join(
        f"[{i}] {d.strip()[:_MAX_CHARS]}" for i, d in enumerate(raw_docs)
    )
    purpose_line = (
        f"\nThe user's business purpose: {business_purpose}\n" if business_purpose else ''
    )
    prompt = f"""You are clustering {n_docs} short documents into EXACTLY {n_clusters} groups.
{purpose_line}
Documents (numbered):
{numbered}

Return ONLY a JSON object with this exact shape, no markdown fences, no prose:
{{
  "labels": [<integer cluster id from 0 to {n_clusters - 1}>, ... one per document, in order],
  "reasoning": "<1-2 sentence summary of what distinguishes the groups>"
}}

Constraints:
- The labels array MUST have exactly {n_docs} integers.
- Each label MUST be between 0 and {n_clusters - 1} inclusive.
- Aim to balance the groups (no single group with > 60% of docs) unless the
  natural grouping really is that lopsided.
"""
    import json as _json
    import re as _re
    raw = bus.ask(
        agent='Clusterer',
        purpose='LLM-as-clusterer (geometric methods failed; semantic grouping fallback)',
        prompt=prompt,
        max_tokens=2000,
        category='pipeline',
    )
    # Strip optional markdown fences before parsing.
    if '```' in raw:
        for chunk in raw.split('```'):
            c = chunk.strip()
            if c.startswith('json'):
                c = c[4:].strip()
            if c.startswith('{'):
                raw = c
                break
    try:
        parsed = _json.loads(raw)
    except _json.JSONDecodeError:
        # Last-ditch: pull the first [...] of ints we can find.
        m = _re.search(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]', raw)
        if not m:
            raise RuntimeError(
                f"llm_cluster: LLM did not return parseable JSON. Got: {raw[:300]!r}"
            )
        parsed = {'labels': _json.loads(m.group(0)), 'reasoning': ''}

    label_list = parsed.get('labels') or []
    if len(label_list) != n_docs:
        raise RuntimeError(
            f"llm_cluster: LLM returned {len(label_list)} labels but corpus has "
            f"{n_docs} docs. Aborting (caller will retry with a geometric algo)."
        )
    labels = np.array([int(x) for x in label_list], dtype=int)
    # Clamp to the valid range — the LLM occasionally returns k itself.
    labels = np.clip(labels, 0, n_clusters - 1)
    reasoning = str(parsed.get('reasoning', '')).strip()[:200]
    return labels, 'LLMCluster', (
        f'LLM-as-clusterer  |  k={n_clusters}  |  docs={n_docs}  |  '
        f'reasoning="{reasoning}"' if reasoning
        else f'LLM-as-clusterer  |  k={n_clusters}  |  docs={n_docs}'
    )


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


def _extract_text_profiles(
    features_df: pd.DataFrame,
    cluster_labels: pd.Series,
    cluster_lineage: dict,
    X_emb: np.ndarray,
    text_artifacts: dict,
    algo_name: str,
    algo_detail: str,
    top_terms_k: int = 12,
    rep_docs_k: int = 3,
) -> dict:
    """Text-mode profiles: c-TF-IDF distinctive terms + representative docs.

    Returns profiles in the SAME schema as `_extract_profiles` so the rest of
    the pipeline (PersonaNamer, UI, cross-cluster comparison) works unchanged:

      - top_above_average ← {term: c_tfidf_score}    (distinctive terms)
      - feature_means     ← {term: c_tfidf_score}    (same dict, for chips)
      - top_terms         ← ordered list of distinctive terms
      - representative_docs ← list of doc strings nearest the cluster centroid
      - top_below_average ← terms that are notably ABSENT from this cluster
                            (always present in other clusters but not here)

    c-TF-IDF (class-based TF-IDF, Grootendorst 2020) per term t in cluster c:
        score(t, c) = tf(t, c) * log(1 + A / sum_c tf(t, c))
    where tf(t, c) is the summed TF-IDF weight of t across docs in c, and A is
    the average per-cluster document count. This pulls out terms that are
    common in cluster c but rare in the rest.
    """
    leaf_ids = sorted([c for c, v in cluster_lineage.items() if 'split_into' not in v])
    n_total = len(features_df)
    profiles: dict = {}

    tfidf_matrix = text_artifacts.get('tfidf_matrix')
    feature_names = list(text_artifacts.get('feature_names') or [])
    raw_docs = list(text_artifacts.get('raw_docs') or [])
    doc_index = list(text_artifacts.get('doc_index') or [])

    have_tfidf = tfidf_matrix is not None and len(feature_names) > 0
    have_docs  = len(raw_docs) == len(doc_index) > 0

    # Per-cluster summed TF-IDF (a sparse matrix → dense array per cluster) +
    # the doc-row index inside the original tfidf_matrix.
    cluster_sums = {}
    cluster_doc_rows: dict[int, list[int]] = {}
    if have_tfidf:
        # Map cluster_labels.index → tfidf_matrix row position.
        # tfidf_matrix rows match the order docs were passed to vectorize_text
        # (i.e. the cleaned subset). text_artifacts['doc_index'] holds the
        # original DataFrame index for each tfidf row.
        if doc_index:
            idx_to_row = {ix: r for r, ix in enumerate(doc_index)}
        else:
            idx_to_row = {ix: r for r, ix in enumerate(cluster_labels.index)}
        for c in leaf_ids:
            mask = cluster_labels == c
            rows = [idx_to_row[ix] for ix in cluster_labels.index[mask] if ix in idx_to_row]
            cluster_doc_rows[c] = rows
            if rows:
                cluster_sums[c] = np.asarray(tfidf_matrix[rows].sum(axis=0)).ravel()
            else:
                cluster_sums[c] = np.zeros(len(feature_names), dtype=float)

    # Per-cluster centroid in the embedding space (for representative docs).
    centroids: dict[int, np.ndarray] = {}
    for c in leaf_ids:
        mask = cluster_labels == c
        rows = mask.to_numpy()
        if rows.any():
            centroids[c] = X_emb[rows].mean(axis=0)
        else:
            centroids[c] = np.zeros(X_emb.shape[1] if X_emb.size else 1, dtype=float)

    avg_docs_per_cluster = max(1.0, n_total / max(len(leaf_ids), 1))

    for c in leaf_ids:
        mask = cluster_labels == c
        n_cluster = int(mask.sum())
        lin = cluster_lineage[c]

        top_above: dict[str, float] = {}
        top_below: dict[str, float] = {}
        feature_means: dict[str, float] = {}
        top_terms: list[str] = []

        if have_tfidf:
            # c-TF-IDF: penalise terms that appear in many other clusters too.
            this_sum = cluster_sums[c]
            other_sum = np.zeros_like(this_sum)
            for cc in leaf_ids:
                if cc != c:
                    other_sum += cluster_sums[cc]
            # Smooth so terms absent elsewhere don't blow up.
            denom = other_sum + 1.0
            scores = this_sum * np.log1p(avg_docs_per_cluster / denom)
            if scores.size:
                # Distinctive (present here, rare elsewhere)
                top_idx = np.argsort(-scores)[:top_terms_k]
                top_above = {
                    str(feature_names[i]): round(float(scores[i]), 4)
                    for i in top_idx if scores[i] > 0
                }
                top_terms = list(top_above.keys())
                # Notably absent: terms that score high in OTHERS but ~0 here.
                # Use a normalised ratio so we don't pick rare-everywhere terms.
                other_strength = other_sum / max(other_sum.max(), 1e-9)
                here_strength  = this_sum / max(this_sum.max(), 1e-9)
                gap = other_strength - here_strength   # positive → absent here
                gap_idx = np.argsort(-gap)[:top_terms_k]
                top_below = {
                    str(feature_names[i]): round(float(gap[i]), 4)
                    for i in gap_idx if gap[i] > 0.1
                }
                feature_means = top_above   # one dict for both UI chips & prompts

        representative_docs: list[str] = []
        if have_docs and n_cluster > 0:
            # Pick the rep_docs_k docs closest to the centroid in cosine space.
            # X_emb is already L2-normalised in the text branch so dot product
            # IS cosine similarity.
            rows = mask.to_numpy()
            in_cluster_rows = np.where(rows)[0]
            if in_cluster_rows.size:
                centroid = centroids[c]
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                sims = X_emb[in_cluster_rows] @ centroid
                # Sort descending; take top rep_docs_k. Map back to raw_docs.
                top_local = in_cluster_rows[np.argsort(-sims)[:rep_docs_k]]
                # raw_docs is aligned to doc_index. If they match, use that;
                # otherwise positional indexing.
                if doc_index and len(raw_docs) == len(doc_index):
                    for row in top_local:
                        if 0 <= row < len(raw_docs):
                            representative_docs.append(raw_docs[row])
                else:
                    for row in top_local:
                        if 0 <= row < len(raw_docs):
                            representative_docs.append(raw_docs[row])

        profiles[str(c)] = {
            "n_entities": n_cluster,
            "pct_total": round(n_cluster / max(n_total, 1) * 100, 1),
            "top_above_average": top_above,
            "top_below_average": top_below,
            "feature_means": feature_means,
            "feature_relative": {k: 1.0 for k in top_above},  # placeholder for UI
            "top_terms": top_terms,
            "representative_docs": [d[:400] for d in representative_docs],
            "modality": "text",
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
        text_artifacts: dict | None = None,
        bypass: bool = False,
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

        # Text modality: embeddings are already a clean numeric matrix produced
        # by TextPreparer. Skip log + StandardScaler (would distort the geometry
        # the embeddings were trained for) and L2-normalise so Euclidean
        # distance becomes cosine — silhouette computed with metric='cosine'.
        text_mode = bool(text_artifacts)
        sil_metric = 'cosine' if text_mode else 'euclidean'

        if text_mode:
            X = X.select_dtypes(include=[np.number])
            X_scaled = normalize(X.to_numpy(dtype=float))
            print(f'  Text mode: skipped log/StandardScaler; L2-normalised '
                  f'{X_scaled.shape[0]}×{X_scaled.shape[1]} embedding matrix. '
                  f'Silhouette metric=cosine.')
        else:
            log_cols = _detect_log_cols(X)
            for col in log_cols:
                X[col] = np.log1p(X[col])
            X = X.select_dtypes(include=[np.number])
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

        # ── Step 2: Algorithm selection ───────────────────────────────────────
        # Geometric algos work for any modality; topic models (lda, nmf) and
        # the LLM-as-clusterer require text artifacts and are silently filtered
        # out of the valid set when running on tabular data.
        valid_algos: tuple[str, ...] = (
            'kmeans', 'hierarchical', 'dbscan', 'gmm', 'fuzzy_cmeans',
        )
        if text_mode:
            valid_algos = valid_algos + ('lda', 'nmf', 'llm_cluster')
        algo_override = str(cfg.get('clustering_algorithm', '')).lower()
        algo_reasoning = ""
        candidate_evidence: dict = {}
        candidate_best = None

        if algo_override in valid_algos:
            algorithm = algo_override
            algo_reasoning = f"Algorithm fixed by config: {algorithm}"
            print(f'  Algorithm: {algorithm} (from config)')
        else:
            # Auto-select based on data shape. In text mode the skewness map
            # on raw embedding dims is meaningless (and the +Hierarchical bias
            # it would trigger isn't appropriate), so we skip it and instead
            # pass the modality hint so the recommender applies its text-aware
            # rule (preferred=kmeans/hierarchical, discouraged=gmm/dbscan/fuzzy).
            _text_ov = dict(cfg.get('text_overrides') or {})
            if text_mode:
                skewness_map = None
                preferred = _text_ov.get('preferred_algorithms') or ['kmeans', 'hierarchical']
                discouraged = _text_ov.get('discouraged_algorithms') or ['gmm', 'dbscan', 'fuzzy_cmeans']
            else:
                skewness_map = {col: float(X[col].skew()) for col in X.columns}
                preferred = None
                discouraged = None
            rec = recommend_algorithm(
                n_rows=len(features_df),
                n_features=len(sel),
                feature_skewness=skewness_map,
                business_purpose=user_intent.business_purpose if user_intent else "",
                verbose=True,
                modality='text' if text_mode else 'tabular',
                preferred_algorithms=preferred,
                discouraged_algorithms=discouraged,
            )
            algorithm = rec.algorithm
            algo_reasoning = rec.reasoning
            print(f'  Algorithm auto-selected: {algorithm}  (confidence={rec.confidence:.2f})')

        # ── Step 2.5: AutoML-as-skill candidate tournament ───────────────────
        # The recommender gives a fast prior; the candidate search gives
        # evidence. It evaluates several algorithm/k combinations with
        # silhouette, bootstrap stability (ARI), and cluster-size balance, then
        # seeds the normal Clusterer path with the winner.
        n_clusters_override = cfg.get('n_clusters', None)
        n_clusters_user = (
            user_intent.n_clusters_requested
            if user_intent and getattr(user_intent, 'n_clusters_requested', None)
            else None
        )
        candidate_search_enabled = bool(cfg.get('enable_automl_candidate_search', True))
        if (
            candidate_search_enabled
            and algo_override not in valid_algos
            and not (n_clusters_user and isinstance(n_clusters_user, int))
            and not (n_clusters_override and isinstance(n_clusters_override, int))
        ):
            k_range_for_search = cfg.get('k_search_range', DEFAULT_K_SEARCH_RANGE)
            configured_algos = cfg.get('candidate_search_algorithms')
            if configured_algos:
                candidate_algos = list(configured_algos)
            elif text_mode:
                candidate_algos = ['kmeans', 'hierarchical']
            else:
                candidate_algos = [algorithm, 'kmeans', 'hierarchical', 'gmm']
            try:
                search_result = search_clustering_candidates(
                    X_scaled,
                    algorithms=candidate_algos,
                    k_range=k_range_for_search,
                    metric=sil_metric,
                    max_cluster_size_pct=max_pct,
                    stability_repeats=int(cfg.get('candidate_stability_repeats', 3)),
                    stability_sample_frac=float(cfg.get('candidate_stability_sample_frac', 0.80)),
                    top_n=int(cfg.get('candidate_search_top_n', 8)),
                    verbose=True,
                )
                candidate_evidence = candidate_search_to_dict(search_result)
                candidate_best = search_result.best
                if candidate_best is not None:
                    algorithm = candidate_best.algorithm
                    algo_reasoning = f"{algo_reasoning}\n{search_result.reasoning}".strip()
                    print(f'  AutoML candidate winner: {search_result.reasoning}')
            except Exception as exc:
                candidate_evidence = {
                    "error": str(exc),
                    "reasoning": "Candidate search failed; falling back to recommender + silhouette optimizer.",
                }
                print(f'  [AutoMLCandidateSearch] failed: {exc}')

        # ── Step 3: K selection ───────────────────────────────────────────────
        # Priority: user_intent.n_clusters_requested > config.n_clusters > silhouette auto-select
        k_scores: dict[int, float] = {}

        if n_clusters_user and isinstance(n_clusters_user, int) and n_clusters_user >= 2:
            n_clusters = n_clusters_user
            print(f'  k: {n_clusters} (requested by user — skipping silhouette optimisation)')
        elif n_clusters_override and isinstance(n_clusters_override, int) and n_clusters_override > 0:
            n_clusters = n_clusters_override
            print(f'  k: {n_clusters} (from config)')
        elif candidate_best is not None and candidate_best.k is not None:
            n_clusters = int(candidate_best.k)
            k_scores = {
                int(c["k"]): float(c["silhouette"])
                for c in candidate_evidence.get("candidates", [])
                if c.get("algorithm") == algorithm and c.get("k") is not None
            }
            print(
                f'  k: {n_clusters} (from AutoML candidate search; '
                f'silhouette={candidate_best.silhouette:.4f}, '
                f'stability_ari={candidate_best.stability_ari:.4f})'
            )
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
                    metric=sil_metric,
                )
                n_clusters = sil_result.best_k
                k_scores = sil_result.scores
                print(
                    f'  k auto-selected: {n_clusters}  '
                    f'composite={sil_result.best_composite:.3f}  '
                    f'silhouette={sil_result.best_silhouette:.4f}  '
                    f'DB={sil_result.best_db:.4f}  CH={sil_result.best_ch:.1f}'
                )

                if sil_result.warning:
                    print(f'  WARNING: {sil_result.warning}')

        # ── Step 4: Initial clustering (with fallback on failure) ─────────────
        attempted_algos: list[str] = []
        cluster_labels_arr = None
        algo_name = ''
        algo_detail = ''

        # Side-input bundle for text-specific algorithms (lda/nmf/llm_cluster).
        # Geometric algorithms ignore these.
        _bp = user_intent.business_purpose if user_intent else ''
        # llm_cluster is gated on small corpora — it grows the LLM prompt
        # linearly with n_docs, so anything past ~200 burns a lot of tokens.
        _llm_cluster_ok = (
            text_mode
            and text_artifacts
            and len(text_artifacts.get('raw_docs') or []) <= 200
        )

        while True:
            try:
                cluster_labels_arr, algo_name, algo_detail = _fit_model(
                    algorithm, n_clusters, X_scaled,
                    text_artifacts=text_artifacts if text_mode else None,
                    bus=self.bus if algorithm == 'llm_cluster' and _llm_cluster_ok else None,
                    business_purpose=_bp,
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
                            algorithm, n_clusters, X_scaled,
                            text_artifacts=text_artifacts if text_mode else None,
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

        sil = silhouette_score(X_scaled, work_df['cluster'], metric=sil_metric)
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
                            context={"llm_decision": decision, "candidate_evidence": candidate_evidence},
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
                            candidate_evidence=candidate_evidence,
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
        #
        # Text mode uses a much smaller default (2) because text corpora often
        # produce meaningful 2-3 document clusters (a topic = a handful of
        # articles), and there is no per-class CV gate downstream to satisfy.
        # The tabular default (5) is calibrated for customer-segmentation
        # F1-validated runs and would erase real text topics.
        _default_min_size = 2 if text_mode else 5
        min_cluster_size = int(cfg.get('min_cluster_size', _default_min_size))
        if text_mode:
            print(f'  Text mode: min_cluster_size={min_cluster_size} '
                  f'(default {_default_min_size}; smaller threshold preserves '
                  f'topic-level groups in short corpora).')
        singleton_merges: list[dict] = []
        if min_cluster_size > 1:
            while True:
                # Preserve ≥2-cluster structure: silhouette and the downstream
                # classifier both require at least 2 distinct labels. If we are
                # already at 2 leaves, stop merging — leaving a tiny leaf is
                # strictly better than collapsing all cluster diversity into 1.
                if int(work_df['cluster'].nunique()) <= 2:
                    break
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
                    # Same ≥2-cluster floor inside the per-pass loop: every
                    # merge reduces the distinct-label count by 1, so stop as
                    # soon as we are about to drop to a single cluster.
                    if int(work_df['cluster'].nunique()) <= 2:
                        break
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

        # Guard: silhouette_score requires ≥2 distinct labels. The singleton-merge
        # step above can collapse every leaf into one target when only a single
        # cluster meets min_cluster_size (all the tiny ones merge into it).
        # Surface this as a blocked run so the orchestrator reselects features
        # rather than crashing inside sklearn.
        n_unique_labels = int(work_df['cluster'].nunique())
        if n_unique_labels < 2:
            print(
                f'  [Clusterer] Singleton-merge collapsed all leaves into '
                f'{n_unique_labels} cluster — min_cluster_size={min_cluster_size} '
                f'too aggressive for this dataset. Asking user how to proceed.'
            )
            from skills.user_decisions import ask_user_decision
            chosen = ask_user_decision(
                self.bus,
                decision_id=f'clusterer_merge_collapse_iter{iteration}',
                agent='Clusterer',
                title='Singleton-merge collapsed all clusters into 1',
                summary=(
                    f"After merging tiny leaves (min_cluster_size="
                    f"{min_cluster_size}), only 1 cluster remained — silhouette "
                    f"can't be computed. The threshold may be too aggressive "
                    f"for this dataset (especially text corpora with small "
                    f"natural topics)."
                ),
                options=[
                    {'key': 'reselect',
                     'label': 'Reselect features (recommended)',
                     'description': 'Default. Pick different features and retry.'},
                    {'key': 'relax_min_size',
                     'label': 'Lower min_cluster_size to 2 and retry',
                     'description': 'Tiny clusters (2 docs) will be preserved instead '
                                    'of merged. Useful when the dataset has many '
                                    'small natural topics.'},
                    {'key': 'abort',
                     'label': 'Abort the run',
                     'description': 'Stop now.'},
                ],
                recommended='reselect',
                bypass=bypass,
                extra={
                    'singleton_merges': len(singleton_merges),
                    'min_cluster_size': min_cluster_size,
                    'algorithm': algo_name,
                    'k': n_clusters,
                    'n_unique_labels': n_unique_labels,
                },
            )
            if chosen == 'relax_min_size':
                # Note: reselect_features is still the orchestrator-level action
                # here because we can't redo the merge inside this run() call
                # without restarting. We surface the chosen tuning through the
                # context so the orchestrator can pass min_cluster_size=2 on
                # the next iteration.
                print('  [Clusterer] User chose to lower min_cluster_size to 2 '
                      '— flagging in context for next iteration.')
                if self.bus:
                    self.bus.report(OrchestratorMessage(
                        agent="Clusterer",
                        iteration=iteration,
                        status="blocked",
                        what_was_done=(
                            f"User chose to relax min_cluster_size from "
                            f"{min_cluster_size} to 2 — needs a fresh iteration."
                        ),
                        what_was_not_done="Did not re-run the merge in this iteration.",
                        doubts="",
                        issues=["Re-run needed with relaxed min_cluster_size."],
                        metrics={"min_cluster_size_was": min_cluster_size,
                                 "min_cluster_size_now": 2},
                        recommendation="retry",
                        context={"action": "reselect_features",
                                 "tuning_overrides": {"min_cluster_size": 2}},
                    ))
                return ClusteringResult(
                    action='reselect_features',
                    cluster_labels=None, profiles=None, lineage=None,
                    silhouette=-1.0, n_leaf=None,
                    reasoning='User requested smaller min_cluster_size — retry.',
                    iteration=iteration, algo_name=algo_name, algo_detail=algo_detail,
                    k_scores=k_scores, algo_reasoning=algo_reasoning,
                    candidate_evidence=candidate_evidence,
                )
            # 'abort' and 'reselect' both fall through to the existing
            # blocked-report path below.
            if self.bus:
                self.bus.report(OrchestratorMessage(
                    agent="Clusterer",
                    iteration=iteration,
                    status="blocked",
                    what_was_done=(
                        f"Used {algo_name}, k={n_clusters}; singleton-merge "
                        f"({len(singleton_merges)} merges, "
                        f"min_cluster_size={min_cluster_size}) collapsed every "
                        f"leaf into 1 cluster."
                    ),
                    what_was_not_done=(
                        "Did not compute silhouette (needs ≥2 distinct labels)."
                    ),
                    doubts="",
                    issues=[
                        f"All clusters merged into 1 — only one leaf met "
                        f"min_cluster_size={min_cluster_size}, so every tiny "
                        f"leaf rolled into it. Features lack separable structure "
                        f"at this granularity."
                    ],
                    metrics={
                        "algorithm": algo_name,
                        "k_selected": n_clusters,
                        "n_unique_labels": n_unique_labels,
                        "singleton_merges": len(singleton_merges),
                        "min_cluster_size": min_cluster_size,
                        "k_scores": {str(k): v for k, v in k_scores.items()},
                        "candidate_search": candidate_evidence.get("best"),
                    },
                    recommendation="retry",
                    context={
                        "action": "reselect_features",
                        "algo_reasoning": algo_reasoning,
                        "k_scores": k_scores,
                        "singleton_merges": singleton_merges,
                        "candidate_evidence": candidate_evidence,
                    },
                ))
            return ClusteringResult(
                action='reselect_features',
                cluster_labels=None,
                profiles=None,
                lineage=None,
                silhouette=-1.0,
                n_leaf=None,
                reasoning=(
                    f"Singleton-merge collapsed all leaves into 1 cluster "
                    f"(min_cluster_size={min_cluster_size}). Need different "
                    f"features or a smaller min_cluster_size."
                ),
                iteration=iteration,
                algo_name=algo_name,
                algo_detail=algo_detail,
                k_scores=k_scores,
                algo_reasoning=algo_reasoning,
                candidate_evidence=candidate_evidence,
            )

        sil = silhouette_score(X_scaled, work_df['cluster'], metric=sil_metric)

        print(f'  Final: {n_leaf} leaf clusters  |  silhouette={sil:.4f}')

        # ── Step 7: Build profiles ─────────────────────────────────────────────
        if text_mode:
            profiles = _extract_text_profiles(
                features_df=work_df.drop(columns=['cluster']),
                cluster_labels=work_df['cluster'],
                cluster_lineage=cluster_lineage,
                X_emb=X_scaled,
                text_artifacts=text_artifacts or {},
                algo_name=algo_name,
                algo_detail=algo_detail,
            )
        else:
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
            # Critical decision point: silhouette is below the near-random floor.
            # Default behavior is `reselect_features`, but for text corpora the
            # floor is often too strict (TF-IDF/SVD silhouette of ~0.04 can
            # still be a meaningful topic split). Surface the choice to the user.
            from skills.user_decisions import ask_user_decision
            chosen = ask_user_decision(
                self.bus,
                decision_id=f'clusterer_sil_floor_iter{iteration}',
                agent='Clusterer',
                title='Silhouette below near-random floor',
                summary=(
                    f"Clustering finished with silhouette={sil:.3f}, which is "
                    f"below the floor ({_min_sil}). Default is to reselect "
                    f"features and retry; you can relax the floor and accept "
                    f"these clusters instead, or abort the run."
                ),
                options=[
                    {'key': 'reselect',
                     'label': 'Reselect features (recommended)',
                     'description': 'Default. The orchestrator will pick a different '
                                    'feature subset / algorithm and retry.'},
                    {'key': 'relax',
                     'label': f'Accept clusters (relax floor to {sil:.3f})',
                     'description': 'Continue with these clusters even though the '
                                    'silhouette is low. Useful for text corpora '
                                    'where small but real topics exist.'},
                    {'key': 'abort',
                     'label': 'Abort the run',
                     'description': 'Stop now and let me look at the data first.'},
                ],
                recommended='reselect',
                bypass=bypass,
                extra={
                    'silhouette': round(sil, 4),
                    'floor': _min_sil,
                    'algorithm': algo_name,
                    'k': n_clusters,
                    'n_leaf_clusters': n_leaf,
                },
            )
            if chosen == 'abort':
                if self.bus:
                    self.bus.report(OrchestratorMessage(
                        agent="Clusterer",
                        iteration=iteration,
                        status="blocked",
                        what_was_done=f"Aborted at user request (silhouette={sil:.4f}).",
                        what_was_not_done="Did not proceed past clustering.",
                        doubts="",
                        issues=["User aborted from threshold-decision modal."],
                        metrics={"silhouette": round(sil, 4)},
                        recommendation="abort",
                        context={"action": "abort", "candidate_evidence": candidate_evidence},
                    ))
                return ClusteringResult(
                    action='reselect_features',  # closest existing terminal action
                    cluster_labels=None, profiles=None, lineage=None,
                    silhouette=sil, n_leaf=None,
                    reasoning='User aborted at silhouette-floor decision.',
                    iteration=iteration, algo_name=algo_name, algo_detail=algo_detail,
                    k_scores=k_scores, algo_reasoning=algo_reasoning,
                    candidate_evidence=candidate_evidence,
                )
            if chosen == 'relax':
                # User accepts the low silhouette — fall through to the normal
                # success path so the run continues with these clusters.
                print(f'  [Clusterer] User relaxed silhouette floor — accepting '
                      f'sil={sil:.4f} and continuing.')
                _min_sil = sil - 1e-6
                # Continue to the normal report path below (sil >= _min_sil now).
            else:  # chosen == 'reselect' — default behavior
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
                            "candidate_search": candidate_evidence.get("best"),
                        },
                        recommendation="retry",
                        context={
                            "action": "reselect_features",
                            "algo_reasoning": algo_reasoning,
                            "k_scores": k_scores,
                            "candidate_evidence": candidate_evidence,
                        },
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
                    candidate_evidence=candidate_evidence,
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

        # Build rich AutoML context for the issue text
        _cs_best = candidate_evidence.get("best") if candidate_evidence else None
        _cs_all = candidate_evidence.get("candidates", []) if candidate_evidence else []
        _automl_ctx = ""
        def _fmt(val, fmt=".3f"):
            try:
                return format(float(val), fmt)
            except (TypeError, ValueError):
                return str(val)

        if _cs_best:
            _automl_ctx = (
                f"AutoML evaluated {len(_cs_all)} candidate(s). "
                f"Best: {_cs_best['algorithm']} k={_cs_best['k']} "
                f"with composite={_fmt(_cs_best.get('composite_score'), '.3f')} "
                f"(sil={_fmt(_cs_best.get('silhouette'))}, "
                f"DB={_fmt(_cs_best.get('davies_bouldin'), '.2f')}, "
                f"CH={_fmt(_cs_best.get('calinski_harabasz'), '.1f')}, "
                f"ARI={_fmt(_cs_best.get('stability_ari'))}). "
            )

        if sil < _target:
            issues.append(
                f"{_automl_ctx}"
                f"Silhouette={sil:.3f} < target {_target:.2f} — orchestrator will "
                "reselect features (or escalate after 3 consecutive misses)."
            )
        elif sil < 0.25:
            issues.append(
                f"{_automl_ctx}"
                f"Silhouette={sil:.3f} < 0.25 — meets dynamic target {_target:.2f} "
                "but clusters may still overlap; consider different k or algorithm."
            )

        # Build doubt + merge diagnosis
        doubt_parts = []
        if _cs_all:
            # Show top-3 candidates for transparency
            _top3 = sorted(_cs_all, key=lambda c: c.get('composite_score', 0), reverse=True)[:3]
            _top3_str = " | ".join(
                f"{c['algorithm']}(k={c['k']}, comp={_fmt(c.get('composite_score'), '.3f')}, sil={_fmt(c.get('silhouette'))})"
                for c in _top3
            )
            doubt_parts.append(f"AutoML top-3: {_top3_str}.")
        elif k_scores:
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
                    + (
                        f"(AutoML-selected from {len(_cs_all)} candidates, composite={_cs_best.get('composite_score', 'N/A')}). "
                        if _cs_best else f"(auto-selected via silhouette). "
                    )
                    + f"Ran deepening loop → {n_leaf} leaf clusters. "
                    + f"Silhouette={sil:.4f} ({sil_quality})."
                    + (f" Merged {len(singleton_merges)} tiny cluster(s) "
                       f"(n<{min_cluster_size}) into nearest leaf."
                       if singleton_merges else "")
                ),
                what_was_not_done=(
                    "Did not run exhaustive hyperparameter optimisation; candidate "
                    "search is bounded by config."
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
                    "candidate_search": candidate_evidence,
                },
                recommendation="proceed" if not issues else "retry",
                context={
                    "algo_reasoning": algo_reasoning,
                    "k_scores": k_scores,
                    "singleton_merges": singleton_merges,
                    "candidate_evidence": candidate_evidence,
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
            candidate_evidence=candidate_evidence,
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
