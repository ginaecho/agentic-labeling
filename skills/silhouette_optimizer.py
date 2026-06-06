"""
Silhouette Optimizer — Data-Driven Cluster Count Selection

Tries a range of k values, computes multiple internal validation metrics
(silhouette, Davies-Bouldin, Calinski-Harabasz, elbow), and returns the
optimal k by a composite score rather than a single metric.

Usage:
    from skills.silhouette_optimizer import optimize_k

    result = optimize_k(
        X_scaled,
        algorithm="hierarchical",
        k_range=[3, 4, 5, 6, 7, 8, 10, 12, 15],
    )
    print(result.best_k, result.best_silhouette)
    print(result.scores)           # {k: silhouette_score}
    print(result.db_scores)        # {k: davies_bouldin}
    print(result.ch_scores)        # {k: calinski_harabasz}
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class SilhouetteResult:
    """Result of the multi-metric optimisation over a range of k."""

    best_k: int
    """k value with the highest composite score."""

    best_silhouette: float
    """Silhouette score at best_k."""

    best_db: float
    """Davies-Bouldin index at best_k (lower is better)."""

    best_ch: float
    """Calinski-Harabasz index at best_k (higher is better)."""

    best_composite: float
    """Composite score at best_k (higher is better)."""

    scores: dict[int, float]
    """Silhouette score for every k tried. {k: score}"""

    db_scores: dict[int, float]
    """Davies-Bouldin for every k tried. {k: db} — lower is better."""

    ch_scores: dict[int, float]
    """Calinski-Harabasz for every k tried. {k: ch} — higher is better."""

    inertias: dict[int, float]
    """K-Means inertia for every k tried (empty for hierarchical). {k: inertia}"""

    algorithm: str
    """Algorithm used ('hierarchical' or 'kmeans')."""

    reasoning: str = ""
    """Human-readable explanation of the selection."""

    warning: str = ""
    """Non-empty if quality is below threshold."""


def _normalize_dict(d: dict[int, float], higher_is_better: bool = True) -> dict[int, float]:
    """Min-max normalise a dict to [0, 1]. Returns all 0.5 if constant."""
    if not d:
        return {}
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 0.5 for k in d}
    if higher_is_better:
        return {k: (v - lo) / (hi - lo) for k, v in d.items()}
    return {k: (hi - v) / (hi - lo) for k, v in d.items()}


def _compute_elbow_score(inertias: dict[int, float]) -> dict[int, float]:
    """Return an elbow-score [0,1] for each k based on inertia curvature.
    Higher = stronger elbow (better k)."""
    if len(inertias) < 3:
        return {k: 0.5 for k in inertias}
    ks = sorted(inertias.keys())
    # Second-difference of log(inertia) approximates curvature
    log_i = [np.log(inertias[k]) if inertias[k] > 0 else 0.0 for k in ks]
    scores = {}
    for i, k in enumerate(ks):
        if i == 0 or i == len(ks) - 1:
            scores[k] = 0.0
            continue
        curv = log_i[i - 1] - 2 * log_i[i] + log_i[i + 1]
        scores[k] = max(0.0, curv)
    # Normalise to [0, 1]
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 0.5 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def optimize_k(
    X_scaled: np.ndarray,
    algorithm: str = "hierarchical",
    k_range: list[int] | None = None,
    random_state: int = 42,
    min_silhouette_warn: float = 0.20,
    verbose: bool = True,
    metric: str = "euclidean",
) -> SilhouetteResult:
    """
    Try each k in k_range, fit the clustering algorithm, compute multiple
    internal validation metrics, and return the k that maximises the
    composite score (silhouette + inverted Davies-Bouldin + normalised
    Calinski-Harabasz + elbow curvature).

    Parameters
    ----------
    X_scaled : np.ndarray
        Pre-processed (scaled) feature matrix.
    algorithm : str
        'hierarchical' (Ward) or 'kmeans'.
    k_range : list[int] | None
        k values to try. Default: [3, 4, 5, 6, 7, 8, 10, 12, 15].
    random_state : int
        Random seed for K-Means.
    min_silhouette_warn : float
        If best silhouette < this, set a warning in the result.
    verbose : bool
        Print progress.
    metric : str
        Distance metric for silhouette_score.

    Returns
    -------
    SilhouetteResult
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    if k_range is None:
        k_range = [3, 4, 5, 6, 7, 8, 10, 12, 15]

    n_samples = X_scaled.shape[0]
    k_range = [k for k in sorted(set(k_range)) if 2 <= k < n_samples]

    if not k_range:
        return SilhouetteResult(
            best_k=2,
            best_silhouette=0.0,
            best_db=0.0,
            best_ch=0.0,
            best_composite=0.0,
            scores={},
            db_scores={},
            ch_scores={},
            inertias={},
            algorithm=algorithm,
            reasoning="No valid k values in range.",
            warning="k_range empty after filtering.",
        )

    if verbose:
        print(f"  [SilhouetteOptimizer] Trying k ∈ {k_range} with algorithm={algorithm}")

    scores: dict[int, float] = {}
    db_scores: dict[int, float] = {}
    ch_scores: dict[int, float] = {}
    inertias: dict[int, float] = {}

    for k in k_range:
        if algorithm == "kmeans":
            model = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
            labels = model.fit_predict(X_scaled)
            inertias[k] = float(model.inertia_)
        elif algorithm == "hierarchical":
            model = AgglomerativeClustering(n_clusters=k, linkage="ward")
            labels = model.fit_predict(X_scaled)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm!r}. Use 'kmeans' or 'hierarchical'.")

        n_unique = len(set(labels))
        if n_unique < 2:
            scores[k] = -1.0
            db_scores[k] = 999.0
            ch_scores[k] = 0.0
            continue

        sil = float(silhouette_score(X_scaled, labels, metric=metric))
        db = float(davies_bouldin_score(X_scaled, labels))
        ch = float(calinski_harabasz_score(X_scaled, labels))
        scores[k] = round(sil, 4)
        db_scores[k] = round(db, 4)
        ch_scores[k] = round(ch, 2)
        if verbose:
            print(f"    k={k:>3}  sil={sil:.4f}  DB={db:.4f}  CH={ch:.1f}")

    if not scores:
        return SilhouetteResult(
            best_k=k_range[0],
            best_silhouette=0.0,
            best_db=0.0,
            best_ch=0.0,
            best_composite=0.0,
            scores=scores,
            db_scores=db_scores,
            ch_scores=ch_scores,
            inertias=inertias,
            algorithm=algorithm,
            reasoning="Could not compute metrics for any k.",
            warning="All k attempts produced < 2 distinct clusters.",
        )

    # Normalise each metric to [0, 1] so they are comparable
    norm_sil = _normalize_dict(scores, higher_is_better=True)
    norm_db = _normalize_dict(db_scores, higher_is_better=False)
    norm_ch = _normalize_dict(ch_scores, higher_is_better=True)
    norm_elbow = _compute_elbow_score(inertias) if inertias else {k: 0.5 for k in scores}

    # Composite: silhouette is primary (0.40), DB + CH each 0.25, elbow 0.10
    composite: dict[int, float] = {}
    for k in scores:
        composite[k] = (
            0.40 * norm_sil.get(k, 0.0)
            + 0.25 * norm_db.get(k, 0.0)
            + 0.25 * norm_ch.get(k, 0.0)
            + 0.10 * norm_elbow.get(k, 0.0)
        )

    best_k = max(composite, key=lambda k: composite[k])
    best_sil = scores[best_k]
    best_db = db_scores[best_k]
    best_ch = ch_scores[best_k]
    best_comp = composite[best_k]

    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    top3 = ", ".join(f"k={k}({s:.3f})" for k, s in sorted_scores[:3])
    reasoning = (
        f"Tried {len(k_range)} values of k. "
        f"Top-3 silhouette: {top3}. "
        f"Selected k={best_k} by composite score={best_comp:.3f} "
        f"(sil={best_sil:.3f}, DB={best_db:.3f}, CH={best_ch:.1f})."
    )

    warning = ""
    if best_sil < min_silhouette_warn:
        warning = (
            f"Best silhouette ({best_sil:.3f}) is below the quality threshold "
            f"({min_silhouette_warn}). Clusters may not be well-separated. "
            "Consider different features or a different algorithm."
        )
        if verbose:
            print(f"  [SilhouetteOptimizer] WARNING: {warning}")

    if verbose:
        print(
            f"  [SilhouetteOptimizer] Best k={best_k}  "
            f"composite={best_comp:.3f}  sil={best_sil:.4f}  "
            f"DB={best_db:.4f}  CH={best_ch:.1f}"
        )

    return SilhouetteResult(
        best_k=best_k,
        best_silhouette=best_sil,
        best_db=best_db,
        best_ch=best_ch,
        best_composite=best_comp,
        scores=scores,
        db_scores=db_scores,
        ch_scores=ch_scores,
        inertias=inertias,
        algorithm=algorithm,
        reasoning=reasoning,
        warning=warning,
    )
