"""
Silhouette Optimizer — Data-Driven Cluster Count Selection

Tries a range of k values, computes the silhouette score for each, and
returns the optimal k. Also records the elbow-method inertia for K-Means
as a secondary signal.

Usage:
    from skills.silhouette_optimizer import optimize_k

    result = optimize_k(
        X_scaled,
        algorithm="hierarchical",
        k_range=[3, 4, 5, 6, 7, 8, 10, 12, 15],
    )
    print(result.best_k, result.best_silhouette)
    print(result.scores)   # {k: silhouette_score}
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class SilhouetteResult:
    """Result of the silhouette optimisation over a range of k."""

    best_k: int
    """k value with the highest silhouette score."""

    best_silhouette: float
    """Silhouette score at best_k."""

    scores: dict[int, float]
    """Silhouette score for every k tried. {k: score}"""

    inertias: dict[int, float]
    """K-Means inertia for every k tried (empty for hierarchical). {k: inertia}"""

    algorithm: str
    """Algorithm used ('hierarchical' or 'kmeans')."""

    reasoning: str = ""
    """Human-readable explanation of the selection."""

    warning: str = ""
    """Non-empty if silhouette is below a quality threshold."""


def optimize_k(
    X_scaled: np.ndarray,
    algorithm: str = "hierarchical",
    k_range: list[int] | None = None,
    random_state: int = 42,
    min_silhouette_warn: float = 0.20,
    verbose: bool = True,
) -> SilhouetteResult:
    """
    Try each k in k_range, fit the clustering algorithm, compute silhouette,
    and return the k that maximises it.

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

    Returns
    -------
    SilhouetteResult
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    if k_range is None:
        k_range = [3, 4, 5, 6, 7, 8, 10, 12, 15]

    # Remove k values that exceed sample size
    n_samples = X_scaled.shape[0]
    k_range = [k for k in sorted(set(k_range)) if 2 <= k < n_samples]

    if not k_range:
        return SilhouetteResult(
            best_k=2,
            best_silhouette=0.0,
            scores={},
            inertias={},
            algorithm=algorithm,
            reasoning="No valid k values in range.",
            warning="k_range empty after filtering.",
        )

    if verbose:
        print(f"  [SilhouetteOptimizer] Trying k ∈ {k_range} with algorithm={algorithm}")

    scores: dict[int, float] = {}
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

        # silhouette_score needs at least 2 distinct labels
        n_unique = len(set(labels))
        if n_unique < 2:
            scores[k] = -1.0
            continue

        sil = float(silhouette_score(X_scaled, labels))
        scores[k] = round(sil, 4)
        if verbose:
            print(f"    k={k:>3}  silhouette={sil:.4f}")

    if not scores:
        return SilhouetteResult(
            best_k=k_range[0],
            best_silhouette=0.0,
            scores=scores,
            inertias=inertias,
            algorithm=algorithm,
            reasoning="Could not compute silhouette for any k.",
            warning="All k attempts produced < 2 distinct clusters.",
        )

    best_k = max(scores, key=lambda k: scores[k])
    best_sil = scores[best_k]

    # Build reasoning string
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    top3 = ", ".join(f"k={k}({s:.3f})" for k, s in sorted_scores[:3])
    reasoning = (
        f"Tried {len(k_range)} values of k. "
        f"Top-3 silhouette: {top3}. "
        f"Selected k={best_k} with silhouette={best_sil:.4f}."
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
        print(f"  [SilhouetteOptimizer] Best k={best_k}  silhouette={best_sil:.4f}")

    return SilhouetteResult(
        best_k=best_k,
        best_silhouette=best_sil,
        scores=scores,
        inertias=inertias,
        algorithm=algorithm,
        reasoning=reasoning,
        warning=warning,
    )
