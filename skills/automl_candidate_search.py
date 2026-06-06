"""
AutoML Candidate Search — model-selection as a reusable clustering skill.

This is intentionally a bounded, deterministic tournament rather than a new
pipeline. Agents call it when clustering is set to auto; the skill evaluates
multiple algorithm/k candidates, adds bootstrap stability evidence, and returns
the strongest candidate for the existing Clusterer to fit and profile.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CandidateResult:
    algorithm: str
    k: int | None
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    stability_ari: float
    max_cluster_pct: float
    n_clusters_found: int
    composite_score: float
    evidence: dict = field(default_factory=dict)


@dataclass
class CandidateSearchResult:
    best: CandidateResult | None
    candidates: list[CandidateResult]
    reasoning: str


def _fit_predict(algorithm: str, X: np.ndarray, k: int | None,
                 random_state: int = 42) -> np.ndarray:
    if algorithm == "kmeans":
        from sklearn.cluster import KMeans
        return KMeans(
            n_clusters=int(k), random_state=random_state, n_init=10, max_iter=300
        ).fit_predict(X)
    if algorithm == "hierarchical":
        from sklearn.cluster import AgglomerativeClustering
        return AgglomerativeClustering(n_clusters=int(k), linkage="ward").fit_predict(X)
    if algorithm == "gmm":
        from sklearn.mixture import GaussianMixture
        return GaussianMixture(
            n_components=int(k), random_state=random_state, n_init=3
        ).fit_predict(X)
    if algorithm == "dbscan":
        from sklearn.cluster import DBSCAN
        min_samples = max(5, X.shape[1] // 10)
        labels = DBSCAN(eps=0.5, min_samples=min_samples).fit_predict(X)
        if (labels == -1).any():
            max_label = int(labels.max()) if labels.max() >= 0 else -1
            labels = labels.copy()
            labels[labels == -1] = max_label + 1
        return labels
    raise ValueError(f"Unsupported candidate algorithm: {algorithm!r}")


def _valid_labels(labels: np.ndarray, n_samples: int) -> bool:
    n_unique = len(set(int(x) for x in labels))
    return 2 <= n_unique < n_samples


def _max_cluster_pct(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / max(1, len(labels)))


def _stability_ari(algorithm: str, X: np.ndarray, full_labels: np.ndarray,
                   k: int | None, metric: str, repeats: int,
                   sample_frac: float, random_state: int) -> float:
    from sklearn.metrics import adjusted_rand_score

    if repeats <= 0 or len(X) < 10:
        return 0.0

    rng = np.random.default_rng(random_state)
    scores: list[float] = []
    n_sub = max(5, int(len(X) * sample_frac))
    n_sub = min(n_sub, len(X) - 1)

    for i in range(repeats):
        idx = np.sort(rng.choice(len(X), size=n_sub, replace=False))
        try:
            sub_labels = _fit_predict(algorithm, X[idx], k, random_state + i + 1)
        except Exception:
            continue
        if not _valid_labels(sub_labels, len(sub_labels)):
            continue
        scores.append(float(adjusted_rand_score(full_labels[idx], sub_labels)))

    if not scores:
        return 0.0
    return round(float(np.mean(scores)), 4)


def search_clustering_candidates(
    X: np.ndarray,
    algorithms: list[str],
    k_range: list[int],
    metric: str = "euclidean",
    max_cluster_size_pct: float = 0.40,
    random_state: int = 42,
    stability_repeats: int = 3,
    stability_sample_frac: float = 0.80,
    top_n: int = 8,
    verbose: bool = True,
) -> CandidateSearchResult:
    """Evaluate clustering candidates and return the best one.

    Composite score rewards separation and repeatability, then penalises
    oversized clusters. This gives agents AutoML-like search while optimising
    for usable segmentation, not just a single metric.
    """
    from sklearn.metrics import silhouette_score

    n_samples = int(X.shape[0])
    valid_k = [int(k) for k in sorted(set(k_range)) if 2 <= int(k) < n_samples]
    if not valid_k:
        return CandidateSearchResult(
            best=None, candidates=[],
            reasoning="No valid k values for candidate search.",
        )

    # Keep the tournament bounded. Hierarchical is costly on large datasets.
    algo_list = []
    for algo in algorithms:
        algo = str(algo).lower()
        if algo == "hierarchical" and n_samples > 10000:
            continue
        if algo in {"kmeans", "hierarchical", "gmm", "dbscan"} and algo not in algo_list:
            algo_list.append(algo)
    if not algo_list:
        algo_list = ["kmeans"]

    candidates: list[CandidateResult] = []
    if verbose:
        print(
            f"  [AutoMLCandidateSearch] algorithms={algo_list}; "
            f"k_range={valid_k}; metric={metric}"
        )

    for algo in algo_list:
        ks = [None] if algo == "dbscan" else valid_k
        for k in ks:
            try:
                labels = _fit_predict(algo, X, k, random_state=random_state)
            except Exception as exc:
                if verbose:
                    print(f"    {algo} k={k}: failed ({exc})")
                continue
            if not _valid_labels(labels, n_samples):
                continue

            try:
                sil = float(silhouette_score(X, labels, metric=metric))
            except Exception:
                continue

            # Multi-metric: Davies-Bouldin (lower=better) + Calinski-Harabasz (higher=better)
            db = ch = None
            try:
                from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
                db = float(davies_bouldin_score(X, labels))
                ch = float(calinski_harabasz_score(X, labels))
            except Exception:
                pass

            max_pct = _max_cluster_pct(labels)
            stability = _stability_ari(
                algo, X, labels, k, metric=metric,
                repeats=stability_repeats,
                sample_frac=stability_sample_frac,
                random_state=random_state,
            )
            oversize_penalty = max(0.0, max_pct - max_cluster_size_pct) * 50.0

            # Normalise DB and CH to [0, 1] across this single candidate
            # so they don't overwhelm silhouette. DB ≈ 0.5-1.5 is typical good;
            # CH scales with n_samples so we cap it at a generous ceiling.
            db_norm = max(0.0, 1.0 - (db / 3.0)) if db is not None else 0.5
            ch_norm = min(1.0, ch / 500.0) if ch is not None else 0.5

            # Composite: silhouette primary (0.45), stability (0.25), DB (0.15), CH (0.15)
            score = (
                max(0.0, sil) * 45.0
                + stability * 25.0
                + db_norm * 15.0
                + ch_norm * 15.0
                - oversize_penalty
            )
            n_found = len(set(int(x) for x in labels))
            result = CandidateResult(
                algorithm=algo,
                k=int(k) if k is not None else None,
                silhouette=round(sil, 4),
                davies_bouldin=round(db, 4) if db is not None else None,
                calinski_harabasz=round(ch, 2) if ch is not None else None,
                stability_ari=round(stability, 4),
                max_cluster_pct=round(max_pct, 4),
                n_clusters_found=n_found,
                composite_score=round(float(score), 4),
                evidence={
                    "oversize_penalty": round(oversize_penalty, 4),
                    "metric": metric,
                    "stability_repeats": stability_repeats,
                    "stability_sample_frac": stability_sample_frac,
                },
            )
            candidates.append(result)
            if verbose:
                db_str = f"DB={db:.3f}" if db is not None else "DB=n/a"
                ch_str = f"CH={ch:.1f}" if ch is not None else "CH=n/a"
                print(
                    f"    {algo:<12} k={str(k):>4}  sil={sil:.4f}  "
                    f"{db_str}  {ch_str}  stability_ari={stability:.4f}  "
                    f"max_pct={max_pct:.2f}  score={score:.2f}"
                )

    candidates.sort(key=lambda c: c.composite_score, reverse=True)
    candidates = candidates[:max(1, int(top_n))]
    best = candidates[0] if candidates else None
    if best is None:
        return CandidateSearchResult(
            best=None, candidates=[],
            reasoning="No candidate produced at least two valid clusters.",
        )

    reasoning = (
        f"Selected {best.algorithm}"
        f"{'' if best.k is None else f' k={best.k}'} by candidate tournament: "
        f"silhouette={best.silhouette:.4f}, "
        f"stability_ari={best.stability_ari:.4f}, "
        f"max_cluster_pct={best.max_cluster_pct:.2f}, "
        f"score={best.composite_score:.2f}."
    )
    return CandidateSearchResult(best=best, candidates=candidates, reasoning=reasoning)


def candidate_search_to_dict(result: CandidateSearchResult) -> dict:
    """JSON-serialisable evidence payload for agent reports and run history."""
    def one(c: CandidateResult) -> dict:
        return {
            "algorithm": c.algorithm,
            "k": c.k,
            "silhouette": c.silhouette,
            "davies_bouldin": c.davies_bouldin,
            "calinski_harabasz": c.calinski_harabasz,
            "stability_ari": c.stability_ari,
            "max_cluster_pct": c.max_cluster_pct,
            "n_clusters_found": c.n_clusters_found,
            "composite_score": c.composite_score,
            "evidence": c.evidence,
        }

    return {
        "best": one(result.best) if result.best else None,
        "candidates": [one(c) for c in result.candidates],
        "reasoning": result.reasoning,
    }
