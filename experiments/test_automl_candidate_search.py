"""
Offline test for AutoML-as-skill clustering candidate search.

Run:
    python experiments/test_automl_candidate_search.py
"""
from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from skills.automl_candidate_search import (
    candidate_search_to_dict,
    search_clustering_candidates,
)


class _StubBus:
    def __init__(self):
        self.reports = []

    def report(self, message):
        self.reports.append(message)

    def ask(self, *args, **kwargs):
        return '{"action":"subcluster","reasoning":"stub"}'


def _check(cond: bool, msg: str) -> None:
    if not cond:
        print(f"  ✗ FAIL: {msg}")
        sys.exit(1)
    print(f"  ✓ {msg}")


def main() -> None:
    print("\n[test_automl_candidate_search] Building synthetic clusters...")
    X, _ = make_blobs(
        n_samples=180,
        centers=4,
        cluster_std=0.55,
        n_features=6,
        random_state=42,
    )
    X = StandardScaler().fit_transform(X)

    result = search_clustering_candidates(
        X,
        algorithms=["kmeans", "hierarchical", "gmm"],
        k_range=[2, 3, 4, 5, 6],
        max_cluster_size_pct=0.50,
        stability_repeats=2,
        verbose=False,
    )

    _check(result.best is not None, "candidate search produced a winner")
    _check(result.best.algorithm in {"kmeans", "hierarchical", "gmm"},
           f"winner algorithm={result.best.algorithm}")
    _check(result.best.k in {3, 4, 5},
           f"winner k={result.best.k} near true structure")
    _check(result.best.silhouette > 0.4,
           f"winner silhouette={result.best.silhouette:.3f} > 0.4")
    _check(result.best.stability_ari >= 0.80,
           f"stability_ari={result.best.stability_ari:.3f} >= 0.80")
    _check(len(result.candidates) >= 3,
           f"candidate evidence contains {len(result.candidates)} rows")

    payload = candidate_search_to_dict(result)
    _check(payload["best"]["composite_score"] == result.best.composite_score,
           "serialisable payload preserves best score")
    _check("candidates" in payload and payload["candidates"],
           "serialisable payload includes candidate list")

    print("\n[2] Clusterer integration uses candidate evidence")
    import pandas as pd
    from agents.clusterer import ClusteringAgent

    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    bus = _StubBus()
    clusterer = ClusteringAgent(
        config={
            "clustering_algorithm": "auto",
            "n_clusters": None,
            "max_depth": 0,
            "max_cluster_size_pct": 0.60,
            "enable_automl_candidate_search": True,
            "candidate_search_algorithms": ["kmeans", "hierarchical", "gmm"],
            "candidate_stability_repeats": 2,
            "candidate_search_top_n": 6,
            "silhouette_target": 0.20,
        },
        bus=bus,
    )
    cr = clusterer.run(
        features_df=df,
        selected_features=list(df.columns),
        user_intent=None,
        dataset_profile=None,
        history=[],
        iteration=1,
        bypass=True,
    )
    _check(cr.action == "proceed", f"clusterer action={cr.action}")
    _check(bool(cr.candidate_evidence.get("best")),
           "ClusteringResult carries candidate_search best evidence")
    _check(cr.algo_name in {"KMeans", "AgglomerativeClustering", "GaussianMixture"},
           f"clusterer fitted winner algo={cr.algo_name}")
    _check(cr.n_leaf is not None and cr.n_leaf >= 2,
           f"clusterer produced {cr.n_leaf} leaf clusters")

    print("\n[test_automl_candidate_search] ALL CHECKS PASSED\n")


if __name__ == "__main__":
    main()
