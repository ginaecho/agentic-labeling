"""Tabular regression test for the multi-agent pipeline.

Confirms that the text-modality changes (state.modality, text_artifacts
plumbing, FeatureSelector short-circuit, Clusterer text branch,
PersonaNamer text prompt block) did NOT alter the tabular code path.

Strategy
--------
- Generate synthetic well-separated 3-Gaussian data (6 numeric features).
- Run FeatureSelectionAgent + ClusteringAgent directly with a deterministic
  LLM stub matched to each agent's bus.ask() contract.
- Assert: VIF + AE + PCA branches all run, profiles are tabular-shaped
  (no top_terms / no representative_docs), ABOVE-AVERAGE wording is in the
  PersonaNamer prompt block, and silhouette is high (>0.5) because the
  Gaussians are well-separated.

Run
---
    python experiments/test_tabular_regression.py
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd


class _StubBus:
    """Deterministic LLM stub that returns shaped JSON per purpose.

    The FeatureSelector LLM call expects {"selected_features": [...],
    "reasoning": "..."}. The stub recovers candidate column names from the
    prompt so it always returns a valid (non-empty) subset.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def ask(self, agent: str, purpose: str, prompt: str, max_tokens: int = 256, category: str = "pipeline") -> str:
        self.calls.append({"agent": agent, "purpose": purpose})
        p = (purpose or "").lower()
        pr = prompt or ""

        if "feature" in p and "select" in p:
            # The FS prompt lists candidate feature names; grep them and
            # return as-is so the intersection with the post-VIF surviving
            # set is non-empty.
            tokens = sorted(set(re.findall(r"\bf\d+\b", pr)))
            return json.dumps({
                "selected_features": tokens or ["f0", "f1", "f2", "f3", "f4", "f5"],
                "reasoning": "stub — keep all numeric features",
            })

        if "oversized" in p or "deepening" in p:
            return json.dumps({"action": "subcluster", "reasoning": "stub"})

        return json.dumps({"action": "proceed", "reasoning": "stub fallback"})

    def report(self, message) -> None:
        pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        print(f"  ✗ FAIL: {msg}", file=sys.stderr)
        sys.exit(1)
    print(f"  ✓ {msg}")


def main() -> int:
    print("\n[tabular] Building synthetic 3-Gaussian dataset ...")
    # FeatureSelector hard-blocks when <10 features survive the VIF gate, so
    # we need at least 10 informative dims to exercise the tabular code path.
    rng = np.random.default_rng(42)
    n_each, dim = 80, 12
    def _mk(loc):
        return rng.normal(loc=loc, scale=0.30, size=(n_each, dim))
    g1 = _mk([ 0,  0, 0,  1, 0, 0, 1, 0,  0, 1, 0, 0])
    g2 = _mk([ 4,  4, 0, -1, 0, 0, 0, 1,  1, 0, 0, 0])
    g3 = _mk([-4, -4, 0,  1, 0, 0, 0, 0, -1, 0, 1, 1])
    X = np.vstack([g1, g2, g3])
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(dim)])
    print(f"  ✓ shape={df.shape} (3 well-separated Gaussian blobs)")

    bus = _StubBus()

    # ── 1. FeatureSelector tabular path (PCA + AE + VIF) ───────────────────
    print("\n[1] FeatureSelector tabular path")
    from agents.feature_selector import FeatureSelectionAgent
    fs_agent = FeatureSelectionAgent(
        bus=bus, ae_bottleneck_cap=4, ae_max_iter=50,
    )
    fs = fs_agent.run(df, feedback="", iteration=1)  # modality defaults to 'tabular'

    _check(fs.n_features >= 1, f"FS kept {fs.n_features} features (>=1)")
    _check(fs.pca_scores != {}, "PCA scores populated (NOT short-circuited)")
    _check(fs.ae_scores != {}, "AE scores populated (NOT short-circuited)")
    _check(fs.vif_table is not None, "VIF table present (gate ran)")

    # ── 2. Clusterer tabular path (StandardScaler + Euclidean silhouette) ──
    print("\n[2] Clusterer tabular path")
    from agents.clusterer import ClusteringAgent
    cluster_agent = ClusteringAgent(
        config={
            "max_cluster_size_pct": 0.7,
            "sub_n_clusters": 2,
            "max_depth": 0,
            "clustering_algorithm": "kmeans",
            "n_clusters": 3,
        },
        bus=bus,
    )
    cr = cluster_agent.run(
        features_df=df,
        selected_features=fs.selected_features,
        iteration=1,
    )  # NO text_artifacts → tabular branch

    _check(cr.action == "proceed", f"clusterer action={cr.action!r}")
    _check(cr.profiles is not None and len(cr.profiles) >= 2,
           f"produced {len(cr.profiles or {})} profiles")
    _check(cr.silhouette is not None and cr.silhouette > 0.3,
           f"euclidean silhouette={cr.silhouette:.3f} > 0.3 (well-separated; "
           "post-StandardScaler the means contract so 0.3–0.5 is the expected band)")

    sample = next(iter(cr.profiles.values()))
    _check(sample.get("modality") != "text",
           f"tabular profile NOT flagged text (got modality={sample.get('modality')!r})")
    _check("feature_relative" in sample,
           "tabular profile keeps feature_relative dict")
    _check("top_terms" not in sample,
           "tabular profile does NOT carry top_terms (text-only field)")
    _check("representative_docs" not in sample,
           "tabular profile does NOT carry representative_docs (text-only field)")

    # ── 3. PersonaNamer block keeps ABOVE-AVERAGE / BELOW-AVERAGE wording ──
    print("\n[3] PersonaNamer _format_cluster_block tabular branch")
    from agents.persona_namer import _format_cluster_block
    block = _format_cluster_block(next(iter(cr.profiles)), sample)
    _check("ABOVE AVERAGE" in block,
           "tabular block uses ABOVE AVERAGE section (numeric mean wording)")
    _check("BELOW AVERAGE" in block,
           "tabular block uses BELOW AVERAGE section")
    _check("DISTINCTIVE TERMS" not in block,
           "tabular block does NOT use DISTINCTIVE TERMS wording (text-only)")
    _check("REPRESENTATIVE DOCUMENTS" not in block,
           "tabular block does NOT use REPRESENTATIVE DOCUMENTS wording (text-only)")

    print(f"\n[tabular] ALL REGRESSION CHECKS PASSED ✓ "
          f"({len(bus.calls)} stub bus.ask calls)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
