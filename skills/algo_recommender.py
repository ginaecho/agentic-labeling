"""
Algorithm Recommender — Clustering Algorithm Selection from Data Shape

Analyses dataset size, feature distribution statistics, and business purpose
to recommend 'hierarchical' or 'kmeans' as the clustering algorithm.

Usage:
    from skills.algo_recommender import recommend_algorithm

    rec = recommend_algorithm(
        n_rows=10_000,
        n_features=45,
        feature_skewness={"travel_spend": 3.2, ...},
        business_purpose="understand customer shopping behaviour",
    )
    print(rec.algorithm, rec.reasoning)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AlgoRecommendation:
    """Output of the algorithm recommender."""

    algorithm: str
    """'hierarchical' or 'kmeans'."""

    reasoning: str
    """Human-readable explanation of the recommendation."""

    confidence: float
    """0–1 confidence score. < 0.6 means the choice is borderline."""

    factors: dict
    """Raw factor values that drove the decision."""


def recommend_algorithm(
    n_rows: int,
    n_features: int,
    feature_skewness: dict[str, float] | None = None,
    business_purpose: str = "",
    X_sample: pd.DataFrame | None = None,
    verbose: bool = True,
) -> AlgoRecommendation:
    """
    Recommend a clustering algorithm based on data characteristics.

    Decision rules (applied in priority order):
    1. n_rows > 100_000  →  K-Means  (speed critical)
    2. Mean feature skewness > 2.0  →  Hierarchical (robust after log-transform)
    3. Business purpose mentions "nested", "hierarchy", "sub-segments"  →  Hierarchical
    4. n_features > 100  →  Hierarchical (handles high-dim better with Ward)
    5. Default  →  Hierarchical (Ward)

    Parameters
    ----------
    n_rows : int
        Number of rows (entities) in the dataset.
    n_features : int
        Number of features to be used for clustering.
    feature_skewness : dict[str, float] | None
        Map of {feature_name: skewness}. If None, skewness is not considered.
    business_purpose : str
        Free-text description of business intent (checked for keywords).
    X_sample : pd.DataFrame | None
        Optional sample of the actual feature data for distribution analysis.
    verbose : bool
        Print the decision to stdout.

    Returns
    -------
    AlgoRecommendation
    """
    reasons: list[str] = []
    kmeans_score = 0      # positive → K-Means
    hierarchical_score = 0  # positive → Hierarchical

    factors: dict = {
        "n_rows": n_rows,
        "n_features": n_features,
    }

    # ── Rule 1: Dataset size ──────────────────────────────────────────────────
    if n_rows > 100_000:
        kmeans_score += 3
        reasons.append(f"n_rows={n_rows:,} > 100k → K-Means preferred for speed")
    elif n_rows < 5_000:
        hierarchical_score += 1
        reasons.append(f"n_rows={n_rows:,} is small → Hierarchical is stable")
    else:
        reasons.append(f"n_rows={n_rows:,} is medium → no strong size preference")

    # ── Rule 2: Feature skewness ──────────────────────────────────────────────
    if feature_skewness is not None:
        skew_values = list(feature_skewness.values())
        mean_skew = float(np.mean(np.abs(skew_values))) if skew_values else 0.0
        max_skew = float(np.max(np.abs(skew_values))) if skew_values else 0.0
        factors["mean_abs_skewness"] = round(mean_skew, 2)
        factors["max_abs_skewness"] = round(max_skew, 2)

        if mean_skew > 2.0:
            hierarchical_score += 2
            reasons.append(
                f"Mean |skewness|={mean_skew:.1f} > 2.0 → "
                "Hierarchical handles skewed distributions better"
            )
        elif mean_skew > 1.0:
            hierarchical_score += 1
            reasons.append(f"Mean |skewness|={mean_skew:.1f} is moderate")
    elif X_sample is not None:
        numeric = X_sample.select_dtypes(include=[np.number])
        skews = numeric.skew().abs()
        mean_skew = float(skews.mean())
        factors["mean_abs_skewness"] = round(mean_skew, 2)
        if mean_skew > 2.0:
            hierarchical_score += 2
            reasons.append(f"Mean |skewness| from sample={mean_skew:.1f} > 2.0 → Hierarchical")

    # ── Rule 3: Business purpose keywords ────────────────────────────────────
    bp_lower = business_purpose.lower()
    hierarchy_keywords = ["nested", "hierarchy", "sub-segment", "sub segment",
                          "group within", "subgroup", "tier", "level"]
    kmeans_keywords = ["simple", "fast", "basic", "broad", "high-level"]

    for kw in hierarchy_keywords:
        if kw in bp_lower:
            hierarchical_score += 2
            reasons.append(f"Business purpose contains '{kw}' → Hierarchical preferred")
            break

    for kw in kmeans_keywords:
        if kw in bp_lower:
            kmeans_score += 1
            reasons.append(f"Business purpose mentions '{kw}' → slight K-Means preference")
            break

    # ── Rule 4: High dimensionality ───────────────────────────────────────────
    if n_features > 100:
        hierarchical_score += 1
        reasons.append(f"n_features={n_features} > 100 → Hierarchical handles high-dim well")

    # ── Rule 5: Multi-modality check (if sample provided) ────────────────────
    if X_sample is not None:
        try:
            from scipy.stats import gaussian_kde
            numeric = X_sample.select_dtypes(include=[np.number])
            multimodal_count = 0
            for col in numeric.columns[:20]:  # check first 20 features
                vals = numeric[col].dropna().values
                if len(vals) < 50:
                    continue
                kde = gaussian_kde(vals)
                x_grid = np.linspace(vals.min(), vals.max(), 200)
                density = kde(x_grid)
                # Count local maxima as proxy for modes
                from scipy.signal import argrelextrema
                maxima = argrelextrema(density, np.greater, order=10)[0]
                if len(maxima) >= 2:
                    multimodal_count += 1
            factors["multimodal_features"] = multimodal_count
            if multimodal_count >= 3:
                hierarchical_score += 2
                reasons.append(
                    f"{multimodal_count} features appear multi-modal → "
                    "Hierarchical can capture sub-group structure"
                )
        except ImportError:
            pass  # scipy optional for this check

    # ── Decision ──────────────────────────────────────────────────────────────
    factors["kmeans_score"] = kmeans_score
    factors["hierarchical_score"] = hierarchical_score

    if kmeans_score > hierarchical_score:
        algorithm = "kmeans"
        margin = kmeans_score - hierarchical_score
    else:
        algorithm = "hierarchical"
        margin = hierarchical_score - kmeans_score

    total_score = kmeans_score + hierarchical_score
    confidence = min(0.5 + (margin / max(total_score, 1)) * 0.5, 1.0)

    if not reasons:
        reasons.append("No strong signal — defaulting to Hierarchical (Ward linkage)")
    reasoning = "; ".join(reasons) + f". → Recommended: {algorithm} (confidence={confidence:.2f})"

    if verbose:
        print(f"  [AlgoRecommender] → {algorithm.upper()}  (confidence={confidence:.2f})")
        for r in reasons:
            print(f"    · {r}")

    return AlgoRecommendation(
        algorithm=algorithm,
        reasoning=reasoning,
        confidence=round(confidence, 2),
        factors=factors,
    )
