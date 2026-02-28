"""
VIF Checker — Multicollinearity & Feature Quality Gates

Implements:
  compute_vif()         — compute VIF for each column of a DataFrame
  remove_high_vif()     — iteratively drop highest-VIF column until all VIF < threshold
  flag_high_correlation() — find feature pairs with |r| above a threshold

Reference:
  https://medium.com/@rasdhar.panchal/feature-selection-using-p-values-and-vif-in-linear-regression-6bf25b652d99

VIF interpretation:
  VIF = 1          → no correlation with other features
  VIF 1 – 5        → moderate, generally acceptable
  VIF 5 – 10       → high, consider removing
  VIF > 10         → severe collinearity, should remove
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for each numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain only numeric columns (no NaNs).

    Returns
    -------
    pd.DataFrame with columns ['feature', 'vif'], sorted descending by vif.
    """
    from numpy.linalg import lstsq

    X = df.select_dtypes(include=[np.number]).dropna(axis=1).copy()
    cols = list(X.columns)
    n = len(cols)

    if n < 2:
        return pd.DataFrame({"feature": cols, "vif": [np.nan] * n})

    X_arr = X.values.astype(float)
    vifs = []

    for i, col in enumerate(cols):
        y = X_arr[:, i]
        # X_other = all columns except i, plus intercept
        other_idx = [j for j in range(n) if j != i]
        X_other = np.column_stack([X_arr[:, other_idx], np.ones(len(y))])

        # R² of regressing col on all others
        coeffs, _, _, _ = lstsq(X_other, y, rcond=None)
        y_hat = X_other @ coeffs
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)

        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
        vifs.append(vif)

    result = pd.DataFrame({"feature": cols, "vif": vifs})
    return result.sort_values("vif", ascending=False).reset_index(drop=True)


def remove_high_vif(
    df: pd.DataFrame,
    threshold: float = 5.0,
    max_iterations: int = 100,
    min_features: int = 5,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Iteratively remove the feature with the highest VIF until all VIF < threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Numeric feature DataFrame.
    threshold : float
        Maximum allowable VIF. Default 5.0.
    max_iterations : int
        Safety limit on removal rounds.
    min_features : int
        Stop removing even if VIF > threshold when this many features remain.
    verbose : bool
        Print which features are removed.

    Returns
    -------
    (cleaned_df, removed_features)
        cleaned_df  : DataFrame with high-VIF features removed.
        removed_features : list of removed column names in removal order.
    """
    work = df.select_dtypes(include=[np.number]).dropna(axis=1).copy()
    removed: list[str] = []

    for _round in range(max_iterations):
        if work.shape[1] <= min_features:
            if verbose:
                print(f"  [VIF] Stopped: only {work.shape[1]} features remain (min={min_features}).")
            break

        vif_df = compute_vif(work)
        max_vif = vif_df["vif"].max()

        if max_vif <= threshold or np.isinf(max_vif) and vif_df.shape[0] <= min_features:
            if verbose:
                print(f"  [VIF] All features have VIF ≤ {threshold:.1f}  (max={max_vif:.2f}).")
            break

        worst_feat = vif_df.iloc[0]["feature"]
        if verbose:
            print(f"  [VIF] Round {_round+1}: removing '{worst_feat}'  VIF={max_vif:.2f}")
        removed.append(worst_feat)
        work = work.drop(columns=[worst_feat])

    else:
        if verbose:
            print(f"  [VIF] Warning: max_iterations ({max_iterations}) reached.")

    return work, removed


def flag_high_correlation(
    df: pd.DataFrame,
    threshold: float = 0.85,
) -> list[tuple[str, str, float]]:
    """
    Find all feature pairs with absolute Pearson correlation above threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Numeric feature DataFrame.
    threshold : float
        |r| threshold. Default 0.85.

    Returns
    -------
    List of (feature_a, feature_b, correlation) tuples, sorted by |r| descending.
    """
    X = df.select_dtypes(include=[np.number]).dropna(axis=1)
    corr = X.corr().abs()
    cols = list(corr.columns)
    pairs = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if r >= threshold:
                pairs.append((cols[i], cols[j], float(round(r, 4))))

    pairs.sort(key=lambda x: -x[2])
    return pairs


def compute_pvalue_scores(
    df: pd.DataFrame,
    target: pd.Series,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute ANOVA F-statistic and p-value for each feature vs. a categorical target.
    Useful for filtering features that have no signal w.r.t. the cluster labels.

    Parameters
    ----------
    df : pd.DataFrame
        Numeric features.
    target : pd.Series
        Categorical labels (e.g. cluster ids).
    alpha : float
        Significance level for flagging.

    Returns
    -------
    pd.DataFrame with columns: feature, f_stat, p_value, significant
    """
    from scipy import stats

    results = []
    classes = target.unique()

    for col in df.select_dtypes(include=[np.number]).columns:
        groups = [df.loc[target == c, col].dropna().values for c in classes]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            results.append((col, np.nan, np.nan, False))
            continue
        f_stat, p_val = stats.f_oneway(*groups)
        results.append((col, float(f_stat), float(p_val), p_val < alpha))

    out = pd.DataFrame(results, columns=["feature", "f_stat", "p_value", "significant"])
    return out.sort_values("p_value").reset_index(drop=True)
