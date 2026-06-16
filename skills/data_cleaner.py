"""
Data Cleaner — Column Sanitisation for Wide / Sparse Feature Matrices

Contract: docs/skills/data_cleaner.md.

Prunes columns that carry no usable signal for clustering BEFORE the heavy
math (scaling, PCA, autoencoder, VIF/OLS) ever sees them:

  - duplicate columns          — identical column name kept once
  - all-null columns           — 100% missing
  - mostly-null columns        — missing fraction > max_null_frac
  - constant / zero-variance   — a single unique non-null value

and (optionally) imputes the remaining gaps so downstream estimators that
reject NaN (StandardScaler, PCA, MLPRegressor, the VIF/OLS gate) don't crash.

Why this exists
───────────────
A raw "feature" table exported from an upstream system often arrives wide and
sparse — e.g. 150+ columns where 60+ are ~99% null and 50+ are constant. Feeding
that straight into the pipeline either:
  * crashes StandardScaler/PCA on NaN, or
  * stalls the VIF gate on rank-deficient, all-zero columns.
Dropping these columns first is faster, avoids the crash/stall, and produces
cleaner clusters because the surviving columns actually vary across entities.

The skill is pure (no I/O, no LLM). It returns the cleaned frame plus a
structured report the Orchestrator surfaces on the bus / Evidence tab so the
user can see exactly which columns were removed and why.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def drop_low_value_columns(
    df: pd.DataFrame,
    *,
    max_null_frac: float = 0.5,
    drop_constant: bool = True,
    drop_duplicate: bool = True,
    protect_cols: tuple[str, ...] | list[str] = (),
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Drop columns that carry no usable clustering signal.

    Parameters
    ----------
    df : pd.DataFrame
        The raw table (entity-level feature matrix OR raw event table).
    max_null_frac : float
        Drop any column whose missing fraction is strictly greater than this.
        0.5 → drop columns that are more than half empty. Set to 1.0 to drop
        only fully-empty (100% null) columns.
    drop_constant : bool
        Drop columns with a single unique non-null value (zero variance).
    drop_duplicate : bool
        Drop duplicate column *names* (keep the first occurrence). Always run
        before the constant check so `df[col]` is unambiguous.
    protect_cols : list[str]
        Column names that must never be dropped (e.g. the entity id `_row_id`,
        or the text column for text modality).
    verbose : bool
        Print a one-line summary of what was removed.

    Returns
    -------
    (cleaned_df, report)
        cleaned_df : DataFrame with the low-value columns removed.
        report     : dict describing exactly what was dropped (see below).
    """
    protect = {str(c) for c in protect_cols}
    report: dict = {
        "n_cols_before": int(df.shape[1]),
        "n_rows": int(df.shape[0]),
        "max_null_frac": float(max_null_frac),
        "dropped_duplicate": [],
        "dropped_all_null": [],
        "dropped_high_null": [],   # list of [name, null_fraction]
        "dropped_constant": [],
    }

    work = df

    # ── 1. Duplicate column names (keep first) ────────────────────────────────
    if drop_duplicate:
        dup_mask = work.columns.duplicated()
        if dup_mask.any():
            report["dropped_duplicate"] = [str(c) for c in work.columns[dup_mask]]
            work = work.loc[:, ~dup_mask]

    null_frac = work.isna().mean()
    drop_set: set = set()

    # ── 2. All-null and mostly-null columns ───────────────────────────────────
    for col in work.columns:
        if str(col) in protect:
            continue
        frac = float(null_frac[col])
        if frac >= 1.0:
            report["dropped_all_null"].append(str(col))
            drop_set.add(col)
        elif frac > max_null_frac:
            report["dropped_high_null"].append([str(col), round(frac, 4)])
            drop_set.add(col)

    # ── 3. Constant / zero-variance columns (among survivors) ─────────────────
    if drop_constant:
        for col in work.columns:
            if col in drop_set or str(col) in protect:
                continue
            # nunique(dropna=True) <= 1 → all non-null values are identical
            # (also catches columns that are entirely null, already caught above).
            if work[col].nunique(dropna=True) <= 1:
                report["dropped_constant"].append(str(col))
                drop_set.add(col)

    if drop_set:
        work = work.drop(columns=[c for c in work.columns if c in drop_set])

    report["n_cols_after"] = int(work.shape[1])
    report["n_dropped"] = report["n_cols_before"] - report["n_cols_after"]

    if verbose and report["n_dropped"] > 0:
        print(
            f"  [DataCleaner] {report['n_cols_before']} → {report['n_cols_after']} columns "
            f"(dropped {len(report['dropped_all_null'])} all-null, "
            f"{len(report['dropped_high_null'])} >{max_null_frac:.0%}-null, "
            f"{len(report['dropped_constant'])} constant, "
            f"{len(report['dropped_duplicate'])} duplicate)."
        )

    return work, report


def impute_missing(
    df: pd.DataFrame,
    *,
    strategy: str = "median",
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Fill remaining NaNs in numeric columns so NaN-intolerant estimators
    (StandardScaler, PCA, MLPRegressor, the VIF/OLS gate) don't crash.

    Non-numeric columns are left untouched — categorical handling belongs to
    the agents that consume them (FeatureEngineer / TextPreparer).

    Parameters
    ----------
    strategy : {"median", "mean", "zero"}
        How to fill numeric gaps. "median" is robust to the skew that is common
        in these feature tables. If the chosen statistic is itself NaN (a column
        that is entirely null), 0.0 is used.

    Returns
    -------
    (filled_df, report) where report['imputed'] maps column → fill value.
    """
    report: dict = {"strategy": strategy, "imputed": {}, "n_inf_replaced": 0}
    work = df.copy()
    num_cols = work.select_dtypes(include=[np.number]).columns

    # ±inf (e.g. divide-by-zero ratios) is just as fatal to the scaler/PCA as
    # NaN — treat it as missing so it is imputed alongside the genuine gaps.
    if num_cols.size:
        inf_mask = np.isinf(work[num_cols].to_numpy(dtype=float, na_value=np.nan))
        n_inf = int(inf_mask.sum())
        if n_inf:
            report["n_inf_replaced"] = n_inf
            work[num_cols] = work[num_cols].replace([np.inf, -np.inf], np.nan)

    for col in num_cols:
        if not work[col].isna().any():
            continue
        if strategy == "mean":
            fill = work[col].mean()
        elif strategy == "zero":
            fill = 0.0
        else:  # median (default)
            fill = work[col].median()
        if pd.isna(fill):
            fill = 0.0
        work[col] = work[col].fillna(fill)
        report["imputed"][str(col)] = float(fill)

    if verbose and report["imputed"]:
        print(f"  [DataCleaner] Median-imputed NaNs in {len(report['imputed'])} numeric column(s).")

    return work, report


def sanitize(
    df: pd.DataFrame,
    *,
    max_null_frac: float = 0.5,
    drop_constant: bool = True,
    impute: str | None = None,
    protect_cols: tuple[str, ...] | list[str] = (),
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Convenience wrapper: drop low-value columns, then (optionally) impute.

    Set ``impute`` to one of {"median", "mean", "zero"} to fill the remaining
    numeric gaps; leave it None to only prune columns (the right choice for raw
    event tables, where imputing a raw measurement would distort aggregates).
    """
    cleaned, drop_report = drop_low_value_columns(
        df,
        max_null_frac=max_null_frac,
        drop_constant=drop_constant,
        protect_cols=protect_cols,
        verbose=verbose,
    )
    report = {"drop": drop_report}
    if impute:
        cleaned, imp_report = impute_missing(cleaned, strategy=impute, verbose=verbose)
        report["impute"] = imp_report
    return cleaned, report
