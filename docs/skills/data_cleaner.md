# data_cleaner — Column Sanitisation for Wide / Sparse Tables

**File:** `skills/data_cleaner.py`
**Used by:** [Orchestrator](../agents/orchestrator.md) (at load), [FeatureSelectionAgent](../agents/feature_selector.md) (imputation)

## Purpose

Prunes columns that carry no usable clustering signal *before* any heavy math
runs, and (optionally) imputes the remaining gaps. A raw "features" export often
arrives wide and sparse — e.g. 150+ columns where 60+ are ~99% null and 50+ are
constant. Feeding that straight into the pipeline either:

- crashes `StandardScaler` / PCA / the autoencoder on NaN, or
- stalls the VIF/OLS gate on rank-deficient, all-zero columns.

Dropping these once, at load, is faster, avoids the crash/stall, and yields
cleaner clusters because the surviving columns actually vary across entities.

## What gets dropped

| Category | Rule |
|----------|------|
| Duplicate columns | identical column **name** — keep the first occurrence |
| All-null columns | missing fraction `>= 1.0` |
| Mostly-null columns | missing fraction `> max_null_frac` |
| Constant columns | a single unique non-null value (zero variance) |

`protect_cols` are never dropped (the Orchestrator protects `_row_id` and the
text column for text modality).

## API

```python
from skills.data_cleaner import drop_low_value_columns, impute_missing, sanitize

cleaned, report = drop_low_value_columns(
    df,
    max_null_frac=0.5,
    drop_constant=True,
    drop_duplicate=True,
    protect_cols=['_row_id'],
)
# report keys: n_cols_before, n_cols_after, n_dropped, n_rows, max_null_frac,
#   dropped_duplicate [name...], dropped_all_null [name...],
#   dropped_high_null [[name, frac]...], dropped_constant [name...]

filled, imp = impute_missing(df, strategy='median')  # numeric NaNs only → imp['imputed']
cleaned, rep = sanitize(df, max_null_frac=0.5, impute='median')  # prune + impute
```

## Configuration (`config.yaml: data_cleaning`)

| Knob | Default | Effect |
|------|---------|--------|
| `enabled` | `true` | Master switch for the load-time prune |
| `max_null_frac` | `0.5` | Drop columns more than this fraction empty |
| `drop_constant` | `true` | Drop single-unique-value columns |

## Imputation policy

The Orchestrator does **not** impute at load: the raw-CSV path feeds
`FeatureEngineerAgent`, which aggregates raw events, so median-filling a raw
measurement would distort sums/means. Instead, `FeatureSelectionAgent` imputes
its own numeric NaNs (median) immediately before scaling/PCA/AE/VIF — the only
place the value actually has to be finite.
