"""
FeatureEngineerAgent

Contract: docs/agents/feature_engineer.md. Skills: docs/skills/orchestrator_bus.md.

Turns any tabular entity-level or event-level data into an entity-level feature matrix.

Architecture:
  - The agent owns a library of feature BUILDERS (generic statistical functions).
  - It asks the Orchestrator (via bus.ask) to act as a data scientist:
    given the schema + business purpose + the idea framework, propose
    which builders to run and with what columns/windows.
  - The agent then executes the plan and prints every step.

The Idea Framework
──────────────────
Features capture a STATISTICAL SUMMARY of an ATTRIBUTE over a TIME WINDOW.

  summary   : count, sum, mean, median, std, max, frequency, recency,
               consecutive_periods
  attribute : any column in the raw data (a category, a label, a signal, etc.)
  window    : recent period (e.g. last 3 / 6 / 12 periods) or all-time

Additionally, TREND features compare two windows:
  trend     : summary_short_window / summary_long_window
              → is the entity's behaviour accelerating or decelerating?

The LLM is shown:
  1. The raw schema and data statistics
  2. The business purpose
  3. The full library of available builders
  4. The idea framework
  ...and asked to produce a structured feature plan using the actual column names
  found in the data.

The agent executes the plan and reports every step to the terminal.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from agents.user_input import UserIntent
from agents.dataset_examiner import DatasetProfile
from skills.orchestrator_bus import OrchestratorBus, OrchestratorMessage


# ── Available builder catalogue (shown to LLM) ────────────────────────────────
BUILDER_CATALOGUE = """
AVAILABLE FEATURE BUILDERS
Each builder is a pure statistical function. The LLM selects which to use and
specifies ACTUAL column names from the dataset schema.

[1] group_aggregate(group_col, value_col, windows, metrics)
    For each unique value in group_col × window × metric, one column per combination.
    Column name: {metric}_{group_col}_{group_val}_{window}
    metrics: count, sum, mean, median, std, max, freq (events per active period),
             pct_count (share of all events), pct_sum (share of total value)

[2] group_trend(group_col, value_col, base_window, compare_window)
    Trend = metric in base_window / metric in compare_window.
    Captures whether an entity's behaviour on this group is growing or shrinking.
    Column names: trend_count_{group_col}_{val}, trend_sum_{group_col}_{val}

[3] group_streak(group_col)
    Consecutive periods (months) with at least one record per group value.
    Useful as a loyalty / engagement signal.
    Column names: streak_{group_col}_{val}

[4] overall_aggregate(value_col, windows, metrics)
    Aggregate statistics across ALL records (no grouping).
    Column names: {metric}_{value_col}_{window}
    metrics: count, sum, mean, median, std, max, pct_high (% above 75th pctile)

[5] frequency_recency(windows)
    Event counts, active periods, recency, and inter-event gap per entity.
    Column names: event_count_{window}, active_periods_{window},
                  days_since_last (all-time only), avg_gap_days_{window}

[6] entity_diversity(cols_to_count, windows)
    Number of unique values per specified column per window.
    Captures breadth / variety of behaviour.
    Column names: n_unique_{col_name}_{window}
    cols_to_count: list of column names whose cardinality to measure

[7] temporal_patterns(windows)
    Time-of-day and day-of-week distributions (only if a timestamp exists).
    Column names: pct_morning_{window}, pct_midday_{window}, pct_evening_{window},
                  pct_night_{window}, pct_weekend_{window}, peak_hour_{window}

[8] static_attributes(cols)
    Copy static (non-windowed) attribute columns directly as entity-level features.
    Handles numeric columns (mean-per-entity) and binary categoricals (first value).
    Column names: same as source column names.
"""

WINDOWS_AVAILABLE = [3, 6, 12]   # months
ALL_TIME_LABEL = "all"


@dataclass
class FeatureEngineeringPlan:
    """Structured plan returned by the LLM."""
    builders: list[dict]        # list of {builder, params, rationale}
    overall_reasoning: str
    expected_n_features: int


@dataclass
class FeatureEngineeringResult:
    n_entities: int
    n_features: int
    feature_names: list[str]
    groups_built: list[str]
    groups_skipped: list[str]
    output_path: str
    reasoning: str


class FeatureEngineerAgent:
    """
    Engineers customer-level features from raw transaction data.

    Workflow:
    1. Inspect schema → describe to LLM
    2. Ask Orchestrator for a feature engineering plan (which builders, which
       topics, which windows) based on the business purpose
    3. Execute the plan builder-by-builder, printing every step
    4. Report the final feature matrix to the Orchestrator
    """

    def __init__(self, bus: OrchestratorBus):
        self.bus = bus

    def run(
        self,
        raw_df: pd.DataFrame,
        user_intent: UserIntent,
        dataset_profile: DatasetProfile,
        output_path: str = "data/processed/engineered_features.parquet",
        iteration: int = 1,
        feedback: str = "",
    ) -> tuple[pd.DataFrame, FeatureEngineeringResult]:
        """
        Engineer features from raw_df and return (customer_df, result).

        Parameters
        ----------
        raw_df : pd.DataFrame
            Raw transaction-level data.
        user_intent : UserIntent
        dataset_profile : DatasetProfile
        output_path : str
            Where to save the customer-level parquet.
        iteration : int
        feedback : str
            Orchestrator feedback from a previous round.

        Returns
        -------
        (customer_feature_df, FeatureEngineeringResult)
        """
        print(f"\n[FeatureEngineer] Iteration {iteration}")
        print(f"  Entity    : {user_intent.target_entity}")
        print(f"  Purpose   : {user_intent.business_purpose}")
        if feedback:
            print(f"  Feedback  : {feedback}")

        # ── 1. Parse timestamps ────────────────────────────────────────────────
        raw_df = raw_df.copy()
        ts_col = self._detect_timestamp_col(raw_df)
        if ts_col:
            raw_df["_ts"] = pd.to_datetime(raw_df[ts_col], errors="coerce")
            raw_df = raw_df.dropna(subset=["_ts"])
        else:
            print("  WARNING: No timestamp column detected — windowed features will use all-time only.")
            raw_df["_ts"] = pd.Timestamp("2020-01-01")

        max_date = raw_df["_ts"].max()
        customer_col = self._detect_entity_col(raw_df)
        amount_col   = self._detect_amount_col(raw_df)
        category_col = self._detect_category_col(raw_df)

        print(f"  Detected  : entity='{customer_col}'  value='{amount_col}'  "
              f"category='{category_col}'  timestamp='{ts_col}'")
        print(f"  Date range: {raw_df['_ts'].min().date()} → {max_date.date()}")
        print(f"  Entities  : {raw_df[customer_col].nunique():,}  |  "
              f"Records: {len(raw_df):,}")

        # Detect categories
        cats = (
            sorted(raw_df[category_col].dropna().unique().tolist())
            if category_col else []
        )
        print(f"  Categories ({len(cats)}): {cats}")

        # ── 2. Ask Orchestrator for feature engineering plan ───────────────────
        plan = self._get_plan_from_orchestrator(
            raw_df=raw_df,
            user_intent=user_intent,
            dataset_profile=dataset_profile,
            customer_col=customer_col,
            amount_col=amount_col,
            category_col=category_col,
            ts_col=ts_col,
            cats=cats,
            max_date=max_date,
            feedback=feedback,
            iteration=iteration,
        )

        # ── 3. Execute the plan ────────────────────────────────────────────────
        customer_df = self._execute_plan(
            raw_df=raw_df,
            plan=plan,
            customer_col=customer_col,
            amount_col=amount_col,
            category_col=category_col,
            cats=cats,
            max_date=max_date,
        )

        # ── 4. Deduplicate columns then save ───────────────────────────────────
        # The LLM plan can reference the same category twice (e.g. 'grocery_net'
        # appearing in multiple builder entries), producing identical column names.
        # Keep only the first occurrence before writing to parquet.
        n_before = len(customer_df.columns)
        customer_df = customer_df.loc[:, ~customer_df.columns.duplicated()]
        n_dropped = n_before - len(customer_df.columns)
        if n_dropped:
            print(f"  [FeatureEngineer] Dropped {n_dropped} duplicate column(s).")

        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        customer_df.to_parquet(output_path, index=True)
        print(f"\n  [FeatureEngineer] Saved {len(customer_df)} customers × "
              f"{len(customer_df.columns)} features → {output_path}")

        n_features = len(customer_df.columns)

        # Failure modes per docs/agents/feature_engineer.md
        if n_features < 20:
            self.bus.report(OrchestratorMessage(
                agent="FeatureEngineer",
                iteration=iteration,
                status="blocked",
                what_was_done=f"Built {n_features} features (below minimum 20).",
                what_was_not_done="Pipeline requires at least 20 features.",
                doubts="",
                issues=[f"Only {n_features} features built; need ≥ 20."],
                metrics={"n_features": n_features, "n_entities": len(customer_df)},
                recommendation="escalate",
            ))
            raise RuntimeError(
                f"FeatureEngineer: fewer than 20 features built ({n_features}). "
                "See docs/agents/feature_engineer.md."
            )
        all_constant_or_binary = all(
            customer_df[col].nunique() <= 2 for col in customer_df.columns
        )
        if all_constant_or_binary:
            self.bus.report(OrchestratorMessage(
                agent="FeatureEngineer",
                iteration=iteration,
                status="blocked",
                what_was_done="Built feature matrix but all columns are constant or binary.",
                what_was_not_done="Need continuous or multi-level features for clustering.",
                doubts="",
                issues=["All features are binary or constant — cannot cluster meaningfully."],
                metrics={"n_features": n_features},
                recommendation="escalate",
            ))
            raise RuntimeError(
                "FeatureEngineer: all features are binary/constant. "
                "See docs/agents/feature_engineer.md."
            )

        result = FeatureEngineeringResult(
            n_entities=len(customer_df),
            n_features=n_features,
            feature_names=list(customer_df.columns),
            groups_built=[b["builder"] for b in plan.builders],
            groups_skipped=[],
            output_path=output_path,
            reasoning=plan.overall_reasoning,
        )

        # ── 5. Report to bus ───────────────────────────────────────────────────
        self.bus.report(OrchestratorMessage(
            agent="FeatureEngineer",
            iteration=iteration,
            status="success",
            what_was_done=(
                f"Engineered {n_features} features for {len(customer_df):,} customers "
                f"from {len(raw_df):,} transactions. "
                f"Builders used: {[b['builder'] for b in plan.builders]}."
            ),
            what_was_not_done=(
                "Did not engineer features requiring external data "
                "(e.g. credit scores, product catalogue)."
            ),
            doubts=(
                f"Some features may have many zeros for customers with short history. "
                f"Log-transform recommended before clustering."
            ),
            issues=[],
            metrics={
                "n_entities": len(customer_df),
                "n_features": n_features,
                "n_builders_run": len(plan.builders),
            },
            recommendation="proceed",
            context={"feature_names": list(customer_df.columns)},
        ))

        return customer_df, result

    # ── Orchestrator consultation ──────────────────────────────────────────────

    def _get_plan_from_orchestrator(
        self,
        raw_df, user_intent, dataset_profile,
        customer_col, amount_col, category_col, ts_col,
        cats, max_date, feedback, iteration,
    ) -> FeatureEngineeringPlan:
        """
        Ask the Orchestrator (→ LLM) for a feature engineering plan.
        The LLM is shown the schema, business purpose, and the builder catalogue.
        It proposes which builders to run, with which topics and windows.
        """
        schema_lines = []
        for col in raw_df.columns:
            if col.startswith("_"):
                continue
            dtype = str(raw_df[col].dtype)
            n_unique = raw_df[col].nunique()
            sample = str(raw_df[col].dropna().iloc[0]) if len(raw_df[col].dropna()) > 0 else "?"
            schema_lines.append(f"  {col:<28} {dtype:<12} unique={n_unique:<8} sample={sample}")
        schema_str = "\n".join(schema_lines)

        feedback_section = f"\nFeedback from previous round:\n{feedback}\n" if feedback else ""

        readme_section = ""
        if dataset_profile and getattr(dataset_profile, 'dataset_readme', ''):
            readme_section = (
                f"\nDATASET README (domain context from the data provider — use this to guide "
                f"which features are meaningful):\n{'─'*60}\n"
                f"{dataset_profile.dataset_readme}\n{'─'*60}\n"
            )

        cats_str = cats if cats else "(no category column detected)"
        prompt = f"""You are a data scientist designing a feature engineering plan for a clustering project.

BUSINESS PURPOSE: {user_intent.business_purpose}
ENTITY TO CLUSTER: {user_intent.target_entity}
{f"CONSTRAINTS: {user_intent.constraints}" if user_intent.constraints else ""}
{readme_section}

RAW DATA SCHEMA ({len(raw_df.columns)} columns):
  entity (grouping) column : {customer_col}
  numeric value column     : {amount_col if amount_col else "(none detected)"}
  grouping/category column : {category_col if category_col else "(none detected)"}
  timestamp column         : {ts_col if ts_col else "(none detected)"}
  date range               : {raw_df['_ts'].min().date()} → {max_date.date()}
  n_records                : {len(raw_df):,}
  n_entities               : {raw_df[customer_col].nunique():,}
  category values found    : {cats_str}

Full column schema:
{schema_str}

THE IDEA FRAMEWORK FOR FEATURE ENGINEERING
───────────────────────────────────────────
A good feature matrix captures HOW entities differ from each other along statistical
dimensions. For each entity (row in the output), compute:

  SUMMARY STATISTICS: count, sum, mean, median, std, max of any numeric column
  GROUPED SUMMARIES:  same statistics broken down by a categorical grouping column
  TIME WINDOWS:       compute the above for recent periods (e.g. last 3/6/12 months)
                      and all-time. Use whatever time periods make sense for this data.
  TRENDS:             ratio of a short-window metric to a long-window metric
                      (growing vs. shrinking behaviour)
  DIVERSITY:          how many unique values appear for a given column (breadth)
  RECENCY/FREQUENCY:  when did the entity last appear? how often? how regular?
  TEMPORAL RHYTHMS:   if time is available, what time-of-day / day-of-week patterns exist?
  STATIC ATTRIBUTES:  any fixed properties of the entity (demographics, type, etc.)

The goal: make entities with DIFFERENT BEHAVIOURS look numerically different from
each other, so that clustering can discover distinct groups.

{BUILDER_CATALOGUE}
{feedback_section}
INSTRUCTIONS:
1. Inspect the schema above. Identify:
   - Which column is the entity ID (rows will be grouped by this)
   - Which column(s) are numeric values to summarise
   - Which column(s) are categorical groupings to break behaviour down by
   - Whether a timestamp enables windowed features
2. Select builders that best capture the behavioural dimensions relevant to the
   business purpose. Use the ACTUAL column names from the schema above.
3. For group_col and value_col parameters, use the real column names from the schema.
4. For group_values ("categories"), use only the actual values listed in
   "category values found" above, or "all" to include all of them.
5. Aim for 60–150 features. Prefer meaningful dimensions over redundant ones.
6. If no timestamp exists, skip windowed builders and use static/overall builders.
7. If no categorical column exists, skip group_aggregate / group_trend / group_streak.

Return ONLY a valid JSON object (no markdown, no extra text):
{{
  "overall_reasoning": "2-3 sentences on your strategy",
  "expected_n_features": <integer>,
  "builders": [
    {{
      "builder": "<builder_name from catalogue>",
      "params": {{
        "group_col": "<actual column name from schema>",
        "value_col": "<actual column name from schema>",
        "cols_to_count": ["<col1>", "<col2>"],
        "cols": ["<col1>"],
        "categories": ["val1", "val2"] or "all",
        "windows": [6, 12] or [3, 6, 12] or ["all"],
        "metrics": ["count", "sum", "mean", "median", "std", "max", "freq", "pct_count", "pct_sum"],
        "base_window": 6,
        "compare_window": 12
      }},
      "rationale": "why this builder adds value for this specific dataset"
    }},
    ...
  ]
}}"""

        raw = self.bus.ask(
            agent="FeatureEngineer",
            purpose="design feature engineering plan (behavior × topic × time_window)",
            prompt=prompt,
            max_tokens=3000,
        ).strip()

        if "```" in raw:
            for part in raw.split("```"):
                p = part.strip()
                if p.startswith("json"):
                    p = p[4:].strip()
                if p.startswith("{"):
                    raw = p
                    break

        try:
            plan_dict = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  [FeatureEngineer] Could not parse LLM plan ({e}) — using default plan.")
            plan_dict = self._default_plan(cats)

        print(f"\n  [FeatureEngineer] LLM Feature Plan:")
        print(f"    Strategy  : {plan_dict.get('overall_reasoning', '?')[:120]}")
        print(f"    Expected  : ~{plan_dict.get('expected_n_features', '?')} features")
        for b in plan_dict.get("builders", []):
            print(f"    Builder   : {b['builder']:30}  → {b.get('rationale','')[:60]}")

        return FeatureEngineeringPlan(
            builders=plan_dict.get("builders", self._default_plan(cats)["builders"]),
            overall_reasoning=plan_dict.get("overall_reasoning", ""),
            expected_n_features=plan_dict.get("expected_n_features", 100),
        )

    # ── Plan execution ─────────────────────────────────────────────────────────

    def _execute_plan(
        self,
        raw_df: pd.DataFrame,
        plan: FeatureEngineeringPlan,
        customer_col: str,
        amount_col: str,
        category_col: str,
        cats: list[str],
        max_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Execute each builder in the plan and join results into a customer DataFrame.
        Prints what is being built at each step.
        """
        print(f"\n  [FeatureEngineer] Executing plan — {len(plan.builders)} builder(s)...\n")

        # Start with a customer index
        all_customers = raw_df[customer_col].unique()
        customer_df = pd.DataFrame(index=all_customers)
        customer_df.index.name = customer_col

        for step, builder_spec in enumerate(plan.builders, 1):
            name = builder_spec.get("builder", "unknown")
            params = builder_spec.get("params", {})
            rationale = builder_spec.get("rationale", "")

            print(f"  Step {step}/{len(plan.builders)}: {name}")
            print(f"    Purpose: {rationale[:80]}")

            try:
                features = self._run_builder(
                    name=name,
                    params=params,
                    raw_df=raw_df,
                    customer_col=customer_col,
                    amount_col=amount_col,
                    category_col=category_col,
                    cats=cats,
                    max_date=max_date,
                )
                if features is not None and not features.empty:
                    before = len(customer_df.columns)
                    customer_df = customer_df.join(features, how="left")
                    after = len(customer_df.columns)
                    n_new = after - before
                    print(f"    Built  : {n_new} features  "
                          f"(total so far: {after})")
                    # Show sample feature names
                    new_names = list(features.columns)[:5]
                    if new_names:
                        print(f"    Sample : {new_names}"
                              + (" ..." if len(features.columns) > 5 else ""))
                else:
                    print(f"    Skipped: builder returned no features.")
            except Exception as e:
                print(f"    ERROR  : {e} — skipping this builder.")

        # ── Fill NaN with 0 (customers with no activity in a window) ──────────
        customer_df = customer_df.fillna(0)

        print(f"\n  [FeatureEngineer] Plan complete.")
        print(f"    Final matrix: {len(customer_df)} customers × {len(customer_df.columns)} features")

        return customer_df

    def _run_builder(
        self,
        name: str,
        params: dict,
        raw_df: pd.DataFrame,
        customer_col: str,
        amount_col: str,
        category_col: str,
        cats: list[str],
        max_date: pd.Timestamp,
    ) -> pd.DataFrame | None:
        """Dispatch to the correct builder function. Accepts both new and legacy names."""

        # Resolve group_col / value_col from params (new API) or fall back to detected cols
        group_col  = params.get("group_col",  category_col)
        value_col  = params.get("value_col",  amount_col)
        categories = params.get("categories", "all")
        # Resolve category values
        avail_cats = (
            sorted(raw_df[group_col].dropna().unique().tolist())
            if group_col and group_col in raw_df.columns
            else cats
        )
        resolved_cats = self._resolve_cats(categories, avail_cats)

        if name in ("group_aggregate", "category_behavior"):
            if not group_col or not value_col:
                return None
            return self._build_category_behavior(
                raw_df, customer_col, value_col, group_col,
                categories=resolved_cats,
                windows=params.get("windows", [6, 12]),
                metrics=params.get("metrics", ["count", "sum", "mean"]),
                max_date=max_date,
            )

        elif name in ("group_trend", "category_trend"):
            if not group_col or not value_col:
                return None
            return self._build_category_trend(
                raw_df, customer_col, value_col, group_col,
                categories=resolved_cats,
                base_window=params.get("base_window", 6),
                compare_window=params.get("compare_window", 12),
                max_date=max_date,
            )

        elif name in ("group_streak", "category_loyalty"):
            if not group_col:
                return None
            return self._build_category_loyalty(
                raw_df, customer_col, group_col,
                categories=resolved_cats,
            )

        elif name in ("overall_aggregate", "overall_value", "overall_spend"):
            if not value_col:
                return None
            return self._build_overall_aggregate(
                raw_df, customer_col, value_col,
                windows=params.get("windows", [6, 12, "all"]),
                max_date=max_date,
            )

        elif name in ("frequency_recency", "overall_frequency"):
            return self._build_overall_frequency(
                raw_df, customer_col,
                windows=params.get("windows", [6, 12, "all"]),
                max_date=max_date,
            )

        elif name in ("entity_diversity",):
            # cols_to_count from params; fall back to all low-cardinality string cols
            cols_to_count = params.get("cols_to_count", [])
            if not cols_to_count and group_col:
                cols_to_count = [group_col]
            return self._build_entity_diversity(
                raw_df, customer_col,
                cols_to_count=cols_to_count,
                windows=params.get("windows", [6, 12, "all"]),
                max_date=max_date,
            )

        elif name in ("temporal_patterns",):
            return self._build_temporal_patterns(
                raw_df, customer_col,
                windows=params.get("windows", ["all"]),
                max_date=max_date,
            )

        elif name in ("static_attributes", "demographic"):
            cols = params.get("cols", [])
            if not cols:
                # Auto-detect: static columns (dob, gender, age, etc.)
                for c in raw_df.columns:
                    if c.startswith("_") or c == customer_col:
                        continue
                    if raw_df[c].dtype == object and raw_df[c].nunique() <= 20:
                        cols.append(c)
                    elif pd.api.types.is_numeric_dtype(raw_df[c]) and c not in (amount_col or ""):
                        # Only add if it looks static (low nunique per customer)
                        avg_unique = raw_df.groupby(customer_col)[c].nunique().mean()
                        if avg_unique < 2:
                            cols.append(c)
            return self._build_static_attributes(raw_df, customer_col, cols=cols)

        else:
            print(f"    Unknown builder '{name}' — skipping.")
            return None

    # ── Builder implementations ────────────────────────────────────────────────

    def _window_mask(self, raw_df: pd.DataFrame, max_date: pd.Timestamp, months: int) -> pd.Series:
        cutoff = max_date - pd.DateOffset(months=months)
        return raw_df["_ts"] >= cutoff

    def _resolve_cats(self, spec, all_cats: list[str]) -> list[str]:
        if spec == "all" or spec is None:
            return all_cats
        return [c for c in spec if c in all_cats]

    def _build_category_behavior(
        self, df, customer_col, amount_col, category_col,
        categories, windows, metrics, max_date,
    ) -> pd.DataFrame:
        frames = []
        for w in windows:
            label = str(w) + "m" if w != "all" else "all"
            sub = df[self._window_mask(df, max_date, w)] if w != "all" else df
            grp = sub.groupby([customer_col, category_col])[amount_col]

            for cat in categories:
                cat_sub = sub[sub[category_col] == cat].groupby(customer_col)[amount_col]
                for metric in metrics:
                    col_name = self._build_col_name(metric, cat, label)
                    if metric in ("count", "n_event"):
                        s = cat_sub.count().rename(col_name)
                    elif metric in ("total", "sum"):
                        s = cat_sub.sum().rename(col_name)
                    elif metric in ("avg", "average", "mean"):
                        s = cat_sub.mean().rename(col_name)
                    elif metric in ("median",):
                        s = cat_sub.median().rename(col_name)
                    elif metric in ("std",):
                        s = cat_sub.std().fillna(0).rename(col_name)
                    elif metric in ("max",):
                        s = cat_sub.max().rename(col_name)
                    elif metric in ("frequency", "freq"):
                        # events per active day
                        n = cat_sub.count()
                        days_active = sub[sub[category_col] == cat].groupby(customer_col)["_ts"].apply(
                            lambda x: max((x.max() - x.min()).days, 1)
                        )
                        s = (n / days_active).rename(col_name)
                    elif metric in ("pct_count",):
                        total_n = sub.groupby(customer_col)[amount_col].count()
                        cat_n = cat_sub.count()
                        s = (cat_n / total_n.replace(0, np.nan)).fillna(0).rename(col_name)
                    elif metric in ("pct_sum",):
                        total_s = sub.groupby(customer_col)[amount_col].sum()
                        cat_s = cat_sub.sum()
                        s = (cat_s / total_s.replace(0, np.nan)).fillna(0).rename(col_name)
                    else:
                        continue
                    frames.append(s)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).fillna(0)

    def _build_col_name(self, metric: str, cat: str, label: str) -> str:
        """
        Build a generic column name from metric, group value, and window label.
        Uses plain statistical terms — no domain-specific prefixes.
        """
        metric_map = {
            "count": "count", "n_event": "count",
            "total": "sum",   "sum": "sum",
            "avg": "mean",    "average": "mean",  "mean": "mean",
            "median": "median",
            "std": "std",
            "max": "max",
            "frequency": "freq", "freq": "freq",
            "pct_count": "pct_count",
            "pct_sum": "pct_sum",
        }
        m = metric_map.get(metric, metric)
        return f"{m}_{cat}_{label}"

    def _build_category_trend(
        self, df, customer_col, amount_col, category_col,
        categories, base_window, compare_window, max_date,
    ) -> pd.DataFrame:
        frames = []
        sub_base = df[self._window_mask(df, max_date, base_window)]
        sub_cmp  = df[self._window_mask(df, max_date, compare_window)]

        for cat in categories:
            cb = sub_base[sub_base[category_col] == cat].groupby(customer_col)[amount_col]
            cc = sub_cmp[sub_cmp[category_col]   == cat].groupby(customer_col)[amount_col]

            n_base = cb.count()
            n_cmp  = cc.count()
            a_base = cb.sum()
            a_cmp  = cc.sum()

            trend_n = (n_base / (n_cmp + 1)).rename(f"trend_count_{cat}")
            trend_a = (a_base / (a_cmp + 1)).rename(f"trend_sum_{cat}")
            frames += [trend_n, trend_a]

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).fillna(1.0)   # 1.0 = no change

    def _build_category_loyalty(
        self, df, customer_col, category_col, categories,
    ) -> pd.DataFrame:
        frames = []
        for cat in categories:
            cat_df = df[df[category_col] == cat].copy()
            cat_df["_ym"] = cat_df["_ts"].dt.to_period("M")

            def consec(x):
                if len(x) == 0:
                    return 0
                periods = sorted(x.unique())
                max_run = run = 1
                for i in range(1, len(periods)):
                    if (periods[i] - periods[i - 1]).n == 1:
                        run += 1
                        max_run = max(max_run, run)
                    else:
                        run = 1
                return max_run

            s = cat_df.groupby(customer_col)["_ym"].apply(consec).rename(
                f"streak_{cat}"
            )
            frames.append(s)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).fillna(0)

    def _build_overall_aggregate(
        self, df, customer_col, amount_col, windows, max_date,
    ) -> pd.DataFrame:
        frames = []
        p75_global = df[amount_col].quantile(0.75)
        val = amount_col  # use actual column name in output column names

        for w in windows:
            sub = df[self._window_mask(df, max_date, w)] if w != "all" else df
            grp = sub.groupby(customer_col)[amount_col]
            label = "all" if w == "all" else f"{w}m"

            frames += [
                grp.sum().rename(f"sum_{val}_{label}"),
                grp.mean().rename(f"mean_{val}_{label}"),
                grp.std().fillna(0).rename(f"std_{val}_{label}"),
                grp.max().rename(f"max_{val}_{label}"),
                grp.median().rename(f"median_{val}_{label}"),
                (sub[sub[amount_col] > p75_global].groupby(customer_col)[amount_col].count()
                 / grp.count().replace(0, np.nan)).fillna(0).rename(f"pct_high_{val}_{label}"),
            ]

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).fillna(0)

    def _build_overall_frequency(
        self, df, customer_col, windows, max_date,
    ) -> pd.DataFrame:
        frames = []
        days_since = (
            (max_date - df.groupby(customer_col)["_ts"].max())
            .dt.days.rename("days_since_last")
        )
        frames.append(days_since)

        # avg days between consecutive records (helper)
        def avg_gap(x):
            s = x.sort_values()
            if len(s) < 2:
                return 0.0
            return float((s.diff().dt.days.dropna()).mean())

        for w in windows:
            sub = df[self._window_mask(df, max_date, w)] if w != "all" else df
            grp = sub.groupby(customer_col)
            label = "all" if w == "all" else f"{w}m"

            frames += [
                grp["_ts"].count().rename(f"event_count_{label}"),
                grp["_ts"].apply(lambda x: x.dt.to_period("M").nunique()).rename(
                    f"active_periods_{label}"
                ),
                grp["_ts"].apply(avg_gap).rename(f"avg_gap_days_{label}"),
            ]

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).fillna(0)

    def _build_entity_diversity(
        self, df, customer_col, cols_to_count, windows, max_date,
    ) -> pd.DataFrame:
        """
        Count unique values per specified column per window.
        Generic — works for any categorical columns (category, merchant, city, tag, etc.).
        """
        frames = []
        # Only use columns that actually exist in df
        valid_cols = [c for c in cols_to_count if c in df.columns]
        if not valid_cols and not cols_to_count:
            # Auto-detect: use all low-cardinality string columns
            for col in df.columns:
                if col.startswith("_") or col == customer_col:
                    continue
                if df[col].dtype == object and df[col].nunique() < 500:
                    valid_cols.append(col)

        for w in windows:
            label = "all" if w == "all" else f"{w}m"
            sub = df[self._window_mask(df, max_date, w)] if w != "all" else df
            grp = sub.groupby(customer_col)
            for col in valid_cols:
                frames.append(
                    grp[col].nunique().rename(f"n_unique_{col}_{label}")
                )

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).fillna(0)

    def _build_temporal_patterns(
        self, df, customer_col, windows, max_date,
    ) -> pd.DataFrame:
        frames = []
        df = df.copy()
        df["_hour"]    = df["_ts"].dt.hour
        df["_weekday"] = df["_ts"].dt.weekday   # 0=Mon, 6=Sun

        for w in windows:
            label = str(w) + "m" if w != "all" else "all"
            sub = df[self._window_mask(df, max_date, w)] if w != "all" else df
            n = sub.groupby(customer_col)["_hour"].count().replace(0, np.nan)

            # Time-of-day buckets
            for bucket, lo, hi, col_label in [
                ("morning", 6, 12, f"pct_morning_{label}"),
                ("midday",  12, 18, f"pct_midday_{label}"),
                ("evening", 18, 24, f"pct_evening_{label}"),
                ("night",   0,   6, f"pct_night_{label}"),
            ]:
                mask = (sub["_hour"] >= lo) & (sub["_hour"] < hi)
                cnt = sub[mask].groupby(customer_col)["_hour"].count()
                frames.append((cnt / n).fillna(0).rename(col_label))

            # Weekend share
            weekend = sub[sub["_weekday"] >= 5].groupby(customer_col)["_hour"].count()
            frames.append((weekend / n).fillna(0).rename(f"pct_weekend_{label}"))

            # Peak hour (most common transaction hour)
            frames.append(
                sub.groupby(customer_col)["_hour"]
                .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else 0)
                .rename(f"peak_hour_{label}")
            )

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).fillna(0)

    def _build_static_attributes(
        self, df, customer_col, cols=None,
    ) -> pd.DataFrame:
        """
        Copy static (non-windowed) attributes as entity-level features.
        For numeric columns: takes the mean per entity.
        For binary-encodable columns (2 unique values): encodes as 0/1.
        For other categoricals: skips (use entity_diversity instead).
        """
        frames = []
        if cols is None:
            cols = []

        for col in cols:
            if col not in df.columns or col == customer_col or col.startswith("_"):
                continue
            s = df.groupby(customer_col)[col]
            if pd.api.types.is_numeric_dtype(df[col]):
                frames.append(s.mean().rename(col))
            else:
                # Try binary encoding
                vals = df[col].dropna().unique()
                if len(vals) == 2:
                    v0, v1 = sorted(str(v) for v in vals)
                    encoded = (df[col].astype(str) == v1).astype(float)
                    df2 = df.copy()
                    df2["_enc"] = encoded
                    frames.append(
                        df2.groupby(customer_col)["_enc"].first().rename(f"{col}_{v1}")
                    )
                # else: skip multi-class categoricals

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1)

    # ── Schema detection helpers ───────────────────────────────────────────────

    def _detect_col(self, df: pd.DataFrame, candidates: list[str]) -> str | None:
        """
        Find the first column matching any candidate.
        Strategy 1 — exact match (case-sensitive).
        Strategy 2 — substring match (case-insensitive): catches e.g. 'event_kind',
                     'product_category', 'item_type' when candidate is 'kind',
                     'category', or 'type'.
        """
        # Exact match first
        for c in candidates:
            if c in df.columns:
                return c
        # Substring match fallback
        lower_map = {col.lower(): col for col in df.columns}
        for kw in candidates:
            kw_l = kw.lower()
            for lc, orig in lower_map.items():
                if kw_l in lc:
                    return orig
        return None

    def _resolve_col(
        self,
        df: pd.DataFrame,
        role: str,
        candidates: list[str],
        required: bool = True,
    ) -> str | None:
        """
        Auto-detect a column for a given role; ask the user if detection fails.

        Parameters
        ----------
        role       : human-readable label, e.g. "entity/ID", "timestamp"
        candidates : ordered list of keywords to try (exact then substring)
        required   : if True, keeps prompting until a valid column is given;
                     if False, allows the user to press Enter to skip
        """
        guess = self._detect_col(df, candidates)
        if guess is not None:
            return guess

        # Auto-detection failed — ask the user
        cols = list(df.columns)
        col_preview = ", ".join(cols[:20])
        if len(cols) > 20:
            col_preview += f" … ({len(cols)} total)"

        print(f"\n  [FeatureEngineer] Could not auto-detect the {role} column.")
        print(f"  Available columns: {col_preview}")
        if not required:
            print(f"  Press Enter to skip {role} (optional).")

        while True:
            try:
                val = input(f"  → Enter column name for {role}: ").strip()
            except EOFError:
                # Non-interactive mode (bypass / detached / piped). Fall back
                # to the caller's default (None → df.columns[0] for required).
                print(f"  [non-interactive] no input — falling back to default for {role}.")
                return None
            if val == "" and not required:
                print(f"  Skipping {role}.")
                return None
            if val in df.columns:
                return val
            print(f"  Column '{val}' not found. Please choose from the list above.")

    def _detect_timestamp_col(self, df: pd.DataFrame) -> str | None:
        return self._resolve_col(df, "timestamp / date", [
            "timestamp", "datetime", "date", "time", "ts",
            "created_at", "occurred_at", "recorded_at", "updated_at",
            "event_time", "event_date", "visit_date", "purchase_date",
            "order_date", "trans_date", "trans_date_trans_time",
        ], required=False)

    def _detect_entity_col(self, df: pd.DataFrame) -> str:
        # Prefer a real ID-like column when present. Otherwise fall back to
        # the auto-injected `_row_id` (loaded by orchestrator._load_df) so
        # each raw row is treated as one entity — better than falling back
        # to columns[0] which is often a non-unique field (e.g. 'Date').
        detected = self._resolve_col(
            df,
            "entity / ID (the column that identifies each entity being clustered)",
            [
                "id", "entity_id", "user_id", "customer_id", "client_id",
                "account_id", "subject_id", "patient_id", "device_id",
                "sensor_id", "item_id", "product_id", "order_id",
                "session_id", "record_id", "uuid", "uid", "pid",
                "card_number", "cc_num",
            ],
            required=False,
        )
        if detected:
            return detected
        if "_row_id" in df.columns:
            print("  [FeatureEngineer] No ID-like column found — using auto-injected `_row_id` "
                  "(each raw row = one entity).")
            return "_row_id"
        # Last-resort fallback (datasets loaded outside _load_df).
        df.insert(0, "_row_id", range(1, len(df) + 1))
        print("  [FeatureEngineer] Injected `_row_id` (no ID-like column detected).")
        return "_row_id"

    def _detect_amount_col(self, df: pd.DataFrame) -> str | None:
        return self._resolve_col(df, "value / amount (the primary numeric measure per event)", [
            "amount", "value", "price", "total", "amt",
            "cost", "revenue", "qty", "quantity", "score",
            "duration", "size", "weight", "measurement", "reading", "level",
        ], required=False)

    def _detect_category_col(self, df: pd.DataFrame) -> str | None:
        return self._resolve_col(df, "category / kind (the column that groups events into types)", [
            "category", "cat", "type", "kind", "label",
            "class", "group", "segment", "tag", "genre",
            "department", "sector", "channel", "mode",
            "event_type", "item_type", "product_type", "product_category",
            "item_category", "subcategory", "transaction_type",
        ], required=False)

    # ── Default plan fallback (if LLM fails) ──────────────────────────────────

    def _default_plan(self, cats: list[str]) -> dict:
        builders = [
            {
                "builder": "overall_aggregate",
                "params": {"windows": [6, 12, "all"], "metrics": ["count", "sum", "mean", "std", "max"]},
                "rationale": "Overall aggregate statistics across all records",
            },
            {
                "builder": "frequency_recency",
                "params": {"windows": [6, 12, "all"]},
                "rationale": "How often and how recently does this entity appear?",
            },
            {
                "builder": "temporal_patterns",
                "params": {"windows": ["all"]},
                "rationale": "Time-of-day and day-of-week patterns",
            },
        ]
        if cats:
            builders = [
                {
                    "builder": "group_aggregate",
                    "params": {
                        "categories": "all",
                        "windows": [6, 12],
                        "metrics": ["count", "sum", "mean"],
                    },
                    "rationale": "Core group × window behaviour",
                },
                {
                    "builder": "group_trend",
                    "params": {"categories": "all", "base_window": 6, "compare_window": 12},
                    "rationale": "Is behaviour growing or shrinking?",
                },
                {
                    "builder": "group_streak",
                    "params": {"categories": "all"},
                    "rationale": "Consecutive active periods per group",
                },
            ] + builders
        return {
            "overall_reasoning": "Default plan: statistical summaries of all available dimensions.",
            "expected_n_features": 80,
            "builders": builders,
        }
