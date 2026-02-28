"""
FeatureEngineerAgent

Contract: docs/agents/feature_engineer.md. Skills: docs/skills/orchestrator_bus.md.

Turns raw transaction-level data into a customer-level feature matrix.

Architecture:
  - The agent owns a library of feature BUILDERS (pure pandas functions).
  - It asks the Orchestrator (via bus.ask) to act as a data scientist:
    given the schema + business purpose + the idea framework below, propose
    which builders to run and with what topics/windows.
  - The agent then executes the plan and prints every step.

The Idea Framework
──────────────────
Features capture BEHAVIOR on a TOPIC over a TIME WINDOW:

  behavior  : count, total, avg, median, std, max, frequency, recency,
               consecutive_months
  topic     : spending category, merchant, geography (state/city), time-of-day
  window    : 3m, 6m, 12m, all_time

Additionally, TREND features show change between windows:
  trend     : metric_6m / metric_12m  →  is the customer accelerating or
               slowing down in this behaviour?

The LLM is shown:
  1. The raw schema and data statistics
  2. The business purpose
  3. The full library of available builders
  4. The idea framework
  ...and asked to produce a structured feature plan.

The agent executes the plan and reports every feature group to the terminal.
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
Each builder is a function the agent can call. The LLM selects which to use.

[1] category_behavior(categories, windows, metrics)
    For each category × window × metric:
    - count      → n_event_{cat}_{w}m     : number of event
    - total      → amt_{cat}_{w}m       : total event value
    - avg        → avg_spend_{cat}_{w}m : mean event value
    - median     → median_spend_{cat}_{w}m : median event value
    - std        → std_spend_{cat}_{w}m : variability of event value
    - frequency  → freq_{cat}_{w}m     : events per active day
    - pct_event  → pct_event_{cat}_{w}m  : share of all events
    - pct_spend  → pct_spend_{cat}_{w}m: share of total event value

[2] category_trend(categories, base_window, compare_window)
    For each category: ratio of metric in short window vs longer window.
    Captures ACCELERATION or DECELERATION of event value.
    - trend_event_{cat}   : n_event_{cat}_{base}m / (n_event_{cat}_{cmp}m + 1)
    - trend_amt_{cat}   : amt_{cat}_{base}m   / (amt_{cat}_{cmp}m   + 1)

[3] category_loyalty(categories)
    For each category: consecutive active months with at least one event (loyalty signal).
    - consec_months_{cat}

[4] overall_spend(windows)
    Aggregate spend behavior across all categories per window:
    - total_event_value_{w}m, avg_event_value_{w}m, std_event_value_{w}m, max_event_value_{w}m,
      median_event_value_{w}m, pct_high_value_{w}m (% events above 75th percentile)

[5] overall_frequency(windows)
    Event frequency and recency:
    - total_event_count_{w}m, active_months_{w}m, avg_days_between_event_{w}m,
      days_since_last_event (all_time only)

[6] merchant_diversity(windows)
    Breadth of merchant and category usage:
    - n_unique_merchants_{w}m, n_unique_categories_{w}m

[7] geographic_mobility(windows)
    Geographic spread of spending:
    - n_unique_states_{w}m, n_unique_cities_{w}m

[8] temporal_patterns(windows)
    Time-of-day and day-of-week patterns:
    - pct_weekend_{w}m, pct_evening_{w}m (18-23h), pct_morning_{w}m (6-12h),
      pct_night_{w}m (0-6h), pct_midday_{w}m (12-18h)
    - peak_hour_{w}m (most common event hour)

[9] demographic()
    Static customer attributes (computed once, not windowed):
    - age, gender_female (1=F, 0=M)
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
    n_customers: int
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
        customer_col = self._detect_customer_col(raw_df)
        amount_col   = self._detect_amount_col(raw_df)
        category_col = self._detect_category_col(raw_df)

        print(f"  Detected  : customer='{customer_col}'  amount='{amount_col}'  "
              f"category='{category_col}'  timestamp='{ts_col}'")
        print(f"  Date range: {raw_df['_ts'].min().date()} → {max_date.date()}")
        print(f"  Customers : {raw_df[customer_col].nunique():,}  |  "
              f"Transactions: {len(raw_df):,}")

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

        # ── 4. Save ────────────────────────────────────────────────────────────
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
                metrics={"n_features": n_features, "n_customers": len(customer_df)},
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
            n_customers=len(customer_df),
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
                "n_customers": len(customer_df),
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

        prompt = f"""You are a data scientist designing a feature engineering plan for a clustering project.

BUSINESS PURPOSE: {user_intent.business_purpose}
ENTITY TO CLUSTER: {user_intent.target_entity}
{f"CONSTRAINTS: {user_intent.constraints}" if user_intent.constraints else ""}

RAW DATA SCHEMA ({len(raw_df.columns)} columns, detected keys below):
  customer_id column : {customer_col}
  amount column      : {amount_col}
  category column    : {category_col}
  timestamp column   : {ts_col}
  date range         : {raw_df['_ts'].min().date()} → {max_date.date()}
  n_transactions     : {len(raw_df):,}
  n_customers        : {raw_df[customer_col].nunique():,}
  categories present : {cats}

Full schema:
{schema_str}

THE IDEA FRAMEWORK FOR FEATURE ENGINEERING
───────────────────────────────────────────
Features should capture BEHAVIOR on a TOPIC over a TIME WINDOW:

  behavior  = what the customer does
              (count, total, avg, median, std, frequency, recency, consecutive months)
  topic     = what they do it on
              (spending category, merchant, geography, time-of-day)
  window    = how recently
              (3m = last 3 months, 6m = last 6 months, 12m = last 12 months, all = all time)

Additionally, TREND features capture CHANGE between windows:
  trend     = behavior_6m / behavior_12m
              → is the customer accelerating or decelerating in this behavior?
              → very powerful for clustering customers by lifecycle stage

The goal: create features that make customers with DIFFERENT SHOPPING BEHAVIORS
look numerically different from each other, so clustering can find distinct personas.

{BUILDER_CATALOGUE}
{feedback_section}
INSTRUCTIONS:
1. Select the builders that will best capture the behavioral dimensions relevant to
   the business purpose.
2. For each selected builder, specify which topics (categories/windows) to use.
3. Include TREND features — they reveal whether behavior is growing or shrinking.
4. If the schema has extra columns (geography, demographics, time), use them.
5. Aim for 80–150 features total. More features = more behavioral nuance, but keep
   each one meaningful (no redundant duplicates).

Return ONLY a valid JSON object (no markdown, no extra text):
{{
  "overall_reasoning": "2-3 sentences on your strategy",
  "expected_n_features": <integer>,
  "builders": [
    {{
      "builder": "<builder_name from catalogue>",
      "params": {{
        "categories": ["cat1", "cat2", ...] or "all",
        "windows": [6, 12] or [3, 6, 12] or ["all"],
        "metrics": ["count", "total", "avg", "median", "std", "frequency", "pct_event", "pct_spend"],
        "base_window": 6,
        "compare_window": 12
      }},
      "rationale": "why this builder is valuable for the business purpose"
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
        """Dispatch to the correct builder function."""

        if name == "category_behavior":
            return self._build_category_behavior(
                raw_df, customer_col, amount_col, category_col,
                categories=self._resolve_cats(params.get("categories", "all"), cats),
                windows=params.get("windows", [6, 12]),
                metrics=params.get("metrics", ["count", "total", "avg"]),
                max_date=max_date,
            )

        elif name == "category_trend":
            return self._build_category_trend(
                raw_df, customer_col, amount_col, category_col,
                categories=self._resolve_cats(params.get("categories", "all"), cats),
                base_window=params.get("base_window", 6),
                compare_window=params.get("compare_window", 12),
                max_date=max_date,
            )

        elif name == "category_loyalty":
            return self._build_category_loyalty(
                raw_df, customer_col, category_col,
                categories=self._resolve_cats(params.get("categories", "all"), cats),
            )

        elif name == "overall_spend":
            return self._build_overall_spend(
                raw_df, customer_col, amount_col,
                windows=params.get("windows", [6, 12, "all"]),
                max_date=max_date,
            )

        elif name == "overall_frequency":
            return self._build_overall_frequency(
                raw_df, customer_col,
                windows=params.get("windows", [6, 12, "all"]),
                max_date=max_date,
            )

        elif name == "merchant_diversity":
            return self._build_merchant_diversity(
                raw_df, customer_col, category_col,
                windows=params.get("windows", [6, 12, "all"]),
                max_date=max_date,
            )

        elif name == "geographic_mobility":
            return self._build_geographic_mobility(
                raw_df, customer_col,
                windows=params.get("windows", [6, 12, "all"]),
                max_date=max_date,
            )

        elif name == "temporal_patterns":
            return self._build_temporal_patterns(
                raw_df, customer_col,
                windows=params.get("windows", [6, 12, "all"]),
                max_date=max_date,
            )

        elif name == "demographic":
            return self._build_demographic(raw_df, customer_col)

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
                    elif metric in ("total", "amt"):
                        s = cat_sub.sum().rename(col_name)
                    elif metric in ("avg", "average", "avg_spend"):
                        s = cat_sub.mean().rename(col_name)
                    elif metric in ("median", "median_spend"):
                        s = cat_sub.median().rename(col_name)
                    elif metric in ("std", "std_spend"):
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
                    elif metric in ("pct_event",):
                        total_n = sub.groupby(customer_col)[amount_col].count()
                        cat_n = cat_sub.count()
                        s = (cat_n / total_n.replace(0, np.nan)).fillna(0).rename(col_name)
                    elif metric in ("pct_spend",):
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
        # Use n_txn (not n_event) so FeatureSelector / Clusterer / Classifier
        # log-transforms and profile lookups work without modification.
        metric_map = {
            "count": "n_txn", "n_event": "n_txn", "n_txn": "n_txn",
            "total": "amt", "amt": "amt",
            "avg": "avg_spend", "average": "avg_spend", "avg_spend": "avg_spend",
            "median": "median_spend", "median_spend": "median_spend",
            "std": "std_spend", "std_spend": "std_spend",
            "max": "max_spend", "max_spend": "max_spend",
            "frequency": "freq", "freq": "freq",
            "pct_event": "pct_event",
            "pct_spend": "pct_spend",
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

            trend_n = (n_base / (n_cmp + 1)).rename(f"trend_event_{cat}")
            trend_a = (a_base / (a_cmp + 1)).rename(f"trend_amt_{cat}")
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
                f"consec_months_{cat}"
            )
            frames.append(s)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).fillna(0)

    def _build_overall_spend(
        self, df, customer_col, amount_col, windows, max_date,
    ) -> pd.DataFrame:
        frames = []
        p75_global = df[amount_col].quantile(0.75)

        for w in windows:
            sub = df[self._window_mask(df, max_date, w)] if w != "all" else df
            grp = sub.groupby(customer_col)[amount_col]

            if w == "all":
                # Use canonical names (no suffix) so clusterer _extract_profiles
                # and run_pipeline.py reporting can find these exact columns.
                frames += [
                    grp.sum().rename("total_spend"),
                    grp.mean().rename("avg_txn_amt"),
                    grp.std().fillna(0).rename("std_txn_amt"),
                    grp.max().rename("max_txn_amt"),
                    grp.median().rename("median_txn_amt"),
                    (sub[sub[amount_col] > p75_global].groupby(customer_col)[amount_col].count()
                     / grp.count().replace(0, np.nan)).fillna(0).rename("pct_high_value"),
                ]
            else:
                label = f"{w}m"
                frames += [
                    grp.sum().rename(f"total_spend_{label}"),
                    grp.mean().rename(f"avg_txn_amt_{label}"),
                    grp.std().fillna(0).rename(f"std_txn_amt_{label}"),
                    grp.max().rename(f"max_txn_amt_{label}"),
                    grp.median().rename(f"median_txn_amt_{label}"),
                    (sub[sub[amount_col] > p75_global].groupby(customer_col)[amount_col].count()
                     / grp.count().replace(0, np.nan)).fillna(0).rename(f"pct_high_value_{label}"),
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
            .dt.days.rename("days_since_last_event")
        )
        frames.append(days_since)

        # avg days between consecutive transactions (helper)
        def avg_gap(x):
            s = x.sort_values()
            if len(s) < 2:
                return 0.0
            return float((s.diff().dt.days.dropna()).mean())

        for w in windows:
            sub = df[self._window_mask(df, max_date, w)] if w != "all" else df
            grp = sub.groupby(customer_col)

            if w == "all":
                # Canonical names — no suffix — so clusterer _extract_profiles
                # and classifier log-transform lists find these columns.
                frames += [
                    grp["_ts"].count().rename("total_txn_count"),
                    grp["_ts"].apply(lambda x: x.dt.to_period("M").nunique()).rename(
                        "active_months"
                    ),
                    grp["_ts"].apply(avg_gap).rename("avg_days_between_txn"),
                ]
            else:
                label = f"{w}m"
                frames += [
                    grp["_ts"].count().rename(f"total_txn_count_{label}"),
                    grp["_ts"].apply(lambda x: x.dt.to_period("M").nunique()).rename(
                        f"active_months_{label}"
                    ),
                    grp["_ts"].apply(avg_gap).rename(f"avg_days_between_txn_{label}"),
                ]

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).fillna(0)

    def _build_merchant_diversity(
        self, df, customer_col, category_col, windows, max_date,
    ) -> pd.DataFrame:
        frames = []
        merch_col = self._detect_col(df, ["merchant", "merchant_name", "store"])

        for w in windows:
            sub = df[self._window_mask(df, max_date, w)] if w != "all" else df
            grp = sub.groupby(customer_col)

            if w == "all":
                # Canonical names — no suffix — expected by clusterer _extract_profiles.
                if category_col:
                    frames.append(
                        grp[category_col].nunique().rename("n_unique_categories")
                    )
                if merch_col:
                    frames.append(
                        grp[merch_col].nunique().rename("n_unique_merchants")
                    )
            else:
                label = f"{w}m"
                if category_col:
                    frames.append(
                        grp[category_col].nunique().rename(f"n_unique_categories_{label}")
                    )
                if merch_col:
                    frames.append(
                        grp[merch_col].nunique().rename(f"n_unique_merchants_{label}")
                    )

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).fillna(0)

    def _build_geographic_mobility(
        self, df, customer_col, windows, max_date,
    ) -> pd.DataFrame:
        frames = []
        state_col = self._detect_col(df, ["state", "province", "region"])
        city_col  = self._detect_col(df, ["city", "town", "location"])

        for w in windows:
            label = str(w) + "m" if w != "all" else "all"
            sub = df[self._window_mask(df, max_date, w)] if w != "all" else df
            grp = sub.groupby(customer_col)

            if state_col:
                frames.append(grp[state_col].nunique().rename(f"n_unique_states_{label}"))
            if city_col:
                frames.append(grp[city_col].nunique().rename(f"n_unique_cities_{label}"))

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

    def _build_demographic(self, df, customer_col) -> pd.DataFrame:
        frames = []

        # Age from dob
        dob_col = self._detect_col(df, ["dob", "date_of_birth", "birth_date", "birthdate"])
        if dob_col:
            ref = df["_ts"].max()
            dob = pd.to_datetime(df[dob_col], errors="coerce")
            df2 = df.copy()
            df2["_age"] = (ref - dob).dt.days / 365.25
            age = df2.groupby(customer_col)["_age"].first().rename("age")
            frames.append(age)

        # Gender (binary)
        gender_col = self._detect_col(df, ["gender", "sex"])
        if gender_col:
            df2 = df.copy()
            df2["_gf"] = (df2[gender_col].astype(str).str.upper().str[0] == "F").astype(float)
            gf = df2.groupby(customer_col)["_gf"].first().rename("gender_female")
            frames.append(gf)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1)

    # ── Schema detection helpers ───────────────────────────────────────────────

    def _detect_col(self, df: pd.DataFrame, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def _detect_timestamp_col(self, df: pd.DataFrame) -> str | None:
        return self._detect_col(df, [
            "trans_date_trans_time", "timestamp", "date", "transaction_date",
            "trans_date", "datetime", "created_at", "order_date",
        ])

    def _detect_customer_col(self, df: pd.DataFrame) -> str:
        return self._detect_col(df, [
            "cc_num", "customer_id", "user_id", "client_id", "account_id",
            "card_number", "id",
        ]) or df.columns[0]

    def _detect_amount_col(self, df: pd.DataFrame) -> str:
        return self._detect_col(df, [
            "amt", "amount", "transaction_amount", "price", "total", "value",
        ]) or "amt"

    def _detect_category_col(self, df: pd.DataFrame) -> str | None:
        return self._detect_col(df, [
            "category", "cat", "type", "transaction_type", "merchant_category",
        ])

    # ── Default plan fallback (if LLM fails) ──────────────────────────────────

    def _default_plan(self, cats: list[str]) -> dict:
        return {
            "overall_reasoning": "Default plan: standard behavior × category × window features.",
            "expected_n_features": 120,
            "builders": [
                {
                    "builder": "category_behavior",
                    "params": {
                        "categories": "all",
                        "windows": [6, 12],
                        "metrics": ["count", "total", "avg"],
                    },
                    "rationale": "Core category × window behavior",
                },
                {
                    "builder": "category_trend",
                    "params": {"categories": "all", "base_window": 6, "compare_window": 12},
                    "rationale": "Trend: is the customer accelerating?",
                },
                {
                    "builder": "category_loyalty",
                    "params": {"categories": "all"},
                    "rationale": "Consecutive months per category",
                },
                {
                    "builder": "overall_spend",
                    "params": {"windows": [6, 12, "all"]},
                    "rationale": "Overall spend behavior",
                },
                {
                    "builder": "overall_frequency",
                    "params": {"windows": [6, 12, "all"]},
                    "rationale": "Transaction frequency and recency",
                },
                {
                    "builder": "merchant_diversity",
                    "params": {"windows": [6, 12, "all"]},
                    "rationale": "Breadth of spending",
                },
                {
                    "builder": "temporal_patterns",
                    "params": {"windows": ["all"]},
                    "rationale": "When does the customer shop?",
                },
                {
                    "builder": "demographic",
                    "params": {},
                    "rationale": "Age and gender",
                },
            ],
        }
