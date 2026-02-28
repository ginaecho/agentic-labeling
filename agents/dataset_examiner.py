"""
DatasetExaminerAgent — Dataset Profiling & Feature Opportunity Discovery

Contract: docs/agents/dataset_examiner.md. Skills: docs/skills/orchestrator_bus.md.

Profiles the raw dataset (schema, distributions, missing rates) and calls
Claude with the schema + business purpose to get a list of suggested feature
groups to engineer.

Reports findings to the orchestrator bus with:
  - SUCCESS if the dataset is clean and has clear feature opportunities
  - WARNING if some columns have high missing rates or unusual distributions
  - BLOCKED if the dataset cannot be used (empty, no numeric columns, missing file)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.user_input import UserIntent
from skills.orchestrator_bus import OrchestratorBus, OrchestratorMessage


@dataclass
class DatasetProfile:
    """Structured profile of the raw dataset, output of DatasetExaminerAgent."""

    n_rows: int
    n_cols: int
    column_types: dict[str, str]
    """{'column_name': 'numeric' | 'categorical' | 'datetime' | 'other'}"""

    missing_rates: dict[str, float]
    """Fraction of missing values per column. 0.0 = no missing."""

    distribution_summary: dict[str, dict[str, float]]
    """Per-column: min, max, mean, std, skewness for numeric columns."""

    high_cardinality_cols: list[str]
    """Categorical columns with > 100 unique values."""

    suggested_feature_groups: list[str]
    """Feature group names recommended by Claude, e.g. ['frequency', 'spend', 'recency']."""

    feature_group_reasoning: str
    """Claude's explanation of why these groups were chosen."""

    warnings: list[str] = field(default_factory=list)
    """Non-fatal issues (e.g. sparse columns)."""

    algo_hint: str = ""
    """Optional algorithm hint based on distribution shape."""


class DatasetExaminerAgent:
    """
    Profiles the raw dataset and asks Claude to suggest feature engineering
    groups aligned with the business purpose.

    Skills used:
      - orchestrator_bus (report to orchestrator)
      - anthropic client (Claude API for feature group suggestions)
    """

    def __init__(self, bus: OrchestratorBus):
        self.bus = bus

    def run(
        self,
        user_intent: UserIntent,
        df: pd.DataFrame | None = None,
        iteration: int = 1,
    ) -> DatasetProfile | None:
        """
        Profile the dataset and suggest feature groups.

        Parameters
        ----------
        user_intent : UserIntent
            Captured clustering intent (business purpose + dataset path).
        df : pd.DataFrame | None
            If provided, use this DataFrame directly instead of loading from disk.
        iteration : int
            Pipeline iteration number.

        Returns
        -------
        DatasetProfile, or None if the dataset is BLOCKED (cannot proceed).
        """
        print(f"\n[DatasetExaminer] Iteration {iteration}")
        print(f"  Target entity : {user_intent.target_entity}")
        print(f"  Purpose       : {user_intent.business_purpose}")

        # ── Load dataset ──────────────────────────────────────────────────────
        if df is None:
            dataset_path = Path(user_intent.dataset_path)
            if not dataset_path.exists():
                self._report_blocked(
                    iteration=iteration,
                    issue=f"Dataset not found: {dataset_path}",
                    what_done="Attempted to load dataset",
                )
                return None

            print(f"  Loading: {dataset_path}")
            try:
                if dataset_path.suffix == ".parquet":
                    df = pd.read_parquet(dataset_path)
                elif dataset_path.suffix == ".csv":
                    df = pd.read_csv(dataset_path)
                else:
                    df = pd.read_csv(dataset_path)
            except Exception as e:
                self._report_blocked(
                    iteration=iteration,
                    issue=f"Failed to load dataset: {e}",
                    what_done="Attempted to load dataset",
                )
                return None

        n_rows, n_cols = df.shape
        print(f"  Shape: {n_rows:,} rows × {n_cols} columns")

        if n_rows == 0:
            self._report_blocked(iteration, "Dataset is empty (0 rows)", "Loaded dataset")
            return None

        # ── Column typing ──────────────────────────────────────────────────────
        column_types: dict[str, str] = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                column_types[col] = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                column_types[col] = "datetime"
            elif df[col].nunique() <= 2 and df[col].nunique() > 0:
                column_types[col] = "binary"
            elif df[col].dtype == object or pd.api.types.is_categorical_dtype(df[col]):
                column_types[col] = "categorical"
            else:
                column_types[col] = "other"

        numeric_cols = [c for c, t in column_types.items() if t == "numeric"]
        if not numeric_cols:
            self._report_blocked(
                iteration,
                "No numeric columns found — cannot build features",
                "Profiled column types",
            )
            return None

        # ── Missing rates ──────────────────────────────────────────────────────
        missing_rates = {col: round(float(df[col].isna().mean()), 4) for col in df.columns}
        high_missing = [c for c, r in missing_rates.items() if r > 0.30]

        # ── Distribution summary ───────────────────────────────────────────────
        distribution_summary: dict[str, dict[str, float]] = {}
        for col in numeric_cols[:50]:  # cap at 50 to keep prompt manageable
            s = df[col].dropna()
            if len(s) == 0:
                continue
            distribution_summary[col] = {
                "min":      round(float(s.min()), 4),
                "max":      round(float(s.max()), 4),
                "mean":     round(float(s.mean()), 4),
                "std":      round(float(s.std()), 4),
                "skewness": round(float(s.skew()), 4),
                "missing":  missing_rates[col],
            }

        # ── High cardinality categoricals ─────────────────────────────────────
        high_card = [
            col for col, t in column_types.items()
            if t == "categorical" and df[col].nunique() > 100
        ]

        # ── Algo hint from skewness ───────────────────────────────────────────
        skewness_values = [v["skewness"] for v in distribution_summary.values()]
        mean_skew = float(np.mean(np.abs(skewness_values))) if skewness_values else 0.0
        algo_hint = (
            "hierarchical"
            if mean_skew > 2.0
            else ("kmeans" if mean_skew < 0.5 else "hierarchical")
        )

        # ── Build schema summary for Claude ───────────────────────────────────
        schema_lines = ["Column name | Type | Missing% | Skewness | Example values"]
        for col in df.columns[:60]:
            ctype = column_types.get(col, "?")
            miss = f"{missing_rates.get(col, 0):.1%}"
            skew = f"{distribution_summary.get(col, {}).get('skewness', '—')}"
            try:
                examples = ", ".join(str(v) for v in df[col].dropna().unique()[:3])
            except Exception:
                examples = "?"
            schema_lines.append(f"{col} | {ctype} | {miss} | {skew} | {examples}")

        schema_str = "\n".join(schema_lines)
        n_schema_cols = min(len(df.columns), 60)

        # ── Ask Orchestrator for LLM guidance on feature groups ───────────────
        # DatasetExaminer does its own profiling (schema, distributions, skewness).
        # It asks the Orchestrator for LLM reasoning only to interpret the schema
        # in the context of the business purpose and suggest feature groups.
        prompt = f"""You are a data scientist examining a dataset that will be clustered by '{user_intent.target_entity}'.

Business purpose: {user_intent.business_purpose}
{f"Constraints: {user_intent.constraints}" if user_intent.constraints else ""}

Dataset shape: {n_rows:,} rows × {n_cols} columns ({len(numeric_cols)} numeric)
Mean feature skewness: {mean_skew:.2f}

Schema (first {n_schema_cols} columns):
{schema_str}

High-missing columns (>30%): {high_missing if high_missing else "none"}
High-cardinality columns (>100 unique values): {high_card if high_card else "none"}

Based on the schema and business purpose, suggest which GROUPS of features should be engineered.
For each group, briefly explain what behavioral dimension it captures and which columns to derive it from.

Return ONLY a valid JSON object (no markdown, no extra text):
{{
  "suggested_feature_groups": ["group_name_1", "group_name_2", ...],
  "group_details": {{
    "group_name_1": {{
      "description": "what this group captures",
      "source_columns": ["col_a", "col_b"],
      "rationale": "why this group is relevant to the business purpose"
    }}
  }},
  "overall_reasoning": "2-3 sentences on your feature engineering strategy",
  "algo_preference": "hierarchical or kmeans",
  "algo_rationale": "one sentence on why"
}}"""

        try:
            raw = self.bus.ask(
                agent="DatasetExaminer",
                purpose="suggest feature engineering groups from schema + business purpose",
                prompt=prompt,
                max_tokens=2048,
            ).strip()
            if "```" in raw:
                for part in raw.split("```"):
                    p = part.strip()
                    if p.startswith("json"):
                        p = p[4:].strip()
                    if p.startswith("{"):
                        raw = p
                        break
            claude_result = json.loads(raw)
        except Exception as e:
            print(f"  [DatasetExaminer] Orchestrator LLM call failed: {e} — using fallback groups")
            claude_result = {
                "suggested_feature_groups": ["frequency", "spend", "recency"],
                "overall_reasoning": f"Fallback groups due to error: {e}",
                "algo_preference": algo_hint,
                "algo_rationale": "Based on mean feature skewness.",
            }

        suggested_groups = claude_result.get("suggested_feature_groups", [])
        reasoning = claude_result.get("overall_reasoning", "")
        claude_algo = claude_result.get("algo_preference", algo_hint)
        if claude_algo in ("hierarchical", "kmeans"):
            algo_hint = claude_algo

        print(f"  Suggested feature groups: {suggested_groups}")

        # ── Build warnings ─────────────────────────────────────────────────────
        warnings: list[str] = []
        if high_missing:
            warnings.append(
                f"High missing rate (>30%) in columns: {high_missing}. "
                "Consider imputation before feature engineering."
            )
        if high_card:
            warnings.append(
                f"High-cardinality columns: {high_card}. "
                "May need grouping or encoding before use."
            )
        if len(numeric_cols) < 5:
            warnings.append(
                f"Only {len(numeric_cols)} numeric columns found. "
                "Feature engineering options are limited."
            )
        if mean_skew > 3.0:
            warnings.append(
                f"Mean feature skewness={mean_skew:.1f} is high. "
                "Log-transform recommended before clustering."
            )

        # ── Build profile ──────────────────────────────────────────────────────
        profile = DatasetProfile(
            n_rows=n_rows,
            n_cols=n_cols,
            column_types=column_types,
            missing_rates=missing_rates,
            distribution_summary=distribution_summary,
            high_cardinality_cols=high_card,
            suggested_feature_groups=suggested_groups,
            feature_group_reasoning=reasoning,
            warnings=warnings,
            algo_hint=algo_hint,
        )

        # ── Report to orchestrator ─────────────────────────────────────────────
        status = "warning" if warnings else "success"
        self.bus.report(OrchestratorMessage(
            agent="DatasetExaminer",
            iteration=iteration,
            status=status,
            what_was_done=(
                f"Profiled {n_rows:,}×{n_cols} dataset. "
                f"Found {len(numeric_cols)} numeric cols. "
                f"Claude suggested {len(suggested_groups)} feature groups."
            ),
            what_was_not_done=(
                "Did not sample sub-populations or run hypothesis tests "
                "for distribution differences."
            ),
            doubts=(
                f"Suggested groups are based on column names; actual feature "
                f"buildability depends on data quality. Mean skewness={mean_skew:.1f}."
            ),
            issues=warnings,
            metrics={
                "n_rows": n_rows,
                "n_numeric_cols": len(numeric_cols),
                "n_suggested_groups": len(suggested_groups),
                "mean_skewness": round(mean_skew, 2),
                "n_high_missing": len(high_missing),
                "algo_hint": algo_hint,
            },
            recommendation="proceed" if not warnings else "proceed",
            context={
                "suggested_feature_groups": suggested_groups,
                "group_details": claude_result.get("group_details", {}),
                "algo_preference": algo_hint,
            },
        ))

        return profile

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _report_blocked(self, iteration: int, issue: str, what_done: str) -> None:
        self.bus.report(OrchestratorMessage(
            agent="DatasetExaminer",
            iteration=iteration,
            status="blocked",
            what_was_done=what_done,
            what_was_not_done="Could not complete dataset profiling.",
            doubts="",
            issues=[issue],
            metrics={},
            recommendation="escalate",
        ))
        print(f"  [DatasetExaminer] BLOCKED: {issue}")
