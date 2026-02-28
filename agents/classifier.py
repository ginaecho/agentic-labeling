"""
ClassifierAgent  (Agent 4)

Contract: docs/agents/classifier.md. Skills: docs/skills/orchestrator_bus.md.

Treats the persona labels produced by PersonaNamingAgent as pseudo ground truth,
trains a Random Forest classifier on the customer features, and evaluates
separability via stratified cross-validation.

If macro-F1 falls below the configured threshold, Claude decides whether the
root cause is:
  (a) bad features  → action = 'reselect_features'
  (b) bad clusters  → action = 'recluster'

If performance is acceptable, action = 'proceed'.

Threshold logic:
  - Because labels come from the same features (pseudo labels), a well-separated
    clustering WILL yield near-perfect CV scores. Low CV F1 means the cluster
    boundaries are not crisp in feature space — a true signal of poor segmentation.
  - Default threshold: macro F1 >= 0.70
    (lower than 1.0 because small, noisy clusters can legitimately score lower)
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from agents.state import ClassifierResult
from skills.orchestrator_bus import OrchestratorBus, OrchestratorMessage

# ── Constants (mirrors notebook 03 / 05) ──────────────────────────────────────
CATEGORIES = [
    'entertainment', 'food_dining', 'gas_transport', 'grocery_net', 'grocery_pos',
    'health_fitness', 'home', 'kids_pets', 'misc_net', 'misc_pos',
    'personal_care', 'shopping_net', 'shopping_pos', 'travel',
]
WINDOWS = [6, 12]

LOG_COLS_BASE = [
    'avg_txn_amt', 'std_txn_amt', 'max_txn_amt',
    'total_spend', 'total_txn_count', 'n_unique_merchants', 'avg_days_between_txn',
]


def _build_log_cols(columns) -> list[str]:
    candidates = (
        [f'n_txn_{cat}_{w}m'      for cat in CATEGORIES for w in WINDOWS]
        + [f'amt_{cat}_{w}m'      for cat in CATEGORIES for w in WINDOWS]
        + [f'avg_spend_{cat}_{w}m' for cat in CATEGORIES for w in WINDOWS]
        + LOG_COLS_BASE
    )
    return [c for c in candidates if c in columns]


class ClassifierAgent:
    """
    Trains a Random Forest on all numeric features (using persona labels),
    evaluates via 5-fold stratified CV, and asks Claude for routing if F1 is low.
    """

    # CV macro-F1 below this triggers Claude consultation
    F1_THRESHOLD = 0.70

    def __init__(self, bus: OrchestratorBus):
        # ClassifierAgent trains and evaluates the RF using its own ML skills.
        # If F1 is low, it asks the Orchestrator for LLM routing diagnosis.
        self.bus = bus

    def run(
        self,
        features_df: pd.DataFrame,
        cluster_labels: pd.Series,
        personas: dict,
        history: list[ClassifierResult] = None,
        feedback: str = '',
        iteration: int = 1,
    ) -> ClassifierResult:
        """
        Parameters
        ----------
        features_df : pd.DataFrame
            Raw customer features (no 'cluster' column).
        cluster_labels : pd.Series
            Integer cluster assignments from ClusteringAgent.
        personas : dict
            cid (str) -> persona dict with at least a 'name' key.
        history : list[ClassifierResult]
            Previous classifier results for context.
        feedback : str
            Free-text guidance from user or previous round.
        iteration : int
        """
        if history is None:
            history = []

        print(f'\n[Classifier] Iteration {iteration}')
        if feedback:
            print(f'  Feedback: {feedback}')

        # ── Map cluster IDs to persona names ──────────────────────────────────
        cluster_to_persona = {
            int(cid): data.get('name', f'Cluster {cid}')
            for cid, data in personas.items()
        }
        y_names = cluster_labels.map(cluster_to_persona)
        missing = y_names.isna().sum()
        if missing > 0:
            print(f'  Warning: {missing} customers have no persona mapping — dropping them.')
            valid = y_names.notna()
            y_names = y_names[valid]
            features_df = features_df[valid]

        n_classes = y_names.nunique()
        n_customers = len(y_names)
        print(f'  Customers: {n_customers}  |  Personas (classes): {n_classes}')
        print(f'  Class distribution:')
        for name, count in y_names.value_counts().items():
            print(f'    {name:<45} {count:>4} ({count/n_customers:.1%})')

        # ── Prepare feature matrix ─────────────────────────────────────────────
        X = features_df.select_dtypes(include=[np.number]).copy()
        log_cols = _build_log_cols(X.columns)
        for col in log_cols:
            X[col] = np.log1p(X[col])

        feature_names = list(X.columns)

        le = LabelEncoder()
        y = le.fit_transform(y_names)
        class_names = list(le.classes_)

        # ── Stratified 5-fold CV ──────────────────────────────────────────────
        # Use at most 5 splits; for tiny classes use min(5, min_class_count)
        min_class_count = int(y_names.value_counts().min())
        n_splits = min(5, max(2, min_class_count))

        print(f'  Running {n_splits}-fold stratified CV ...')
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

        y_pred_cv = cross_val_predict(rf, X, y, cv=cv)

        cv_accuracy   = float(accuracy_score(y, y_pred_cv))
        cv_f1_macro   = float(f1_score(y, y_pred_cv, average='macro',    zero_division=0))
        cv_f1_weighted = float(f1_score(y, y_pred_cv, average='weighted', zero_division=0))

        # Per-class F1
        f1_per_class = f1_score(y, y_pred_cv, average=None, zero_division=0)
        per_class_f1 = {
            class_names[i]: round(float(f1_per_class[i]), 4)
            for i in range(len(class_names))
        }

        print(f'  CV accuracy    : {cv_accuracy:.4f}')
        print(f'  CV F1 (macro)  : {cv_f1_macro:.4f}  (threshold ≥ {self.F1_THRESHOLD})')
        print(f'  CV F1 (weighted): {cv_f1_weighted:.4f}')
        print(f'  Per-class F1:')
        for name, score in sorted(per_class_f1.items(), key=lambda x: x[1]):
            bar = '█' * int(score * 20)
            print(f'    {name:<45} {score:.3f}  {bar}')

        # ── Fit final model on all data for feature importances ───────────────
        rf.fit(X, y)
        importances = {
            feature_names[i]: round(float(rf.feature_importances_[i]), 6)
            for i in range(len(feature_names))
        }
        top10 = sorted(importances.items(), key=lambda x: -x[1])[:10]
        print(f'  Top 10 features: {[f for f, _ in top10]}')

        # ── Decide: proceed or route back? ────────────────────────────────────
        if cv_f1_macro >= self.F1_THRESHOLD:
            print(f'  Performance OK — proceeding.')
            result = ClassifierResult(
                action='proceed',
                cv_accuracy=cv_accuracy,
                cv_f1_macro=cv_f1_macro,
                cv_f1_weighted=cv_f1_weighted,
                feature_importances=importances,
                per_class_f1=per_class_f1,
                reasoning=f'CV macro-F1 {cv_f1_macro:.3f} ≥ threshold {self.F1_THRESHOLD}.',
                iteration=iteration,
                model=rf,
                label_encoder=le,
            )
            if self.bus:
                worst_3 = sorted(per_class_f1.items(), key=lambda x: x[1])[:3]
                self.bus.report(OrchestratorMessage(
                    agent="Classifier",
                    iteration=iteration,
                    status="success",
                    what_was_done=(
                        f"Trained RF, {n_splits}-fold stratified CV. "
                        f"CV macro-F1={cv_f1_macro:.3f} ≥ {self.F1_THRESHOLD} threshold."
                    ),
                    what_was_not_done="Did not compute SHAP values.",
                    doubts=(
                        f"Lowest-F1 personas: "
                        + ", ".join(f"{n}({s:.2f})" for n, s in worst_3)
                        if worst_3 else ""
                    ),
                    issues=[],
                    metrics={
                        "cv_f1_macro": round(cv_f1_macro, 4),
                        "cv_accuracy": round(cv_accuracy, 4),
                        "n_classes": n_classes,
                    },
                    recommendation="proceed",
                ))
            return result

        # ── Performance is poor — ask Claude for routing ───────────────────────
        print(f'  Performance below threshold ({cv_f1_macro:.4f} < {self.F1_THRESHOLD}). Consulting Claude...')
        decision = self._ask_claude_routing(
            cv_f1_macro=cv_f1_macro,
            cv_accuracy=cv_accuracy,
            per_class_f1=per_class_f1,
            n_classes=n_classes,
            top_features=[f for f, _ in top10],
            history=history,
            feedback=feedback,
        )

        action = decision.get('action', 'recluster')
        reasoning = decision.get('reasoning', '')
        print(f'  Claude decision: {action}  |  {reasoning}')

        if self.bus:
            worst_3 = sorted(per_class_f1.items(), key=lambda x: x[1])[:3]
            self.bus.report(OrchestratorMessage(
                agent="Classifier",
                iteration=iteration,
                status="warning",
                what_was_done=(
                    f"Trained RF, {n_splits}-fold CV. "
                    f"CV macro-F1={cv_f1_macro:.3f} < {self.F1_THRESHOLD}. "
                    f"Claude routed: {action}."
                ),
                what_was_not_done="Did not compute SHAP values.",
                doubts=f"Hardest personas: " + ", ".join(f"{n}({s:.2f})" for n, s in worst_3),
                issues=[
                    f"CV macro-F1={cv_f1_macro:.3f} below threshold {self.F1_THRESHOLD}. "
                    f"Clusters may not be well-separated."
                ],
                metrics={
                    "cv_f1_macro": round(cv_f1_macro, 4),
                    "cv_accuracy": round(cv_accuracy, 4),
                    "n_classes": n_classes,
                    "claude_action": action,
                },
                recommendation="retry",
                context={"claude_routing": decision},
            ))

        return ClassifierResult(
            action=action,
            cv_accuracy=cv_accuracy,
            cv_f1_macro=cv_f1_macro,
            cv_f1_weighted=cv_f1_weighted,
            feature_importances=importances,
            per_class_f1=per_class_f1,
            reasoning=reasoning,
            iteration=iteration,
            model=rf,
            label_encoder=le,
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _ask_claude_routing(
        self,
        cv_f1_macro: float,
        cv_accuracy: float,
        per_class_f1: dict,
        n_classes: int,
        top_features: list[str],
        history: list[ClassifierResult],
        feedback: str,
    ) -> dict:
        """
        Ask Claude whether the poor classifier performance is due to bad features
        or bad clusters, and which agent should be re-run.
        """
        history_lines = []
        for r in history:
            history_lines.append(
                f'  Iteration {r.iteration}: action={r.action}  '
                f'cv_f1={r.cv_f1_macro:.3f}  reason={r.reasoning}'
            )
        history_str = '\n'.join(history_lines) if history_lines else '  No prior classifier runs.'

        worst_classes = sorted(per_class_f1.items(), key=lambda x: x[1])[:3]
        worst_str = ', '.join(f'{name}(F1={sc:.2f})' for name, sc in worst_classes)

        feedback_section = f'\nUser / system feedback: {feedback}\n' if feedback else ''

        prompt = f"""You are diagnosing a customer segmentation pipeline.

A Random Forest classifier was trained on customer behavioral features using
persona labels (from clustering) as pseudo ground truth. Poor classifier
performance means the clusters are NOT well-separated in feature space.

Results:
  CV macro-F1  : {cv_f1_macro:.4f}  (threshold ≥ {self.F1_THRESHOLD})
  CV accuracy  : {cv_accuracy:.4f}
  N classes    : {n_classes}
  Worst personas (hardest to predict): {worst_str}
  Top predictive features: {', '.join(top_features[:5])}

History of attempts:
{history_str}
{feedback_section}
Diagnose the root cause and recommend ONE of:
  (a) reselect_features — the current features don't capture enough signal to
      distinguish the personas. Going back to feature selection might reveal
      better dimensions (e.g. time-window ratios, category interaction features).
  (b) recluster — the number of clusters or the clustering algorithm is wrong.
      The clusters overlap; changing n_clusters or the algorithm may help.

Return ONLY a valid JSON object (no markdown, no extra text):
{{"action": "reselect_features" or "recluster", "reasoning": "2-3 sentences"}}"""

        # ClassifierAgent has done its own ML work (CV, F1, feature importances).
        # It now asks the Orchestrator to diagnose the root cause and route.
        raw = self.bus.ask(
            agent="Classifier",
            purpose=f"diagnose low CV F1={cv_f1_macro:.3f} — route to reselect_features or recluster",
            prompt=prompt,
            max_tokens=512,
        ).strip()
        if '```' in raw:
            for part in raw.split('```'):
                p = part.strip()
                if p.startswith('json'):
                    p = p[4:].strip()
                if p.startswith('{'):
                    raw = p
                    break
        return json.loads(raw)
