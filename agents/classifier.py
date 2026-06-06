"""
ClassifierAgent  (Agent 4)

Contract: docs/agents/classifier.md. Skills: docs/skills/orchestrator_bus.md.

Treats the persona labels produced by PersonaNamingAgent as pseudo ground truth,
trains a classifier on the customer features, and evaluates separability via
stratified cross-validation.

If macro-F1 falls below the configured threshold, the LLM decides whether the
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

Supported classifier models:
  - random_forest      : robust baseline, handles high-dim, little tuning
  - xgboost            : powerful for tabular data, handles class imbalance
  - lightgbm           : very fast gradient boosting, excellent for large tabular
  - gradient_boosting  : good for mid-size data, often more accurate than RF
  - logistic_regression: fast, interpretable, best for linearly separable classes
  - knn                : simple non-parametric, good for small well-separated data
  - svm                : powerful for high-dimensional data with clear margins
  - naive_bayes        : extremely fast baseline, surprisingly effective for text
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from agents.state import ClassifierResult
from skills.orchestrator_bus import OrchestratorBus, OrchestratorMessage

# ── Log-transform detection ────────────────────────────────────────────────────


def _detect_log_cols(df, skewness_threshold: float = 2.0) -> list[str]:
    """Return non-negative numeric columns whose |skewness| exceeds the threshold."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return []
    # Deduplicate column names (keep first occurrence) to avoid ambiguous comparisons
    numeric = numeric.loc[:, ~numeric.columns.duplicated()]
    non_neg = [col for col in numeric.columns if float(numeric[col].min()) >= 0]
    if not non_neg:
        return []
    skews = numeric[non_neg].skew().abs()
    return list(skews[skews > skewness_threshold].index)


# ── Model construction ─────────────────────────────────────────────────────────

def _build_model(model_name: str):
    """Return the appropriate sklearn estimator for the given model name."""
    if model_name == 'xgboost':
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss',
                use_label_encoder=False,
                verbosity=0,
            )
        except ImportError:
            print('  [Classifier] xgboost not installed — falling back to random_forest')
            return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    elif model_name == 'lightgbm':
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            )
        except ImportError:
            print('  [Classifier] lightgbm not installed — falling back to gradient_boosting')
            return GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_name == 'gradient_boosting':
        return GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_name == 'logistic_regression':
        # sklearn >=1.5 removed `multi_class` (lbfgs handles multinomial
        # automatically when n_classes > 2). Older versions tolerated it.
        # Build kwargs dynamically so this works on both old and new sklearn.
        _kwargs = dict(max_iter=1000, random_state=42, solver='lbfgs', C=1.0)
        try:
            import inspect as _inspect
            if 'multi_class' in _inspect.signature(LogisticRegression).parameters:
                _kwargs['multi_class'] = 'auto'
        except Exception:  # noqa: BLE001
            pass
        return LogisticRegression(**_kwargs)
    elif model_name == 'knn':
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    elif model_name == 'svm':
        return SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
    elif model_name == 'naive_bayes':
        return GaussianNB()
    else:
        # Default: random_forest
        return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)


class ClassifierAgent:
    """
    Trains a classifier on all numeric features (using persona labels),
    evaluates via 5-fold stratified CV, and asks the LLM for routing if F1 is low.

    The classifier model is selected dynamically based on dataset characteristics,
    or can be specified via config (classifier_model key).
    """

    # CV macro-F1 below this triggers LLM consultation
    F1_THRESHOLD = 0.70

    def __init__(self, bus: OrchestratorBus):
        # ClassifierAgent trains and evaluates the model using its own ML skills.
        # If F1 is low, it asks the Orchestrator for LLM routing diagnosis.
        self.bus = bus

    def _select_model(
        self,
        n_samples: int,
        n_features: int,
        n_classes: int,
        history: list[ClassifierResult],
        feedback: str,
        config_model: str = 'auto',
    ) -> str:
        """
        Ask the LLM to select the best classifier model, or use config setting.

        Returns model name string: 'random_forest', 'xgboost', 'lightgbm',
        'gradient_boosting', 'logistic_regression', 'knn', 'svm', 'naive_bayes'.
        """
        _VALID_MODELS = {
            'random_forest', 'xgboost', 'lightgbm', 'gradient_boosting',
            'logistic_regression', 'knn', 'svm', 'naive_bayes',
        }
        if config_model and config_model != 'auto':
            if config_model in _VALID_MODELS:
                print(f'  [Classifier] Model fixed by config: {config_model}')
                return config_model

        # If bus is not available, default to random_forest
        if not self.bus:
            return 'random_forest'

        prev_model = None
        prev_f1 = None
        if history:
            last = history[-1]
            prev_model = getattr(last, 'model_name', None)
            prev_f1 = getattr(last, 'cv_f1_macro', None)

        history_str = ''
        if prev_model and prev_f1 is not None:
            history_str = f"\nPrevious model: {prev_model} → CV macro-F1={prev_f1:.3f}"

        prompt = f"""You are selecting a classifier model for a customer segmentation validation task.

Dataset characteristics:
  n_samples  : {n_samples}
  n_features : {n_features}
  n_classes  : {n_classes}
{history_str}
{f"Additional feedback: {feedback}" if feedback else ""}

Available models and their strengths:
  random_forest     : Robust baseline, handles high-dim, little tuning needed.
                      Good for most situations. DEFAULT safe choice.
  xgboost           : Best for tabular data, handles class imbalance well.
                      Slightly slower to train but often more accurate.
  lightgbm          : Very fast gradient boosting, excellent for large tabular
                      datasets (n_samples > 20k). Often beats XGBoost on speed.
  gradient_boosting : Good for mid-size data (n_samples < 50k), slower than RF
                      but often more accurate than RF.
  logistic_regression: Fast, interpretable, best for linearly separable classes.
                       Works well when n_features << n_samples.
  knn               : Simple non-parametric classifier. Good for small datasets
                      (n_samples < 5k) with well-separated clusters.
  svm               : Powerful for high-dimensional data (n_features > n_samples).
                      Good when clusters have clear margins. Slower on large data.
  naive_bayes       : Extremely fast, good baseline for text data or when
                      features are roughly independent. Often surprisingly effective.

Guidelines:
  - If n_samples > 50k and n_features > 100: prefer random_forest, xgboost, or lightgbm
  - If n_classes > 10: prefer random_forest, xgboost, or lightgbm (robust to many classes)
  - If previous model gave F1 < 0.5: try a different model family (e.g. tree → linear)
  - If n_features > n_samples: prefer svm or logistic_regression
  - If dataset is text / sparse: prefer naive_bayes or logistic_regression
  - If n_samples < 5k and clusters look well-separated: try knn
  - Default safe choice: random_forest

Return ONLY a valid JSON object (no markdown, no extra text):
{{"model": "<model_name>", "reasoning": "<1-2 sentences>"}}"""

        _VALID_MODELS = {
            'random_forest', 'xgboost', 'lightgbm', 'gradient_boosting',
            'logistic_regression', 'knn', 'svm', 'naive_bayes',
        }

        try:
            raw = self.bus.ask(
                agent="Classifier",
                purpose="select classifier model based on dataset characteristics",
                prompt=prompt,
                max_tokens=256,
            ).strip()
            if '```' in raw:
                for part in raw.split('```'):
                    p = part.strip()
                    if p.startswith('json'):
                        p = p[4:].strip()
                    if p.startswith('{'):
                        raw = p
                        break
            resp = json.loads(raw)
            model_choice = resp.get('model', 'random_forest')
            if model_choice not in _VALID_MODELS:
                model_choice = 'random_forest'
            print(f'  [Classifier] LLM selected model: {model_choice}  ({resp.get("reasoning", "")})')
            return model_choice
        except Exception as e:
            print(f'  [Classifier] Model selection failed ({e}) — defaulting to random_forest')
            return 'random_forest'

    def run(
        self,
        features_df: pd.DataFrame,
        cluster_labels: pd.Series,
        personas: dict,
        history: list[ClassifierResult] = None,
        feedback: str = '',
        iteration: int = 1,
        config: dict | None = None,
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
        config : dict | None
            Pipeline config (used to read classifier_model setting).
        """
        if history is None:
            history = []
        if config is None:
            config = {}

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
        n_entities = len(y_names)
        print(f'  Entities: {n_entities}  |  Personas (classes): {n_classes}')
        print(f'  Class distribution:')
        for name, count in y_names.value_counts().items():
            print(f'    {name:<45} {count:>4} ({count/n_entities:.1%})')

        # ── Prepare feature matrix ─────────────────────────────────────────────
        X = features_df.select_dtypes(include=[np.number]).copy()
        log_cols = _detect_log_cols(X)
        for col in log_cols:
            X[col] = np.log1p(X[col])

        feature_names = list(X.columns)

        le = LabelEncoder()
        y = le.fit_transform(y_names)
        class_names = list(le.classes_)

        # ── Select model ──────────────────────────────────────────────────────
        config_model = config.get('classifier_model', 'auto')
        model_name = self._select_model(
            n_samples=n_entities,
            n_features=len(feature_names),
            n_classes=n_classes,
            history=history,
            feedback=feedback,
            config_model=config_model,
        )
        clf_model = _build_model(model_name)
        print(f'  Using classifier: {model_name}')

        # ── Stratified 5-fold CV ──────────────────────────────────────────────
        min_class_count = int(y_names.value_counts().min())
        n_splits = min(5, max(2, min_class_count))

        print(f'  Running {n_splits}-fold stratified CV ...')
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        y_pred_cv = cross_val_predict(clf_model, X, y, cv=cv)

        cv_accuracy    = float(accuracy_score(y, y_pred_cv))
        cv_f1_macro    = float(f1_score(y, y_pred_cv, average='macro',    zero_division=0))
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
        clf_model.fit(X, y)
        if hasattr(clf_model, 'feature_importances_'):
            importances = {
                feature_names[i]: round(float(clf_model.feature_importances_[i]), 6)
                for i in range(len(feature_names))
            }
        elif hasattr(clf_model, 'coef_'):
            # Logistic regression: use mean absolute coefficient
            coefs = np.abs(clf_model.coef_).mean(axis=0)
            importances = {
                feature_names[i]: round(float(coefs[i]), 6)
                for i in range(len(feature_names))
            }
        else:
            importances = {f: 0.0 for f in feature_names}

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
                model=clf_model,
                label_encoder=le,
            )
            # Store model_name as attribute for history access
            result.model_name = model_name  # type: ignore[attr-defined]

            if self.bus:
                worst_3 = sorted(per_class_f1.items(), key=lambda x: x[1])[:3]
                self.bus.report(OrchestratorMessage(
                    agent="Classifier",
                    iteration=iteration,
                    status="success",
                    what_was_done=(
                        f"Trained {model_name}, {n_splits}-fold stratified CV. "
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
                        "model_name": model_name,
                    },
                    recommendation="proceed",
                ))
            return result

        # ── Performance is poor — ask LLM for routing ─────────────────────────
        print(f'  Performance below threshold ({cv_f1_macro:.4f} < {self.F1_THRESHOLD}). Consulting LLM...')
        decision = self._ask_llm_routing(
            cv_f1_macro=cv_f1_macro,
            cv_accuracy=cv_accuracy,
            per_class_f1=per_class_f1,
            n_classes=n_classes,
            top_features=[f for f, _ in top10],
            history=history,
            feedback=feedback,
            model_name=model_name,
        )

        action = decision.get('action', 'recluster')
        reasoning = decision.get('reasoning', '')
        print(f'  LLM decision: {action}  |  {reasoning}')

        if self.bus:
            worst_3 = sorted(per_class_f1.items(), key=lambda x: x[1])[:3]
            self.bus.report(OrchestratorMessage(
                agent="Classifier",
                iteration=iteration,
                status="warning",
                what_was_done=(
                    f"Trained {model_name}, {n_splits}-fold CV. "
                    f"CV macro-F1={cv_f1_macro:.3f} < {self.F1_THRESHOLD}. "
                    f"LLM routed: {action}."
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
                    "model_name": model_name,
                    "llm_action": action,
                },
                recommendation="retry",
                context={"llm_routing": decision},
            ))

        result = ClassifierResult(
            action=action,
            cv_accuracy=cv_accuracy,
            cv_f1_macro=cv_f1_macro,
            cv_f1_weighted=cv_f1_weighted,
            feature_importances=importances,
            per_class_f1=per_class_f1,
            reasoning=reasoning,
            iteration=iteration,
            model=clf_model,
            label_encoder=le,
        )
        result.model_name = model_name  # type: ignore[attr-defined]
        return result

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _ask_llm_routing(
        self,
        cv_f1_macro: float,
        cv_accuracy: float,
        per_class_f1: dict,
        n_classes: int,
        top_features: list[str],
        history: list[ClassifierResult],
        feedback: str,
        model_name: str = 'classifier',
    ) -> dict:
        """
        Ask the LLM whether the poor classifier performance is due to bad features
        or bad clusters, and which agent should be re-run.
        """
        history_lines = []
        for r in history:
            m = getattr(r, 'model_name', 'classifier')
            history_lines.append(
                f'  Iteration {r.iteration}: action={r.action}  '
                f'cv_f1={r.cv_f1_macro:.3f}  model={m}  reason={r.reasoning}'
            )
        history_str = '\n'.join(history_lines) if history_lines else '  No prior classifier runs.'

        worst_classes = sorted(per_class_f1.items(), key=lambda x: x[1])[:3]
        worst_str = ', '.join(f'{name}(F1={sc:.2f})' for name, sc in worst_classes)

        feedback_section = f'\nUser / system feedback: {feedback}\n' if feedback else ''

        prompt = f"""You are diagnosing a customer segmentation pipeline.

A {model_name} classifier was trained on customer behavioral features using
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
