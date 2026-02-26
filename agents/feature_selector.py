"""
FeatureSelectionAgent

Uses PCA importance + autoencoder reconstruction error to score all 108 features,
then asks Claude which subset to keep for discovering distinct spending personas.
"""
from __future__ import annotations

import json
import numpy as np
import anthropic

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from agents.state import FeatureSelectionResult

# ── Constants (mirrored from notebook 03) ─────────────────────────────────────
CATEGORIES = [
    'entertainment', 'food_dining', 'gas_transport', 'grocery_net', 'grocery_pos',
    'health_fitness', 'home', 'kids_pets', 'misc_net', 'misc_pos',
    'personal_care', 'shopping_net', 'shopping_pos', 'travel',
]
WINDOWS = [6, 12]


def _build_log_cols(columns) -> list[str]:
    """Build LOG_COLS list, filtering to columns actually present in the DataFrame."""
    candidates = (
        [f'n_txn_{cat}_{w}m'     for cat in CATEGORIES for w in WINDOWS]
        + [f'amt_{cat}_{w}m'     for cat in CATEGORIES for w in WINDOWS]
        + [f'avg_spend_{cat}_{w}m' for cat in CATEGORIES for w in WINDOWS]
        + ['total_txn_count', 'total_spend', 'avg_txn_amt', 'std_txn_amt',
           'max_txn_amt', 'n_unique_merchants', 'avg_days_between_txn']
    )
    return [c for c in candidates if c in columns]


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise a 1-D array to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


class FeatureSelectionAgent:
    """
    Scores all features with PCA importance + AE reconstruction error,
    then asks Claude to choose the best subset for persona discovery.
    """

    def __init__(self, client: anthropic.Anthropic):
        self.client = client

    def run(self, features_df, feedback: str = '', iteration: int = 1) -> FeatureSelectionResult:
        """
        Parameters
        ----------
        features_df : pd.DataFrame
            The raw customer features table (108 columns, no 'cluster' column).
        feedback : str
            Free-text feedback from the previous round (empty on first run).
        iteration : int
            Which iteration this is (for logging and state tracking).

        Returns
        -------
        FeatureSelectionResult
        """
        print(f'\n[FeatureSelector] Iteration {iteration}')
        if feedback:
            print(f'  Feedback from previous round: {feedback}')

        # ── Step 1: Preprocess (mirrors notebook 03 cell d1f6406a) ─────────────
        log_cols = _build_log_cols(features_df.columns)
        X = features_df.copy()
        for col in log_cols:
            X[col] = np.log1p(X[col])

        # Drop non-numeric / cluster columns if present
        X = X.select_dtypes(include=[np.number])
        feature_names = list(X.columns)
        n_features = len(feature_names)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        print(f'  Features: {n_features}  |  Customers: {X_scaled.shape[0]}')

        # ── Step 2: PCA importance score ───────────────────────────────────────
        n_components = min(50, n_features)
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X_scaled)

        # Weighted sum of squared loadings: shape (n_features,)
        pca_score = np.sum(
            pca.components_ ** 2 * pca.explained_variance_ratio_[:, None],
            axis=0,
        )

        # ── Step 3: Autoencoder reconstruction score ───────────────────────────
        bottleneck = max(8, n_features // 5)
        ae = MLPRegressor(
            hidden_layer_sizes=(64, bottleneck, 64),
            activation='relu',
            max_iter=300,
            early_stopping=True,
            random_state=42,
            verbose=False,
        )
        ae.fit(X_scaled, X_scaled)
        recon_error = np.mean((X_scaled - ae.predict(X_scaled)) ** 2, axis=0)
        # High error = feature is unique/hard to reconstruct = keep it

        # ── Step 4: Combined score ─────────────────────────────────────────────
        combined = 0.5 * _normalise(pca_score) + 0.5 * _normalise(recon_error)
        ranked_idx = np.argsort(combined)[::-1]  # highest first

        # Build a summary table for Claude (top 40 + bottom 10)
        rows = []
        for rank, idx in enumerate(ranked_idx):
            rows.append({
                'rank': rank + 1,
                'feature': feature_names[idx],
                'pca_score': round(float(pca_score[idx]), 6),
                'ae_recon_error': round(float(recon_error[idx]), 6),
                'combined_score': round(float(combined[idx]), 6),
            })

        table_lines = ['rank | feature | pca_score | ae_error | combined']
        for r in rows[:40]:
            table_lines.append(
                f'{r["rank"]:>4} | {r["feature"]:<35} | '
                f'{r["pca_score"]:.5f} | {r["ae_recon_error"]:.5f} | {r["combined_score"]:.5f}'
            )
        table_lines.append('  ... (showing top 40 of {n_features})')
        table_str = '\n'.join(table_lines)

        # ── Step 5: Ask Claude ─────────────────────────────────────────────────
        feedback_section = (
            f'\nFeedback from previous round:\n{feedback}\n'
            if feedback else ''
        )

        prompt = f"""You are a data scientist helping to select the best features for
customer segmentation. We have {n_features} behavioral features from bank transaction data.

Goal: find the most DISTINCT customer spending personas — groups that behave very
differently from each other.

Feature ranking (PCA importance × autoencoder uniqueness score):
{table_str}
{feedback_section}
All 14 spending categories: {CATEGORIES}
Feature naming convention:
  n_txn_{{cat}}_{{w}}m  = number of transactions in that category in past w months
  amt_{{cat}}_{{w}}m    = total spend in that category in past w months
  avg_spend_{{cat}}_{{w}}m = average spend per transaction in past w months

Instructions:
1. Choose a subset of features that will produce the most distinct, interpretable clusters.
2. Prefer features that capture DIFFERENT behavioral dimensions (frequency, spend level,
   category mix, transaction size, recency). Avoid keeping many redundant features that
   say the same thing (e.g. n_txn_cat_6m AND n_txn_cat_12m for every category — pick one window
   unless both windows add real signal).
3. Keep enough features to represent all 14 spending categories but trim the redundant ones.
4. Typical good range: 30–60 features.

Return ONLY a valid JSON object (no markdown, no extra text):
{{
  "selected_features": ["feature_name_1", "feature_name_2", ...],
  "reasoning": "2-3 sentences explaining your selection strategy"
}}"""

        print(f'  Calling Claude for feature selection...')
        response = self.client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=2048,
            messages=[{'role': 'user', 'content': prompt}],
        )

        raw = response.content[0].text.strip()
        if '```' in raw:
            for part in raw.split('```'):
                p = part.strip()
                if p.startswith('json'):
                    p = p[4:].strip()
                if p.startswith('{'):
                    raw = p
                    break

        result = json.loads(raw)
        selected = [f for f in result['selected_features'] if f in feature_names]
        reasoning = result.get('reasoning', '')

        # Build score dicts for the returned result
        pca_scores_dict = {feature_names[i]: float(pca_score[i]) for i in range(n_features)}
        ae_scores_dict = {feature_names[i]: float(recon_error[i]) for i in range(n_features)}

        print(f'  Selected {len(selected)} features. Reasoning: {reasoning}')

        return FeatureSelectionResult(
            selected_features=selected,
            n_features=len(selected),
            pca_scores=pca_scores_dict,
            ae_scores=ae_scores_dict,
            reasoning=reasoning,
            iteration=iteration,
        )
