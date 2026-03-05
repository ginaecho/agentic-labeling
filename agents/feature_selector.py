"""
FeatureSelectionAgent

Contract: docs/agents/feature_selector.md. Skills: docs/skills/orchestrator_bus.md,
docs/skills/vif_checker.md.

Scores all features with PCA importance + autoencoder reconstruction error,
applies VIF and correlation gates (via skills.vif_checker), then asks Claude
which subset to keep for discovering distinct spending personas.

Pipeline:
  1. Log-transform skewed columns
  2. Scale to zero-mean, unit-variance
  3. PCA importance score (weighted squared loadings)
  4. Autoencoder reconstruction error score
  5. Combined score = 0.5 × PCA + 0.5 × AE
  6. VIF gate  — iteratively remove highest-VIF feature until all VIF < threshold
  7. Correlation gate — flag pairs with |r| > corr_threshold
  8. Claude selects final subset from VIF-filtered ranked list

Skills used: skills.vif_checker, skills.orchestrator_bus
"""
from __future__ import annotations

import json
import numpy as np

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from agents.state import FeatureSelectionResult
from agents.user_input import UserIntent
from skills.vif_checker import compute_vif, remove_high_vif, flag_high_correlation
from skills.orchestrator_bus import OrchestratorBus, OrchestratorMessage

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
    Scores features with PCA + AE, applies VIF and correlation gates,
    then asks Claude to choose the best subset for persona discovery.
    """

    def __init__(
        self,
        bus: OrchestratorBus,
        vif_threshold: float = 10.0,
        corr_threshold: float = 0.85,
    ):
        # FeatureSelectionAgent owns its ML skills (PCA, AE, VIF).
        # LLM reasoning is requested through the bus → Orchestrator.
        self.bus = bus
        self.vif_threshold = vif_threshold
        self.corr_threshold = corr_threshold

    def run(
        self,
        features_df,
        user_intent: UserIntent | None = None,
        feedback: str = '',
        iteration: int = 1,
        vif_threshold: float | None = None,
        feature_focus: str = '',
    ) -> FeatureSelectionResult:
        """
        Parameters
        ----------
        features_df : pd.DataFrame
            The raw customer features table (no 'cluster' column).
        user_intent : UserIntent | None
            Clustering intent for context-aware Claude prompting.
        feedback : str
            Free-text feedback from the previous round (empty on first run).
        iteration : int
            Which iteration this is.
        vif_threshold : float | None
            Override the instance VIF threshold (set by orchestrator tuning).
        feature_focus : str
            Orchestrator hint injected into the Claude prompt to guide selection.

        Returns
        -------
        FeatureSelectionResult
        """
        # Resolve effective thresholds — orchestrator may override per-iteration
        effective_vif = vif_threshold if vif_threshold is not None else self.vif_threshold

        print(f'\n[FeatureSelector] Iteration {iteration}  (vif_threshold={effective_vif})')
        if feedback:
            print(f'  Feedback from previous round: {feedback}')

        # ── Step 1: Preprocess ─────────────────────────────────────────────────
        log_cols = _build_log_cols(features_df.columns)
        X = features_df.copy()
        for col in log_cols:
            X[col] = np.log1p(X[col])

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

        # ── Step 4: Combined score ─────────────────────────────────────────────
        combined = 0.5 * _normalise(pca_score) + 0.5 * _normalise(recon_error)
        ranked_idx = np.argsort(combined)[::-1]  # highest first

        # ── Step 5: VIF gate ───────────────────────────────────────────────────
        import pandas as pd
        X_for_vif = pd.DataFrame(X_scaled, columns=feature_names)

        print(f'  Running VIF gate (threshold={effective_vif})...')
        X_clean, removed_by_vif = remove_high_vif(
            X_for_vif,
            threshold=effective_vif,
            min_features=10,
            verbose=True,
        )
        clean_feature_names = list(X_clean.columns)
        n_after_vif = len(clean_feature_names)
        print(f'  After VIF gate: {n_after_vif} features  (removed {len(removed_by_vif)})')

        # Failure mode per docs/agents/feature_selector.md: < 10 features survive VIF → blocked
        if n_after_vif < 10:
            if self.bus:
                self.bus.report(OrchestratorMessage(
                    agent="FeatureSelector",
                    iteration=iteration,
                    status="blocked",
                    what_was_done=(
                        f"Scored {n_features} features, ran VIF gate; "
                        f"only {n_after_vif} survived (need ≥ 10)."
                    ),
                    what_was_not_done="Cannot select a valid subset with < 10 features.",
                    doubts="Consider relaxing VIF threshold (e.g. to 8) or engineering more features.",
                    issues=[f"Only {n_after_vif} features after VIF gate (min 10)."],
                    metrics={
                        "n_input_features": n_features,
                        "n_after_vif": n_after_vif,
                        "n_removed_by_vif": len(removed_by_vif),
                    },
                    recommendation="escalate",
                    context={"vif_removed": removed_by_vif},
                ))
            vif_df_early = compute_vif(X_clean)
            vif_table_early = dict(zip(vif_df_early["feature"], vif_df_early["vif"].round(2)))
            return FeatureSelectionResult(
                selected_features=[],
                n_features=0,
                pca_scores={},
                ae_scores={},
                vif_table=vif_table_early,
                removed_by_vif=removed_by_vif,
                reasoning="Blocked: fewer than 10 features survived VIF gate. See docs/agents/feature_selector.md.",
                iteration=iteration,
            )

        # VIF table for surviving features
        vif_df = compute_vif(X_clean)
        vif_table = dict(zip(vif_df["feature"], vif_df["vif"].round(2)))

        # ── Step 6: Correlation gate ───────────────────────────────────────────
        high_corr_pairs = flag_high_correlation(X_clean, threshold=self.corr_threshold)
        if high_corr_pairs:
            print(f'  High-correlation pairs (|r|>{self.corr_threshold}):')
            for a, b, r in high_corr_pairs[:5]:
                print(f'    {a} ↔ {b}  r={r:.3f}')

        # ── Step 7: Build ranking table for VIF-surviving features ─────────────
        # Re-rank among surviving features using combined score
        surviving_set = set(clean_feature_names)
        rows = []
        for rank_pos, idx in enumerate(ranked_idx):
            feat = feature_names[idx]
            if feat in surviving_set:
                rows.append({
                    'rank': rank_pos + 1,  # original rank before VIF removal
                    'feature': feat,
                    'pca_score': round(float(pca_score[idx]), 6),
                    'ae_recon_error': round(float(recon_error[idx]), 6),
                    'combined_score': round(float(combined[idx]), 6),
                    'vif': vif_table.get(feat, '?'),
                })

        # Sort by combined score descending
        rows.sort(key=lambda r: -r['combined_score'])

        table_lines = ['rank | feature | pca_score | ae_error | combined | vif']
        for i, r in enumerate(rows[:40]):
            table_lines.append(
                f'{r["rank"]:>4} | {r["feature"]:<35} | '
                f'{r["pca_score"]:.5f} | {r["ae_recon_error"]:.5f} | '
                f'{r["combined_score"]:.5f} | {r["vif"]}'
            )
        if len(rows) > 40:
            table_lines.append(f'  ... ({len(rows) - 40} more survived VIF gate)')
        table_str = '\n'.join(table_lines)

        # ── Step 8: Ask Claude ─────────────────────────────────────────────────
        feedback_section = (
            f'\nFeedback from previous round:\n{feedback}\n'
            if feedback else ''
        )
        intent_section = ""
        if user_intent:
            intent_section = (
                f"\nClustering intent:\n"
                f"  Target: {user_intent.target_entity}\n"
                f"  Purpose: {user_intent.business_purpose}\n"
            )

        high_corr_note = ""
        if high_corr_pairs:
            high_corr_note = (
                f"\nHigh-correlation pairs still present (|r|>{self.corr_threshold}) "
                f"— avoid selecting both in a pair:\n"
                + "\n".join(f"  {a} ↔ {b}  r={r:.3f}" for a, b, r in high_corr_pairs[:5])
                + "\n"
            )

        focus_section = (
            f'\nOrchestrator guidance: {feature_focus}\n'
            if feature_focus else ''
        )

        prompt = f"""You are a data scientist selecting features for customer segmentation.

{intent_section}
We started with {n_features} features. After VIF gate (threshold={effective_vif}):
  Removed {len(removed_by_vif)} high-VIF features: {removed_by_vif[:10]}{"..." if len(removed_by_vif) > 10 else ""}
  Remaining: {n_after_vif} features

Feature ranking (PCA importance × autoencoder uniqueness score) — VIF-filtered:
{table_str}
{high_corr_note}
{feedback_section}
{focus_section}
Spending categories: {CATEGORIES}
Feature naming: n_txn_{{cat}}_{{w}}m = transactions, amt_{{cat}}_{{w}}m = total spend,
avg_spend_{{cat}}_{{w}}m = avg per transaction, w is months (6 or 12).

Instructions:
1. Choose a subset that produces the MOST DISTINCT, INTERPRETABLE clusters.
2. Prefer features capturing DIFFERENT behavioral dimensions (frequency, spend,
   category mix, transaction size, recency). Avoid redundant pairs flagged above.
3. Represent all 14 spending categories but trim redundant windows.
4. Typical good range: 25–55 features.

Return ONLY a valid JSON object (no markdown, no extra text):
{{
  "selected_features": ["feature_name_1", "feature_name_2", ...],
  "reasoning": "2-3 sentences explaining your selection strategy"
}}"""

        # FeatureSelector has done all its own work: PCA scores, AE scores,
        # VIF filtering, correlation flagging. Now it asks the Orchestrator
        # to use LLM reasoning to make the final feature subset decision.
        raw = self.bus.ask(
            agent="FeatureSelector",
            purpose=f"select best feature subset from {n_after_vif} VIF-filtered candidates",
            prompt=prompt,
            max_tokens=2048,
        ).strip()
        if '```' in raw:
            for part in raw.split('```'):
                p = part.strip()
                if p.startswith('json'):
                    p = p[4:].strip()
                if p.startswith('{'):
                    raw = p
                    break

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as e:
            # Per docs/agents/feature_selector.md: invalid JSON → warning, retry
            if self.bus:
                self.bus.report(OrchestratorMessage(
                    agent="FeatureSelector",
                    iteration=iteration,
                    status="warning",
                    what_was_done="Scored features, ran VIF gate; Claude response was not valid JSON.",
                    what_was_not_done="Could not parse selected_features from Claude.",
                    doubts=str(e),
                    issues=["Claude returned invalid JSON — orchestrator should retry."],
                    metrics={"n_after_vif": n_after_vif},
                    recommendation="retry",
                ))
            raise RuntimeError(
                f"FeatureSelector: Claude returned invalid JSON ({e}). "
                "See docs/agents/feature_selector.md."
            ) from e
        # Only keep features that survived the VIF gate
        selected = [f for f in result['selected_features'] if f in surviving_set]
        reasoning = result.get('reasoning', '')

        # Build score dicts
        pca_scores_dict = {feature_names[i]: float(pca_score[i]) for i in range(n_features)}
        ae_scores_dict = {feature_names[i]: float(recon_error[i]) for i in range(n_features)}

        print(f'  Selected {len(selected)} features.')
        print(f'  VIF removed: {len(removed_by_vif)}  |  Max VIF remaining: {max(vif_table.values(), default=0):.2f}')
        print(f'  Reasoning: {reasoning}')

        # ── Report to orchestrator ─────────────────────────────────────────────
        max_vif = max(vif_table.values(), default=0.0) if vif_table else 0.0
        n_blocked = max(0, n_features - n_after_vif)
        status = "success"
        issues = []
        if len(selected) < 10:
            status = "warning"
            issues.append(f"Only {len(selected)} features selected — may be too few.")
        if max_vif > effective_vif:
            status = "warning"
            issues.append(f"Some features still have VIF > {effective_vif} (max={max_vif:.1f}).")

        if self.bus:
            self.bus.report(OrchestratorMessage(
                agent="FeatureSelector",
                iteration=iteration,
                status=status,
                what_was_done=(
                    f"Scored {n_features} features (PCA+AE). "
                    f"Removed {len(removed_by_vif)} via VIF gate. "
                    f"Claude selected {len(selected)} from {n_after_vif} survivors."
                ),
                what_was_not_done=(
                    "Did not apply p-value gate "
                    f"(sufficient features remain after VIF gate)."
                    if len(selected) >= 10 else
                    "P-value gate skipped due to too few features."
                ),
                doubts=(
                    f"{len(high_corr_pairs)} high-correlation pairs still present "
                    f"after VIF gate — Claude was asked to avoid both in each pair."
                    if high_corr_pairs else ""
                ),
                issues=issues,
                metrics={
                    "n_input_features": n_features,
                    "n_after_vif": n_after_vif,
                    "n_removed_by_vif": len(removed_by_vif),
                    "n_selected": len(selected),
                    "max_vif_remaining": round(max_vif, 2),
                    "n_high_corr_pairs": len(high_corr_pairs),
                },
                recommendation="proceed" if not issues else "retry",
                context={
                    "vif_table": vif_table,
                    "removed_by_vif": removed_by_vif,
                    "high_corr_pairs": [(a, b, r) for a, b, r in high_corr_pairs[:10]],
                },
            ))

        return FeatureSelectionResult(
            selected_features=selected,
            n_features=len(selected),
            pca_scores=pca_scores_dict,
            ae_scores=ae_scores_dict,
            vif_table=vif_table,
            removed_by_vif=removed_by_vif,
            reasoning=reasoning,
            iteration=iteration,
        )
