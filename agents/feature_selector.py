"""
FeatureSelectionAgent

Contract: docs/agents/feature_selector.md. Skills: docs/skills/orchestrator_bus.md,
docs/skills/vif_checker.md.

Scores all features with PCA importance + autoencoder reconstruction error,
applies VIF and correlation gates (via skills.vif_checker), then asks the LLM
which subset to keep for discovering distinct entity personas.

Pipeline:
  1. Log-transform skewed columns
  2. Scale to zero-mean, unit-variance
  3. PCA importance score (weighted squared loadings)
  4. Autoencoder reconstruction error score
  5. Combined score = 0.5 × PCA + 0.5 × AE
  6. VIF gate  — iteratively remove highest-VIF feature until all VIF < threshold
  7. Correlation gate — flag pairs with |r| > corr_threshold
  8. LLM selects final subset from VIF-filtered ranked list

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


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise a 1-D array to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


class FeatureSelectionAgent:
    """
    Scores features with PCA + AE, applies VIF and correlation gates,
    then asks the LLM to choose the best subset for persona discovery.
    """

    def __init__(
        self,
        bus: OrchestratorBus,
        vif_threshold: float = 10.0,
        corr_threshold: float = 0.85,
        ae_bottleneck_cap: int = 32,
        ae_max_iter: int = 200,
        max_features_for_vif: int = 150,
    ):
        # FeatureSelectionAgent owns its ML skills (PCA, AE, VIF).
        # LLM reasoning is requested through the bus → Orchestrator.
        self.bus = bus
        self.vif_threshold = vif_threshold
        self.corr_threshold = corr_threshold
        self.ae_bottleneck_cap = ae_bottleneck_cap
        self.ae_max_iter = ae_max_iter
        # Generic high-dimensionality guard: the VIF gate is O(rounds × cols ×
        # OLS), so when a DENSE matrix has more than this many columns we first
        # keep only the top-N features by combined PCA+AE importance, then run
        # VIF on that bounded set. Feature identity is preserved (unlike a PCA
        # projection) so the LLM still selects real, interpretable columns.
        # Set to 0/None to disable the cap.
        self.max_features_for_vif = max_features_for_vif

    def run(
        self,
        features_df,
        user_intent: UserIntent | None = None,
        dataset_profile=None,
        feedback: str = '',
        iteration: int = 1,
        vif_threshold: float | None = None,
        feature_focus: str = '',
        modality: str = 'tabular',
    ) -> FeatureSelectionResult:
        """
        Parameters
        ----------
        features_df : pd.DataFrame
            The raw customer features table (no 'cluster' column).
        user_intent : UserIntent | None
            Clustering intent for context-aware LLM prompting.
        feedback : str
            Free-text feedback from the previous round (empty on first run).
        iteration : int
            Which iteration this is.
        vif_threshold : float | None
            Override the instance VIF threshold (set by orchestrator tuning).
        feature_focus : str
            Orchestrator hint injected into the LLM prompt to guide selection.

        Returns
        -------
        FeatureSelectionResult
        """
        # ── Text-modality short-circuit ────────────────────────────────────────
        # Embedding dims (emb_0..emb_n) are already a compact, decorrelated
        # representation (TruncatedSVD on TF-IDF, or a sentence-transformer
        # output). PCA / autoencoder / VIF don't apply cleanly and would just
        # discard useful dims. Keep every embedding column; return the SAME
        # FeatureSelectionResult shape so the loop + run_history stay valid.
        if modality == 'text':
            emb_cols = [c for c in features_df.columns if str(c).startswith('emb_')] \
                       or list(features_df.columns)
            print(f'\n[FeatureSelector] Iteration {iteration}  '
                  f'(text modality — skipping PCA/AE/VIF, keeping all {len(emb_cols)} dims)')
            return FeatureSelectionResult(
                selected_features=emb_cols,
                n_features=len(emb_cols),
                pca_scores={},
                ae_scores={},
                vif_table={},
                removed_by_vif=[],
                reasoning=(
                    f"Text modality: kept all {len(emb_cols)} embedding dimensions "
                    "(TruncatedSVD / transformer output is already compact and decorrelated; "
                    "PCA/AE/VIF would just throw away useful signal)."
                ),
                iteration=iteration,
            )

        # Resolve effective thresholds — orchestrator may override per-iteration
        effective_vif = vif_threshold if vif_threshold is not None else self.vif_threshold

        print(f'\n[FeatureSelector] Iteration {iteration}  (vif_threshold={effective_vif})')
        if feedback:
            print(f'  Feedback from previous round: {feedback}')

        # ── Step 1: Preprocess ─────────────────────────────────────────────────
        X = features_df.copy()
        log_cols = _detect_log_cols(X)
        for col in log_cols:
            X[col] = np.log1p(X[col])

        X = X.select_dtypes(include=[np.number])

        # ── Defensive NaN handling ─────────────────────────────────────────────
        # StandardScaler / PCA / MLPRegressor / the VIF OLS gate all reject NaN.
        # The orchestrator already prunes all-null and mostly-null columns at
        # load, but features can still arrive with sparse gaps (e.g. a parquet
        # loaded directly, or engineered ratios). Drop any fully-empty column
        # that slipped through, then median-impute the rest so the math below
        # never crashes.
        from skills.data_cleaner import impute_missing
        # ±inf (divide-by-zero ratios) is as fatal as NaN — treat it as missing.
        X = X.replace([np.inf, -np.inf], np.nan)
        all_nan_cols = [c for c in X.columns if X[c].isna().all()]
        if all_nan_cols:
            print(f'  [FeatureSelector] Dropping {len(all_nan_cols)} all-NaN/all-inf column(s) before scaling.')
            X = X.drop(columns=all_nan_cols)
        if X.isna().any().any():
            X, _imp = impute_missing(X, strategy='median', verbose=False)
            print(f'  [FeatureSelector] Median-imputed NaN/inf in {len(_imp["imputed"])} column(s).')

        feature_names = list(X.columns)
        n_features = len(feature_names)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        print(f'  Features: {n_features}  |  Entities: {X_scaled.shape[0]}')

        # ── Step 2: PCA importance score ───────────────────────────────────────
        n_components = min(50, n_features)
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X_scaled)

        pca_score = np.sum(
            pca.components_ ** 2 * pca.explained_variance_ratio_[:, None],
            axis=0,
        )

        # ── Step 3: Autoencoder reconstruction score ───────────────────────────
        # Bottleneck = min(n_features // 5, ae_bottleneck_cap).
        # The cap prevents the bottleneck from exceeding the hidden-layer width (64),
        # which would make the network expand rather than compress and stall convergence
        # (e.g. 379 // 5 = 75 > 64 without the cap).
        # Both values come from config.yaml (ae_bottleneck_cap, ae_max_iter).
        bottleneck = min(max(8, n_features // 5), self.ae_bottleneck_cap)
        print(f'  AE bottleneck={bottleneck}  (cap={self.ae_bottleneck_cap}, max_iter={self.ae_max_iter})')
        ae = MLPRegressor(
            hidden_layer_sizes=(64, bottleneck, 64),
            activation='relu',
            max_iter=self.ae_max_iter,
            early_stopping=True,
            random_state=42,
            verbose=False,
        )
        ae.fit(X_scaled, X_scaled)
        recon_error = np.mean((X_scaled - ae.predict(X_scaled)) ** 2, axis=0)

        # ── Step 4: Combined score ─────────────────────────────────────────────
        combined = 0.5 * _normalise(pca_score) + 0.5 * _normalise(recon_error)
        ranked_idx = np.argsort(combined)[::-1]  # highest first

        # ── Step 4.5: High-dimensionality prefilter (generic) ──────────────────
        # The VIF gate is O(rounds × cols × OLS); on a wide DENSE matrix that
        # dominates the runtime. When there are more than max_features_for_vif
        # columns, keep only the top-N by combined PCA+AE importance BEFORE the
        # gate. This bounds VIF cost for ANY dataset while preserving real
        # feature names so the LLM still selects interpretable columns.
        import pandas as pd
        cap = self.max_features_for_vif
        removed_by_prefilter: list[str] = []
        if cap and n_features > cap:
            keep_idx = ranked_idx[:cap]
            keep_names = [feature_names[i] for i in keep_idx]
            keep_set = set(keep_names)
            removed_by_prefilter = [f for f in feature_names if f not in keep_set]
            print(f'  High-dim prefilter: {n_features} → {cap} features '
                  f'(top by PCA+AE importance) before VIF gate.')
            X_for_vif = pd.DataFrame(X_scaled[:, keep_idx], columns=keep_names)
        else:
            X_for_vif = pd.DataFrame(X_scaled, columns=feature_names)

        # ── Step 5: VIF gate ───────────────────────────────────────────────────
        print(f'  Running VIF gate (threshold={effective_vif})...')
        X_clean, removed_by_vif = remove_high_vif(
            X_for_vif,
            threshold=effective_vif,
            min_features=10,
            verbose=True,
        )
        clean_feature_names = list(X_clean.columns)
        n_after_vif = len(clean_feature_names)
        print(f'  After VIF gate: {n_after_vif} features  (removed {len(removed_by_vif)}'
              + (f', prefiltered {len(removed_by_prefilter)}' if removed_by_prefilter else '')
              + ')')

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

        # ── Step 8: Ask LLM ────────────────────────────────────────────────────
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
        if dataset_profile and getattr(dataset_profile, 'dataset_readme', ''):
            intent_section += (
                f"\nDataset README (domain context from the data provider):\n"
                f"{'─'*60}\n{dataset_profile.dataset_readme}\n{'─'*60}\n"
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

        prompt = f"""You are a data scientist selecting features for entity segmentation.

{intent_section}
We started with {n_features} features. After VIF gate (threshold={effective_vif}):
  Removed {len(removed_by_vif)} high-VIF features: {removed_by_vif[:10]}{"..." if len(removed_by_vif) > 10 else ""}
  Remaining: {n_after_vif} features

Feature ranking (PCA importance × autoencoder uniqueness score) — VIF-filtered:
{table_str}
{high_corr_note}
{feedback_section}
{focus_section}
Instructions:
1. Choose a subset that produces the MOST DISTINCT, INTERPRETABLE clusters.
2. Prefer features capturing DIFFERENT behavioral dimensions (e.g. frequency,
   magnitude, diversity, recency, trends). Avoid redundant pairs flagged above.
3. Cover as many distinct behavioral dimensions as possible; trim redundant
   time-window variants (e.g. if 6m and 12m versions are highly correlated, keep one).
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
                    what_was_done="Scored features, ran VIF gate; LLM response was not valid JSON.",
                    what_was_not_done="Could not parse selected_features from LLM.",
                    doubts=str(e),
                    issues=["LLM returned invalid JSON — orchestrator should retry."],
                    metrics={"n_after_vif": n_after_vif},
                    recommendation="retry",
                ))
            raise RuntimeError(
                f"FeatureSelector: LLM returned invalid JSON ({e}). "
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
                    f"LLM selected {len(selected)} from {n_after_vif} survivors."
                ),
                what_was_not_done=(
                    "Did not apply p-value gate "
                    f"(sufficient features remain after VIF gate)."
                    if len(selected) >= 10 else
                    "P-value gate skipped due to too few features."
                ),
                doubts=(
                    f"{len(high_corr_pairs)} high-correlation pairs still present "
                    f"after VIF gate — LLM was asked to avoid both in each pair."
                    if high_corr_pairs else ""
                ),
                issues=issues,
                metrics={
                    "n_input_features": n_features,
                    "n_prefiltered_high_dim": len(removed_by_prefilter),
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
