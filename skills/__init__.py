"""
Skills package — atomic, reusable building blocks for the agentic clustering pipeline.

Each module in this package implements one or more skills as described in skill.md.
Agents import skills directly from this package.
"""
from skills.orchestrator_bus import OrchestratorBus, OrchestratorMessage
from skills.vif_checker import compute_vif, remove_high_vif, flag_high_correlation
from skills.silhouette_optimizer import optimize_k, SilhouetteResult
from skills.algo_recommender import recommend_algorithm, AlgoRecommendation
from skills.data_cleaner import drop_low_value_columns, impute_missing, sanitize

__all__ = [
    # Bus
    "OrchestratorBus",
    "OrchestratorMessage",
    # VIF
    "compute_vif",
    "remove_high_vif",
    "flag_high_correlation",
    # Data cleaning
    "drop_low_value_columns",
    "impute_missing",
    "sanitize",
    # Silhouette
    "optimize_k",
    "SilhouetteResult",
    # Algo
    "recommend_algorithm",
    "AlgoRecommendation",
]
