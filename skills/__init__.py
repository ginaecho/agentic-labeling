"""
Skills package — atomic, reusable building blocks for the agentic clustering pipeline.

Each module in this package implements one or more skills as described in skill.md.
Agents import skills directly from this package.
"""
from skills.orchestrator_bus import OrchestratorBus, OrchestratorMessage
from skills.vif_checker import compute_vif, remove_high_vif, flag_high_correlation
from skills.silhouette_optimizer import optimize_k, SilhouetteResult
from skills.algo_recommender import recommend_algorithm, AlgoRecommendation

__all__ = [
    # Bus
    "OrchestratorBus",
    "OrchestratorMessage",
    # VIF
    "compute_vif",
    "remove_high_vif",
    "flag_high_correlation",
    # Silhouette
    "optimize_k",
    "SilhouetteResult",
    # Algo
    "recommend_algorithm",
    "AlgoRecommendation",
]
