# Multi-Agent Clustering & Persona Discovery Pipeline
#
# Agent registry — see agent.md for full documentation.
#
# Pipeline order:
#   0. UserInputAgent         (agents/user_input.py)
#   1. DatasetExaminerAgent   (agents/dataset_examiner.py)
#   2. FeatureEngineerAgent   (agents/feature_engineer.py)  ← NEW
#   3. FeatureSelectionAgent  (agents/feature_selector.py)
#   4. ClusteringAgent        (agents/clusterer.py)
#   5. PersonaNamingAgent     (agents/persona_namer.py)
#   6. ClassifierAgent        (agents/classifier.py)
#   ↑  Orchestrator           (agents/orchestrator.py)  — coordinates all

from agents.user_input import UserInputAgent, UserIntent
from agents.dataset_examiner import DatasetExaminerAgent, DatasetProfile
from agents.feature_engineer import FeatureEngineerAgent, FeatureEngineeringResult
from agents.feature_selector import FeatureSelectionAgent
from agents.clusterer import ClusteringAgent
from agents.persona_namer import PersonaNamingAgent
from agents.classifier import ClassifierAgent
from agents.orchestrator import Orchestrator
from agents.state import PipelineState, HumanDecision

__all__ = [
    "UserInputAgent",
    "UserIntent",
    "DatasetExaminerAgent",
    "DatasetProfile",
    "FeatureEngineerAgent",
    "FeatureEngineeringResult",
    "FeatureSelectionAgent",
    "ClusteringAgent",
    "PersonaNamingAgent",
    "ClassifierAgent",
    "Orchestrator",
    "PipelineState",
    "HumanDecision",
]
