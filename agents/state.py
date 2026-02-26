"""
Shared state dataclasses for the multi-agent persona discovery pipeline.
No logic lives here — only data containers.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class FeatureSelectionResult:
    selected_features: list[str]
    n_features: int
    pca_scores: dict[str, float]       # feature -> pca importance score
    ae_scores: dict[str, float]        # feature -> autoencoder reconstruction error
    reasoning: str
    iteration: int


@dataclass
class ClusteringResult:
    action: str                        # 'proceed' | 'reselect_features'
    cluster_labels: Optional[Any]      # pd.Series with cluster assignments (None if reselecting)
    profiles: Optional[dict]           # cluster_profiles dict (None if reselecting)
    lineage: Optional[dict]            # cluster_lineage dict (None if reselecting)
    silhouette: Optional[float]
    n_leaf: Optional[int]
    reasoning: str
    iteration: int
    algo_name: str = ''
    algo_detail: str = ''


@dataclass
class NamingResult:
    action: str                        # 'proceed' | 'recluster'
    personas: Optional[dict]           # cid -> persona dict (None if gate failed)
    passed: bool
    issues: list[str]
    avg_confidence: float
    reasoning: str
    iteration: int


@dataclass
class ClassifierResult:
    action: str                        # 'proceed' | 'reselect_features' | 'recluster'
    cv_accuracy: float
    cv_f1_macro: float
    cv_f1_weighted: float
    feature_importances: dict[str, float]  # feature -> importance score
    per_class_f1: dict[str, float]         # persona_name -> f1 score
    reasoning: str
    iteration: int
    model: Optional[Any] = None            # fitted RandomForestClassifier
    label_encoder: Optional[Any] = None    # fitted LabelEncoder


@dataclass
class HumanDecision:
    action: str                        # 'approve' | 'recluster' | 'reselect_features' | 'quit'
    feedback: str = ''


@dataclass
class PipelineState:
    config: dict

    # Current feature selection
    selected_features: list[str] = field(default_factory=list)
    needs_feature_selection: bool = True

    # Feedback strings routed to specific agents
    fs_feedback: str = ''
    cluster_feedback: str = ''
    naming_feedback: str = ''
    classifier_feedback: str = ''

    # Full history of each agent's results
    fs_history: list[FeatureSelectionResult] = field(default_factory=list)
    clustering_history: list[ClusteringResult] = field(default_factory=list)
    naming_history: list[NamingResult] = field(default_factory=list)
    classifier_history: list[ClassifierResult] = field(default_factory=list)

    # Overall iteration counter
    total_iterations: int = 0

    # Best result seen so far (saved even if we keep looping)
    best_naming_result: Optional[NamingResult] = None
    best_clustering_result: Optional[ClusteringResult] = None
    best_classifier_result: Optional[ClassifierResult] = None

    def update_features(self, fs_result: FeatureSelectionResult) -> None:
        self.selected_features = fs_result.selected_features
        self.needs_feature_selection = False
        self.fs_history.append(fs_result)

    def request_feature_reselection(self, reason: str) -> None:
        self.needs_feature_selection = True
        self.fs_feedback = reason
        self.cluster_feedback = ''   # reset cluster feedback after going back

    def update_best(
        self,
        nr: NamingResult,
        cr: ClusteringResult,
        clf: Optional[ClassifierResult] = None,
    ) -> None:
        """Keep track of the best (highest avg_confidence + passed gate) result."""
        if nr.passed:
            if (self.best_naming_result is None
                    or nr.avg_confidence > self.best_naming_result.avg_confidence):
                self.best_naming_result = nr
                self.best_clustering_result = cr
                self.best_classifier_result = clf
