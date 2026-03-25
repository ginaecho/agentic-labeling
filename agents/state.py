"""
Shared state dataclasses for the multi-agent persona discovery pipeline.
No logic lives here — only data containers.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


# ── New: intent & dataset profile ─────────────────────────────────────────────

@dataclass
class UserIntent:
    """Clustering intent captured from the user (from UserInputAgent)."""
    target_entity: str         # what is being clustered
    business_purpose: str      # why we are clustering
    dataset_path: str          # path to the feature parquet/CSV
    constraints: str = ""      # optional free-text constraints
    n_clusters_requested: Optional[int] = None
    """If the user specifies an exact cluster count, ClusteringAgent uses it directly."""
    must_have_clusters: list = field(default_factory=list)
    """Cluster attribute labels that must appear in the final result, e.g. ['traveller', 'VIP']."""


@dataclass
class DatasetProfile:
    """Structured profile of the raw dataset (from DatasetExaminerAgent)."""
    n_rows: int
    n_cols: int
    column_types: dict[str, str]
    missing_rates: dict[str, float]
    distribution_summary: dict[str, dict[str, float]]
    high_cardinality_cols: list[str]
    suggested_feature_groups: list[str]
    feature_group_reasoning: str
    warnings: list[str] = field(default_factory=list)
    algo_hint: str = ""        # 'hierarchical' | 'kmeans'
    dataset_readme: str = ""   # contents of README.md in the dataset folder (if present)


# ── Existing result dataclasses ────────────────────────────────────────────────

@dataclass
class FeatureSelectionResult:
    selected_features: list[str]
    n_features: int
    pca_scores: dict[str, float]       # feature -> pca importance score
    ae_scores: dict[str, float]        # feature -> autoencoder reconstruction error
    vif_table: dict[str, float]        # feature -> VIF value (NEW)
    removed_by_vif: list[str]          # features removed by VIF gate (NEW)
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
    k_scores: dict = field(default_factory=dict)   # {k: silhouette_score} from optimizer (NEW)
    algo_reasoning: str = ''                        # from algo_recommender (NEW)


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


# ── Pipeline state ─────────────────────────────────────────────────────────────

@dataclass
class PipelineState:
    config: dict

    # ── Intent & dataset (NEW) ──────────────────────────────────────────────
    user_intent: Optional[UserIntent] = None
    dataset_profile: Optional[DatasetProfile] = None

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

    # Best clustering by raw silhouette — tracked even when naming/classifier fail.
    # Used as fallback at max_iterations when no naming result was ever approved.
    best_silhouette_cluster: Optional[ClusteringResult] = None
    best_silhouette_value: float = -1.0
    best_silhouette_features: list[str] = field(default_factory=list)

    # Dynamic tuning parameters — LLM adjusts these after each failed iteration
    # so agents are NOT locked into hardcoded thresholds.
    tuning_params: dict = field(default_factory=lambda: {
        'vif_threshold': 10.0,    # higher = keep more correlated features
        'k_range': None,          # None = use config default
        'algorithm': None,        # None = use config/auto-select
        'min_silhouette': 0.05,   # hard-block below this; LLM may raise/lower
        'feature_focus': '',      # hint injected into FeatureSelector prompt
    })

    def update_features(self, fs_result: FeatureSelectionResult) -> None:
        self.selected_features = fs_result.selected_features
        self.needs_feature_selection = False
        self.fs_history.append(fs_result)

    def request_feature_reselection(self, reason: str) -> None:
        self.needs_feature_selection = True
        self.fs_feedback = reason
        self.cluster_feedback = ''   # reset cluster feedback after going back

    def update_best_silhouette(self, cr: ClusteringResult, selected_features: list[str]) -> None:
        """Track the best clustering by raw silhouette, regardless of naming/classifier outcome.
        Used at max_iterations to deliver a full analysis on the most-separated result."""
        sil = cr.silhouette if cr.silhouette is not None else -1.0
        if cr.profiles is not None and sil > self.best_silhouette_value:
            self.best_silhouette_value = sil
            self.best_silhouette_cluster = cr
            self.best_silhouette_features = list(selected_features)

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
