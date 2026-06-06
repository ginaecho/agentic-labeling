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
    max_cluster_size_pct: Optional[float] = None
    """Parsed from intent text: 'max cluster size 25%' -> 0.25."""
    modality: str = "auto"
    """Data modality: 'auto', 'tabular', or 'text'."""
    text_column: Optional[str] = None
    """For text modality: the column holding documents. None = auto-detect."""
    text_columns: list = field(default_factory=list)
    """For text modality: ordered columns to concatenate before embedding."""
    max_total_iterations: Optional[int] = None
    """Maximum orchestrator iterations for this run. None means use CLI default."""


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
    candidate_evidence: dict = field(default_factory=dict)
    """AutoML-as-skill candidate tournament evidence, if enabled."""


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

    # ── Modality routing ─────────────────────────────────────────────────────
    # Mirrors dataset_profile.modality once detected. The agents key off this
    # to take the text-specific branch (cosine clustering, c-TF-IDF profiles,
    # term-based PersonaNamer prompts) without each one re-running detection.
    modality: str = "tabular"
    # Stashed by TextPreparerAgent after vectorization. Carries the raw docs
    # (aligned to embedding row index), the fitted TfidfVectorizer + matrix,
    # and the embedding method actually used. Downstream stages read this to
    # build per-cluster distinctive terms + representative documents.
    text_artifacts: dict = field(default_factory=dict)

    # Current feature selection
    selected_features: list[str] = field(default_factory=list)
    needs_feature_selection: bool = True

    # Escalation rules (silhouette < target → reselect; N failures → re-engineer)
    needs_feature_engineering: bool = False
    consecutive_silhouette_failures: int = 0   # → re-engineer at max_reselect_failures (3)
    silhouette_fail_for_relax: int = 0         # → relax target at max_relax_failures (5)
    silhouette_target_override: Optional[float] = None   # dynamic, set by relax logic

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

    # Composite score of the all-time best iteration (F1 + Silhouette − VIF penalty).
    # Recomputed every call to update_best so the per-iteration comparison is fair.
    best_composite_score: float = float('-inf')
    best_max_vif: float = 0.0

    # Dynamic tuning parameters — LLM adjusts these after each failed iteration
    # so agents are NOT locked into hardcoded thresholds.
    tuning_params: dict = field(default_factory=lambda: {
        'vif_threshold': 10.0,    # higher = keep more correlated features
        'k_range': None,          # None = use config default
        'algorithm': None,        # None = use config/auto-select
        'min_silhouette': 0.05,   # hard-block below this; LLM may raise/lower
        'feature_focus': '',      # hint injected into FeatureSelector prompt
        'text_vectorizer': None,  # text-mode: None=auto / 'tfidf_svd' / 'transformer'
    })

    # Case-memory recall from skills.case_memory (CaseRecall or None).
    # Set once at the start of run() and consumed by _ask_parameter_tuning
    # to render a "prior experience" hint block for the tuning LLM.
    case_recall: object = None

    # Column resolution — entity_id, timestamp, amount, category — resolved
    # by the orchestrator after dataset examination (smart detection + LLM
    # fallback in bypass, user modal in interactive).
    resolved_columns: dict = field(default_factory=dict)

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

    @staticmethod
    def composite_score(silhouette: Optional[float],
                         cv_f1_macro: Optional[float],
                         max_vif: Optional[float]) -> float:
        """Decision metric for the best iteration. Higher is better.

        Weighting (all three deliberately on different scales so each one matters
        without any single metric dominating):
          • F1 macro      → ×100  (primary signal: classifier learnability of the labels)
          • Silhouette    → ×30   (cluster separation quality)
          • VIF penalty   → −log10(max(1, max_vif)) × 5  (feature multicollinearity)

        A perfect iteration (F1=1.0, Sil=1.0, VIF=1.0) ≈ 100 + 30 − 0 = 130.
        A terrible iteration (F1=0.2, Sil=0.1, VIF=50) ≈ 20 + 3 − 8.5 ≈ 14.5.
        Negative silhouette or F1 contribute 0 (not negative) — we don't want
        a single dead-bad signal to dominate when the others are fine.
        """
        import math
        f1_part  = max(0.0, float(cv_f1_macro or 0.0)) * 100.0
        sil_part = max(0.0, float(silhouette or 0.0)) * 30.0
        vif_part = -math.log10(max(1.0, float(max_vif or 1.0))) * 5.0
        return f1_part + sil_part + vif_part

    def update_best(
        self,
        nr: NamingResult,
        cr: ClusteringResult,
        clf: Optional[ClassifierResult] = None,
        max_vif: Optional[float] = None,
    ) -> bool:
        """Keep the iteration with the highest composite score (F1 + Sil − VIF penalty).

        Unlike before, this runs for EVERY iteration that produced a Classifier
        result — not only iterations whose PersonaNamer cleared the Clarity
        Gate. That's because the user-facing best-iteration decision combines
        three signals (F1↑, Silhouette↑, VIF↓), so we must score every iter
        with a Classifier result, not gate by naming quality alone.

        Returns True if this iteration is the new best (helps the caller log it).
        """
        if cr is None or clf is None:
            return False

        sil = cr.silhouette if cr.silhouette is not None else 0.0
        f1  = clf.cv_f1_macro if clf.cv_f1_macro is not None else 0.0
        vif = float(max_vif) if max_vif is not None else self.best_max_vif or 1.0
        new_score = self.composite_score(sil, f1, vif)

        if new_score > self.best_composite_score:
            self.best_composite_score = new_score
            self.best_naming_result = nr
            self.best_clustering_result = cr
            self.best_classifier_result = clf
            self.best_max_vif = vif
            return True
        return False

    def current_max_vif(self) -> float:
        """Largest VIF among the currently-selected features (1.0 if unknown).
        Sourced from the most recent FeatureSelectionResult; used as the VIF
        component of the composite score for the current iteration."""
        if not self.fs_history:
            return 1.0
        last = self.fs_history[-1]
        vt = getattr(last, 'vif_table', None) or {}
        if not vt:
            return 1.0
        try:
            return max(float(v) for v in vt.values() if v is not None)
        except (ValueError, TypeError):
            return 1.0
