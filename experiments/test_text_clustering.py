"""
Offline end-to-end test for the TEXT clustering modality (Stage 1).

Runs without any LLM or network: a stub bus answers `ask` deterministically and
`report` is a no-op. Requires scikit-learn (TF-IDF path). sentence-transformers
is NOT required — the TF-IDF fallback is exercised.

Run:
    python experiments/test_text_clustering.py
Exits non-zero on the first failed assertion.
"""
from __future__ import annotations

import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from skills.text_vectorizer import recommend_text_vectorizer, vectorize_text
from agents.dataset_examiner import DatasetExaminerAgent
from agents.text_preparer import TextPreparerAgent
from agents.user_input import UserIntent


SAMPLE = _ROOT / "data" / "raw" / "text_articles" / "text_articles.csv"


class _StubBus:
    """Minimal OrchestratorBus stand-in: deterministic ask, no-op report."""

    def __init__(self, method: str = "tfidf_svd"):
        self._method = method
        self.reports = []

    def ask(self, agent, purpose, prompt, max_tokens=256, category="pipeline"):
        return json.dumps({"method": self._method, "reasoning": "stub"})

    def report(self, message):
        self.reports.append(message)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        print(f"  ✗ FAIL: {msg}")
        sys.exit(1)
    print(f"  ✓ {msg}")


def main() -> None:
    print("\n[test_text_clustering] Loading sample corpus...")
    df = pd.read_csv(SAMPLE)
    _check(len(df) >= 30, f"loaded {len(df)} sample articles")

    # ── 1. Recommender prefers TF-IDF for short docs ───────────────────────────
    print("\n[1] recommend_text_vectorizer")
    rec = recommend_text_vectorizer(n_docs=len(df), avg_doc_len=18, verbose=False)
    _check(rec.method in ("tfidf_svd", "transformer"), f"method={rec.method}")
    _check(rec.method == "tfidf_svd", "short docs → tfidf_svd recommended")

    # ── 2. Vectorizer produces a finite matrix + retains vocab ─────────────────
    print("\n[2] vectorize_text (tfidf_svd)")
    docs = df["body"].tolist()
    X, art = vectorize_text(docs, method="tfidf_svd", verbose=False)
    import numpy as np
    _check(X.shape[0] == len(docs), f"rows={X.shape[0]} match docs")
    _check(X.shape[1] >= 2, f"dims={X.shape[1]} (>=2)")
    _check(np.isfinite(X).all(), "embedding matrix is all finite")
    _check("feature_names" in art and len(art["feature_names"]) > 0,
           f"tfidf vocab retained ({len(art.get('feature_names', []))} terms)")

    # ── 3. Transformer request falls back to TF-IDF when unavailable ───────────
    print("\n[3] transformer fallback")
    X2, art2 = vectorize_text(docs[:30], method="transformer", verbose=False)
    _check(art2["method"] == "tfidf_svd" or art2["method"] == "transformer",
           f"resolved method={art2['method']}")

    # ── 4. DatasetExaminer detects the text column ─────────────────────────────
    print("\n[4] DatasetExaminer._detect_text_column")
    col = DatasetExaminerAgent._detect_text_column(df)
    _check(col == "body", f"detected text column = {col!r}")

    # ── 5. TextPreparer produces an embedding DataFrame ────────────────────────
    print("\n[5] TextPreparer.run")
    bus = _StubBus(method="tfidf_svd")
    prep = TextPreparerAgent(bus)
    intent = UserIntent(
        target_entity="documents",
        business_purpose="discover distinct themes in the articles",
        dataset_path=str(SAMPLE),
        modality="text",
        text_column="body",
    )
    emb_df, result = prep.run(
        raw_df=df, user_intent=intent,
        output_path=str(_ROOT / "data" / "processed" / "_test_text_embeddings.parquet"),
        iteration=0,
    )
    _check(emb_df.shape[0] == len(df), f"embedding rows={emb_df.shape[0]}")
    _check(all(c.startswith("emb_") for c in emb_df.columns),
           "embedding columns are emb_*")
    _check(result.text_column == "body", "result records text column")
    _check(len(result.raw_docs) == len(df), "raw docs retained for labelling")

    # ── 6. Themes actually separate (sanity: KMeans finds structure) ───────────
    print("\n[6] KMeans sanity on embeddings")
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(emb_df.values)
    sil = silhouette_score(emb_df.values, labels)
    _check(len(set(labels)) == 3, "formed 3 clusters")
    _check(sil > 0, f"silhouette={sil:.3f} > 0 (themes separate)")

    # ── 7. FeatureSelector text short-circuit keeps every dim ──────────────────
    print("\n[7] FeatureSelector text short-circuit")
    from agents.feature_selector import FeatureSelectionAgent
    fs_agent = FeatureSelectionAgent(bus=bus)
    fs = fs_agent.run(
        emb_df, user_intent=intent, feedback="", iteration=1, modality="text"
    )
    _check(fs.n_features == emb_df.shape[1],
           f"kept all {fs.n_features} embedding dims (no PCA/AE/VIF)")
    _check(fs.removed_by_vif == [], "no VIF removals in text mode")
    _check(fs.pca_scores == {}, "PCA scores empty (skipped)")

    # ── 8. Clusterer runs cosine + emits text profiles ─────────────────────────
    print("\n[8] Clusterer text mode (cosine + c-TF-IDF profiles)")
    from agents.clusterer import ClusteringAgent
    cluster_agent = ClusteringAgent(
        config={"max_cluster_size_pct": 0.95, "sub_n_clusters": 3, "max_depth": 0,
                "clustering_algorithm": "kmeans", "n_clusters": 3},
        bus=bus,
    )
    text_artifacts = {
        "method": result.method,
        "text_column": result.text_column,
        "raw_docs": result.raw_docs,
        "feature_names": result.artifacts.get("feature_names", []),
        "tfidf": result.artifacts.get("tfidf"),
        "tfidf_matrix": result.artifacts.get("tfidf_matrix"),
        "doc_index": list(emb_df.index),
    }
    cr = cluster_agent.run(
        features_df=emb_df,
        selected_features=fs.selected_features,
        user_intent=intent,
        history=[],
        iteration=1,
        text_artifacts=text_artifacts,
    )
    _check(cr.action == "proceed", f"clusterer action=proceed (got {cr.action!r})")
    _check(cr.profiles is not None and len(cr.profiles) >= 2,
           f"produced {len(cr.profiles or {})} profiles")
    _check(cr.silhouette is not None and cr.silhouette > -1.0,
           f"cosine silhouette={cr.silhouette}")

    sample_cid = next(iter(cr.profiles))
    sample = cr.profiles[sample_cid]
    _check(sample.get("modality") == "text", "profile is flagged modality=text")
    _check(bool(sample.get("top_terms")), f"top_terms present ({sample.get('top_terms', [])[:5]})")
    _check(bool(sample.get("representative_docs")),
           f"representative_docs present ({len(sample.get('representative_docs', []))} docs)")
    _check(bool(sample.get("top_above_average")),
           "top_above_average mirrors c-TF-IDF terms for UI compatibility")

    # ── 9. PersonaNamer block uses text branch ─────────────────────────────────
    print("\n[9] PersonaNamer _format_cluster_block (text branch)")
    from agents.persona_namer import _format_cluster_block
    block = _format_cluster_block(sample_cid, sample)
    _check("DISTINCTIVE TERMS" in block, "block has DISTINCTIVE TERMS section")
    _check("REPRESENTATIVE DOCUMENTS" in block, "block has REPRESENTATIVE DOCUMENTS section")
    _check("ABOVE AVERAGE" not in block, "block does NOT use tabular ABOVE-AVERAGE wording")

    print("\n[test_text_clustering] ALL CHECKS PASSED ✓\n")


if __name__ == "__main__":
    main()
