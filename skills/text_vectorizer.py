"""
Text Vectorizer — turn raw documents into a numeric embedding matrix.

This is the text-modality analog of skills.algo_recommender + the numeric
feature builders: it (a) RECOMMENDS an embedding method from the corpus shape
(short text → TF-IDF + TruncatedSVD; long-form → sentence-transformer), and
(b) VECTORIZES the documents with the chosen method.

Design notes
------------
- Mirrors algo_recommender's interface: a `recommend_text_vectorizer(...)`
  function returning a small dataclass with `method`, `reasoning`, `confidence`.
- `vectorize_text(...)` returns (X, artifacts) where X is a dense float matrix
  (n_docs × n_dims) and `artifacts` retains the fitted TF-IDF vocabulary so a
  later step can compute per-cluster distinctive terms (c-TF-IDF).
- Heavyweight imports (sklearn, sentence-transformers) are deferred to call
  time, and the transformer path FALLS BACK to TF-IDF when the optional
  `sentence-transformers` package is not installed — so the pipeline always
  runs offline with scikit-learn alone.

Usage
-----
    from skills.text_vectorizer import recommend_text_vectorizer, vectorize_text

    rec = recommend_text_vectorizer(n_docs=2000, avg_doc_len=180)
    X, artifacts = vectorize_text(docs, method=rec.method)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Valid embedding methods the agent may choose between.
VALID_METHODS = ("tfidf_svd", "transformer")
DEFAULT_SVD_DIMS = 100
DEFAULT_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class VectorizerRecommendation:
    """Output of recommend_text_vectorizer (mirrors AlgoRecommendation)."""

    method: str
    """One of VALID_METHODS: 'tfidf_svd' or 'transformer'."""

    reasoning: str
    """Human-readable explanation of the recommendation."""

    confidence: float
    """0–1 confidence score. < 0.6 means the choice is borderline."""

    factors: dict = field(default_factory=dict)
    """Raw factor values that drove the decision."""


def transformer_available() -> bool:
    """True when sentence-transformers can be imported (optional dependency)."""
    try:
        import importlib.util
        return importlib.util.find_spec("sentence_transformers") is not None
    except Exception:
        return False


def recommend_text_vectorizer(
    n_docs: int,
    avg_doc_len: float,
    business_purpose: str = "",
    offline_only: bool | None = None,
    verbose: bool = True,
) -> VectorizerRecommendation:
    """Recommend an embedding method from corpus characteristics.

    Heuristics
    ----------
    - Long, prose-like documents (articles, reviews) benefit most from semantic
      sentence-transformer embeddings → 'transformer' (if available).
    - Short documents (titles, tweets, tags) cluster well on lexical overlap and
      are cheaper with TF-IDF + TruncatedSVD → 'tfidf_svd'.
    - Semantic intent keywords ("meaning", "topic", "semantic", "theme") nudge
      toward the transformer.
    - If sentence-transformers is unavailable, we always fall back to tfidf_svd
      so the pipeline runs offline.

    Parameters
    ----------
    n_docs : int                Number of documents in the corpus.
    avg_doc_len : float         Mean token (whitespace) count per document.
    business_purpose : str      Free-text intent (checked for semantic keywords).
    offline_only : bool | None  Force the offline TF-IDF path. None = auto-detect
                                from whether transformers is importable.
    verbose : bool              Print the decision.
    """
    factors: dict[str, Any] = {"n_docs": n_docs, "avg_doc_len": round(avg_doc_len, 1)}
    have_transformer = transformer_available()
    factors["transformer_available"] = have_transformer

    if offline_only is None:
        offline_only = not have_transformer

    reasons: list[str] = []
    scores = {"tfidf_svd": 0.0, "transformer": 0.0}

    # Document length — the dominant signal.
    if avg_doc_len >= 40:
        scores["transformer"] += 2
        reasons.append(
            f"avg_doc_len={avg_doc_len:.0f} tokens — long/prose text benefits "
            "from semantic embeddings"
        )
    else:
        scores["tfidf_svd"] += 2
        reasons.append(
            f"avg_doc_len={avg_doc_len:.0f} tokens — short text clusters well on "
            "lexical TF-IDF"
        )

    # Corpus size — transformers cost more per doc; TF-IDF scales cheaply.
    if n_docs > 50_000:
        scores["tfidf_svd"] += 1
        reasons.append(f"n_docs={n_docs:,} is large — TF-IDF is cheaper at scale")

    # Intent keywords.
    bp = business_purpose.lower()
    if any(k in bp for k in ("semantic", "meaning", "topic", "theme", "intent")):
        scores["transformer"] += 1
        reasons.append("business purpose implies semantic grouping → transformer")

    method = "transformer" if scores["transformer"] > scores["tfidf_svd"] else "tfidf_svd"

    # Hard constraint: no transformer package → must use TF-IDF.
    if method == "transformer" and (offline_only or not have_transformer):
        method = "tfidf_svd"
        reasons.append(
            "sentence-transformers not available (or offline_only) — "
            "falling back to TF-IDF + SVD"
        )

    total = sum(scores.values()) or 1.0
    margin = abs(scores["transformer"] - scores["tfidf_svd"])
    confidence = round(min(0.5 + (margin / total) * 0.5, 1.0), 2)
    factors["scores"] = scores

    reasoning = "; ".join(reasons) + f". → {method} (confidence={confidence})"
    if verbose:
        print(f"  [TextVectorizer] → {method.upper()}  (confidence={confidence})")
        for r in reasons:
            print(f"    · {r}")

    return VectorizerRecommendation(
        method=method, reasoning=reasoning, confidence=confidence, factors=factors
    )


def vectorize_text(
    docs: list[str],
    method: str = "tfidf_svd",
    svd_dims: int = DEFAULT_SVD_DIMS,
    max_features: int = 20_000,
    transformer_model: str = DEFAULT_TRANSFORMER_MODEL,
    verbose: bool = True,
) -> tuple["Any", dict]:
    """Vectorize `docs` into a dense float matrix.

    Returns
    -------
    (X, artifacts)
        X         : np.ndarray of shape (n_docs, n_dims), float32/64.
        artifacts : dict with keys describing how the matrix was produced:
            'method'         — the method actually used (may differ on fallback)
            'n_dims'         — embedding dimensionality
            'tfidf'          — fitted TfidfVectorizer (tfidf_svd path; for c-TF-IDF)
            'tfidf_matrix'   — sparse TF-IDF matrix (tfidf_svd path)
            'feature_names'  — TF-IDF vocabulary terms (tfidf_svd path)
            'fallback'       — True if a transformer request fell back to TF-IDF
    """
    import numpy as np

    if not docs:
        raise ValueError("vectorize_text: empty document list")
    # Normalize to non-empty strings; preserve positions so callers can re-align.
    clean = [("" if d is None else str(d)).strip() for d in docs]

    if method not in VALID_METHODS:
        raise ValueError(
            f"Unknown text vectorizer method {method!r}. "
            f"Valid: {', '.join(VALID_METHODS)}."
        )

    artifacts: dict[str, Any] = {"method": method, "fallback": False}

    if method == "transformer":
        if transformer_available():
            try:
                from sentence_transformers import SentenceTransformer
                if verbose:
                    print(f"  [TextVectorizer] Encoding {len(clean):,} docs with {transformer_model} ...")
                model = SentenceTransformer(transformer_model)
                X = np.asarray(
                    model.encode(clean, show_progress_bar=False, convert_to_numpy=True),
                    dtype=float,
                )
                artifacts.update(n_dims=int(X.shape[1]), transformer_model=transformer_model)
                return X, artifacts
            except Exception as exc:  # noqa: BLE001 — any failure → TF-IDF fallback
                if verbose:
                    print(f"  [TextVectorizer] transformer failed ({exc}); falling back to TF-IDF")
                artifacts["fallback"] = True
        else:
            if verbose:
                print("  [TextVectorizer] sentence-transformers not installed; using TF-IDF + SVD")
            artifacts["fallback"] = True
        method = "tfidf_svd"
        artifacts["method"] = "tfidf_svd"

    # ── TF-IDF + TruncatedSVD path (default / fallback) ───────────────────────
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    tfidf_matrix = tfidf.fit_transform(clean)
    vocab_size = tfidf_matrix.shape[1]

    # SVD can output at most min(n_docs, vocab) - 1 components.
    n_comp = max(2, min(svd_dims, vocab_size - 1, len(clean) - 1))
    if verbose:
        print(
            f"  [TextVectorizer] TF-IDF vocab={vocab_size:,} → "
            f"TruncatedSVD to {n_comp} dims"
        )
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X = svd.fit_transform(tfidf_matrix)
    # L2-normalize so downstream Euclidean distance approximates cosine.
    X = normalize(X)

    artifacts.update(
        method="tfidf_svd",
        n_dims=int(X.shape[1]),
        tfidf=tfidf,
        tfidf_matrix=tfidf_matrix,
        feature_names=list(tfidf.get_feature_names_out()),
        svd=svd,
        explained_variance=float(svd.explained_variance_ratio_.sum()),
    )
    return X, artifacts
