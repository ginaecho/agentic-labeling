"""
TextPreparerAgent — Text → embedding matrix (text-modality analog of FeatureEngineer)

When the dataset is text-dominant (articles, documents, reviews), this agent
replaces FeatureEngineerAgent. It:

  1. Pulls the raw documents from the dataset's text column.
  2. Asks the text_vectorizer skill to RECOMMEND an embedding method from the
     corpus shape (short text → TF-IDF + SVD; long-form → transformer), letting
     the LLM (via the bus) override when it has a reason to.
  3. VECTORIZES the documents into a dense numeric embedding matrix.
  4. Returns that matrix as a DataFrame of columns emb_0..emb_n — a plain
     numeric feature table that flows through the EXISTING FeatureSelector →
     Clusterer → Classifier → PersonaNamer stages unchanged.

The raw documents (aligned to the embedding rows by index) and the TF-IDF
artifacts are stashed on the result so a later, text-aware labelling step can
compute distinctive terms + representative documents per cluster (Stage 2).

Reports to the OrchestratorBus exactly like FeatureEngineerAgent:
  - success  when a usable embedding matrix is produced
  - blocked  when no text column / too few documents
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from agents.user_input import UserIntent
from skills.orchestrator_bus import OrchestratorBus, OrchestratorMessage
from skills.text_vectorizer import (
    VALID_METHODS,
    recommend_text_vectorizer,
    vectorize_text,
)

# A document shorter than this (chars) after stripping is treated as empty.
_MIN_DOC_CHARS = 1
# Need at least this many usable documents to cluster meaningfully.
_MIN_DOCS = 20


@dataclass
class TextPreparationResult:
    n_docs: int
    n_dims: int
    method: str
    feature_names: list[str]          # emb_0..emb_n (the embedding columns)
    text_column: str
    output_path: str
    reasoning: str
    artifacts: dict = field(default_factory=dict)   # tfidf vocab etc. (for labelling)
    raw_docs: list = field(default_factory=list)     # aligned to embedding rows


class TextPreparerAgent:
    """Turns a text column into an embedding matrix for the clustering loop."""

    def __init__(self, bus: OrchestratorBus):
        self.bus = bus

    # ── Text column detection ──────────────────────────────────────────────────

    @staticmethod
    def detect_text_column(df: pd.DataFrame, hint: str | None = None) -> str | None:
        """Pick the most text-like column.

        A text column is an object/string column whose values are long and
        highly unique (free prose), not a low-cardinality category. We score
        each candidate by mean token count and pick the longest.
        """
        if hint and hint in df.columns:
            return hint

        best_col, best_score = None, 0.0
        n = max(1, min(len(df), 2000))
        sample = df.head(n)
        for col in df.columns:
            s = sample[col]
            if s.dtype != object and str(s.dtype) not in ("string", "str", "category"):
                continue
            vals = s.dropna().astype(str)
            if vals.empty:
                continue
            avg_tokens = float(vals.str.split().str.len().mean() or 0)
            uniqueness = vals.nunique() / max(len(vals), 1)
            # Free text: many tokens AND mostly unique values.
            score = avg_tokens * (0.5 + 0.5 * uniqueness)
            if avg_tokens >= 5 and score > best_score:
                best_col, best_score = col, score
        return best_col

    # ── Main entry ─────────────────────────────────────────────────────────────

    def run(
        self,
        raw_df: pd.DataFrame,
        user_intent: UserIntent,
        dataset_profile=None,
        output_path: str = "data/processed/text_embeddings.parquet",
        iteration: int = 1,
        method: str | None = None,
        feedback: str = "",
    ) -> tuple[pd.DataFrame, TextPreparationResult]:
        """Vectorize the text column and return (embeddings_df, result).

        Parameters
        ----------
        raw_df : pd.DataFrame      The raw documents table.
        user_intent : UserIntent   Has .text_column / .modality when known.
        method : str | None        Force an embedding method; None = auto-recommend.
        """
        print(f"\n[TextPreparer] Iteration {iteration}")
        print(f"  Purpose : {user_intent.business_purpose}")
        if feedback:
            print(f"  Feedback: {feedback}")

        # ── 1. Locate the text column ──────────────────────────────────────────
        hint = getattr(user_intent, "text_column", None) or (
            getattr(dataset_profile, "text_column", None) if dataset_profile else None
        )
        text_col = self.detect_text_column(raw_df, hint=hint)
        if not text_col:
            self._report_blocked(
                iteration,
                "No free-text column detected — cannot build text embeddings.",
                "Inspected column types for text content",
            )
            raise RuntimeError(
                "TextPreparer: no text column found. Set text_column in config/intent."
            )

        docs_series = raw_df[text_col].fillna("").astype(str)
        mask = docs_series.str.strip().str.len() >= _MIN_DOC_CHARS
        docs_series = docs_series[mask]
        docs = docs_series.tolist()
        n_docs = len(docs)
        print(f"  Text column : {text_col!r}  |  usable docs: {n_docs:,}")

        if n_docs < _MIN_DOCS:
            self._report_blocked(
                iteration,
                f"Only {n_docs} usable documents (need ≥ {_MIN_DOCS}).",
                f"Read text column {text_col!r}",
            )
            raise RuntimeError(
                f"TextPreparer: only {n_docs} documents; need ≥ {_MIN_DOCS}."
            )

        avg_len = float(pd.Series(docs).str.split().str.len().mean() or 0)

        # ── 2. Choose embedding method (recommend → LLM may override) ───────────
        if method in VALID_METHODS:
            chosen, reasoning = method, f"Embedding method fixed: {method}"
            print(f"  Method  : {chosen} (forced)")
        else:
            rec = recommend_text_vectorizer(
                n_docs=n_docs,
                avg_doc_len=avg_len,
                business_purpose=user_intent.business_purpose,
                verbose=True,
            )
            chosen, reasoning = rec.method, rec.reasoning
            chosen = self._maybe_llm_override(chosen, rec, n_docs, avg_len, feedback)

        # ── 3. Vectorize ───────────────────────────────────────────────────────
        X, artifacts = vectorize_text(docs, method=chosen, verbose=True)
        n_dims = int(X.shape[1])
        emb_cols = [f"emb_{i}" for i in range(n_dims)]
        embeddings_df = pd.DataFrame(X, columns=emb_cols, index=docs_series.index)
        print(f"  Embeddings  : {n_docs:,} docs × {n_dims} dims  (method={artifacts['method']})")

        # ── 4. Persist embeddings (best-effort) ────────────────────────────────
        try:
            pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            embeddings_df.to_parquet(output_path, index=True)
            print(f"  Saved → {output_path}")
        except Exception as exc:  # noqa: BLE001 — non-fatal
            print(f"  [TextPreparer] Could not save embeddings: {exc}")

        result = TextPreparationResult(
            n_docs=n_docs,
            n_dims=n_dims,
            method=artifacts["method"],
            feature_names=emb_cols,
            text_column=text_col,
            output_path=output_path,
            reasoning=reasoning,
            artifacts=artifacts,
            raw_docs=docs,
        )

        # ── 5. Report to bus ───────────────────────────────────────────────────
        if self.bus:
            self.bus.report(OrchestratorMessage(
                agent="TextPreparer",
                iteration=iteration,
                status="success",
                what_was_done=(
                    f"Vectorized {n_docs:,} documents from column {text_col!r} into a "
                    f"{n_dims}-dim embedding matrix using {artifacts['method']}."
                ),
                what_was_not_done=(
                    "Did not perform topic modeling or named-entity extraction; "
                    "per-cluster distinctive terms are computed at labelling time."
                ),
                doubts=(
                    "TF-IDF fallback used (sentence-transformers unavailable)."
                    if artifacts.get("fallback") else ""
                ),
                issues=[],
                metrics={
                    "n_docs": n_docs,
                    "n_dims": n_dims,
                    "method": artifacts["method"],
                    "avg_doc_len": round(avg_len, 1),
                },
                recommendation="proceed",
                context={"text_column": text_col, "embedding_method": artifacts["method"]},
            ))

        return embeddings_df, result

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def _maybe_llm_override(self, chosen, rec, n_docs, avg_len, feedback) -> str:
        """Let the orchestrator LLM confirm or override the recommended method.

        Falls back silently to the heuristic recommendation on any failure, so
        the agent works even without a live LLM (e.g. in offline tests).
        """
        if not self.bus:
            return chosen
        prompt = f"""You are selecting how to embed a text corpus for clustering.

Corpus: {n_docs:,} documents, avg length ≈ {avg_len:.0f} tokens.
Heuristic recommendation: {rec.method} (confidence={rec.confidence}).
Reasoning: {rec.reasoning}
{f'Feedback from a previous round: {feedback}' if feedback else ''}

Options:
  tfidf_svd   — TF-IDF + TruncatedSVD. Fast, offline, great for short/keyword text.
  transformer — sentence-transformer embeddings. Best semantic quality for long prose
                (only if installed; otherwise the system falls back to tfidf_svd).

Return ONLY a valid JSON object: {{"method": "tfidf_svd" or "transformer", "reasoning": "<1 sentence>"}}"""
        try:
            raw = self.bus.ask(
                agent="TextPreparer",
                purpose="select text embedding method for clustering",
                prompt=prompt,
                max_tokens=128,
            ).strip()
            if "```" in raw:
                for part in raw.split("```"):
                    p = part.strip()
                    if p.startswith("json"):
                        p = p[4:].strip()
                    if p.startswith("{"):
                        raw = p
                        break
            choice = json.loads(raw).get("method", chosen)
            if choice in VALID_METHODS:
                if choice != chosen:
                    print(f"  [TextPreparer] LLM overrode method: {chosen} → {choice}")
                return choice
        except Exception as exc:  # noqa: BLE001
            print(f"  [TextPreparer] LLM method selection failed ({exc}) — keeping {chosen}")
        return chosen

    def _report_blocked(self, iteration: int, issue: str, what_done: str) -> None:
        if not self.bus:
            return
        self.bus.report(OrchestratorMessage(
            agent="TextPreparer",
            iteration=iteration,
            status="blocked",
            what_was_done=what_done,
            what_was_not_done="Could not produce a text embedding matrix.",
            doubts="",
            issues=[issue],
            metrics={},
            recommendation="escalate",
        ))
        print(f"  [TextPreparer] BLOCKED: {issue}")
