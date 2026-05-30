"""Offline benchmark of the text-clustering pipeline on 20 Newsgroups.

Validates the new text modality end-to-end on a *real* public dataset
without hitting the LLM (a stub bus answers `ask` deterministically).
The benchmark answers three questions:

  1. Does the text pipeline RUN on a realistic corpus
     (TextPreparer → FeatureSelector → ClusteringAgent)?
  2. Are the resulting profiles SHAPED CORRECTLY for the rest of the
     pipeline (UI / PersonaNamer / cross-cluster comparison)?
  3. Does the unsupervised clustering RECOVER MEANING — i.e. align with
     the known newsgroup labels above chance?

Run
---
    python data/raw/twenty_newsgroups/download.py   # one-time
    python experiments/benchmark_text_clustering.py

The benchmark subsamples 5 newsgroups × 200 posts so the full run completes
in well under a minute, even on the TF-IDF + SVD path with no transformer.

Adjustable knobs (CLI):
    --categories 4   how many newsgroups to include
    --per-cat 200    posts per category (after the body-length filter)
    --k 5            target number of clusters
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import Counter

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from agents.text_preparer import TextPreparerAgent
from agents.feature_selector import FeatureSelectionAgent
from agents.clusterer import ClusteringAgent
from agents.persona_namer import _format_cluster_block
from agents.user_input import UserIntent


DATA = _ROOT / "data" / "raw" / "twenty_newsgroups" / "twenty_newsgroups.csv"


class _StubBus:
    """No-op LLM bus — every `ask` answers a deterministic JSON shaped
    for whichever caller might invoke it. `report` is a sink."""

    def __init__(self, fs_selected: list[str] | None = None):
        self._fs_selected = fs_selected
        self.reports = []

    def ask(self, agent, purpose, prompt, max_tokens=256, category="pipeline"):
        if "embedding" in (purpose or "").lower():
            return json.dumps({"method": "tfidf_svd", "reasoning": "stub"})
        if "feature" in (purpose or "").lower() and self._fs_selected is not None:
            return json.dumps(
                {"selected_features": list(self._fs_selected), "reasoning": "stub"}
            )
        # Default: empty proceed
        return json.dumps({"action": "proceed", "reasoning": "stub"})

    def report(self, message):
        self.reports.append(message)


def _purity(labels: list[int], ground: list[str]) -> float:
    """Per-cluster majority-class purity: 1.0 = each cluster maps to one
    label, 1/N_categories = random. Reported as a sanity check, not a gate
    (the pipeline is UNsupervised — purity uses labels only for evaluation)."""
    n = len(labels)
    if n == 0:
        return 0.0
    correct = 0
    for c in set(labels):
        in_c = [ground[i] for i, l in enumerate(labels) if l == c]
        if in_c:
            top_label, top_n = Counter(in_c).most_common(1)[0]
            correct += top_n
    return correct / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", type=int, default=5,
                    help="How many newsgroups to use (default 5).")
    ap.add_argument("--per-cat", type=int, default=200,
                    help="Posts per category (after filtering, default 200).")
    ap.add_argument("--k", type=int, default=5,
                    help="Target cluster count (default 5).")
    ap.add_argument("--min-chars", type=int, default=120,
                    help="Drop posts shorter than this (default 120).")
    args = ap.parse_args()

    if not DATA.exists():
        print(f"ERROR: {DATA} not found. Run "
              "`python data/raw/twenty_newsgroups/download.py` first.",
              file=sys.stderr)
        return 1

    print(f"[benchmark] Loading {DATA.name} ...")
    df = pd.read_csv(DATA)
    print(f"[benchmark] {len(df):,} posts, {df['category'].nunique()} categories")

    # Subsample: pick 5 visually-distinct categories so the unsupervised signal
    # is reasonable. These are commonly used in the 20NG literature.
    target_cats = [
        "rec.autos", "sci.space", "rec.sport.baseball",
        "comp.graphics", "talk.politics.guns",
    ][: args.categories]
    df = df[df["category"].isin(target_cats)]
    df = df[df["text"].fillna("").str.len() >= args.min_chars]
    # Take the first N posts per category — index slicing keeps the column.
    df = (df.groupby("category", as_index=False, group_keys=False)
            .head(args.per_cat)
            .reset_index(drop=True))
    print(f"[benchmark] Subsampled to {len(df):,} posts across "
          f"{df['category'].nunique()} categories: {sorted(df['category'].unique())}")

    intent = UserIntent(
        target_entity="documents",
        business_purpose="discover distinct themes in the newsgroup posts",
        dataset_path=str(DATA),
        modality="text",
        text_column="text",
    )

    # ── 1. Vectorize ────────────────────────────────────────────────────────
    print("\n[1] TextPreparer.run")
    bus = _StubBus()
    prep = TextPreparerAgent(bus)
    t0 = time.perf_counter()
    emb_df, prep_result = prep.run(
        raw_df=df, user_intent=intent,
        output_path=str(_ROOT / "data" / "processed" / "_benchmark_text_embeddings.parquet"),
        iteration=0,
    )
    print(f"  → {emb_df.shape[0]}×{emb_df.shape[1]} embeddings in "
          f"{time.perf_counter() - t0:.1f}s")

    # ── 2. FeatureSelector text short-circuit ──────────────────────────────
    print("\n[2] FeatureSelector (text short-circuit)")
    fs_bus = _StubBus(fs_selected=list(emb_df.columns))
    fs = FeatureSelectionAgent(bus=fs_bus).run(
        emb_df, user_intent=intent, feedback="", iteration=1, modality="text",
    )
    assert fs.n_features == emb_df.shape[1], "FS dropped dims in text mode (should not)"

    # ── 3. Clusterer (cosine + c-TF-IDF profiles) ──────────────────────────
    print(f"\n[3] ClusteringAgent (cosine, k={args.k})")
    text_artifacts = {
        "method": prep_result.method,
        "text_column": prep_result.text_column,
        "raw_docs": prep_result.raw_docs,
        "feature_names": prep_result.artifacts.get("feature_names", []),
        "tfidf": prep_result.artifacts.get("tfidf"),
        "tfidf_matrix": prep_result.artifacts.get("tfidf_matrix"),
        "doc_index": list(emb_df.index),
    }
    t0 = time.perf_counter()
    cr = ClusteringAgent(
        config={
            "max_cluster_size_pct": 0.95,
            "sub_n_clusters": 3,
            "max_depth": 0,
            "clustering_algorithm": "kmeans",
            "n_clusters": args.k,
        },
        bus=_StubBus(),
    ).run(
        features_df=emb_df,
        selected_features=fs.selected_features,
        user_intent=intent,
        history=[],
        iteration=1,
        text_artifacts=text_artifacts,
    )
    print(f"  → {len(cr.profiles)} clusters in {time.perf_counter() - t0:.1f}s · "
          f"silhouette={cr.silhouette:.3f} (cosine)")

    assert cr.action == "proceed", f"clusterer action={cr.action}"
    assert cr.profiles, "no profiles produced"

    # ── 4. Each profile carries text-mode evidence ─────────────────────────
    print("\n[4] Cluster profile sanity")
    for cid, p in cr.profiles.items():
        assert p.get("modality") == "text", f"cluster {cid} missing modality=text"
        assert p.get("top_terms"), f"cluster {cid} has no top_terms"
        assert p.get("representative_docs"), f"cluster {cid} has no rep docs"
    print("  ✓ every cluster has top_terms + representative_docs")

    # ── 5. Cluster purity vs ground truth (sanity, not a gate) ─────────────
    # Use the cluster labels stored on cr.cluster_labels (aligned to emb_df.index).
    labels = list(cr.cluster_labels)
    ground = df["category"].tolist()[: len(labels)]
    p = _purity(labels, ground)
    baseline = 1.0 / df["category"].nunique()
    print(f"\n[5] Cluster purity = {p:.3f}  (random baseline = {baseline:.3f})")
    if p < baseline + 0.05:
        print("  ⚠ WARNING: clustering barely above random baseline. "
              "On 20NG with TF-IDF+SVD we usually see ≥ 0.50.")
    else:
        print("  ✓ unsupervised clustering recovers meaningful topical structure")

    # ── 6. Show the named-cluster prompt block the LLM would receive ───────
    print("\n[6] Sample PersonaNamer prompt block (first cluster)")
    first_cid = next(iter(cr.profiles))
    block = _format_cluster_block(first_cid, cr.profiles[first_cid])
    # Indent for readability
    for line in block.splitlines()[:20]:
        print("    " + line)
    if len(block.splitlines()) > 20:
        print(f"    … ({len(block.splitlines()) - 20} more lines)")

    print("\n[benchmark] DONE ✓\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
