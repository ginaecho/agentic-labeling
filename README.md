# Automated Cluster Interpretation with a Multi-Agent Pipeline

> **The hard part of clustering is not the math — it's the meaning.**

## TL;DR

Seven specialised agents + an LLM Decision Maker run a feedback-driven clustering pipeline that ends with **named, validated personas** and a full reasoning trace. Every quality gate (silhouette, Clarity, classifier F1, VIF) can push the pipeline backward; the Decision Maker tunes parameters and routes each retry. A live web UI lets you watch, edit, chat with the agents per cluster, run **cross-cluster contrasting analysis**, and feed corrections back into the next run (adaptive learning). The pipeline supports **tabular** and **text** modalities. One run typically completes in under an hour and costs under one dollar of API.

---

## Architecture

<img src="docs/screenshots/00_architecture.png" alt="Seven agents arranged left-to-right (UserInput → DatasetExaminer → FeatureEngineer → Clusterer → PersonaNamer → Classifier) with dotted feedback arrows from each quality-gate back down to a central Orchestrator + LLM Decision Maker box" width="1100"/>

Solid arrows = forward path; dotted arrows = feedback loops. The Orchestrator + LLM Decision Maker reads every status report, diagnoses failures, tunes the next iteration's parameters, and routes the pipeline back to whichever step needs to re-run. The best iteration across all 10 attempts is picked by composite score: **F1 ↑ · Silhouette ↑ · max-VIF ↓**.

### What each agent does

| # | Agent | Role |
|---|-------|------|
| ⓪ | **UserInputAgent** | Clustering intent (target entity, business purpose, dataset path, optional `modality` / `text_column`). |
| ① | **DatasetExaminerAgent** | Profiles schema, missingness, skewness; suggests feature groups. **Auto-detects text-dominant** datasets and routes to the text branch. |
| ② | **FeatureEngineerAgent** *(tabular)* | Entity-level feature matrix from raw event data via LLM-planned statistical ops. Saves to `data/processed/`. |
| ② | **TextPreparerAgent** *(text)* | Replaces FeatureEngineer for document corpora. Embeds via `text_vectorizer` (TF-IDF+SVD or sentence-transformers); stashes docs + vocab for c-TF-IDF terms downstream. |
| ③ | **FeatureSelectionAgent** | PCA + autoencoder + VIF gate → LLM picks a subset. **Short-circuits in text mode** (keeps all embedding dims). |
| ④ | **ClusteringAgent** | AutoML-as-skill candidate tournament, five algorithms, silhouette k-opt, oversized-cluster deepening. **Text mode:** L2-normalize, cosine silhouette, c-TF-IDF distinctive terms + representative docs. |
| ⑤ | **PersonaNamingAgent** | LLM names clusters from numeric deviations or, in text mode, distinctive terms + doc snippets. Clarity Gate must pass. |
| ⑥ | **ClassifierAgent** | LLM picks classifier; 5-fold CV. Routes back on low F1 (threshold 0.70 tabular / 0.60 text). |

---

## Quick Start

```bash
pip install -r requirements.txt
export LLM_API_KEY="sk-ant-..."        # or add to .env
python run_pipeline.py                  # opens the live UI in your browser
```

Flags: `--no-ui` (headless), `--ui-port 5057` (change port), `--data path/to.csv` (override config dataset), `--modality text`, `--text-column text`.

Demo dataset (optional): `kaggle datasets download -d kartik2112/fraud-detection -p data/raw --unzip` → `data/raw/fraudTrain.csv`.

---

## Interactive UI + Adaptive Learning

The UI streams every agent step, LLM call, gate decision, and escalation over Server-Sent Events.

**Named Clusters tab + Adaptive Memory** — every cluster is an editable card. Open one and start a multi-turn conversation with the agent about why it picked those features, then **Conclude → propose action** to rename, merge, or save guidance for the next run. Every rename, merge, hint, and chat conclusion lands in the **Adaptive Memory drawer** (right side of the topbar) as a prioritised rule; the next pipeline run reads `outputs/user_feedback_log.jsonl` and the Decision Maker prompts adapt.

![Named Clusters tab — chat with an agent, conclude, save guidance to Adaptive Memory](docs/screenshots/01_named_clusters.gif)

**Data & Evidence tab** — per-iteration 2-D PCA projection of the clustered data, collapsible feature-engineering builders, orchestrator adaptive-escalation warnings, and an **Explain** button (LLM evidence ledger) that becomes active once iteration evidence is available. Includes **cross-cluster comparison**: an LLM contrasting analysis across all named clusters (not one cluster at a time), cached on the evidence ledger.

![Data & Evidence tab — per-iteration PCA projection with adaptive-escalation warnings](docs/screenshots/02_data_evidence.gif)

**Case Memory — recall a prior winning recipe** — every successful run is fingerprinted (column set, row count, business purpose) and saved to `outputs/case_memory.json` along with the winning recipe (algorithm, k, vif_threshold, feature focus, min_silhouette) and the outcome (silhouette, CV-F1). On the next run the Decision Maker looks for a match — `exact` or `similar` — and, in interactive mode, pauses the pipeline to ask how to use it:

  1. DatasetExaminer finishes profiling.
  2. Modal: **"🧠 Memory match — reuse the prior winning recipe?"** with recipe + outcome.
  3. **Reuse** — seed iteration 1 tuning params verbatim; drop conflicting LLM hints.
  4. **Modify (hint only)** — prior recipe injected into failure-tuning prompts only.
  5. **Ignore** — fresh run.
  6. Live tab + `case_memory_decision` event record the choice.

Bypass / headless mode auto-picks **Modify**. Interactive timeout (5 min) also defaults to **Modify**.

---

## Text Modality (document / article clustering)

Same pipeline, routed through `TextPreparerAgent` instead of `FeatureEngineerAgent`:

```bash
python run_pipeline.py --data data/raw/twenty_newsgroups/twenty_newsgroups.csv
python run_pipeline.py --data path/to.csv --modality text --text-column text
```

| Stage | Text-mode behaviour |
|-------|---------------------|
| DatasetExaminer | Skips "no numeric columns" block; profiles text column. |
| TextPreparer | Embeds docs → `data/processed/text_embeddings.parquet`. |
| FeatureSelector | Skips PCA/AE/VIF; keeps all dims. |
| Clusterer | Cosine silhouette; c-TF-IDF terms + representative docs per cluster. |
| Orchestrator | `min_silhouette=0.01`, classifier F1 gate `0.60`; can swap `text_vectorizer` on retry. |

**Benchmark:** `python data/raw/twenty_newsgroups/download.py` then `python experiments/benchmark_text_clustering.py`.

---

## Configuration (`config.yaml`)

```yaml
n_clusters: ~
clustering_algorithm: auto
classifier_model: auto
max_cluster_size_pct: 0.40
silhouette_target: 0.5
persona_tone: easy
modality: auto              # auto | tabular | text
text_column: ~              # for text modality
text_vectorizer: auto       # auto | tfidf_svd | transformer
```

All values are starting points — the Decision Maker tunes them per iteration.

---

## Outputs

Written to `outputs/` after each run:

- `personas.json` · `persona_summary.txt` · `persona_metrics.csv` — named clusters + distinguishing features
- `classifier_metrics.json` — CV accuracy, macro-F1, per-class F1, importances
- `cluster_profiles.json` · `cluster_lineage.json` · `silhouette_curve.json` — cluster stats, deepening tree, k-curve
- `pipeline_events.jsonl` · `agents_conversation.txt` — events + LLM prompt/response log
- `user_feedback_log.jsonl` — UI rules that adapt the next run
- `case_memory.json` — winning recipes for Case Memory recall (Reuse / Modify / Ignore)
- `data/processed/engineered_features.parquet` — tabular feature matrix (when starting from CSV)
- `data/processed/text_embeddings.parquet` — document embeddings (text mode)

If 10 iterations finish without passing all gates, the pipeline enters **best-effort mode**: highest-silhouette clustering, force-named personas, classifier run, `status='best_effort'`.

---

## Skills

| Skill | File | Used by |
|-------|------|---------|
| OrchestratorBus | `skills/orchestrator_bus.py` | All agents — LLM gateway + event log |
| Case memory | `skills/case_memory.py` | Orchestrator — fingerprint datasets, recall/save winning recipes |
| VIF checker | `skills/vif_checker.py` | FeatureSelector |
| Silhouette optimizer | `skills/silhouette_optimizer.py` | Clusterer (euclidean or cosine) |
| Algorithm recommender | `skills/algo_recommender.py` | Clusterer |
| AutoML candidate search | `skills/automl_candidate_search.py` | Clusterer — bounded algorithm/k tournament with stability evidence |
| Text vectorizer | `skills/text_vectorizer.py` | TextPreparer |

---

## Appendix: Agentic Workflow vs AutoML

AutoML automates model selection: it searches preprocessing choices, algorithms,
hyperparameters, and validation metrics. This workflow uses that idea, but treats
AutoML as one skill inside a broader agentic analysis loop. The goal is not only
to find a cluster assignment with a good score; the goal is to produce clusters
that are stable, explainable, nameable, aligned with the user's intent, and usable
for a business decision.

### Key differences

| Dimension | Typical AutoML | This agentic workflow |
|-----------|----------------|-----------------------|
| Starting point | Dataset + metric | User intent, target entity, business purpose, constraints, and optional must-have cluster types |
| Search mechanism | Pipeline/model/hyperparameter search | AutoML-style candidate search plus agent routing, feature loops, naming gates, classifier validation, and human checkpoint |
| Objective | Optimise one or a few ML metrics | Optimise usable segmentation: separation, stability, feature quality, persona clarity, size balance, business fit, and user feedback |
| Unsupervised labelling | Usually absent or shallow | Dedicated `PersonaNamingAgent` turns cluster evidence into human-readable personas |
| Failure handling | Try another model or report best score | Diagnose the failure and route back to feature engineering, feature selection, clustering, threshold relaxation, or human review |
| Validation | Metric-driven, often silhouette/inertia/CV | Multi-gate: VIF/correlation, silhouette, oversized-cluster deepening, Clarity Gate, pseudo-label classifier F1, and human approval |
| Memory | Usually starts fresh | Case Memory and Adaptive Memory reuse prior recipes and user corrections |
| Output | Best model/pipeline | Personas, profiles, labels, lineage, metrics, evidence ledger, reasoning trace, feedback log, and reusable memory |

### Where AutoML lives in this system

The Clusterer now has an AutoML-as-skill candidate tournament:

```text
skills/automl_candidate_search.py
```

When `clustering_algorithm: auto` and `n_clusters` is unset, the skill evaluates
a bounded set of algorithm/k candidates and ranks them by:

```text
max(0, silhouette) * 70
+ bootstrap_stability_ari * 25
- oversized_cluster_penalty
```

The agent uses the winning candidate as evidence-backed input to the normal
clustering, profiling, naming, and validation path. This keeps brute-force search
in deterministic code while leaving judgment, diagnosis, and interpretation to
the agents.

### Why it can do better than plain AutoML

1. **It optimises the real deliverable.** For unsupervised clustering, the useful
   deliverable is not a model alone. It is a set of meaningful groups a human can
   understand and act on.

2. **It combines quantitative and semantic gates.** A candidate can have a good
   silhouette and still be useless if the personas are vague, duplicated, too
   broad, or misaligned with the stated business purpose.

3. **It tests repeatability, not just fit.** Candidate search includes bootstrap
   stability via ARI, so a slightly lower-silhouette but more stable solution can
   beat a fragile one.

4. **It can recover from the right layer.** If clustering fails, the orchestrator
   can change features, vectorizers, algorithms, k-ranges, thresholds, or route
   to human review instead of blindly continuing the same search space.

5. **It turns feedback into future performance.** Human renames, merge decisions,
   hints, and successful recipes are saved and reused, so the system improves on
   similar future datasets instead of starting from zero.

6. **It preserves evidence.** The final output includes what was tried, what won,
   what failed, why the agent routed backward, and which evidence supports each
   cluster label.

In short: AutoML helps find candidate models. This workflow uses AutoML as a
skill, then adds agentic diagnosis, semantic interpretation, memory, and human
validation so the result is not just statistically acceptable but operationally
usable.
