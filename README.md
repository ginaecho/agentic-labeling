# Automated Cluster Interpretation with a Multi-Agent Pipeline

> **The hard part of clustering is not the math — it's the meaning.**

## TL;DR

Seven specialised agents + an LLM Decision Maker run a feedback-driven clustering pipeline that ends with **named, validated personas** and a full reasoning trace. Every quality gate (silhouette, Clarity, classifier F1, VIF) can push the pipeline backward; the Decision Maker tunes parameters and routes each retry. A live web UI lets you watch, edit, chat with the agents per cluster, and feed corrections back into the next run (adaptive learning). One run typically completes in under an hour and costs under one dollar of API.

---

## Architecture

<img src="docs/screenshots/00_architecture.png" alt="Seven agents arranged left-to-right (UserInput → DatasetExaminer → FeatureEngineer → FeatureSelector → Clusterer → PersonaNamer → Classifier) with dotted feedback arrows from each quality-gate back down to a central Orchestrator + LLM Decision Maker box" width="1100"/>

Solid arrows = forward path; dotted arrows = feedback loops. The Orchestrator + LLM Decision Maker reads every status report, diagnoses failures, tunes the next iteration's parameters, and routes the pipeline back to whichever step needs to re-run. The best iteration across all 10 attempts is picked by composite score: **F1 ↑ · Silhouette ↑ · max-VIF ↓**.

---

## Quick Start

```bash
pip install -r requirements.txt
export LLM_API_KEY="sk-ant-..."        # or add to .env
python run_pipeline.py                  # opens the live UI in your browser
```

Flags: `--no-ui` (headless), `--ui-port 5090` (change port), `--data path/to.csv` (override config dataset).

Demo dataset (optional): `kaggle datasets download -d kartik2112/fraud-detection -p data/raw --unzip` → `data/raw/fraudTrain.csv`.

---

## Interactive UI + Adaptive Learning

The UI streams every agent step, LLM call, gate decision, and escalation over Server-Sent Events. Two things make it more than a viewer:

**Named Clusters tab + Adaptive Memory** — every cluster is an editable card. Open one and start a multi-turn conversation with the agent about why it picked those features, then **Conclude → propose action** to rename, merge, or save guidance for the next run. Every rename, merge, hint, and chat conclusion lands in the **Adaptive Memory drawer** (right side of the topbar) as a prioritised rule; the next pipeline run reads `outputs/user_feedback_log.jsonl` and the Decision Maker prompts adapt — that is the adaptive-learning loop, made literal.

https://github.com/ginaecho/Agentic_Labelling/raw/main/recordings/named_cluster.mp4

**Data & Evidence tab** — per-iteration 2-D PCA projection of the clustered data, with the orchestrator's adaptive-escalation warning surfaced in line: *"Silhouette=0.142 < target 0.40 — orchestrator will reselect features (or escalate after 3 consecutive misses)"*.

https://github.com/ginaecho/Agentic_Labelling/raw/main/recordings/data_evidence.mp4

---

## Configuration (`config.yaml`)

```yaml
n_clusters: ~                # null = auto-select k via silhouette optimizer
clustering_algorithm: auto   # auto | kmeans | hierarchical | dbscan | gmm | fuzzy_cmeans
classifier_model: auto       # auto | random_forest | xgboost | gradient_boosting | logistic_regression
max_cluster_size_pct: 0.40   # split any cluster above this share
silhouette_target: 0.5       # starts here; auto-relaxes by 0.1 after 3 consecutive misses
persona_tone: easy           # easy | professional | data-driven | creative
```

All of these are tuned dynamically per-iteration by the Decision Maker — config values are starting points, not locks.

---

## Outputs

Written to `outputs/` after each run:

- `personas.json` · `persona_summary.txt` · `persona_metrics.csv` — named clusters + per-cluster distinguishing features
- `classifier_metrics.json` — CV accuracy, macro-F1, per-class F1, top feature importances
- `cluster_profiles.json` · `cluster_lineage.json` · `silhouette_curve.json` — cluster stats, deepening tree, k-curve
- `pipeline_events.jsonl` · `agents_conversation.txt` — every event + every LLM prompt/response
- `user_feedback_log.jsonl` — rules from the UI that adapt the next run

If 10 iterations finish without any result passing all gates, the pipeline falls into **best-effort mode**: it takes the highest-silhouette clustering, force-names it, runs the classifier, and saves with `status='best_effort'` so a usable result is always delivered.
