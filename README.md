# Agentic Clustering & Auto-Labeling: Autonomous Cluster Interpretation with a Multi-Agent System

[![GitHub Stars](https://img.shields.io/github/stars/yourusername/your-repo-name?style=flat-square)](https://github.com/yourusername/your-repo-name)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)

> **The hard part of unsupervised clustering is not the mathematics — it's extracting the meaning.**

**Agentic Clustering & Auto-Labeling** is an autonomous machine learning pipeline that uses an LLM-driven multi-agent architecture to automatically cluster datasets, engineer features, interpret the results, and generate human-readable cluster personas. It bridges the gap between raw statistical grouping and actionable data insights.

---

## ⚡ TL;DR

Seven specialized agents + an LLM Decision Maker run a feedback-driven clustering pipeline that ends with **named, validated clusters** and a full reasoning trace. Every quality gate (silhouette, Clarity, classifier F1, VIF) can push the pipeline backward; the Decision Maker tunes parameters and routes each retry. A live web UI lets you watch, edit, chat with the agents per cluster, and feed corrections back into the next run (adaptive learning). One run typically completes in under an hour and costs under one dollar of API.

---

## 🏗️ Architecture & Agent Roles

<img src="docs/screenshots/00_architecture.png" alt="Seven agents arranged left-to-right (UserInput → DatasetExaminer → FeatureEngineer → FeatureSelector → Clusterer → PersonaNamer → Classifier) with dotted feedback arrows from each quality-gate back down to a central Orchestrator + LLM Decision Maker box" width="1100"/>

Solid arrows = forward path; dotted arrows = feedback loops. The Orchestrator + LLM Decision Maker reads every status report, diagnoses failures, tunes the next iteration's parameters, and routes the pipeline back to whichever step needs to re-run. The best iteration across all 10 attempts is picked by a composite score balancing accuracy, separation, and non-redundancy: 

$$\text{Composite Score} = F_1 \cdot \text{Silhouette} \cdot \frac{1}{\text{max-VIF}}$$

### Core Multi-Agent Breakdown

To optimize performance and handle bottlenecks, tasks are delegated to specialized agents:

* **Dataset Examiner:** Profiles distributions, identifies data types, and flags initial anomalies.
* **Feature Engineer:** Proposes and applies domain-specific mathematical transformations autonomously.
* **Feature Selector:** Detects multi-collinearity and optimizes feature importance, keeping Variance Inflation Factor (VIF) < 5.0.
* **Clusterer:** Sweeps multiple algorithms (K-Means, DBSCAN, GMM) and optimizes hyperparameter $k$.
* **Persona Namer:** Translates cluster centroids and distinct feature patterns into human-readable archetypes.
* **Classifier:** Trains an internal proxy model (e.g., XGBoost, Random Forest) to verify if the clusters are distinct and mathematically reproducible ($F_1$ score gate).

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
cd your-repo-name

# Install dependencies
pip install -r requirements.txt

# Configure your environment variables
export LLM_API_KEY="sk-ant-..."        # or add to .env
```

Run the Pipeline
Executing the script automatically triggers the pipeline and provisions the web interface in your default browser:

Bash
python run_pipeline.py
CLI Flags:

--no-ui: Runs the agentic pipeline in headless mode directly inside the terminal.

--ui-port 5090: Modifies the default web UI hosting port.

--data path/to/dataset.csv: Dynamically overrides the default configuration dataset path.

💡 Optional Demo Dataset: Test the adaptive learning loop out of the box using Kaggle's Fraud Detection data:
kaggle datasets download -d kartik2112/fraud-detection -p data/raw --unzip

---

## Interactive UI + Adaptive Learning

The UI streams every agent step, LLM call, gate decision, and escalation over Server-Sent Events. Two things make it more than a viewer:

**Named Clusters tab + Adaptive Memory** — every cluster is an editable card. Open one and start a multi-turn conversation with the agent about why it picked those features, then **Conclude → propose action** to rename, merge, or save guidance for the next run. Every rename, merge, hint, and chat conclusion lands in the **Adaptive Memory drawer** (right side of the topbar) as a prioritised rule; the next pipeline run reads `outputs/user_feedback_log.jsonl` and the Decision Maker prompts adapt — that is the adaptive-learning loop, made literal.

![Named Clusters tab — chat with an agent, conclude, save guidance to Adaptive Memory](docs/screenshots/01_named_clusters.gif)

**Data & Evidence tab** — per-iteration 2-D PCA projection of the clustered data, with the orchestrator's adaptive-escalation warning surfaced in line: *"Silhouette=0.142 < target 0.40 — orchestrator will reselect features (or escalate after 3 consecutive misses)"*.

![Data & Evidence tab — per-iteration PCA projection with adaptive-escalation warnings](docs/screenshots/02_data_evidence.gif)

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
