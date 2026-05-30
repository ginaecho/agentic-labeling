# Agents Index

Each agent has a dedicated doc: role, inputs/outputs, skills used (with links to skill docs), communication contract, and failure/retry behaviour.

| Agent | File | Skills used |
|-------|------|-------------|
| [Orchestrator](orchestrator.md) | `agents/orchestrator.py` | — |
| [UserInputAgent](user_input.md) | `agents/user_input.py` | — |
| [DatasetExaminerAgent](dataset_examiner.md) | `agents/dataset_examiner.py` | [orchestrator_bus](../skills/orchestrator_bus.md) |
| [FeatureEngineerAgent](feature_engineer.md) *(tabular modality)* | `agents/feature_engineer.py` | [orchestrator_bus](../skills/orchestrator_bus.md) |
| [TextPreparerAgent](text_preparer.md) *(text modality)* | `agents/text_preparer.py` | [orchestrator_bus](../skills/orchestrator_bus.md), [text_vectorizer](../skills/text_vectorizer.md) |
| [FeatureSelectionAgent](feature_selector.md) | `agents/feature_selector.py` | [orchestrator_bus](../skills/orchestrator_bus.md), [vif_checker](../skills/vif_checker.md) — *short-circuits in text mode* |
| [ClusteringAgent](clusterer.md) | `agents/clusterer.py` | [orchestrator_bus](../skills/orchestrator_bus.md), [algo_recommender](../skills/algo_recommender.md), [silhouette_optimizer](../skills/silhouette_optimizer.md) |
| [PersonaNamingAgent](persona_namer.md) | `agents/persona_namer.py` | [orchestrator_bus](../skills/orchestrator_bus.md) |
| [ClassifierAgent](classifier.md) | `agents/classifier.py` | [orchestrator_bus](../skills/orchestrator_bus.md) |

**Pipeline flow:**
- *Tabular:* UserInput → DatasetExaminer → FeatureEngineer → FeatureSelector → Clusterer → PersonaNamer → Classifier → Orchestrator (human checkpoint).
- *Text:* UserInput → DatasetExaminer *(detects text-dominant)* → **TextPreparer** → FeatureSelector *(short-circuit)* → Clusterer *(cosine + c-TF-IDF profiles)* → PersonaNamer *(text prompt block)* → Classifier → Orchestrator (human checkpoint).

Feedback can route back to FeatureSelector or Clusterer; the failure-tuning LLM can also swap `text_vectorizer` (tfidf_svd ↔ transformer) which the orchestrator picks up at the top of the next iteration and re-runs TextPreparer with — the text-mode analog of "re-engineer features".
