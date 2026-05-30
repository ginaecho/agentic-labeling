# Skills Index

Skills are atomic, testable Python functions or classes used by agents. Each skill has its own doc with purpose, API, and which agents use it.

| Skill | File | Used by |
|-------|------|---------|
| [orchestrator_bus](orchestrator_bus.md) | `skills/orchestrator_bus.py` | All agents |
| [vif_checker](vif_checker.md) | `skills/vif_checker.py` | FeatureSelectionAgent |
| [silhouette_optimizer](silhouette_optimizer.md) | `skills/silhouette_optimizer.py` | ClusteringAgent |
| [algo_recommender](algo_recommender.md) | `skills/algo_recommender.py` | ClusteringAgent |
| [text_vectorizer](text_vectorizer.md) | `skills/text_vectorizer.py` | TextPreparerAgent — recommends embedding method (TF-IDF + SVD or sentence-transformer) and vectorizes documents, with offline TF-IDF fallback |

**Inline skills** (implemented inside agents, not in `skills/`):

- PCA / autoencoder scoring → [FeatureSelectionAgent](../agents/feature_selector.md)
- Clarity gate → [PersonaNamingAgent](../agents/persona_namer.md)
- Classifier training (LLM-selected model) → [ClassifierAgent](../agents/classifier.md)
- Feature-engineering builders → [FeatureEngineerAgent](../agents/feature_engineer.md)

**Adding a new skill:** implement in `skills/<name>.py`, export from `skills/__init__.py`, add a doc here and in each agent that uses it.
