"""
Orchestrator

Contract: docs/agents/orchestrator.md. Consumes docs/skills/orchestrator_bus.md.

Coordinates the full multi-agent pipeline in a feedback loop:

  (0) UserInputAgent         — collect clustering intent
  (1) DatasetExaminerAgent   — profile dataset, suggest feature groups
  (2) FeatureEngineerAgent   — engineer customer features from raw CSV
                               (skipped if a pre-engineered parquet is given)
  (3) FeatureSelectionAgent  — PCA + AE + VIF → LLM picks feature subset
  (4) ClusteringAgent        — silhouette k-opt + auto algo + deepening loop
  (5) PersonaNamingAgent     — LLM names clusters; Clarity Gate validates
  (6) ClassifierAgent        — classifier CV validates cluster separability;
                               LLM routes back to (3) or (4) if F1 is low
  ↓
  Human Checkpoint           — user approves, requests re-run, or quits

All agents report to a shared OrchestratorBus. The orchestrator uses the
bus log when calling the LLM for routing decisions.
"""
from __future__ import annotations

import json
import pathlib
import time
from collections import defaultdict

import anthropic
import pandas as pd

from agents.state import HumanDecision, PipelineState
from agents.user_input import UserInputAgent, UserIntent
from agents.dataset_examiner import DatasetExaminerAgent
from agents.feature_engineer import FeatureEngineerAgent, FeatureEngineeringResult
from agents.feature_selector import FeatureSelectionAgent
from agents.clusterer import ClusteringAgent
from agents.persona_namer import PersonaNamingAgent
from agents.classifier import ClassifierAgent
from skills.orchestrator_bus import OrchestratorBus, OrchestratorMessage


def _load_df(path: str) -> pd.DataFrame:
    """Load a DataFrame from parquet or CSV, detected by file extension.

    Always injects a `_row_id` column (1..N) so datasets without an obvious
    ID column still have a unique per-row identifier the agents can use as
    the entity key (treating each raw row as one entity to classify).
    """
    import pathlib
    suffix = pathlib.Path(path).suffix.lower()
    if suffix == '.parquet':
        df = pd.read_parquet(path)
    elif suffix in ('.csv', '.tsv'):
        sep = '\t' if suffix == '.tsv' else ','
        df = pd.read_csv(path, sep=sep, low_memory=False)
    else:
        try:
            df = pd.read_parquet(path)
        except Exception:
            df = pd.read_csv(path, low_memory=False)

    if '_row_id' not in df.columns:
        df.insert(0, '_row_id', range(1, len(df) + 1))
    return df


def human_checkpoint(personas: dict, cluster_result, classifier_result, bus: OrchestratorBus) -> HumanDecision:
    """
    Print persona + classifier summary and ask the user what to do next.

    Returns HumanDecision with action in:
      'approve' | 'recluster' | 'reselect_features' | 'quit'
    """
    print('\n' + '=' * 65)
    print('HUMAN CHECKPOINT — Persona & Classifier Review')
    print('=' * 65)
    print(f'Silhouette score   : {cluster_result.silhouette:.4f}')
    print(f'Leaf clusters      : {cluster_result.n_leaf}')
    print(f'Algorithm used     : {cluster_result.algo_name}')
    if cluster_result.k_scores:
        top3 = sorted(cluster_result.k_scores.items(), key=lambda x: -x[1])[:3]
        print(f'Top-3 k values     : ' + ', '.join(f'k={k}({s:.3f})' for k, s in top3))
    print(f'CV accuracy        : {classifier_result.cv_accuracy:.4f}')
    print(f'CV F1 (macro)      : {classifier_result.cv_f1_macro:.4f}')
    print(f'CV F1 (weighted)   : {classifier_result.cv_f1_weighted:.4f}')
    print()

    # Persona table
    print(f'{"Cluster":<10} {"Conf":>4}  {"CV-F1":>6}  {"Persona Name":<45}  Tagline')
    print('-' * 115)
    for cid, p in personas.items():
        name    = p.get('name', '?')
        conf    = p.get('confidence', '?')
        tagline = p.get('tagline', '')
        cv_f1   = classifier_result.per_class_f1.get(name, None)
        cv_f1_str = f'{cv_f1:.3f}' if cv_f1 is not None else '  n/a'
        print(f'  C{cid:<7}  {conf:>4}  {cv_f1_str:>6}  {name:<45}  {tagline}')

    # Highlight worst-performing personas
    worst = sorted(classifier_result.per_class_f1.items(), key=lambda x: x[1])[:3]
    print()
    print('Hardest-to-predict personas (CV F1):')
    for name, score in worst:
        bar = '█' * int(score * 20)
        print(f'  {name:<45}  {score:.3f}  {bar}')

    # Pipeline log summary
    print()
    print('Pipeline Agent Log (recent):')
    print(bus.summary_for_llm(last_n=10))

    print()
    print('Options:')
    print('  [1] Approve — save results and finish')
    print('  [2] Re-cluster — try different clustering parameters')
    print('  [3] Re-select features — go back to feature selection')
    print('  [4] Quit without saving')
    print()

    while True:
        try:
            choice = input('Your choice [1/2/3/4]: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nNo input received — defaulting to Approve.')
            return HumanDecision(action='approve')

        if choice == '1':
            return HumanDecision(action='approve')

        elif choice == '2':
            try:
                reason = input('  Reason / feedback for re-clustering (or Enter to skip): ').strip()
            except (EOFError, KeyboardInterrupt):
                reason = ''
            return HumanDecision(action='recluster', feedback=reason)

        elif choice == '3':
            try:
                reason = input('  Reason / feedback for feature re-selection (or Enter to skip): ').strip()
            except (EOFError, KeyboardInterrupt):
                reason = ''
            return HumanDecision(action='reselect_features', feedback=reason)

        elif choice == '4':
            return HumanDecision(action='quit')

        else:
            print('  Invalid choice. Please enter 1, 2, 3, or 4.')


def save_outputs(cluster_result, naming_result, classifier_result, bus: OrchestratorBus) -> None:
    """Save all pipeline outputs including the orchestrator bus log."""
    pathlib.Path('outputs').mkdir(exist_ok=True)

    with open('outputs/cluster_profiles.json', 'w') as f:
        json.dump(cluster_result.profiles, f, indent=2)
    print('  Saved outputs/cluster_profiles.json')

    with open('outputs/cluster_lineage.json', 'w') as f:
        lineage_str = {str(k): v for k, v in cluster_result.lineage.items()}
        json.dump(lineage_str, f, indent=2)
    print('  Saved outputs/cluster_lineage.json')

    combined = {
        cid: {
            'cluster_stats': cluster_result.profiles[cid],
            'persona': naming_result.personas.get(cid, {}),
        }
        for cid in cluster_result.profiles
    }
    with open('outputs/personas.json', 'w') as f:
        json.dump(combined, f, indent=2)
    print('  Saved outputs/personas.json')

    if cluster_result.cluster_labels is not None:
        try:
            import pandas as _pd
            _labels = cluster_result.cluster_labels
            _df = _pd.DataFrame({'row_index': range(len(_labels)),
                                  'cluster_id': list(_labels)})
            _df.to_csv('outputs/cluster_labels.csv', index=False)
            print('  Saved outputs/cluster_labels.csv')
        except Exception as _e:
            print(f'  [warn] could not save cluster_labels.csv: {_e}')

    metrics = {
        'cv_accuracy':     classifier_result.cv_accuracy,
        'cv_f1_macro':     classifier_result.cv_f1_macro,
        'cv_f1_weighted':  classifier_result.cv_f1_weighted,
        'per_class_f1':    classifier_result.per_class_f1,
        'top20_features':  dict(
            sorted(classifier_result.feature_importances.items(),
                   key=lambda x: -x[1])[:20]
        ),
        'reasoning': classifier_result.reasoning,
    }
    with open('outputs/classifier_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print('  Saved outputs/classifier_metrics.json')

    # Save k-selection silhouette scores
    if cluster_result.k_scores:
        with open('outputs/silhouette_curve.json', 'w') as f:
            json.dump({
                "algorithm": cluster_result.algo_name,
                "best_k": max(cluster_result.k_scores, key=lambda k: cluster_result.k_scores[k]),
                "scores": {str(k): v for k, v in cluster_result.k_scores.items()},
                "algo_reasoning": cluster_result.algo_reasoning,
            }, f, indent=2)
        print('  Saved outputs/silhouette_curve.json')

    # Save pipeline log (JSON + human-readable .txt for agent conversations)
    bus.save_log('outputs/pipeline_log.json')
    bus.save_log_txt('outputs/agents_conversation.txt')


class Orchestrator:
    """
    Main pipeline coordinator.

    Loop order each iteration:
      0. User intent (first run only)
      1. Dataset examination (first run only, or if orchestrator requests re-exam)
      2. Feature selection (first run, or when flagged for re-selection)
      3. Clustering
         → if oversized cluster + LLM says reselect: loop to 2
      4. Persona naming (Clarity Gate)
         → if gate fails: loop to 3
      5. Classifier validation (CV F1 gate)
         → if poor: LLM routes to 2 or 3
      6. Human checkpoint
         → approve  : save + exit
         → recluster: loop to 3
         → reselect : loop to 2
         → quit     : exit without saving
    """

    def __init__(self, config: dict):
        self.config = config
        # Deep-copy the original config so per-run intent overrides (e.g.
        # max_cluster_size_pct from user_intent text) can be undone on the
        # next run() call. Without this, a 25% cap from one run would persist
        # silently into the next.
        import copy as _copy
        self._config_baseline = _copy.deepcopy(config)
        self.client = anthropic.Anthropic()   # ONLY the Orchestrator holds this

        # ── Shared bus — the single communication channel for all agents ────────
        self.bus = OrchestratorBus()

        # ── Load skill & agent catalogs ────────────────────────────────────────
        # Agents "know their skills" by having the relevant skill and agent
        # documentation injected into their LLM calls as system context.
        # This implements plan.md P8 — agents select skills based on the catalog.
        self._skill_catalog = self._load_catalog('skill.md')
        self._agent_catalog = self._load_catalog('agent.md')
        if self._skill_catalog:
            print(f'  [Orchestrator] Skill catalog loaded: {len(self._skill_catalog)} chars (skill.md)')
        if self._agent_catalog:
            print(f'  [Orchestrator] Agent catalog loaded: {len(self._agent_catalog)} chars (agent.md)')

        # Skills injected into system context by calling-agent type:
        # Routing agents (Clusterer, Classifier) get a brief pipeline skills summary
        # so the LLM understands what options are available when it makes routing decisions.
        # Planning agents (FeatureEngineer, FeatureSelector, PersonaNamer) get their
        # agent-specific role description so the LLM adopts the right persona.
        _ROUTING_PURPOSES = frozenset(['route', 'diagnose', 'routing'])
        _skill_summary = self._skill_catalog[:3000] if self._skill_catalog else ''
        _agent_summary = self._agent_catalog[:2000] if self._agent_catalog else ''

        # ── LLM usage log ──────────────────────────────────────────────────────
        self._llm_calls: list[dict] = []

        # ── Register LLM handler on the bus ───────────────────────────────────
        # Agents call bus.ask(agent, purpose, prompt) when they need LLM help.
        # The Orchestrator intercepts, calls the LLM, logs usage, returns the answer.
        # The system parameter carries the skill/agent context so every call
        # is aware of the pipeline's capability catalog (P8 — Modular Skills).

        _CLUSTERING_ALGO_KNOWLEDGE = """
CLUSTERING ALGORITHMS KNOWLEDGE:
- KMeans: Best for large datasets (>100k), spherical clusters, fast. Sensitive to outliers.
- Hierarchical (Ward): Best for nested/hierarchical structure, medium datasets, high-skewness data. Deterministic.
- DBSCAN: Best when clusters have irregular shapes, data has noise/outliers, density-based. Does not require k. Returns noise label -1.
- GMM (Gaussian Mixture): Best for overlapping clusters, soft/probabilistic assignments, elliptical clusters. Requires k.
- Fuzzy C-Means: Similar to GMM, partial memberships. Best when boundaries are gradual.

CLASSIFIER ALGORITHMS KNOWLEDGE:
- Random Forest: Robust baseline, handles high-dim, little tuning needed.
- XGBoost: Best for tabular data, handles class imbalance, slightly slower to train.
- Gradient Boosting: Good for mid-size data, slower than RF but often more accurate.
- Logistic Regression: Fast, interpretable, best for linearly separable personas.
"""

        def _llm_handler(agent: str, purpose: str, prompt: str, max_tokens: int,
                         category: str = 'pipeline') -> str:
            print(f"\n  [Orchestrator] ← {agent} requests LLM help: {purpose} ({category})")
            # Stream the request so the UI can show "Agent → Decision Maker: …"
            try:
                self.bus.emit(
                    'llm_call_started',
                    agent=agent,
                    purpose=purpose,
                    prompt_chars=len(prompt),
                    prompt=prompt,
                    category=category,
                )
            except Exception:
                pass

            # Build system context: routing decisions get the full skill catalog + ML knowledge;
            # all other calls get a brief agent-role description.
            purpose_lower = purpose.lower()
            is_routing = any(kw in purpose_lower for kw in _ROUTING_PURPOSES)
            if is_routing and _skill_summary:
                system_ctx = (
                    "You are an AI orchestrator for a multi-agent customer segmentation pipeline.\n"
                    "The following skills and agents are available to you when making decisions.\n\n"
                    f"{_skill_summary}\n\n"
                    f"{_CLUSTERING_ALGO_KNOWLEDGE}\n"
                    "Use your knowledge of these skills and algorithms when deciding how to route the pipeline."
                )
            elif is_routing:
                system_ctx = (
                    "You are an AI orchestrator for a multi-agent customer segmentation pipeline.\n"
                    f"{_CLUSTERING_ALGO_KNOWLEDGE}"
                )
            elif _agent_summary:
                system_ctx = (
                    "You are an AI component in a multi-agent customer segmentation pipeline.\n"
                    f"Pipeline context:\n{_agent_summary}"
                )
            else:
                system_ctx = (
                    "You are an AI component in a multi-agent customer segmentation pipeline."
                )

            t0 = time.perf_counter()
            resp = self.client.messages.create(
                model='claude-sonnet-4-6',
                max_tokens=max_tokens,
                system=system_ctx,
                messages=[{'role': 'user', 'content': prompt}],
            )
            elapsed = round(time.perf_counter() - t0, 2)
            self._llm_calls.append({
                'agent':         agent,
                'purpose':       purpose,
                'category':      category,
                'input_tokens':  resp.usage.input_tokens,
                'output_tokens': resp.usage.output_tokens,
                'time_s':        elapsed,
            })
            print(
                f"  [Orchestrator] → {agent}: LLM answered "
                f"(in={resp.usage.input_tokens} out={resp.usage.output_tokens} {elapsed}s)"
            )
            try:
                self.bus.emit(
                    'llm_call_finished',
                    agent=agent,
                    purpose=purpose,
                    input_tokens=resp.usage.input_tokens,
                    output_tokens=resp.usage.output_tokens,
                    time_s=elapsed,
                    response=resp.content[0].text,
                    category=category,
                )
            except Exception:
                pass
            return resp.content[0].text

        self.bus.set_llm_handler(_llm_handler)

        # ── Instantiate agents — they receive ONLY the bus, not the client ──────
        # Each agent uses its own ML/stats skills and calls bus.ask() when it
        # needs LLM reasoning. The Orchestrator remains the sole LLM gateway.
        self.input_agent            = UserInputAgent(self.bus)
        self.examiner_agent         = DatasetExaminerAgent(self.bus)
        self.feature_engineer_agent = FeatureEngineerAgent(self.bus)
        self.feature_agent          = FeatureSelectionAgent(
            self.bus,
            ae_bottleneck_cap=config.get('ae_bottleneck_cap', 32),
            ae_max_iter=config.get('ae_max_iter', 200),
        )
        self.cluster_agent          = ClusteringAgent(config, self.bus)
        self.naming_agent           = PersonaNamingAgent(self.bus)
        self.classifier_agent       = ClassifierAgent(self.bus)

        # Telemetry
        self._timings: dict[str, list[float]] = defaultdict(list)
        self._pipeline_start: float = 0.0

    def run(
        self,
        features_path: str = 'data/processed/customer_features.parquet',
        max_total_iterations: int = 10,
        skip_user_input: bool = False,
        user_intent: UserIntent | None = None,
    ) -> dict:
        """
        Run the full multi-agent pipeline.

        Parameters
        ----------
        features_path : str
            Path to the engineered features parquet file.
        max_total_iterations : int
            Maximum number of full pipeline loops before saving best and exiting.
        skip_user_input : bool
            If True, skip the UserInputAgent (use user_intent or defaults).
        user_intent : UserIntent | None
            Pre-built intent (used if skip_user_input=True).

        Returns
        -------
        dict with keys:
          'status'      : 'success' | 'quit' | 'max_iterations_reached'
          'personas'    : dict or None
          'run_history' : list of dicts summarising each stage/iteration
        """
        # Restore config from the baseline so any per-run overrides from a
        # previous .run() call don't carry into this one (fully-fresh restart).
        import copy as _copy
        self.config = _copy.deepcopy(self._config_baseline)

        # Clear any stale abort flag from a prior run (defensive — the file
        # is consumed when honoured, but a crash mid-run could leave it).
        try:
            pathlib.Path('outputs/pipeline_abort.json').unlink(missing_ok=True)
        except OSError:
            pass

        # Reset bus + per-run accumulators so a second .run() in the same
        # process (e.g. after a blocked/aborted restart) starts with no
        # leakage from the previous attempt — the UI sees a fresh run_started
        # event and wipes its in-browser state.
        self.bus.reset_for_new_run()
        self._llm_calls = []

        print('\n' + '=' * 65)
        print('Multi-Agent Clustering & Persona Discovery Pipeline')
        print('=' * 65)
        print(f'Config       : {self.config}')
        print(f'Features     : {features_path}')
        print(f'Max iters    : {max_total_iterations}')
        print(f'Classifier F1 threshold: {ClassifierAgent.F1_THRESHOLD}')

        run_history = []
        self._timings = defaultdict(list)
        self._pipeline_start = time.perf_counter()

        # Announce pipeline start so the UI can switch to "live" mode immediately
        self.bus.emit(
            'pipeline_started',
            features_path=features_path,
            max_total_iterations=max_total_iterations,
            f1_threshold=ClassifierAgent.F1_THRESHOLD,
        )

        state = PipelineState(config=self.config)

        # ── Step 0: User Intent ────────────────────────────────────────────────
        if not skip_user_input:
            self._active_agent = 'UserInput'
            _t0 = time.perf_counter()
            state.user_intent = self.input_agent.run(iteration=0)
            self._timings['UserInput'].append(time.perf_counter() - _t0)
        elif user_intent:
            state.user_intent = user_intent

        # ── Apply user-intent overrides to pipeline config ──────────────────
        # max_cluster_size_pct: if the user said "max cluster <X%>" in their
        # intent text, the UserInputAgent parsed it into this field; we
        # propagate it to self.config so Clusterer's oversized-cluster guard
        # uses the user's limit instead of the default 40%.
        if state.user_intent and state.user_intent.max_cluster_size_pct is not None:
            old = float(self.config.get('max_cluster_size_pct', 0.40))
            new = float(state.user_intent.max_cluster_size_pct)
            if abs(new - old) > 1e-6:
                self.config['max_cluster_size_pct'] = new
                print(
                    f'  [Orchestrator] max_cluster_size_pct overridden by user intent: '
                    f'{old:.0%} → {new:.0%}.'
                )
                self.bus.emit(
                    'config_override_from_intent',
                    field='max_cluster_size_pct',
                    old=old,
                    new=new,
                    source='user_intent_text',
                )

        # ── Validate user intent path; fall back to features_path if missing ──
        # UserInputAgent may default to a pre-built parquet that no longer exists.
        # When that happens, substitute the features_path supplied to run().
        if state.user_intent:
            _ui_path = pathlib.Path(state.user_intent.dataset_path)
            if not _ui_path.exists():
                _fallback = pathlib.Path(features_path)
                if _fallback.exists():
                    print(
                        f'  [Orchestrator] Dataset path not found: {state.user_intent.dataset_path!r}\n'
                        f'  [Orchestrator] Falling back to: {features_path!r}'
                    )
                    state.user_intent = UserIntent(
                        target_entity=state.user_intent.target_entity,
                        business_purpose=state.user_intent.business_purpose,
                        dataset_path=features_path,
                        constraints=state.user_intent.constraints,
                    )
                else:
                    print(
                        f'  [Orchestrator] WARNING: neither {state.user_intent.dataset_path!r} '
                        f'nor {features_path!r} found on disk.'
                    )

        # ── Resolve raw data path ──────────────────────────────────────────────
        # user_intent.dataset_path may point to a raw transaction CSV (preferred)
        # or a pre-engineered customer-feature parquet (backward compat).
        raw_data_path = (
            state.user_intent.dataset_path
            if state.user_intent and state.user_intent.dataset_path
            else None
        )

        # Detect whether to run FeatureEngineerAgent.
        # We run it when the user gave us a raw transaction CSV.
        # If they gave a parquet (or nothing), we load features_path directly.
        _raw_suffix = pathlib.Path(raw_data_path).suffix.lower() if raw_data_path else ''
        need_feature_engineering = raw_data_path is not None and _raw_suffix in ('.csv', '.tsv')

        if need_feature_engineering:
            # Load the full raw CSV — FeatureEngineer needs all rows.
            print(f'\nLoading raw transaction data: {raw_data_path}')
            full_raw_df = _load_df(raw_data_path)
            print(f'  {len(full_raw_df):,} transactions × {len(full_raw_df.columns)} columns')
            # Subsample for DatasetExaminer only (it only needs schema + stats)
            if len(full_raw_df) > 50_000:
                raw_df = full_raw_df.sample(50_000, random_state=42)
                print(f'  (subsampled to 50,000 rows for DatasetExaminer)')
            else:
                raw_df = full_raw_df
            features_df = None   # will be produced by FeatureEngineerAgent
        else:
            # Pre-engineered parquet — load directly and skip FeatureEngineer.
            load_path = raw_data_path or features_path
            print(f'\nLoading pre-engineered features: {load_path}')
            features_df = _load_df(load_path)
            if 'cluster' in features_df.columns:
                features_df = features_df.drop(columns=['cluster'])
            print(f'  {len(features_df)} customers × {len(features_df.columns)} features')
            raw_df = features_df
            full_raw_df = None

        # ── Step 1: Dataset Examination (once per pipeline run) ────────────────
        self._active_agent = 'DatasetExaminer'
        _t0 = time.perf_counter()
        dataset_profile = self.examiner_agent.run(
            user_intent=state.user_intent or UserIntent(
                target_entity="entities",
                business_purpose="discover distinct groups in the data",
                dataset_path=features_path,
            ),
            df=raw_df,
            iteration=0,
        )
        self._timings['DatasetExaminer'].append(time.perf_counter() - _t0)
        state.dataset_profile = dataset_profile

        if dataset_profile is None:
            print('\n[Orchestrator] DatasetExaminer BLOCKED — cannot proceed.')
            self.bus.emit('pipeline_complete', status='blocked',
                          reason='DatasetExaminer blocked')
            return {'status': 'blocked', 'personas': None, 'run_history': run_history}

        run_history.append({
            'iteration': 0,
            'stage': 'dataset_examination',
            'n_rows': dataset_profile.n_rows,
            'suggested_groups': dataset_profile.suggested_feature_groups,
            'algo_hint': dataset_profile.algo_hint,
        })

        # ── Decision-Maker case-memory recall ──────────────────────────────────
        # Look up any prior successful run whose dataset+goal matches this one
        # (exact = same column-set + row count; similar = looser overlap).
        # The match is stored on `state` and surfaced as a HINT block inside
        # _ask_parameter_tuning — never a hard override.
        state.case_recall = None
        try:
            from skills.case_memory import find_case
            _ui = state.user_intent
            _ds_name = pathlib.Path(raw_data_path).name if raw_data_path else features_path
            _cols = list(raw_df.columns) if raw_df is not None else []
            recall = find_case(
                dataset_name=_ds_name,
                columns=_cols,
                n_rows=len(raw_df) if raw_df is not None else 0,
                business_purpose=(_ui.business_purpose if _ui else ''),
                target_entity=(_ui.target_entity if _ui else ''),
                n_clusters_requested=(getattr(_ui, 'n_clusters_requested', None) if _ui else None),
            )
            if recall is not None:
                state.case_recall = recall
                print(
                    f'\n[Orchestrator] 🧠 Case-memory recall ({recall.match_type.upper()}): '
                    f'{recall.notes}'
                )
                strat = recall.case.get('winning_strategy', {})
                print(
                    f'  [Orchestrator] Prior winning recipe → algo={strat.get("algorithm","?")}, '
                    f'k={strat.get("k","?")}, vif={strat.get("vif_threshold","?")}, '
                    f'features={strat.get("n_features_kept","?")}.'
                )
                if recall.match_type == 'similar':
                    print('  [Orchestrator] ⚠ NOT the same case — recall used as inspiration only.')
                # Emit so the UI can surface it as a chip too.
                self.bus.emit(
                    'case_memory_recall',
                    match_type=recall.match_type,
                    notes=recall.notes,
                    prior_silhouette=recall.case.get('outcome', {}).get('silhouette'),
                    prior_cv_f1_macro=recall.case.get('outcome', {}).get('cv_f1_macro'),
                    prior_algorithm=strat.get('algorithm'),
                    prior_k=strat.get('k'),
                )
            else:
                print('\n[Orchestrator] 🧠 Case-memory: no matching prior case found.')
        except Exception as _exc:  # noqa: BLE001
            print(f'  [Orchestrator] (case-memory lookup failed: {_exc})')

        # ── Step 2: Feature Engineering (only when a raw CSV was provided) ─────
        # When the user gave a .csv path, the FeatureEngineerAgent turns the
        # event-level data into an entity-level feature matrix and saves
        # it to data/processed/. Downstream agents then use that parquet.
        if need_feature_engineering:
            print('\n[Orchestrator] Launching FeatureEngineerAgent on raw transaction data...')
            _t0 = time.perf_counter()
            self._active_agent = 'FeatureEngineer'
            try:
                features_df, fe_result = self.feature_engineer_agent.run(
                    raw_df=full_raw_df,
                    user_intent=state.user_intent or UserIntent(
                        target_entity='entities',
                        business_purpose='discover distinct groups in the data',
                        dataset_path=raw_data_path,
                    ),
                    dataset_profile=dataset_profile,
                    output_path='data/processed/engineered_features.parquet',
                    iteration=0,
                )
                self._timings['FeatureEngineer'].append(time.perf_counter() - _t0)
                run_history.append({
                    'iteration': 0,
                    'stage': 'feature_engineering',
                    'n_entities': fe_result.n_entities,
                    'n_features': fe_result.n_features,
                    'groups_built': fe_result.groups_built,
                    'elapsed_s': round(self._timings['FeatureEngineer'][-1], 1),
                })
                print(
                    f'  [Orchestrator] Feature engineering done: '
                    f'{fe_result.n_features} features × {fe_result.n_entities} entities'
                )
            except RuntimeError as e:
                print(f'\n[Orchestrator] FeatureEngineer BLOCKED: {e}')
                self.bus.emit('pipeline_complete', status='blocked',
                              reason=f'FeatureEngineer blocked: {e}')
                return {'status': 'blocked', 'personas': None, 'run_history': run_history}

        # ── Escalation thresholds (configurable) ───────────────────────────────
        # NOTE: silhouette_target is read DYNAMICALLY inside the loop because the
        # relax logic can lower it on the fly (state.silhouette_target_override).
        config_silhouette_target = float(self.config.get('silhouette_target', 0.5))
        max_reselect_failures = int(self.config.get('max_reselect_failures', 3))
        max_relax_failures = int(self.config.get('max_relax_failures', 3))

        def _current_silhouette_target() -> float:
            return state.silhouette_target_override \
                if state.silhouette_target_override is not None \
                else config_silhouette_target

        # ── Main pipeline loop ─────────────────────────────────────────────────
        while state.total_iterations < max_total_iterations:
            # ── Abort check (UI-driven): if outputs/pipeline_abort.json exists,
            # stop the run cleanly between iterations. run_pipeline.py loops and
            # picks up a fresh intent on the next pass.
            _abort_path = pathlib.Path('outputs/pipeline_abort.json')
            if _abort_path.exists():
                try:
                    _abort_payload = json.loads(_abort_path.read_text(encoding='utf-8'))
                except (OSError, json.JSONDecodeError):
                    _abort_payload = {}
                _reason = _abort_payload.get('reason', 'user_abort')
                _restart = bool(_abort_payload.get('restart', True))
                print(f'\n[Orchestrator] 🛑 Abort signal received '
                      f'(reason={_reason!r}, restart={_restart}).')
                try:
                    _abort_path.unlink(missing_ok=True)
                except OSError:
                    pass
                self.bus.emit(
                    'pipeline_complete',
                    status='aborted',
                    reason=_reason,
                    restart=_restart,
                    completed_iterations=state.total_iterations,
                )
                return {
                    'status': 'aborted',
                    'reason': _reason,
                    'restart': _restart,
                    'personas': None,
                    'run_history': run_history,
                }

            state.total_iterations += 1
            iteration = state.total_iterations
            print(f'\n{"─"*65}')
            print(f'ITERATION {iteration} / {max_total_iterations}')
            print(f'{"─"*65}')
            self.bus.emit(
                'iteration_started',
                iteration=iteration,
                max_total_iterations=max_total_iterations,
            )

            # ── ESCALATION: re-engineer features from scratch ─────────────────
            # Triggered when we've had N consecutive low-silhouette iterations.
            if state.needs_feature_engineering and full_raw_df is not None:
                print(f'\n[Orchestrator] ESCALATION — {state.consecutive_silhouette_failures} '
                      f'failures in a row. Re-running FeatureEngineer from raw data + '
                      f'asking Decision Maker for a fresh algorithm pick.')
                _target = _current_silhouette_target()
                self.bus.emit(
                    'feature_re_engineering',
                    consecutive_failures=state.consecutive_silhouette_failures,
                    silhouette_target=_target,
                )
                # Clear stale tuning so the LLM picks a different algorithm
                state.tuning_params['algorithm'] = None
                state.tuning_params['feature_focus'] = (
                    f"Previous engineered features gave silhouette < {_target} "
                    f"across {state.consecutive_silhouette_failures} iterations — try a "
                    f"fundamentally different set of features."
                )
                _t0 = time.perf_counter()
                self._active_agent = 'FeatureEngineer'
                try:
                    features_df, fe_result = self.feature_engineer_agent.run(
                        raw_df=full_raw_df,
                        user_intent=state.user_intent or UserIntent(
                            target_entity='entities',
                            business_purpose='discover distinct groups in the data',
                            dataset_path=raw_data_path or features_path,
                        ),
                        dataset_profile=state.dataset_profile,
                        output_path='data/processed/engineered_features.parquet',
                        iteration=iteration,
                    )
                    self._timings['FeatureEngineer'].append(time.perf_counter() - _t0)
                    state.needs_feature_engineering = False
                    state.consecutive_silhouette_failures = 0
                    state.needs_feature_selection = True   # force fresh selection
                except RuntimeError as e:
                    print(f'[Orchestrator] FeatureEngineer escalation failed: {e}')
                    state.needs_feature_engineering = False  # don't loop forever

            # Check for hard blocks from the bus
            if self.bus.has_hard_block():
                print('\n[Orchestrator] Hard block detected — triggering human checkpoint.')
                break

            # ── (2) Feature Selection ──────────────────────────────────────────
            if state.needs_feature_selection:
                _t0 = time.perf_counter()
                self._active_agent = 'FeatureSelector'
                # Build combined feedback: agent feedback + orchestrator feature focus hint
                _fs_feedback = state.fs_feedback
                if state.tuning_params.get('feature_focus'):
                    _fs_feedback = (
                        f"{_fs_feedback}\nOrchestrator guidance: {state.tuning_params['feature_focus']}"
                    ).strip()
                fs = self.feature_agent.run(
                    features_df,
                    user_intent=state.user_intent,
                    dataset_profile=state.dataset_profile,
                    feedback=_fs_feedback,
                    iteration=iteration,
                    vif_threshold=state.tuning_params.get('vif_threshold'),
                    feature_focus=state.tuning_params.get('feature_focus', ''),
                )
                self._timings['FeatureSelector'].append(time.perf_counter() - _t0)
                state.update_features(fs)
                run_history.append({
                    'iteration': iteration,
                    'stage': 'feature_selection',
                    'n_features': fs.n_features,
                    'n_removed_vif': len(fs.removed_by_vif),
                    'elapsed_s': round(self._timings['FeatureSelector'][-1], 1),
                    'reasoning': fs.reasoning,
                    'vif_threshold_used': state.tuning_params.get('vif_threshold', 10.0),
                })

            # ── (3) Clustering ─────────────────────────────────────────────────
            _t0 = time.perf_counter()
            self._active_agent = 'Clusterer'
            # Build per-iteration config overrides from tuning params
            _cluster_override: dict = {}
            if state.tuning_params.get('algorithm') is not None:
                _cluster_override['clustering_algorithm'] = state.tuning_params['algorithm']
            if state.tuning_params.get('k_range') is not None:
                _cluster_override['k_search_range'] = state.tuning_params['k_range']
            cr = self.cluster_agent.run(
                features_df,
                selected_features=state.selected_features,
                user_intent=state.user_intent,
                dataset_profile=state.dataset_profile,
                history=state.clustering_history,
                feedback=state.cluster_feedback,
                iteration=iteration,
                config_override=_cluster_override or None,
                min_silhouette=state.tuning_params.get('min_silhouette'),
                silhouette_target=_current_silhouette_target(),
            )
            self._timings['Clusterer'].append(time.perf_counter() - _t0)
            state.clustering_history.append(cr)

            # Track best silhouette regardless of whether clustering "passed"
            state.update_best_silhouette(cr, state.selected_features)

            # ── Per-iteration PCA snapshot (for the Evidence tab visual) ───────
            # Save a 2-D PCA projection of the selected-feature matrix coloured
            # by cluster id. Skipped on reselect (no clustering produced).
            if cr.cluster_labels is not None and cr.profiles is not None:
                try:
                    self._save_pca_projection(features_df, state.selected_features,
                                              cr.cluster_labels, iteration, cr)
                except Exception as _exc:  # noqa: BLE001
                    print(f'  [Orchestrator] PCA snapshot skipped: {_exc}')

            # ── HARD STOP: Clusterer itself bailed out (sil < hard min) ────────
            # When the clusterer's internal min_silhouette gate (~0.05) hits, no
            # cluster labels were produced. Without labels we can't run Naming or
            # Classifier. Skip the rest of this iteration and reselect features.
            if cr.action == 'reselect_features':
                print(f'\n[Orchestrator] Clustering → reselect features: {cr.reasoning}')
                new_params = self._ask_parameter_tuning(iteration, run_history, state)
                state.tuning_params.update(new_params)
                state.request_feature_reselection(cr.reasoning)
                run_history.append({
                    'iteration': iteration,
                    'stage': 'clustering',
                    'action': 'reselect_features',
                    'elapsed_s': round(self._timings['Clusterer'][-1], 1),
                    'reasoning': cr.reasoning,
                    'tuning_applied': new_params,
                })
                continue

            _sil = cr.silhouette if cr.silhouette is not None else -1.0
            _target_now = _current_silhouette_target()
            _silhouette_missed = _sil < _target_now

            # ── (4) Persona Naming — ALWAYS run ─────────────────────────────────
            # User spec: every iteration must produce a Classifier F1 so the
            # best-iteration decision (composite: F1↑ + Silhouette↑ + VIF↓) is
            # comparable across all 10 iterations — not only iterations that
            # cleared the Clarity Gate.
            #
            # We pass force_proceed=True whenever silhouette is below target so
            # naming returns personas even when the Clarity Gate would normally
            # block; otherwise Classifier has no labels to learn.
            _t0 = time.perf_counter()
            self._active_agent = 'PersonaNamer'
            nr = self.naming_agent.run(
                profiles=cr.profiles,
                lineage=cr.lineage,
                tone=self.config.get('persona_tone', 'easy'),
                feedback=state.naming_feedback,
                iteration=iteration,
                user_intent=state.user_intent,
                force_proceed=_silhouette_missed,
            )
            self._timings['PersonaNamer'].append(time.perf_counter() - _t0)
            state.naming_history.append(nr)

            # If naming failed *and* we didn't force, fall back: re-run with
            # force so Classifier still has labels. Cheap insurance.
            if (not nr.passed) and (not _silhouette_missed) and nr.personas is None:
                self._active_agent = 'PersonaNamer'
                nr = self.naming_agent.run(
                    profiles=cr.profiles,
                    lineage=cr.lineage,
                    tone=self.config.get('persona_tone', 'easy'),
                    feedback='Clarity Gate failed once; running with force_proceed so '
                             'Classifier can still score this iteration.',
                    iteration=iteration,
                    user_intent=state.user_intent,
                    force_proceed=True,
                )
                state.naming_history.append(nr)

            run_history.append({
                'iteration': iteration,
                'stage': 'naming',
                'passed': nr.passed,
                'avg_confidence': nr.avg_confidence,
                'silhouette': cr.silhouette,
                'n_leaf': cr.n_leaf,
                'algo': cr.algo_name,
                'elapsed_s': round(self._timings['PersonaNamer'][-1], 1),
                'issues': nr.issues,
                'force_proceed': _silhouette_missed,
            })

            # ── (5) Classifier — ALWAYS run (per user spec) ─────────────────────
            # Even when silhouette missed or Clarity Gate failed, we still need
            # F1 to score this iteration against the others.
            clf = None
            if nr.personas:
                _t0 = time.perf_counter()
                self._active_agent = 'Classifier'
                try:
                    clf = self.classifier_agent.run(
                        features_df=features_df,
                        cluster_labels=cr.cluster_labels,
                        personas=nr.personas,
                        history=state.classifier_history,
                        feedback=state.classifier_feedback,
                        iteration=iteration,
                        config=self.config,
                    )
                except Exception as _clf_exc:  # noqa: BLE001
                    print(f'  [Orchestrator] Classifier raised on iter {iteration}: {_clf_exc}'
                          f' — iteration scored without F1.')
                    clf = None
                self._timings['Classifier'].append(time.perf_counter() - _t0)
                if clf is not None:
                    state.classifier_history.append(clf)

            # ── Composite score → best iteration tracker ───────────────────────
            # F1↑ + Silhouette↑ − VIF penalty. See PipelineState.composite_score.
            max_vif_now = state.current_max_vif()
            became_best = False
            if clf is not None:
                became_best = state.update_best(nr, cr, clf, max_vif=max_vif_now)
            state.update_best_silhouette(cr, state.selected_features)

            run_history.append({
                'iteration': iteration,
                'stage': 'classifier',
                'action': clf.action if clf else 'skipped',
                'cv_accuracy': clf.cv_accuracy if clf else None,
                'cv_f1_macro': clf.cv_f1_macro if clf else None,
                'max_vif': round(max_vif_now, 3),
                'composite_score': round(state.composite_score(
                    cr.silhouette,
                    clf.cv_f1_macro if clf else None,
                    max_vif_now,
                ), 3),
                'became_best': became_best,
                'elapsed_s': round(self._timings['Classifier'][-1], 1) if clf else 0,
                'reasoning': clf.reasoning if clf else 'classifier skipped (no personas)',
            })
            if became_best:
                print(
                    f'  [Orchestrator] iter {iteration} is new BEST '
                    f'(F1={clf.cv_f1_macro:.3f}, Sil={cr.silhouette:.3f}, '
                    f'maxVIF={max_vif_now:.2f}, score={state.best_composite_score:.2f})'
                )

            # ── Apply escalation rules AFTER scoring ───────────────────────────
            # (a) silhouette miss → reselect features (+ tiered escalations)
            if _silhouette_missed:
                state.consecutive_silhouette_failures += 1
                state.silhouette_fail_for_relax += 1
                print(f'\n[Orchestrator] Silhouette {_sil:.3f} < target {_target_now:.2f} '
                      f'(re-engineer counter {state.consecutive_silhouette_failures}/{max_reselect_failures} · '
                      f'relax counter {state.silhouette_fail_for_relax}/{max_relax_failures}).')
                self.bus.emit(
                    'silhouette_target_missed',
                    silhouette=_sil,
                    target=_target_now,
                    consecutive_failures=state.consecutive_silhouette_failures,
                    max_failures=max_reselect_failures,
                    relax_failures=state.silhouette_fail_for_relax,
                    max_relax_failures=max_relax_failures,
                )
                if state.silhouette_fail_for_relax >= max_relax_failures:
                    self._relax_silhouette_target(state, _target_now)
                if state.consecutive_silhouette_failures >= max_reselect_failures:
                    state.needs_feature_engineering = True
                    state.cluster_feedback = (
                        f'{max_reselect_failures} consecutive iterations with silhouette '
                        f'< target. Pick a completely different algorithm next round '
                        f'based on the whole history.'
                    )
                else:
                    new_params = self._ask_parameter_tuning(iteration, run_history, state)
                    state.tuning_params.update(new_params)
                    state.request_feature_reselection(
                        f'silhouette {_sil:.3f} < target {_target_now:.2f}'
                    )
                continue

            # silhouette passed → reset counters
            state.consecutive_silhouette_failures = 0
            state.silhouette_fail_for_relax = 0

            # (b) Clarity Gate failed → re-cluster (but the iteration's F1 has
            # already been recorded into the best-iteration tracker above).
            if not nr.passed:
                state.cluster_feedback = f'Clarity Gate failed: {"; ".join(nr.issues)}'
                state.naming_feedback = ''
                print(f'\n[Orchestrator] Clarity Gate failed → re-clustering.')
                new_params = self._ask_parameter_tuning(iteration, run_history, state)
                state.tuning_params.update(new_params)
                continue

            # (c) Classifier asked for reselect / recluster
            if clf is not None and clf.action == 'reselect_features':
                print(f'\n[Orchestrator] Classifier → reselect features: {clf.reasoning}')
                new_params = self._ask_parameter_tuning(iteration, run_history, state)
                state.tuning_params.update(new_params)
                state.request_feature_reselection(clf.reasoning)
                state.classifier_feedback = clf.reasoning
                continue
            if clf is not None and clf.action == 'recluster':
                print(f'\n[Orchestrator] Classifier → re-cluster: {clf.reasoning}')
                new_params = self._ask_parameter_tuning(iteration, run_history, state)
                state.tuning_params.update(new_params)
                state.cluster_feedback = f'Classifier CV F1={clf.cv_f1_macro:.3f} too low: {clf.reasoning}'
                state.classifier_feedback = clf.reasoning
                continue

            # ── (6) Human Checkpoint ───────────────────────────────────────────
            decision = human_checkpoint(nr.personas, cr, clf, self.bus)

            run_history.append({
                'iteration': iteration,
                'stage': 'human_checkpoint',
                'decision': decision.action,
                'feedback': decision.feedback,
            })

            if decision.action == 'approve':
                # Make sure the CURRENT iteration competes for best (it just passed
                # all gates) so the all-time best comparison is fair.
                state.update_best(nr, cr, clf)
                # SAVE THE ALL-TIME BEST, not necessarily the current iteration.
                # If an earlier iteration scored higher F1, that one wins — so the
                # Named Clusters tab always shows the actual best-performing run.
                best_cr  = state.best_clustering_result or cr
                best_nr  = state.best_naming_result or nr
                best_clf = state.best_classifier_result or clf
                if best_nr is not nr:
                    print(
                        f'\n[Orchestrator] Approved at iter {cr.iteration}, but iter '
                        f'{best_nr.iteration} scored higher (F1={best_clf.cv_f1_macro:.3f} '
                        f'vs {clf.cv_f1_macro:.3f}). Saving iter {best_nr.iteration} as the winner.'
                    )
                else:
                    print('\n[Orchestrator] Approved! Saving outputs...')
                save_outputs(best_cr, best_nr, best_clf, self.bus)
                self._persist_case_memory(
                    state=state,
                    raw_df=raw_df,
                    raw_data_path=raw_data_path or features_path,
                    best_cr=best_cr,
                    best_clf=best_clf,
                    run_history=run_history,
                    status='success',
                )
                self._print_timing_summary()
                self.bus.emit(
                    'pipeline_complete',
                    status='success',
                    n_clusters=len(best_nr.personas) if best_nr.personas else 0,
                    silhouette=getattr(best_cr, 'silhouette', None),
                    cv_f1_macro=best_clf.cv_f1_macro,
                    winning_iteration=getattr(best_nr, 'iteration', None),
                )
                return {
                    'status': 'success',
                    'personas': best_nr.personas,
                    'classifier': {
                        'cv_accuracy': best_clf.cv_accuracy,
                        'cv_f1_macro': best_clf.cv_f1_macro,
                        'cv_f1_weighted': best_clf.cv_f1_weighted,
                        'per_class_f1': best_clf.per_class_f1,
                    },
                    'winning_iteration': getattr(best_nr, 'iteration', None),
                    'run_history': run_history,
                    'timing': self._timing_dict(),
                    'llm_usage': self._llm_usage_dict(),
                }

            elif decision.action == 'quit':
                print('\n[Orchestrator] Quit without saving.')
                self.bus.emit('pipeline_complete', status='quit')
                return {
                    'status': 'quit',
                    'personas': None,
                    'run_history': run_history,
                }

            elif decision.action == 'recluster':
                # When the checkpoint asks for another iteration, tune params first
                # so the next round tries a different algorithm/k — otherwise we'd
                # just re-cluster with identical settings and produce identical
                # personas. This is what gives the orchestrator iteration diversity
                # when run_pipeline.py defers approval to collect more candidates.
                new_params = self._ask_parameter_tuning(iteration, run_history, state)
                state.tuning_params.update(new_params)
                state.cluster_feedback = decision.feedback
                state.naming_feedback = ''
                state.classifier_feedback = ''
                print(f'\n[Orchestrator] Human → re-cluster: {decision.feedback!r}')

            elif decision.action == 'reselect_features':
                state.request_feature_reselection(decision.feedback)
                state.classifier_feedback = ''
                print(f'\n[Orchestrator] Human → reselect features: {decision.feedback!r}')

        # ── Max iterations reached ─────────────────────────────────────────────
        self._print_timing_summary()
        print(f'\n[Orchestrator] Max iterations ({max_total_iterations}) reached.')

        # ── Path A: a naming result passed the Clarity Gate at some point ──────
        if state.best_naming_result is not None:
            best_personas = state.best_naming_result.personas
            best_clf = state.best_classifier_result
            print(
                f'  Best approved result: avg_confidence={state.best_naming_result.avg_confidence:.1f}'
                + (f'  cv_f1={best_clf.cv_f1_macro:.3f}' if best_clf else '')
            )
            print('  Saving best result...')
            save_outputs(
                state.best_clustering_result,
                state.best_naming_result,
                best_clf,
                self.bus,
            )
            self._persist_case_memory(
                state=state,
                raw_df=raw_df,
                raw_data_path=raw_data_path or features_path,
                best_cr=state.best_clustering_result,
                best_clf=best_clf,
                run_history=run_history,
                status='max_iterations_reached',
            )
            self.bus.emit(
                'pipeline_complete',
                status='max_iterations_reached',
                n_clusters=len(best_personas) if best_personas else 0,
                silhouette=getattr(state.best_clustering_result, 'silhouette', None),
                cv_f1_macro=getattr(best_clf, 'cv_f1_macro', None),
            )
            return {
                'status': 'max_iterations_reached',
                'personas': best_personas,
                'run_history': run_history,
                'timing': self._timing_dict(),
                'llm_usage': self._llm_usage_dict(),
            }

        # ── Path B: no naming result ever passed — use the best-silhouette cluster ─
        if state.best_silhouette_cluster is not None:
            best_cr = state.best_silhouette_cluster
            print(
                f'\n[Orchestrator] No approved result — delivering best-effort analysis '
                f'(silhouette={state.best_silhouette_value:.4f}, '
                f'k={best_cr.n_leaf} leaf clusters).'
            )
            print('  Running PersonaNamer (force_proceed=True) on best clustering...')
            self._active_agent = 'PersonaNamer'
            best_nr = self.naming_agent.run(
                profiles=best_cr.profiles,
                lineage=best_cr.lineage,
                tone=self.config.get('persona_tone', 'easy'),
                user_intent=state.user_intent,
                feedback='Best-effort fallback: deliver the best personas available.',
                iteration=state.total_iterations + 1,
                force_proceed=True,
            )

            print('  Running Classifier on best clustering...')
            self._active_agent = 'Classifier'
            best_clf = self.classifier_agent.run(
                features_df=features_df,
                cluster_labels=best_cr.cluster_labels,
                personas=best_nr.personas,
                history=state.classifier_history,
                feedback='Best-effort fallback run.',
                iteration=state.total_iterations + 1,
                config=self.config,
            )

            print('  Saving best-effort result...')
            save_outputs(best_cr, best_nr, best_clf, self.bus)

            self.bus.emit(
                'pipeline_complete',
                status='best_effort',
                n_clusters=len(best_nr.personas) if best_nr.personas else 0,
                silhouette=state.best_silhouette_value,
                cv_f1_macro=best_clf.cv_f1_macro,
            )
            return {
                'status': 'best_effort',
                'personas': best_nr.personas,
                'classifier': {
                    'cv_accuracy':    best_clf.cv_accuracy,
                    'cv_f1_macro':    best_clf.cv_f1_macro,
                    'cv_f1_weighted': best_clf.cv_f1_weighted,
                    'per_class_f1':   best_clf.per_class_f1,
                },
                'silhouette': state.best_silhouette_value,
                'run_history': run_history,
                'timing': self._timing_dict(),
                'llm_usage': self._llm_usage_dict(),
            }

        # ── Path C: no usable result at all ────────────────────────────────────
        print('\n[Orchestrator] No usable clustering result found after all iterations.')
        self.bus.emit(
            'pipeline_complete',
            status='max_iterations_reached',
            reason='no usable clustering result',
        )
        return {
            'status': 'max_iterations_reached',
            'personas': None,
            'run_history': run_history,
            'timing': self._timing_dict(),
            'llm_usage': self._llm_usage_dict(),
        }

    # ── Per-iteration PCA projection (Evidence tab visual) ────────────────────

    def _save_pca_projection(self, features_df, selected_features, cluster_labels,
                              iteration: int, cr) -> None:
        """Append a 2-D PCA projection of this iteration's clustered data."""
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        cols = [c for c in selected_features if c in features_df.columns]
        if not cols or len(features_df) == 0:
            return
        X = features_df[cols].fillna(0).to_numpy(dtype=float)
        labels = np.asarray(list(cluster_labels))
        if len(X) != len(labels):
            return
        # Sub-sample to keep the JSON small + browser snappy
        MAX_POINTS = 1500
        if len(X) > MAX_POINTS:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), MAX_POINTS, replace=False)
            X = X[idx]; labels = labels[idx]

        try:
            X_scaled = StandardScaler().fit_transform(X)
            pca = PCA(n_components=2)
            Z = pca.fit_transform(X_scaled)
        except Exception:
            return

        points = [{'x': round(float(z[0]), 4),
                   'y': round(float(z[1]), 4),
                   'c': int(c)} for z, c in zip(Z, labels)]
        out = pathlib.Path('outputs/pca_iterations.json')
        try:
            existing = json.loads(out.read_text(encoding='utf-8')) if out.exists() else []
        except Exception:
            existing = []
        existing.append({
            'iteration': iteration,
            'algorithm': getattr(cr, 'algo_name', ''),
            'silhouette': float(cr.silhouette) if cr.silhouette is not None else None,
            'k': int(cr.n_leaf or 0),
            'n_points': len(points),
            'explained_variance_ratio': [round(float(v), 4) for v in pca.explained_variance_ratio_.tolist()],
            'points': points,
        })
        try:
            out.write_text(json.dumps(existing, ensure_ascii=False), encoding='utf-8')
            print(f'  [Orchestrator] PCA snapshot saved (iter {iteration}, {len(points)} points)')
        except OSError as e:
            print(f'  [Orchestrator] PCA snapshot save failed: {e}')

    # ── Adaptive silhouette target relaxation ──────────────────────────────────

    def _relax_silhouette_target(self, state, current_target: float) -> None:
        """After max_relax_failures consecutive misses, lower the bar.

        - BYPASS mode: auto-drop by 0.1.
        - INTERACTIVE mode: pause, wait for the user to type a new target via
          the UI (POST /api/silhouette-target → outputs/pending_target_change.json).
        Resets the relax counter either way.
        """
        from skills.orchestrator_bus import read_pipeline_mode
        mode = read_pipeline_mode()
        suggested = round(max(0.05, current_target - 0.1), 3)

        if mode == 'interactive':
            new_target = self._wait_for_target_change(current_target, suggested, state)
        else:
            new_target = suggested
            print(f'  [Orchestrator] BYPASS — auto-lowering silhouette_target '
                  f'{current_target:.2f} → {new_target:.2f}')

        state.silhouette_target_override = float(new_target)
        state.silhouette_fail_for_relax = 0
        self.bus.emit(
            'silhouette_target_changed',
            previous=current_target,
            new=new_target,
            mode=mode,
        )
        # Emit a HIGH-VISIBILITY Orchestrator agent_report so the degrade lands in
        # the right-column outputs panel as a warning chip — not just a buried
        # event log line. This makes the "0.5 → 0.4" relax decision impossible
        # to miss when reviewing why a run accepted a lower-quality clustering.
        try:
            from skills.orchestrator_bus import OrchestratorMessage
            self.bus.report(OrchestratorMessage(
                agent="Orchestrator",
                iteration=state.total_iterations,
                status="warning",
                what_was_done=(
                    f"Silhouette target relaxed {current_target:.2f} → {new_target:.2f} "
                    f"({mode} mode) after 3 consecutive iterations failed to clear the bar."
                ),
                what_was_not_done=(
                    "Did not raise the target back; future iterations only need to "
                    f"clear {new_target:.2f} to be accepted."
                ),
                doubts=(
                    "Lower target accepts weaker cluster separation — interpretability "
                    "of the resulting personas may drop."
                ),
                issues=[
                    f"⚠ silhouette_target degraded {current_target:.2f}→{new_target:.2f} "
                    f"(step #{int(round((0.5 - new_target) / 0.1))}, {mode})"
                ],
                metrics={
                    "silhouette_target_previous": round(current_target, 3),
                    "silhouette_target_new": round(new_target, 3),
                    "mode": mode,
                },
                recommendation="proceed",
                context={"reason": "max_relax_failures reached"},
            ))
        except Exception:  # noqa: BLE001
            pass

    def _wait_for_target_change(self, current_target: float, suggested: float, state) -> float:
        """Block until the user submits a new silhouette_target via /api/silhouette-target."""
        import pathlib, time as _time
        pending = pathlib.Path('outputs/pending_target_change.json')
        try:
            pending.unlink(missing_ok=True)
        except OSError:
            pass
        self.bus.emit(
            'awaiting_silhouette_relaxation',
            current_target=current_target,
            suggested_target=suggested,
            consecutive_failures=state.silhouette_fail_for_relax,
            timeout_s=300,
        )
        print(f'\n  [INTERACTIVE MODE] {state.silhouette_fail_for_relax} silhouette '
              f'misses in a row. Pipeline paused for you to set a new target.')
        print(f'  [INTERACTIVE MODE] Current target: {current_target:.2f} · '
              f'suggested: {suggested:.2f}.')

        deadline = _time.time() + 300
        try:
            while _time.time() < deadline:
                if pending.exists():
                    try:
                        payload = json.loads(pending.read_text(encoding='utf-8'))
                        v = float(payload.get('target'))
                        if 0.05 <= v <= 1.0:
                            try: pending.unlink(missing_ok=True)
                            except OSError: pass
                            return v
                    except (OSError, ValueError, TypeError, json.JSONDecodeError):
                        pass
                _time.sleep(0.6)
        except KeyboardInterrupt:
            pass
        print(f'  [INTERACTIVE MODE] Timed out — auto-lowering to {suggested:.2f}.')
        return suggested

    # ── Dynamic parameter tuning ───────────────────────────────────────────────

    def _ask_parameter_tuning(
        self,
        iteration: int,
        run_history: list[dict],
        state,
    ) -> dict:
        """
        After a failed iteration, ask the LLM to suggest improved pipeline parameters.

        The LLM sees the history of what happened (silhouette scores, VIF removals,
        feature counts) and proposes new values for vif_threshold, k_range,
        algorithm, min_silhouette, and feature_focus.

        Returns a (possibly partial) dict of new parameter values.
        """
        # Compact history summary — last 8 pipeline events
        history_lines = []
        for h in run_history[-8:]:
            stage = h.get('stage', '')
            it    = h.get('iteration', '?')
            if stage == 'feature_selection':
                history_lines.append(
                    f"  Iter {it} FeatureSelector: kept {h.get('n_features','?')} features, "
                    f"VIF removed {h.get('n_removed_vif','?')}"
                )
            elif stage == 'clustering':
                history_lines.append(
                    f"  Iter {it} Clusterer: action={h.get('action','?')}, "
                    f"reason={str(h.get('reasoning',''))[:100]}"
                )
            elif stage == 'naming':
                history_lines.append(
                    f"  Iter {it} PersonaNamer: passed={h.get('passed','?')}, "
                    f"sil={h.get('silhouette','?')}, issues={h.get('issues','[]')}"
                )
            elif stage == 'classifier':
                f1 = h.get('cv_f1_macro')
                f1_str = f"{f1:.3f}" if isinstance(f1, (int, float)) else "n/a"
                history_lines.append(
                    f"  Iter {it} Classifier: f1_macro={f1_str}, action={h.get('action','?')}"
                )

        best_sil_str = (
            f"{state.best_silhouette_value:.4f}" if state.best_silhouette_value > -1 else "none yet"
        )
        k_scores_str = ""
        if state.best_silhouette_cluster and state.best_silhouette_cluster.k_scores:
            ks = state.best_silhouette_cluster.k_scores
            k_scores_str = "\nBest k-curve: " + ", ".join(
                f"k={k}:{v:.3f}" for k, v in sorted(ks.items())
            )

        # ── Adaptive learning: prepend persistent user feedback ──────────────
        # Same source PersonaNamer reads (outputs/user_feedback_log.jsonl).
        # Surfaces global rules and high-priority overrides so the Decision
        # Maker's parameter tuning respects choices the user made in the UI.
        prefs_preamble = ''
        try:
            from ui.feedback_store import build_preferences_block
            prefs_block = build_preferences_block(
                types=('global_rule', 'manual_override', 'merge', 'naming_hint'),
            )
            if prefs_block:
                prefs_preamble = (
                    prefs_block
                    + 'These are durable user preferences from prior UI sessions — '
                    + 'honour them when tuning.\n\n'
                )
                print(f'  [Orchestrator] Injected user-preference block into tuning prompt.')
        except Exception as _exc:  # noqa: BLE001
            print(f'  [Orchestrator] (no UI feedback memory loaded: {_exc})')

        # ── Case-memory hint (HINT-ONLY — LLM may ignore) ──────────────────
        # If find_case() matched a prior successful run, render it as a
        # paragraph the tuning LLM can use as a starting point. For 'similar'
        # matches the block explicitly warns this is NOT the same case.
        case_preamble = ''
        if getattr(state, 'case_recall', None) is not None:
            try:
                from skills.case_memory import build_hint_block
                case_preamble = build_hint_block(state.case_recall) + '\n'
                print(
                    f'  [Orchestrator] Injected case-memory hint block '
                    f'(match={state.case_recall.match_type}) into tuning prompt.'
                )
            except Exception as _exc:  # noqa: BLE001
                print(f'  [Orchestrator] (case-memory hint render failed: {_exc})')

        cur = state.tuning_params
        prompt = prefs_preamble + case_preamble + f"""You are orchestrating a customer-clustering pipeline. Iteration {iteration} just failed.

Current parameters:
  vif_threshold  : {cur.get('vif_threshold', 10.0)}   (higher = keep more correlated features; range 5–25)
  algorithm      : {cur.get('algorithm') or 'auto'}
  k_range        : {cur.get('k_range') or 'default [3,4,5,6,7,8,10,12,15]'}
  min_silhouette : {cur.get('min_silhouette', 0.05)}  (hard-block; range 0.02–0.12)
  feature_focus  : "{cur.get('feature_focus', '')}"

Best silhouette achieved so far: {best_sil_str}{k_scores_str}

Pipeline history (recent):
{chr(10).join(history_lines)}

Dataset: ~983 bank customers, ~232 transaction-ratio/spend/frequency features.
Banking features typically yield silhouette 0.08–0.20 even in good segmentations.

Available clustering algorithms:
  kmeans, hierarchical, dbscan, gmm, fuzzy_cmeans, or null (auto-select)

Tuning guidelines:
- If VIF gate removes >60 features or hits max_iterations: raise vif_threshold (try 12–18)
- If silhouette consistently <0.10 with hierarchical: switch to kmeans or gmm
- If silhouette is low across all k values: narrow k_range to [3,4,5,6] or [4,5,6,7]
- If many features selected but silhouette stays low: add feature_focus to guide selector
- Lower min_silhouette only if data genuinely resists clustering (e.g. keep at 0.03)
- Do NOT lower min_silhouette below 0.02
- dbscan is good when you suspect outlier-heavy data or irregular cluster shapes

Return ONLY a valid JSON object — no markdown fences, no extra text:
{{
  "vif_threshold": <float 5–25>,
  "k_range": [<int>, ...],
  "algorithm": "kmeans" | "hierarchical" | "dbscan" | "gmm" | "fuzzy_cmeans" | null,
  "min_silhouette": <float 0.02–0.12>,
  "feature_focus": "<short hint for FeatureSelector, or empty string>",
  "reasoning": "<1-2 sentences explaining the change>"
}}"""

        print(f'\n  [Orchestrator] Asking LLM to tune parameters for iteration {iteration + 1}...')
        raw = self.bus.ask(
            agent="Orchestrator",
            purpose="tune pipeline parameters based on iteration failures",
            prompt=prompt,
            max_tokens=512,
        ).strip()

        # Strip markdown fences
        if '```' in raw:
            for part in raw.split('```'):
                p = part.strip()
                if p.startswith('json'):
                    p = p[4:].strip()
                if p.startswith('{'):
                    raw = p
                    break

        try:
            params = json.loads(raw)
        except json.JSONDecodeError:
            print('  [Orchestrator] Parameter tuning: invalid JSON response — keeping current params.')
            return {}

        # Validate and clamp each field
        result: dict = {}
        if 'vif_threshold' in params:
            result['vif_threshold'] = float(max(5.0, min(25.0, params['vif_threshold'])))
        if 'k_range' in params and isinstance(params['k_range'], list) and len(params['k_range']) >= 2:
            result['k_range'] = [int(k) for k in params['k_range'] if isinstance(k, (int, float))]
        if params.get('algorithm') in ('kmeans', 'hierarchical', 'dbscan', 'gmm', 'fuzzy_cmeans', None):
            result['algorithm'] = params['algorithm']
        if 'min_silhouette' in params:
            result['min_silhouette'] = float(max(0.02, min(0.12, params['min_silhouette'])))
        if 'feature_focus' in params:
            result['feature_focus'] = str(params.get('feature_focus', ''))

        reasoning = params.get('reasoning', '')
        print(
            f'  [Orchestrator] Tuned → vif={result.get("vif_threshold","—")}, '
            f'algo={result.get("algorithm","—")}, '
            f'k_range={result.get("k_range","—")}, '
            f'min_sil={result.get("min_silhouette","—")}'
        )
        if reasoning:
            print(f'  [Orchestrator] Reasoning: {reasoning}')
        return result

    # ── Case-memory persistence ────────────────────────────────────────────────
    def _persist_case_memory(
        self,
        *,
        state,
        raw_df,
        raw_data_path: str,
        best_cr,
        best_clf,
        run_history: list[dict],
        status: str,
    ) -> None:
        """Save the winning iteration's recipe to the decision-maker case book.

        Called on success (human-approve) and on max-iterations-reached when a
        usable best result exists. Failures here are non-fatal — we never want
        a memory-write to derail a successful run.
        """
        try:
            from skills.case_memory import save_case

            ui = state.user_intent
            # Recover the feature list that produced the winning clustering.
            features_used = list(state.best_silhouette_features) \
                if getattr(state, 'best_silhouette_features', None) else \
                list(state.selected_features or [])

            # Track which tuning params were live when the best run completed.
            tp = dict(state.tuning_params or {})

            # Lessons learned — derived mechanically from run_history.
            lessons: list[str] = []
            algo_seen = []
            singleton_iters = []
            relax_steps = 0
            vif_changes = []
            for h in run_history:
                if h.get('stage') == 'clustering':
                    a = h.get('algorithm') or h.get('algo_name')
                    if a:
                        algo_seen.append(a)
                    if h.get('singleton_merges'):
                        singleton_iters.append(h.get('iteration'))
                if h.get('stage') == 'tuning' and h.get('vif_threshold') is not None:
                    vif_changes.append(h.get('vif_threshold'))
                if 'silhouette_target_new' in h:
                    relax_steps += 1

            winning_algo = getattr(best_cr, 'algo_name', '') or (algo_seen[-1] if algo_seen else '')
            lessons.append(
                f"Winning algorithm was '{winning_algo}' "
                f"(silhouette={getattr(best_cr,'silhouette',None)}, "
                f"k={getattr(best_cr,'n_leaf',None)})."
            )
            if singleton_iters:
                lessons.append(
                    f"Singleton-cluster merges fired in iter(s) {singleton_iters} — "
                    f"watch for tiny clusters when n is small or k is high."
                )
            if relax_steps:
                lessons.append(
                    f"Silhouette target was relaxed {relax_steps} time(s) before "
                    f"this dataset converged — expect <0.5 silhouette as normal."
                )
            if vif_changes:
                lessons.append(
                    f"VIF threshold was tuned across iterations to {vif_changes[-1]} — "
                    f"useful starting point for similar data."
                )
            if getattr(best_clf, 'cv_f1_macro', None) is not None:
                worst_three = sorted(
                    (best_clf.per_class_f1 or {}).items(), key=lambda x: x[1]
                )[:3]
                if worst_three:
                    lessons.append(
                        "Hardest-to-predict personas in the winning run: "
                        + ", ".join(f"{n}({s:.2f})" for n, s in worst_three)
                    )

            winning_strategy = {
                'iteration': getattr(best_cr, 'iteration', None),
                'total_iterations': len(state.clustering_history or []),
                'algorithm': winning_algo,
                'k': getattr(best_cr, 'n_leaf', None),
                'vif_threshold': tp.get('vif_threshold'),
                'min_silhouette': tp.get('min_silhouette'),
                'min_cluster_size': int(self.config.get('min_cluster_size', 5)),
                'n_features_kept': len(features_used),
                'selected_features': features_used[:80],   # cap to keep file small
                'k_scores': {str(k): v for k, v in (getattr(best_cr, 'k_scores', {}) or {}).items()},
                'feature_focus': tp.get('feature_focus', ''),
            }
            outcome = {
                'status': status,
                'silhouette': getattr(best_cr, 'silhouette', None),
                'cv_f1_macro': getattr(best_clf, 'cv_f1_macro', None),
                'cv_accuracy': getattr(best_clf, 'cv_accuracy', None),
                'n_leaf_clusters': getattr(best_cr, 'n_leaf', None),
                'silhouette_target_at_finish': (
                    state.silhouette_target_override
                    if state.silhouette_target_override is not None
                    else float(self.config.get('silhouette_target', 0.5))
                ),
            }

            ds_name = pathlib.Path(raw_data_path).name if raw_data_path else 'unknown'
            cols = list(raw_df.columns) if raw_df is not None else []
            case_id = save_case(
                dataset_name=ds_name,
                dataset_path=raw_data_path,
                columns=cols,
                n_rows=len(raw_df) if raw_df is not None else 0,
                n_cols=len(cols),
                business_purpose=(ui.business_purpose if ui else ''),
                target_entity=(ui.target_entity if ui else ''),
                n_clusters_requested=(getattr(ui, 'n_clusters_requested', None) if ui else None),
                winning_strategy=winning_strategy,
                outcome=outcome,
                lessons=lessons,
            )
            print(
                f'  [Orchestrator] 🧠 Saved case to memory '
                f'(case_id={case_id[:8]}…, status={status}, '
                f'algo={winning_algo}, k={winning_strategy["k"]}, '
                f'F1={outcome["cv_f1_macro"]}).'
            )
            self.bus.emit(
                'case_memory_saved',
                case_id=case_id,
                status=status,
                algorithm=winning_algo,
                k=winning_strategy['k'],
                silhouette=outcome['silhouette'],
                cv_f1_macro=outcome['cv_f1_macro'],
            )
        except Exception as exc:  # noqa: BLE001
            print(f'  [Orchestrator] (case-memory save failed: {exc})')

    # ── Catalog helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _load_catalog(path: str) -> str:
        """Load a markdown catalog file, returning '' if not found."""
        p = pathlib.Path(path)
        return p.read_text(encoding='utf-8') if p.exists() else ''

    # ── Telemetry helpers ──────────────────────────────────────────────────────

    def _timing_dict(self) -> dict:
        total_s = time.perf_counter() - self._pipeline_start
        agent_order = ['UserInput', 'DatasetExaminer', 'FeatureEngineer', 'FeatureSelector', 'Clusterer', 'PersonaNamer', 'Classifier']
        agents = {}
        for name in agent_order:
            runs = self._timings.get(name, [])
            agents[name] = {
                'calls':      len(runs),
                'total_s':    round(sum(runs), 1),
                'per_call_s': [round(r, 1) for r in runs],
            }
        return {
            'total_s': round(total_s, 1),
            'agents':  agents,
        }

    def _llm_usage_dict(self) -> dict:
        by_agent: dict = defaultdict(lambda: {'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'time_s': 0.0, 'detail': []})
        for c in self._llm_calls:
            a = c['agent']
            by_agent[a]['calls']         += 1
            by_agent[a]['input_tokens']  += c['input_tokens']
            by_agent[a]['output_tokens'] += c['output_tokens']
            by_agent[a]['time_s']        += c['time_s']
            by_agent[a]['detail'].append({
                'input_tokens':  c['input_tokens'],
                'output_tokens': c['output_tokens'],
                'time_s':        c['time_s'],
            })
        total_in  = sum(c['input_tokens']  for c in self._llm_calls)
        total_out = sum(c['output_tokens'] for c in self._llm_calls)
        total_t   = sum(c['time_s']        for c in self._llm_calls)
        return {
            'by_agent':           dict(by_agent),
            'total_calls':        len(self._llm_calls),
            'total_input_tokens': total_in,
            'total_output_tokens': total_out,
            'total_llm_time_s':   round(total_t, 1),
            'raw_calls':          list(self._llm_calls),
        }

    def _print_timing_summary(self) -> None:
        td = self._timing_dict()
        total_s = td['total_s']

        print('\n' + '=' * 65)
        print('PIPELINE TIMING SUMMARY')
        print('=' * 65)
        print(f'{"Agent":<22}  {"Calls":>5}  {"Total":>8}  {"Per call":>10}  {"% of total":>10}')
        print('-' * 65)

        AGENT_LABELS = {
            'UserInput':       '(0) UserInput',
            'DatasetExaminer': '(1) DatasetExaminer',
            'FeatureEngineer': '(2) FeatureEngineer',
            'FeatureSelector': '(3) FeatureSelector',
            'Clusterer':       '(4) Clusterer',
            'PersonaNamer':    '(5) PersonaNamer',
            'Classifier':      '(6) Classifier',
        }
        for name, label in AGENT_LABELS.items():
            info = td['agents'].get(name, {'calls': 0, 'total_s': 0.0, 'per_call_s': []})
            calls = info['calls']
            tot   = info['total_s']
            avg   = round(tot / calls, 1) if calls else 0.0
            pct   = round(tot / total_s * 100, 1) if total_s > 0 else 0.0
            per_calls = ', '.join(f'{r}s' for r in info['per_call_s']) or '—'
            print(f'{label:<22}  {calls:>5}  {tot:>7.1f}s  {avg:>8.1f}s  {pct:>9.1f}%')

        print('-' * 65)
        print(f'{"TOTAL":<22}  {"":>5}  {total_s:>7.1f}s')
        print('=' * 65)
