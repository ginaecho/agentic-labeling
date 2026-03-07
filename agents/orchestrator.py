"""
Orchestrator

Contract: docs/agents/orchestrator.md. Consumes docs/skills/orchestrator_bus.md.

Coordinates the full multi-agent pipeline in a feedback loop:

  (0) UserInputAgent         — collect clustering intent
  (1) DatasetExaminerAgent   — profile dataset, suggest feature groups
  (2) FeatureEngineerAgent   — engineer customer features from raw CSV
                               (skipped if a pre-engineered parquet is given)
  (3) FeatureSelectionAgent  — PCA + AE + VIF → Claude picks feature subset
  (4) ClusteringAgent        — silhouette k-opt + auto algo + deepening loop
  (5) PersonaNamingAgent     — Claude names clusters; Clarity Gate validates
  (6) ClassifierAgent        — Random Forest CV validates cluster separability;
                               Claude routes back to (3) or (4) if F1 is low
  ↓
  Human Checkpoint           — user approves, requests re-run, or quits

All agents report to a shared OrchestratorBus. The orchestrator uses the
bus log when calling Claude for routing decisions.
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
    """Load a DataFrame from parquet or CSV, detected by file extension."""
    import pathlib
    suffix = pathlib.Path(path).suffix.lower()
    if suffix == '.parquet':
        return pd.read_parquet(path)
    elif suffix in ('.csv', '.tsv'):
        sep = '\t' if suffix == '.tsv' else ','
        return pd.read_csv(path, sep=sep, low_memory=False)
    else:
        # Try parquet first, fall back to CSV
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path, low_memory=False)


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
    print(bus.summary_for_claude(last_n=10))

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
         → if oversized cluster + Claude says reselect: loop to 2
      4. Persona naming (Clarity Gate)
         → if gate fails: loop to 3
      5. Classifier validation (CV F1 gate)
         → if poor: Claude routes to 2 or 3
      6. Human checkpoint
         → approve  : save + exit
         → recluster: loop to 3
         → reselect : loop to 2
         → quit     : exit without saving
    """

    def __init__(self, config: dict):
        self.config = config
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
        # so Claude understands what options are available when it makes routing decisions.
        # Planning agents (FeatureEngineer, FeatureSelector, PersonaNamer) get their
        # agent-specific role description so Claude adopts the right persona.
        _ROUTING_PURPOSES = frozenset(['route', 'diagnose', 'routing'])
        _skill_summary = self._skill_catalog[:3000] if self._skill_catalog else ''
        _agent_summary = self._agent_catalog[:2000] if self._agent_catalog else ''

        # ── Claude usage log ───────────────────────────────────────────────────
        self._claude_calls: list[dict] = []

        # ── Register LLM handler on the bus ───────────────────────────────────
        # Agents call bus.ask(agent, purpose, prompt) when they need LLM help.
        # The Orchestrator intercepts, calls Claude, logs usage, returns the answer.
        # The system parameter carries the skill/agent context so every call
        # is aware of the pipeline's capability catalog (P8 — Modular Skills).
        def _llm_handler(agent: str, purpose: str, prompt: str, max_tokens: int) -> str:
            print(f"\n  [Orchestrator] ← {agent} requests LLM help: {purpose}")

            # Build system context: routing decisions get the full skill catalog;
            # all other calls get a brief agent-role description.
            purpose_lower = purpose.lower()
            is_routing = any(kw in purpose_lower for kw in _ROUTING_PURPOSES)
            if is_routing and _skill_summary:
                system_ctx = (
                    "You are an AI orchestrator for a multi-agent customer segmentation pipeline.\n"
                    "The following skills and agents are available to you when making decisions.\n\n"
                    f"{_skill_summary}\n\n"
                    "Use your knowledge of these skills when deciding how to route the pipeline."
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
            self._claude_calls.append({
                'agent':         agent,
                'purpose':       purpose,
                'input_tokens':  resp.usage.input_tokens,
                'output_tokens': resp.usage.output_tokens,
                'time_s':        elapsed,
            })
            print(
                f"  [Orchestrator] → {agent}: LLM answered "
                f"(in={resp.usage.input_tokens} out={resp.usage.output_tokens} {elapsed}s)"
            )
            return resp.content[0].text

        self.bus.set_llm_handler(_llm_handler)

        # ── Instantiate agents — they receive ONLY the bus, not the client ──────
        # Each agent uses its own ML/stats skills and calls bus.ask() when it
        # needs LLM reasoning. The Orchestrator remains the sole LLM gateway.
        self.input_agent            = UserInputAgent(self.bus)
        self.examiner_agent         = DatasetExaminerAgent(self.bus)
        self.feature_engineer_agent = FeatureEngineerAgent(self.bus)
        self.feature_agent          = FeatureSelectionAgent(self.bus)
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

        state = PipelineState(config=self.config)

        # ── Step 0: User Intent ────────────────────────────────────────────────
        if not skip_user_input:
            self._active_agent = 'UserInput'
            _t0 = time.perf_counter()
            state.user_intent = self.input_agent.run(iteration=0)
            self._timings['UserInput'].append(time.perf_counter() - _t0)
        elif user_intent:
            state.user_intent = user_intent

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
                target_entity="customers",
                business_purpose="understand spending behaviour",
                dataset_path=features_path,
            ),
            df=raw_df,
            iteration=0,
        )
        self._timings['DatasetExaminer'].append(time.perf_counter() - _t0)
        state.dataset_profile = dataset_profile

        if dataset_profile is None:
            print('\n[Orchestrator] DatasetExaminer BLOCKED — cannot proceed.')
            return {'status': 'blocked', 'personas': None, 'run_history': run_history}

        run_history.append({
            'iteration': 0,
            'stage': 'dataset_examination',
            'n_rows': dataset_profile.n_rows,
            'suggested_groups': dataset_profile.suggested_feature_groups,
            'algo_hint': dataset_profile.algo_hint,
        })

        # ── Step 2: Feature Engineering (only when a raw CSV was provided) ─────
        # When the user gave a .csv path, the FeatureEngineerAgent turns the
        # transaction-level data into a customer-level feature matrix and saves
        # it to data/processed/. Downstream agents then use that parquet.
        if need_feature_engineering:
            print('\n[Orchestrator] Launching FeatureEngineerAgent on raw transaction data...')
            _t0 = time.perf_counter()
            self._active_agent = 'FeatureEngineer'
            try:
                features_df, fe_result = self.feature_engineer_agent.run(
                    raw_df=full_raw_df,
                    user_intent=state.user_intent or UserIntent(
                        target_entity='customers',
                        business_purpose='understand spending behaviour to personalise offers',
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
                    'n_customers': fe_result.n_customers,
                    'n_features': fe_result.n_features,
                    'groups_built': fe_result.groups_built,
                    'elapsed_s': round(self._timings['FeatureEngineer'][-1], 1),
                })
                print(
                    f'  [Orchestrator] Feature engineering done: '
                    f'{fe_result.n_features} features × {fe_result.n_customers} customers'
                )
            except RuntimeError as e:
                print(f'\n[Orchestrator] FeatureEngineer BLOCKED: {e}')
                return {'status': 'blocked', 'personas': None, 'run_history': run_history}

        # ── Main pipeline loop ─────────────────────────────────────────────────
        while state.total_iterations < max_total_iterations:
            state.total_iterations += 1
            iteration = state.total_iterations
            print(f'\n{"─"*65}')
            print(f'ITERATION {iteration} / {max_total_iterations}')
            print(f'{"─"*65}')

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
            )
            self._timings['Clusterer'].append(time.perf_counter() - _t0)
            state.clustering_history.append(cr)

            # Track best silhouette regardless of whether clustering "passed"
            state.update_best_silhouette(cr, state.selected_features)

            if cr.action == 'reselect_features':
                print(f'\n[Orchestrator] Clustering → reselect features: {cr.reasoning}')
                # Ask Claude to tune parameters before the next iteration
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

            # ── (4) Persona Naming ─────────────────────────────────────────────
            _t0 = time.perf_counter()
            self._active_agent = 'PersonaNamer'
            nr = self.naming_agent.run(
                profiles=cr.profiles,
                lineage=cr.lineage,
                tone=self.config.get('persona_tone', 'easy'),
                feedback=state.naming_feedback,
                iteration=iteration,
            )
            self._timings['PersonaNamer'].append(time.perf_counter() - _t0)
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
            })

            if not nr.passed:
                state.cluster_feedback = f'Clarity Gate failed: {"; ".join(nr.issues)}'
                state.naming_feedback = ''
                print(f'\n[Orchestrator] Clarity Gate failed → re-clustering.')
                # Tune params — clarity failures usually mean clusters are too similar;
                # Claude may suggest reducing k, switching algorithm, or refocusing features.
                new_params = self._ask_parameter_tuning(iteration, run_history, state)
                state.tuning_params.update(new_params)
                continue

            # ── (5) Classifier Validation ──────────────────────────────────────
            _t0 = time.perf_counter()
            self._active_agent = 'Classifier'
            clf = self.classifier_agent.run(
                features_df=features_df,
                cluster_labels=cr.cluster_labels,
                personas=nr.personas,
                history=state.classifier_history,
                feedback=state.classifier_feedback,
                iteration=iteration,
            )
            self._timings['Classifier'].append(time.perf_counter() - _t0)
            state.classifier_history.append(clf)

            run_history.append({
                'iteration': iteration,
                'stage': 'classifier',
                'action': clf.action,
                'cv_accuracy': clf.cv_accuracy,
                'cv_f1_macro': clf.cv_f1_macro,
                'elapsed_s': round(self._timings['Classifier'][-1], 1),
                'reasoning': clf.reasoning,
            })

            if clf.action == 'reselect_features':
                print(f'\n[Orchestrator] Classifier → reselect features: {clf.reasoning}')
                new_params = self._ask_parameter_tuning(iteration, run_history, state)
                state.tuning_params.update(new_params)
                state.request_feature_reselection(clf.reasoning)
                state.classifier_feedback = clf.reasoning
                continue

            elif clf.action == 'recluster':
                print(f'\n[Orchestrator] Classifier → re-cluster: {clf.reasoning}')
                new_params = self._ask_parameter_tuning(iteration, run_history, state)
                state.tuning_params.update(new_params)
                state.cluster_feedback = f'Classifier CV F1={clf.cv_f1_macro:.3f} too low: {clf.reasoning}'
                state.classifier_feedback = clf.reasoning
                continue

            # clf.action == 'proceed' — track best result
            state.update_best(nr, cr, clf)
            # Also update silhouette-based tracker (already called above, but refresh with full profiles)
            state.update_best_silhouette(cr, state.selected_features)

            # ── (6) Human Checkpoint ───────────────────────────────────────────
            decision = human_checkpoint(nr.personas, cr, clf, self.bus)

            run_history.append({
                'iteration': iteration,
                'stage': 'human_checkpoint',
                'decision': decision.action,
                'feedback': decision.feedback,
            })

            if decision.action == 'approve':
                print('\n[Orchestrator] Approved! Saving outputs...')
                save_outputs(cr, nr, clf, self.bus)
                self._print_timing_summary()
                return {
                    'status': 'success',
                    'personas': nr.personas,
                    'classifier': {
                        'cv_accuracy': clf.cv_accuracy,
                        'cv_f1_macro': clf.cv_f1_macro,
                        'cv_f1_weighted': clf.cv_f1_weighted,
                        'per_class_f1': clf.per_class_f1,
                    },
                    'run_history': run_history,
                    'timing': self._timing_dict(),
                    'claude_usage': self._claude_usage_dict(),
                }

            elif decision.action == 'quit':
                print('\n[Orchestrator] Quit without saving.')
                return {
                    'status': 'quit',
                    'personas': None,
                    'run_history': run_history,
                }

            elif decision.action == 'recluster':
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
            return {
                'status': 'max_iterations_reached',
                'personas': best_personas,
                'run_history': run_history,
                'timing': self._timing_dict(),
                'claude_usage': self._claude_usage_dict(),
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
            )

            print('  Saving best-effort result...')
            save_outputs(best_cr, best_nr, best_clf, self.bus)

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
                'claude_usage': self._claude_usage_dict(),
            }

        # ── Path C: no usable result at all ────────────────────────────────────
        print('\n[Orchestrator] No usable clustering result found after all iterations.')
        return {
            'status': 'max_iterations_reached',
            'personas': None,
            'run_history': run_history,
            'timing': self._timing_dict(),
            'claude_usage': self._claude_usage_dict(),
        }

    # ── Dynamic parameter tuning ───────────────────────────────────────────────

    def _ask_parameter_tuning(
        self,
        iteration: int,
        run_history: list[dict],
        state,
    ) -> dict:
        """
        After a failed iteration, ask Claude to suggest improved pipeline parameters.

        Claude sees the history of what happened (silhouette scores, VIF removals,
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
                f1 = h.get('cv_f1_macro', 0)
                history_lines.append(
                    f"  Iter {it} Classifier: f1_macro={f1:.3f}, action={h.get('action','?')}"
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

        cur = state.tuning_params
        prompt = f"""You are orchestrating a customer-clustering pipeline. Iteration {iteration} just failed.

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

Tuning guidelines:
- If VIF gate removes >60 features or hits max_iterations: raise vif_threshold (try 12–18)
- If silhouette consistently <0.10 with hierarchical: switch to kmeans
- If silhouette is low across all k values: narrow k_range to [3,4,5,6] or [4,5,6,7]
- If many features selected but silhouette stays low: add feature_focus to guide selector
- Lower min_silhouette only if data genuinely resists clustering (e.g. keep at 0.03)
- Do NOT lower min_silhouette below 0.02

Return ONLY a valid JSON object — no markdown fences, no extra text:
{{
  "vif_threshold": <float 5–25>,
  "k_range": [<int>, ...],
  "algorithm": "kmeans" | "hierarchical" | null,
  "min_silhouette": <float 0.02–0.12>,
  "feature_focus": "<short hint for FeatureSelector, or empty string>",
  "reasoning": "<1-2 sentences explaining the change>"
}}"""

        print(f'\n  [Orchestrator] Asking Claude to tune parameters for iteration {iteration + 1}...')
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
        if params.get('algorithm') in ('kmeans', 'hierarchical', None):
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

    def _claude_usage_dict(self) -> dict:
        by_agent: dict = defaultdict(lambda: {'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'time_s': 0.0, 'detail': []})
        for c in self._claude_calls:
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
        total_in  = sum(c['input_tokens']  for c in self._claude_calls)
        total_out = sum(c['output_tokens'] for c in self._claude_calls)
        total_t   = sum(c['time_s']        for c in self._claude_calls)
        return {
            'by_agent':            dict(by_agent),
            'total_calls':         len(self._claude_calls),
            'total_input_tokens':  total_in,
            'total_output_tokens': total_out,
            'total_claude_time_s': round(total_t, 1),
            'raw_calls':           list(self._claude_calls),
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
