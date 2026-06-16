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
import numpy as np
import pandas as pd

from agents.state import HumanDecision, PipelineState
from agents.user_input import UserInputAgent, UserIntent
from agents.dataset_examiner import DatasetExaminerAgent
from agents.feature_engineer import FeatureEngineerAgent, FeatureEngineeringResult
from agents.text_preparer import TextPreparerAgent
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


PENDING_HUMAN_CHECKPOINT_PATH = pathlib.Path('outputs/pending_human_checkpoint.json')
HUMAN_CHECKPOINT_TIMEOUT_S = 600  # 10 min — UI modal gives the user plenty of time


def _display_checkpoint_summary(personas: dict, cluster_result, classifier_result,
                                bus: OrchestratorBus) -> dict:
    """Print persona + classifier summary to stdout and return a structured
    payload the UI modal can render. Pure side-effect printing + payload —
    no input, no blocking.

    Returned dict shape (also emitted via SSE `awaiting_human_checkpoint`):
        {
          "silhouette": float, "n_leaf": int, "algorithm": str,
          "cv_accuracy": float|None, "cv_f1_macro": float|None,
          "cv_f1_weighted": float|None,
          "personas": [{"cid", "name", "tagline", "confidence", "cv_f1"}, ...],
          "worst_personas": [(name, f1), ...],
          "agent_log_summary": str,
        }
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
    # Classifier can be None when it crashed (e.g. sklearn version mismatch on
    # logistic_regression). Surface that clearly instead of attribute-erroring.
    if classifier_result is not None:
        print(f'CV accuracy        : {classifier_result.cv_accuracy:.4f}')
        print(f'CV F1 (macro)      : {classifier_result.cv_f1_macro:.4f}')
        print(f'CV F1 (weighted)   : {classifier_result.cv_f1_weighted:.4f}')
    else:
        print('CV metrics         : (classifier failed this iteration — no F1 available)')
    print()

    # Persona table
    print(f'{"Cluster":<10} {"Conf":>4}  {"CV-F1":>6}  {"Persona Name":<45}  Tagline')
    print('-' * 115)
    _per_class = (classifier_result.per_class_f1 if classifier_result is not None else {}) or {}
    persona_rows = []
    for cid, p in personas.items():
        name    = p.get('name', '?')
        conf    = p.get('confidence', '?')
        tagline = p.get('tagline', '')
        cv_f1   = _per_class.get(name, None)
        cv_f1_str = f'{cv_f1:.3f}' if cv_f1 is not None else '  n/a'
        print(f'  C{cid:<7}  {conf:>4}  {cv_f1_str:>6}  {name:<45}  {tagline}')
        persona_rows.append({
            'cid': str(cid), 'name': name, 'tagline': tagline,
            'confidence': conf,
            'cv_f1': float(cv_f1) if isinstance(cv_f1, (int, float)) else None,
        })

    worst = []
    if _per_class:
        worst = sorted(_per_class.items(), key=lambda x: x[1])[:3]
        print()
        print('Hardest-to-predict personas (CV F1):')
        for name, score in worst:
            bar = '█' * int(score * 20)
            print(f'  {name:<45}  {score:.3f}  {bar}')

    print()
    print('Pipeline Agent Log (recent):')
    log_summary = bus.summary_for_llm(last_n=10)
    print(log_summary)

    return {
        'silhouette': float(cluster_result.silhouette) if cluster_result.silhouette is not None else None,
        'n_leaf': cluster_result.n_leaf,
        'algorithm': cluster_result.algo_name,
        'cv_accuracy': float(classifier_result.cv_accuracy) if classifier_result is not None else None,
        'cv_f1_macro': float(classifier_result.cv_f1_macro) if classifier_result is not None else None,
        'cv_f1_weighted': float(classifier_result.cv_f1_weighted) if classifier_result is not None else None,
        'personas': persona_rows,
        'worst_personas': [{'name': n, 'cv_f1': float(s)} for n, s in worst],
        'agent_log_summary': log_summary[:4000],
    }


def _collect_human_decision(summary: dict, bus: OrchestratorBus,
                            timeout_s: float = HUMAN_CHECKPOINT_TIMEOUT_S) -> HumanDecision:
    """Wait for the user's decision via the UI modal, falling back to terminal.

    Flow:
      1. Emit `awaiting_human_checkpoint` SSE event carrying the summary.
      2. Poll outputs/pending_human_checkpoint.json for up to timeout_s seconds.
      3. If the file appears, parse {action, feedback} and return.
      4. If polling times out OR stdin has terminal input first, fall through
         to the existing terminal prompt so this still works headless.

    The polling loop checks stdin every tick (non-blocking via select) so a
    user typing 1/2/3/4 in the terminal also unblocks the run.
    """
    import select
    import sys as _sys

    # Defensive cleanup: stale file from a prior checkpoint must not auto-resolve us.
    try:
        if PENDING_HUMAN_CHECKPOINT_PATH.exists():
            PENDING_HUMAN_CHECKPOINT_PATH.unlink(missing_ok=True)
    except OSError:
        pass

    try:
        bus.emit('awaiting_human_checkpoint', timeout_s=timeout_s, **summary)
    except Exception:  # noqa: BLE001
        pass  # bus problems shouldn't block the terminal fallback

    print('\nOptions:')
    print('  [1] Approve — save results and finish')
    print('  [2] Re-cluster — try different clustering parameters')
    print('  [3] Re-select features — go back to feature selection')
    print('  [4] Quit without saving')
    print('\n  (or click an option in the browser modal — pipeline is paused)')
    print('Your choice [1/2/3/4]: ', end='', flush=True)

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        # 1) UI modal wrote a decision file?
        if PENDING_HUMAN_CHECKPOINT_PATH.exists():
            try:
                payload = json.loads(PENDING_HUMAN_CHECKPOINT_PATH.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                payload = {}
            try:
                PENDING_HUMAN_CHECKPOINT_PATH.unlink(missing_ok=True)
            except OSError:
                pass
            action = str(payload.get('action') or 'approve').lower()
            feedback = str(payload.get('feedback') or '').strip()
            if action not in ('approve', 'recluster', 'reselect_features', 'quit'):
                action = 'approve'
            try:
                bus.emit('human_checkpoint_resolved', source='ui',
                         action=action, feedback=feedback)
            except Exception:  # noqa: BLE001
                pass
            print(f'\n  [UI] {action!r} received from browser modal.')
            return HumanDecision(action=action, feedback=feedback)
        # 2) Stdin readable? (terminal fallback)
        try:
            ready, _, _ = select.select([_sys.stdin], [], [], 0.4)
        except (ValueError, OSError):
            ready = []
        if ready:
            try:
                choice = _sys.stdin.readline().strip()
            except (EOFError, KeyboardInterrupt):
                print('\nNo input received — defaulting to Approve.')
                return HumanDecision(action='approve')
            return _handle_terminal_choice(choice)
    # Timeout — default to approve so unattended runs still finish.
    try:
        bus.emit('human_checkpoint_resolved', source='timeout', action='approve')
    except Exception:  # noqa: BLE001
        pass
    print(f'\n  No decision received within {int(timeout_s)}s — defaulting to Approve.')
    return HumanDecision(action='approve')


def _handle_terminal_choice(choice: str) -> HumanDecision:
    """Map a typed 1/2/3/4 to a HumanDecision, prompting for a reason when needed."""
    choice = (choice or '').strip()
    if choice == '1':
        return HumanDecision(action='approve')
    if choice == '2':
        try:
            reason = input('  Reason / feedback for re-clustering (or Enter to skip): ').strip()
        except (EOFError, KeyboardInterrupt):
            reason = ''
        return HumanDecision(action='recluster', feedback=reason)
    if choice == '3':
        try:
            reason = input('  Reason / feedback for feature re-selection (or Enter to skip): ').strip()
        except (EOFError, KeyboardInterrupt):
            reason = ''
        return HumanDecision(action='reselect_features', feedback=reason)
    if choice == '4':
        return HumanDecision(action='quit')
    # Anything else: treat as approve to avoid spinning further on bad input.
    print(f'  Unrecognised choice {choice!r} — defaulting to Approve.')
    return HumanDecision(action='approve')


def human_checkpoint(personas: dict, cluster_result, classifier_result, bus: OrchestratorBus) -> HumanDecision:
    """Display the checkpoint and collect the user's decision (UI or terminal).

    Returns HumanDecision with action in:
      'approve' | 'recluster' | 'reselect_features' | 'quit'

    Composed of two pure halves so run_pipeline.py's `_auto_approve` wrapper
    can call only `_display_checkpoint_summary` and skip the wait.
    """
    summary = _display_checkpoint_summary(personas, cluster_result, classifier_result, bus)
    return _collect_human_decision(summary, bus)


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
        self.text_preparer_agent    = TextPreparerAgent(self.bus)
        self.feature_agent          = FeatureSelectionAgent(
            self.bus,
            ae_bottleneck_cap=config.get('ae_bottleneck_cap', 32),
            ae_max_iter=config.get('ae_max_iter', 200),
            max_features_for_vif=config.get('max_features_for_vif', 150),
        )
        self.cluster_agent          = ClusteringAgent(config, self.bus)
        self.naming_agent           = PersonaNamingAgent(self.bus)
        self.classifier_agent       = ClassifierAgent(self.bus)

        # Telemetry
        self._timings: dict[str, list[float]] = defaultdict(list)
        self._pipeline_start: float = 0.0

    def _sanitize_loaded_df(self, df, *, source_label: str, user_intent=None):
        """Drop low-value columns from a freshly-loaded dataset before any agent
        touches it.

        Wide, sparse exports (e.g. a 150-column feature CSV where 60+ columns are
        ~99% null and 50+ are constant) otherwise stall the pipeline: NaNs crash
        StandardScaler/PCA in FeatureSelector, and constant/rank-deficient columns
        slow the VIF gate. Pruning them here is the single, visible place that
        guards every downstream stage.

        Controlled by config['data_cleaning']:
          enabled       (bool, default True)
          max_null_frac (float, default 0.5)  — drop columns more than this empty
          drop_constant (bool, default True)  — drop zero-variance columns

        Imputation is intentionally NOT done here: the raw-CSV path feeds
        FeatureEngineer (which aggregates raw events, so a median-filled raw
        measurement would distort sums/means). FeatureSelector imputes its own
        NaNs just before the math that needs it.
        """
        from skills.data_cleaner import drop_low_value_columns

        dc_cfg = dict(self.config.get('data_cleaning') or {})
        if dc_cfg.get('enabled', True) is False:
            return df

        protect = ['_row_id']
        text_col = getattr(user_intent, 'text_column', None) if user_intent else None
        if text_col:
            protect.append(text_col)

        cleaned, report = drop_low_value_columns(
            df,
            max_null_frac=float(dc_cfg.get('max_null_frac', 0.5)),
            drop_constant=bool(dc_cfg.get('drop_constant', True)),
            protect_cols=protect,
            verbose=False,
        )
        if report['n_dropped'] > 0:
            print(
                f"  [Orchestrator] Data cleaning ({source_label}): "
                f"{report['n_cols_before']} → {report['n_cols_after']} columns "
                f"(dropped {len(report['dropped_all_null'])} all-null, "
                f"{len(report['dropped_high_null'])} >{report['max_null_frac']:.0%}-null, "
                f"{len(report['dropped_constant'])} constant, "
                f"{len(report['dropped_duplicate'])} duplicate)."
            )
            try:
                self.bus.emit('data_cleaning', source=source_label, **report)
            except Exception:  # noqa: BLE001
                pass
        return cleaned

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

        if state.user_intent and getattr(state.user_intent, 'max_total_iterations', None):
            old_max = int(max_total_iterations)
            new_max = int(state.user_intent.max_total_iterations)
            if new_max > 0 and new_max != old_max:
                max_total_iterations = new_max
                print(
                    f'  [Orchestrator] max_total_iterations overridden by user intent: '
                    f'{old_max} → {new_max}.'
                )
                self.bus.emit(
                    'config_override_from_intent',
                    field='max_total_iterations',
                    old=old_max,
                    new=new_max,
                    source='user_intent',
                )

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

        # ── Inject modality from config when intent leaves it on 'auto' ────────
        # Lets `modality: text` / `text_column:` in config.yaml (or --modality)
        # drive routing without the interactive UserInputAgent needing to ask.
        if state.user_intent is not None:
            if (getattr(state.user_intent, 'modality', 'auto') or 'auto') == 'auto':
                state.user_intent.modality = str(
                    self.config.get('modality', 'auto') or 'auto'
                ).lower()
            if not getattr(state.user_intent, 'text_column', None) and self.config.get('text_column'):
                state.user_intent.text_column = self.config.get('text_column')

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
            full_raw_df = self._sanitize_loaded_df(
                full_raw_df, source_label='raw_csv', user_intent=state.user_intent,
            )
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
            features_df = self._sanitize_loaded_df(
                features_df, source_label='pre_engineered', user_intent=state.user_intent,
            )
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
            n_rows_source=len(full_raw_df) if full_raw_df is not None else None,
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

        # ── Control-gate tuning (max_cluster_size_pct, sub_n_clusters, max_depth) ─
        # The Decision Maker (bypass) or the user (interactive) sets these based
        # on dataset stats rather than hard-coding 0.40 / 3 / 2.
        _tuned_gates = self._tune_control_gates(
            raw_df=raw_df if raw_df is not None else features_df,
            dataset_profile=dataset_profile,
            user_intent=state.user_intent,
        )
        self.config['max_cluster_size_pct'] = _tuned_gates['max_cluster_size_pct']
        self.config['sub_n_clusters'] = _tuned_gates['sub_n_clusters']
        self.config['max_depth'] = _tuned_gates['max_depth']

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
                    prior_dataset=recall.case.get('dataset', {}).get('name'),
                    prior_purpose=recall.case.get('intent', {}).get('business_purpose'),
                    prior_silhouette=recall.case.get('outcome', {}).get('silhouette'),
                    prior_cv_f1_macro=recall.case.get('outcome', {}).get('cv_f1_macro'),
                    prior_algorithm=strat.get('algorithm'),
                    prior_k=strat.get('k'),
                    prior_vif_threshold=strat.get('vif_threshold'),
                    prior_min_silhouette=strat.get('min_silhouette'),
                    prior_n_features_kept=strat.get('n_features_kept'),
                )

                # Ask the user how to use the recall (interactive only).
                # Three options:
                #   reuse  — seed iteration 1's tuning_params with the prior
                #            recipe AND drop the LLM hint block so it doesn't
                #            second-guess the user's chosen recipe.
                #   modify — keep defaults but inject the recall as a HINT in
                #            the failure-tuning prompt (the historical behaviour).
                #   ignore — clear the recall entirely; pretend we never matched.
                #
                # Bypass mode auto-picks 'modify' so headless runs behave as
                # they did before this gate existed.
                from skills.orchestrator_bus import read_pipeline_mode
                _mode = read_pipeline_mode()
                if _mode == 'interactive':
                    _decision = self._wait_for_case_recall_decision(recall, state)
                else:
                    _decision = 'modify'
                    print(f'  [Orchestrator] BYPASS — auto-applying case recall as a hint (decision=modify).')

                self.bus.emit('case_memory_decision', decision=_decision,
                              match_type=recall.match_type)

                if _decision == 'reuse':
                    # Seed iteration 1's tuning params with the prior recipe.
                    # Each value is only applied when present in the case so we
                    # don't clobber defaults with None.
                    seeded: dict = {}
                    if strat.get('algorithm'):
                        state.tuning_params['algorithm'] = strat.get('algorithm')
                        seeded['algorithm'] = strat.get('algorithm')
                    if strat.get('k') is not None:
                        try:
                            _k = int(strat.get('k'))
                            state.tuning_params['k_range'] = [_k]
                            seeded['k_range'] = [_k]
                        except (TypeError, ValueError):
                            pass
                    if strat.get('vif_threshold') is not None:
                        try:
                            state.tuning_params['vif_threshold'] = float(strat.get('vif_threshold'))
                            seeded['vif_threshold'] = float(strat.get('vif_threshold'))
                        except (TypeError, ValueError):
                            pass
                    if strat.get('min_silhouette') is not None:
                        try:
                            state.tuning_params['min_silhouette'] = float(strat.get('min_silhouette'))
                            seeded['min_silhouette'] = float(strat.get('min_silhouette'))
                        except (TypeError, ValueError):
                            pass
                    if strat.get('feature_focus'):
                        state.tuning_params['feature_focus'] = strat.get('feature_focus')
                        seeded['feature_focus'] = strat.get('feature_focus')
                    # Drop the recall so the failure-tuning LLM doesn't get a
                    # contradictory hint block AND so case_recall isn't
                    # re-applied later.
                    state.case_recall = None
                    print(f'  [Orchestrator] 🧠 REUSE — seeded iteration 1 with prior recipe: {seeded}')
                elif _decision == 'ignore':
                    state.case_recall = None
                    print('  [Orchestrator] 🧠 IGNORE — discarding recall; fresh run.')
                else:
                    # modify — keep recall on state so the failure-tuning prompt
                    # gets the hint block. No tuning_params seeded.
                    print('  [Orchestrator] 🧠 MODIFY — recall stays as a hint for failure-tuning only.')
            else:
                print('\n[Orchestrator] 🧠 Case-memory: no matching prior case found.')
        except Exception as _exc:  # noqa: BLE001
            print(f'  [Orchestrator] (case-memory lookup failed: {_exc})')

        # ── Step 2a: TEXT modality — TextPreparer replaces FeatureEngineer ─────
        # When the dataset is text-dominant, vectorize the documents into an
        # embedding matrix. That matrix is a plain numeric feature table, so the
        # SAME FeatureSelector → Clusterer → PersonaNamer → Classifier loop and
        # its control gates run unchanged on it.
        modality = getattr(dataset_profile, 'modality', 'tabular')
        if modality == 'text':
            print('\n[Orchestrator] Text modality — launching TextPreparerAgent...')
            self._active_agent = 'TextPreparer'
            _t0 = time.perf_counter()
            source_df = full_raw_df if full_raw_df is not None else features_df
            _tv = str(self.config.get('text_vectorizer', 'auto') or 'auto').lower()
            try:
                features_df, tp_result = self.text_preparer_agent.run(
                    raw_df=source_df,
                    user_intent=state.user_intent or UserIntent(
                        target_entity='documents',
                        business_purpose='discover distinct themes in the documents',
                        dataset_path=raw_data_path or features_path,
                    ),
                    dataset_profile=dataset_profile,
                    output_path='data/processed/text_embeddings.parquet',
                    iteration=0,
                    method=_tv if _tv in ('tfidf_svd', 'transformer') else None,
                )
            except RuntimeError as e:
                print(f'\n[Orchestrator] TextPreparer BLOCKED: {e}')
                return {'status': 'blocked', 'personas': None, 'run_history': run_history}
            self._timings['TextPreparer'].append(time.perf_counter() - _t0)
            # Stash text artifacts so Clusterer/PersonaNamer/FeatureSelector can
            # access raw docs + TF-IDF vocab for c-TF-IDF distinctive terms +
            # representative documents per cluster. `modality` mirrors the
            # detected profile so each downstream agent can take the text branch.
            state.modality = 'text'
            state.text_artifacts = {
                'method': tp_result.method,
                'text_column': tp_result.text_column,
                'raw_docs': tp_result.raw_docs,
                'feature_names': tp_result.artifacts.get('feature_names', []),
                'tfidf': tp_result.artifacts.get('tfidf'),
                'tfidf_matrix': tp_result.artifacts.get('tfidf_matrix'),
                'doc_index': list(features_df.index),
            }
            state.text_prep = tp_result  # type: ignore[attr-defined]
            need_feature_engineering = False

            # Relax control gates for text: topic clusters are fuzzier than
            # tabular RFM clusters, so the same thresholds would loop needlessly.
            # Values come from config.yaml:text_overrides, with hardcoded
            # fallbacks so existing tests that don't load the YAML still pass.
            txt_ov = dict(self.config.get('text_overrides') or {})
            min_sil_text = float(txt_ov.get('min_silhouette') or 0.01)
            sil_target_text = float(txt_ov.get('silhouette_target') or 0.18)
            f1_text = float(txt_ov.get('classifier_f1_threshold') or 0.60)
            max_pct_text = float(txt_ov.get('max_cluster_size_pct') or 0.60)
            state.tuning_params['min_silhouette'] = min_sil_text
            # Drive the orchestrator's dynamic silhouette target through the
            # override mechanism so _current_silhouette_target() picks it up
            # for *every* iteration (not just for the first relax cycle).
            state.silhouette_target_override = sil_target_text
            self.classifier_agent.F1_THRESHOLD = f1_text
            # Also widen max_cluster_size_pct so Clusterer doesn't over-split
            # natural topic groups (the tabular 0.40 cap was tuned for behavioral
            # features; text topics commonly occupy 40-55% of a small corpus).
            # Only widen, never tighten — respect a user-set explicit value.
            if float(self.config.get('max_cluster_size_pct', 0.40)) < max_pct_text:
                self.config['max_cluster_size_pct'] = max_pct_text
            print(
                f'  [Orchestrator] Text mode gates from config.text_overrides: '
                f'min_silhouette={min_sil_text}, silhouette_target={sil_target_text}, '
                f'classifier F1 threshold={f1_text}, '
                f'max_cluster_size_pct={self.config["max_cluster_size_pct"]}'
            )

            run_history.append({
                'iteration': 0,
                'stage': 'text_preparation',
                'n_docs': tp_result.n_docs,
                'n_dims': tp_result.n_dims,
                'method': tp_result.method,
                'text_column': tp_result.text_column,
                'elapsed_s': round(self._timings['TextPreparer'][-1], 1),
            })

        # ── Step 2: Feature Engineering (only when a raw CSV was provided) ─────
        # When the user gave a .csv path, the FeatureEngineerAgent turns the
        # event-level data into an entity-level feature matrix and saves
        # it to data/processed/. Downstream agents then use that parquet.
        if need_feature_engineering:
            # Resolve entity/timestamp/amount/category columns before engineering
            _resolved_cols = self._resolve_columns(
                raw_df=raw_df if raw_df is not None else full_raw_df,
                dataset_profile=dataset_profile,
                user_intent=state.user_intent,
            )
            state.resolved_columns = _resolved_cols

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
                    resolved_columns=_resolved_cols,
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
                self._save_pre_modelling_preview(
                    features_df, list(features_df.columns),
                    stage='feature_engineering', iteration=0,
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
            # SKIP for text modality: FeatureEngineer is a tabular RFM/aggregate
            # builder and has nothing useful to do with raw text columns — it
            # would hang on the heavy LLM call looking for "behavioral features"
            # in title/body strings. For text, the escalation path is instead
            # handled by _ask_parameter_tuning: it picks a different algorithm
            # (kmeans ↔ hierarchical ↔ nmf ↔ lda) or a different text_vectorizer
            # (tfidf_svd ↔ transformer). We just clear the flag and let the
            # main loop continue.
            _is_text = getattr(state, 'modality', 'tabular') == 'text'
            if state.needs_feature_engineering and _is_text:
                print(
                    f'\n[Orchestrator] ESCALATION (text mode) — '
                    f'{state.consecutive_silhouette_failures} silhouette failures. '
                    f'Skipping FeatureEngineer re-run (tabular-only); the Decision '
                    f'Maker will pick a different algorithm / text_vectorizer on '
                    f'this iteration instead.'
                )
                self.bus.emit(
                    'feature_re_engineering',
                    consecutive_failures=state.consecutive_silhouette_failures,
                    silhouette_target=_current_silhouette_target(),
                    modality='text',
                    skipped='FeatureEngineer is tabular-only; tuning algorithm instead',
                )
                # Force a fresh algorithm pick on this iteration by clearing
                # the cached one — _ask_parameter_tuning will be invoked later
                # in the loop via the silhouette-miss branch.
                state.tuning_params['algorithm'] = None
                state.tuning_params['feature_focus'] = (
                    f"Previous text clustering gave silhouette < target across "
                    f"{state.consecutive_silhouette_failures} iterations — try a "
                    f"fundamentally different algorithm (kmeans/hierarchical/nmf/lda) "
                    f"or text_vectorizer (tfidf_svd↔transformer)."
                )
                state.needs_feature_engineering = False
                state.consecutive_silhouette_failures = 0
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
                        resolved_columns=state.resolved_columns,
                    )
                    self._timings['FeatureEngineer'].append(time.perf_counter() - _t0)
                    state.needs_feature_engineering = False
                    state.consecutive_silhouette_failures = 0
                    state.needs_feature_selection = True   # force fresh selection
                    self._save_pre_modelling_preview(
                        features_df, list(features_df.columns),
                        stage='feature_engineering', iteration=iteration,
                    )
                except RuntimeError as e:
                    print(f'[Orchestrator] FeatureEngineer escalation failed: {e}')
                    state.needs_feature_engineering = False  # don't loop forever

            # Check for hard blocks from the bus
            if self.bus.has_hard_block():
                print('\n[Orchestrator] Hard block detected — triggering human checkpoint.')
                break

            # ── (1b) Cross-modal loopback: re-vectorise text if the failure-
            # tuning LLM picked a different embedding method last round. This is
            # the text-mode analog of FeatureEngineer re-engineering — the LLM
            # decided tfidf_svd was the wrong call and asked for transformer
            # (or vice-versa).
            if (getattr(state, 'modality', 'tabular') == 'text'
                    and state.tuning_params.get('text_vectorizer') is not None):
                _requested = state.tuning_params['text_vectorizer']
                _current = (state.text_artifacts or {}).get('method')
                if _requested != _current and _requested in ('tfidf_svd', 'transformer'):
                    print(f'\n[Orchestrator] Text re-vectorisation — switching '
                          f'{_current!r} → {_requested!r} (LLM-driven).')
                    self._active_agent = 'TextPreparer'
                    _t0 = time.perf_counter()
                    _source_df = full_raw_df if full_raw_df is not None else features_df
                    try:
                        features_df, tp_result = self.text_preparer_agent.run(
                            raw_df=_source_df,
                            user_intent=state.user_intent,
                            dataset_profile=dataset_profile,
                            output_path='data/processed/text_embeddings.parquet',
                            iteration=iteration,
                            method=_requested,
                            feedback=(f"Failure-tuning LLM asked to switch embedding "
                                      f"method to {_requested!r} after the previous "
                                      f"method produced low silhouette."),
                        )
                        self._timings['TextPreparer'].append(time.perf_counter() - _t0)
                        state.text_artifacts = {
                            'method': tp_result.method,
                            'text_column': tp_result.text_column,
                            'raw_docs': tp_result.raw_docs,
                            'feature_names': tp_result.artifacts.get('feature_names', []),
                            'tfidf': tp_result.artifacts.get('tfidf'),
                            'tfidf_matrix': tp_result.artifacts.get('tfidf_matrix'),
                            'doc_index': list(features_df.index),
                        }
                        state.text_prep = tp_result  # type: ignore[attr-defined]
                        state.needs_feature_selection = True  # re-select on the new matrix
                        run_history.append({
                            'iteration': iteration,
                            'stage': 'text_re_vectorisation',
                            'from_method': _current,
                            'to_method': tp_result.method,
                            'n_dims': tp_result.n_dims,
                            'elapsed_s': round(self._timings['TextPreparer'][-1], 1),
                        })
                    except RuntimeError as e:
                        print(f'  [Orchestrator] Re-vectorisation failed ({e}); '
                              f'keeping previous embeddings.')

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
                    modality=getattr(state, 'modality', 'tabular'),
                )
                self._timings['FeatureSelector'].append(time.perf_counter() - _t0)
                state.update_features(fs)
                self._save_pre_modelling_preview(
                    features_df, list(state.selected_features or []),
                    stage='feature_selection', iteration=iteration,
                )
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
            # Threshold-decision modal only pauses the pipeline when the user
            # has explicitly toggled the UI mode to 'interactive'. Otherwise
            # (default + CLI --bypass) the clusterer auto-applies the
            # recommended option and emits an event the UI surfaces as a
            # warning banner. This mirrors the existing _relax_silhouette_target
            # flow so the two pause paths behave consistently.
            from skills.orchestrator_bus import read_pipeline_mode
            _bypass_for_decisions = (
                read_pipeline_mode() != 'interactive'
                or bool(self.config.get('bypass_mode', False))
            )
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
                text_artifacts=(state.text_artifacts
                                if getattr(state, 'modality', 'tabular') == 'text'
                                else None),
                bypass=_bypass_for_decisions,
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
                # Pull candidate info from the clustering report if available
                _cand = cr.candidate_evidence if cr else {}
                _cand_best = _cand.get("best") if isinstance(_cand, dict) else None
                _cand_all = _cand.get("candidates", []) if isinstance(_cand, dict) else []
                _algos = sorted(set(c["algorithm"] for c in _cand_all))
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
                    candidate_best=_cand_best,
                    algorithms=_algos,
                    n_candidates=len(_cand_all),
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

    # ── Pre-modelling preview (Evidence tab: dataset after transformations) ───

    def _save_pre_modelling_preview(self, features_df, selected_cols,
                                     stage: str, iteration: int) -> None:
        """Snapshot the dataset state after FeatureEngineer / FeatureSelector.

        Writes outputs/pre_modelling_preview.json with head rows, basic column
        stats, and a stage label. The UI's "Pre-modelling dataset" card reads
        this file so the user can see what the dataset looks like AFTER each
        transformation step, without having to load the full parquet.
        """
        import numpy as np
        if features_df is None or len(features_df) == 0:
            return
        if selected_cols:
            cols = [c for c in selected_cols if c in features_df.columns]
        else:
            cols = list(features_df.columns)
        if not cols:
            return
        df = features_df[cols]
        n_rows, n_cols = df.shape

        head = df.head(8)
        rows = [[('' if v is None or (isinstance(v, float) and np.isnan(v))
                  else (round(float(v), 4) if isinstance(v, (int, float, np.floating, np.integer))
                        else str(v)))
                 for v in row] for row in head.itertuples(index=False, name=None)]

        col_stats = []
        for i, c in enumerate(cols[:200]):
            # Positional access: df[c] returns a DataFrame when column names duplicate.
            s = df.iloc[:, i]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            try:
                is_numeric = bool(pd.api.types.is_numeric_dtype(s))
            except (TypeError, AttributeError):
                is_numeric = False
            entry = {
                'name': str(c),
                'numeric': is_numeric,
                'missing_pct': round(float(s.isna().mean() * 100.0), 2),
            }
            if is_numeric:
                try:
                    sk = float(s.dropna().skew())
                    if not np.isnan(sk):
                        entry['skew'] = round(sk, 3)
                except Exception:
                    pass
            col_stats.append(entry)

        from datetime import datetime, timezone
        snapshot = {
            'stage': stage,
            'iteration': iteration,
            'ts': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'n_rows': int(n_rows),
            'n_cols': int(n_cols),
            'columns': [str(c) for c in cols],
            'rows': rows,
            'col_stats': col_stats,
        }

        # Append to a list of snapshots so the user can browse every
        # FeatureEngineer / FeatureSelector run in the Data & Evidence tab.
        # Replace any existing entry with the same (stage, iteration) key so
        # re-runs of the same step don't accumulate duplicates. Cap to the
        # most recent MAX_SNAPSHOTS entries to keep the file small.
        MAX_SNAPSHOTS = 20
        out_path = pathlib.Path('outputs/pre_modelling_preview.json')
        try:
            existing = json.loads(out_path.read_text(encoding='utf-8')) if out_path.exists() else []
        except (OSError, json.JSONDecodeError):
            existing = []
        # Back-compat: file previously held a single dict, not a list.
        if isinstance(existing, dict):
            existing = [existing]
        elif not isinstance(existing, list):
            existing = []
        existing = [s for s in existing
                    if not (s.get('stage') == stage and s.get('iteration') == iteration)]
        existing.append(snapshot)
        if len(existing) > MAX_SNAPSHOTS:
            existing = existing[-MAX_SNAPSHOTS:]
        payload = existing
        try:
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding='utf-8'
            )
            print(f'  [Orchestrator] Pre-modelling preview saved '
                  f'(stage={stage}, iter {iteration}, {n_rows}×{n_cols})')
        except OSError as e:
            print(f'  [Orchestrator] Pre-modelling preview save failed: {e}')

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
            # Also surface through the new threshold-decision banner so the
            # Evidence tab shows every bypass-applied relaxation in one place.
            try:
                self.bus.emit(
                    'threshold_decision_auto_applied',
                    mode='bypass',
                    chosen='relax',
                    decision_id=f'orch_relax_target_iter{state.total_iterations}',
                    agent='Orchestrator',
                    title='Silhouette target auto-relaxed',
                    summary=(
                        f"3 consecutive iterations missed silhouette target "
                        f"{current_target:.2f}. Bypass mode auto-dropped it to "
                        f"{new_target:.2f}."
                    ),
                    options=[
                        {'key': 'relax', 'label': f'Drop to {new_target:.2f}',
                         'description': 'Lower the bar; future iterations only need to clear the new target.'},
                        {'key': 'keep', 'label': f'Keep at {current_target:.2f}',
                         'description': 'Demand the original quality; may exhaust iterations.'},
                    ],
                    recommended='relax',
                    extra={'previous': current_target, 'new': new_target,
                           'iterations_failed': 3},
                )
            except Exception:  # noqa: BLE001
                pass

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

    def _wait_for_case_recall_decision(self, recall, state) -> str:
        """Block until the user decides what to do with a case-memory recall.

        Reads outputs/pending_case_recall.json (written by POST
        /api/case-recall-decision). Valid decisions: 'reuse' | 'modify' |
        'ignore'. Defaults to 'modify' if the user doesn't respond in time —
        same behaviour the system had before this gate existed, so a missed
        click never blocks a run forever.
        """
        import pathlib, time as _time
        pending = pathlib.Path('outputs/pending_case_recall.json')
        try:
            pending.unlink(missing_ok=True)
        except OSError:
            pass

        strat = recall.case.get('winning_strategy', {}) or {}
        outcome = recall.case.get('outcome', {}) or {}
        ds = recall.case.get('dataset', {}) or {}
        intent = recall.case.get('intent', {}) or {}

        self.bus.emit(
            'awaiting_case_recall_decision',
            match_type=recall.match_type,
            notes=recall.notes,
            prior_dataset=ds.get('name'),
            prior_purpose=intent.get('business_purpose'),
            prior_algorithm=strat.get('algorithm'),
            prior_k=strat.get('k'),
            prior_vif_threshold=strat.get('vif_threshold'),
            prior_min_silhouette=strat.get('min_silhouette'),
            prior_n_features_kept=strat.get('n_features_kept'),
            prior_silhouette=outcome.get('silhouette'),
            prior_cv_f1_macro=outcome.get('cv_f1_macro'),
            timeout_s=300,
        )
        print(f'\n  [INTERACTIVE MODE] Case memory matched a prior run '
              f'({recall.match_type}). Pipeline paused — waiting for you to '
              f'pick Reuse / Modify / Ignore in the UI banner.')

        valid = {'reuse', 'modify', 'ignore'}
        deadline = _time.time() + 300
        try:
            while _time.time() < deadline:
                if pending.exists():
                    try:
                        payload = json.loads(pending.read_text(encoding='utf-8'))
                        d = str(payload.get('decision') or '').strip().lower()
                        if d in valid:
                            try: pending.unlink(missing_ok=True)
                            except OSError: pass
                            return d
                    except (OSError, ValueError, TypeError, json.JSONDecodeError):
                        pass
                _time.sleep(0.6)
        except KeyboardInterrupt:
            pass
        print('  [INTERACTIVE MODE] Case-recall decision timed out — defaulting to MODIFY.')
        return 'modify'

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

    # ── Control-gate tuning (max_cluster_size_pct, sub_n_clusters, max_depth) ──

    def _tune_control_gates(
        self,
        raw_df: pd.DataFrame,
        dataset_profile,
        user_intent: UserIntent | None,
    ) -> dict:
        """Decide max_cluster_size_pct, sub_n_clusters, and max_depth.

        Bypass mode    → LLM decides from dataset stats (transparent print-out).
        Interactive mode → UI modal asks the user; falls back to defaults on timeout.

        Returns a dict with the three keys. The orchestrator should merge this
        into self.config before clustering.
        """
        import pathlib, time as _time

        # Gather lightweight dataset stats for the prompt / UI
        n_rows = len(raw_df)
        n_features = len(raw_df.columns)
        modality = getattr(dataset_profile, 'modality', 'tabular') if dataset_profile else 'tabular'
        purpose = (user_intent.business_purpose if user_intent else '') or ''
        target = (user_intent.target_entity if user_intent else 'entities') or 'entities'

        # Quick skewness read
        numeric = raw_df.select_dtypes(include=[np.number])
        mean_skew = 0.0
        if not numeric.empty:
            try:
                mean_skew = float(numeric.skew().abs().mean())
            except Exception:
                pass

        # Quick PCA — just first 2 components' explained variance
        pca_ev = []
        if not numeric.empty and len(numeric.columns) >= 2:
            try:
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                X_num = numeric.dropna().to_numpy(dtype=float)
                if len(X_num) > 2:
                    X_s = StandardScaler().fit_transform(X_num)
                    pca = PCA(n_components=min(2, X_s.shape[1]))
                    pca.fit(X_s)
                    pca_ev = [round(float(v), 4) for v in pca.explained_variance_ratio_]
            except Exception:
                pass

        # ── Data-driven defaults (ignore config.yaml — the Decision Maker decides) ─
        if n_rows < 1_000 or modality == 'text':
            _dd_max_pct = 0.55
            _dd_sub_k = 2
            _dd_depth = 1
        elif n_rows < 50_000:
            _dd_max_pct = 0.40
            _dd_sub_k = 3
            _dd_depth = 2
        else:
            _dd_max_pct = 0.30
            _dd_sub_k = 3
            _dd_depth = 2

        if mean_skew > 2.0:
            _dd_max_pct = max(0.25, _dd_max_pct - 0.10)
            _dd_depth = min(3, _dd_depth + 1)

        # Honor explicit user-intent overrides (e.g. "max cluster size 25%" in text)
        _user_max_pct = (
            float(user_intent.max_cluster_size_pct)
            if user_intent and user_intent.max_cluster_size_pct is not None
            else None
        )

        defaults = {
            'max_cluster_size_pct': _user_max_pct if _user_max_pct is not None else _dd_max_pct,
            'sub_n_clusters': _dd_sub_k,
            'max_depth': _dd_depth,
        }

        print(
            f"\n[Orchestrator] Control-gate data-driven defaults:\n"
            f"  max_cluster_size_pct = {defaults['max_cluster_size_pct']:.0%}\n"
            f"  sub_n_clusters       = {defaults['sub_n_clusters']}\n"
            f"  max_depth            = {defaults['max_depth']}"
            f"{'  (user override)' if _user_max_pct is not None else ''}"
        )

        from skills.orchestrator_bus import read_pipeline_mode
        mode = read_pipeline_mode()

        # ── BYPASS MODE: ask the LLM Decision Maker ─────────────────────────────
        if mode == 'bypass':
            prompt = f"""You are tuning three control-gate parameters for a clustering pipeline.

Dataset snapshot:
  target_entity      : {target}
  business_purpose   : {purpose}
  modality           : {modality}
  n_rows             : {n_rows:,}
  n_features         : {n_features}
  mean_abs_skewness  : {mean_skew:.2f}
  PCA EV (1st 2)     : {pca_ev if pca_ev else 'N/A'}

{_user_max_pct is not None and f"NOTE: user explicitly requested max_cluster_size_pct = {_user_max_pct:.0%} in their intent. Respect this exact value." or ""}

Data-driven starting point (from dataset stats — you may adjust):
  max_cluster_size_pct = {defaults['max_cluster_size_pct']:.0%}
  sub_n_clusters       = {defaults['sub_n_clusters']}
  max_depth            = {defaults['max_depth']}

Data-science guidelines:
  max_cluster_size_pct:
    - Small datasets (< 1k) or text topics  → 0.50–0.60 (tight splitting is risky)
    - Medium (1k–50k) tabular               → 0.35–0.45
    - Large (> 50k) tabular                 → 0.25–0.35
    - Highly skewed features                → 0.25–0.30 (natural groups are uneven)
  sub_n_clusters:
    - Small datasets                        → 2 (conservative)
    - Medium                                → 2–3
    - Large or many expected personas       → 3–4
  max_depth:
    - Small datasets                        → 1 (avoid over-splitting)
    - Medium                                → 2
    - Large or hierarchical purpose         → 2–3

Return ONLY a JSON object (no markdown, no prose):
{{
  "max_cluster_size_pct": <float 0.15–0.80>,
  "sub_n_clusters": <int 2–6>,
  "max_depth": <int 1–4>,
  "reasoning": "<1 sentence per parameter>"
}}"""
            try:
                raw = self.bus.ask(
                    agent='Orchestrator',
                    purpose='tune control gates from dataset stats (bypass mode)',
                    prompt=prompt,
                    max_tokens=256,
                    category='pipeline',
                ).strip()
                if '```' in raw:
                    for part in raw.split('```'):
                        p = part.strip()
                        if p.startswith('json'):
                            p = p[4:].strip()
                        if p.startswith('{'):
                            raw = p
                            break
                parsed = json.loads(raw)
                result = {
                    'max_cluster_size_pct': float(parsed.get('max_cluster_size_pct', defaults['max_cluster_size_pct'])),
                    'sub_n_clusters': int(parsed.get('sub_n_clusters', defaults['sub_n_clusters'])),
                    'max_depth': int(parsed.get('max_depth', defaults['max_depth'])),
                }
                # Clamp to sane ranges
                result['max_cluster_size_pct'] = max(0.15, min(0.80, result['max_cluster_size_pct']))
                result['sub_n_clusters'] = max(2, min(6, result['sub_n_clusters']))
                result['max_depth'] = max(1, min(4, result['max_depth']))

                print(
                    f"\n[Orchestrator] Control gates tuned by Decision Maker (bypass):\n"
                    f"  max_cluster_size_pct = {result['max_cluster_size_pct']:.0%}\n"
                    f"  sub_n_clusters       = {result['sub_n_clusters']}\n"
                    f"  max_depth            = {result['max_depth']}\n"
                    f"  Reasoning: {parsed.get('reasoning', 'N/A')}"
                )
                self.bus.emit(
                    'control_gates_tuned',
                    mode='bypass',
                    source='llm',
                    **result,
                    reasoning=parsed.get('reasoning', ''),
                )
                return result
            except Exception as exc:
                print(f'  [Orchestrator] Control-gate tuning failed ({exc}) — using data-driven defaults.')
                self.bus.emit('control_gates_tuned', mode='bypass', source='data_driven_default', **defaults)
                return defaults

        # ── INTERACTIVE MODE: ask the user via UI modal ─────────────────────────
        pending = pathlib.Path('outputs/pending_control_gates.json')
        try:
            pending.unlink(missing_ok=True)
        except OSError:
            pass

        self.bus.emit(
            'awaiting_control_gates',
            defaults=defaults,
            dataset_stats={
                'n_rows': n_rows,
                'n_features': n_features,
                'modality': modality,
                'mean_abs_skewness': round(mean_skew, 2),
                'pca_ev': pca_ev,
                'target_entity': target,
                'business_purpose': purpose,
            },
            timeout_s=300,
        )
        print(
            f'\n  [INTERACTIVE MODE] Control-gate tuning — pipeline paused.\n'
            f'  Please set max_cluster_size_pct, sub_n_clusters, and max_depth '
            f'in the UI modal (or wait 5 min to accept data-driven defaults).'
        )

        deadline = _time.time() + 300
        try:
            while _time.time() < deadline:
                if pending.exists():
                    try:
                        payload = json.loads(pending.read_text(encoding='utf-8'))
                        result = {
                            'max_cluster_size_pct': float(
                                payload.get('max_cluster_size_pct', defaults['max_cluster_size_pct'])
                            ),
                            'sub_n_clusters': int(
                                payload.get('sub_n_clusters', defaults['sub_n_clusters'])
                            ),
                            'max_depth': int(
                                payload.get('max_depth', defaults['max_depth'])
                            ),
                        }
                        # Clamp
                        result['max_cluster_size_pct'] = max(0.15, min(0.80, result['max_cluster_size_pct']))
                        result['sub_n_clusters'] = max(2, min(6, result['sub_n_clusters']))
                        result['max_depth'] = max(1, min(4, result['max_depth']))
                        try:
                            pending.unlink(missing_ok=True)
                        except OSError:
                            pass
                        print(
                            f'\n[Orchestrator] Control gates set by user:\n'
                            f"  max_cluster_size_pct = {result['max_cluster_size_pct']:.0%}\n"
                            f"  sub_n_clusters       = {result['sub_n_clusters']}\n"
                            f"  max_depth            = {result['max_depth']}"
                        )
                        self.bus.emit(
                            'control_gates_tuned',
                            mode='interactive',
                            source='user',
                            **result,
                        )
                        return result
                    except (OSError, ValueError, TypeError, json.JSONDecodeError):
                        pass
                _time.sleep(0.6)
        except KeyboardInterrupt:
            pass
        print('  [INTERACTIVE MODE] Control-gate tuning timed out — using data-driven defaults.')
        self.bus.emit('control_gates_tuned', mode='interactive', source='data_driven_default', **defaults)
        return defaults

    # ── Column resolution (entity_id, timestamp, amount, category) ─────────────

    def _resolve_columns(
        self,
        raw_df: pd.DataFrame,
        dataset_profile,
        user_intent,
    ) -> dict:
        """
        Resolve entity_id, timestamp, amount, and category columns using
        data-science heuristics.  If ambiguous:
          - bypass mode   → ask the LLM Decision Maker to decide
          - interactive   → pause and ask the user via UI modal
        Returns a dict with keys: entity_id, timestamp, amount, category.
        """
        import time as _time

        n_rows = len(raw_df)
        results: dict[str, str | None] = {}
        ambiguous: dict[str, list[str]] = {}

        def _score_name(col: str, keywords: list[str]) -> int:
            col_l = col.lower().replace('_', '').replace('-', '')
            score = 0
            for kw in keywords:
                kw_l = kw.lower().replace('_', '').replace('-', '')
                if kw_l == col_l:
                    score += 100
                elif kw_l in col_l:
                    score += 50
            return score

        # 1. Entity ID
        entity_candidates = []
        for col in raw_df.columns:
            if col.startswith('_'):
                continue
            nuniq = raw_df[col].nunique()
            ratio = nuniq / max(n_rows, 1)
            score = _score_name(col, [
                'id', 'entityid', 'userid', 'customerid', 'clientid',
                'accountid', 'subjectid', 'patientid', 'deviceid',
                'sensorid', 'itemid', 'productid', 'orderid',
                'sessionid', 'recordid', 'uuid', 'uid', 'pid',
                'cardnumber', 'ccnum',
            ])
            if ratio >= 0.90:
                score += 80
            elif ratio >= 0.70:
                score += 40
            elif ratio >= 0.50:
                score += 10
            if pd.api.types.is_integer_dtype(raw_df[col]):
                score += 20
            if score > 0:
                entity_candidates.append((col, score, ratio))
        entity_candidates.sort(key=lambda x: (-x[1], -x[2]))
        if entity_candidates and entity_candidates[0][1] >= 80:
            results['entity_id'] = entity_candidates[0][0]
        elif entity_candidates:
            ambiguous['entity_id'] = [c[0] for c in entity_candidates[:5]]
        else:
            results['entity_id'] = '_row_id'

        # 2. Timestamp
        ts_candidates = []
        for col in raw_df.columns:
            if col.startswith('_'):
                continue
            score = _score_name(col, [
                'timestamp', 'datetime', 'date', 'time', 'ts',
                'createdat', 'occurredat', 'recordedat', 'updatedat',
                'eventtime', 'eventdate', 'visitdate', 'purchasedate',
                'orderdate', 'transdate', 'transdatetranstime',
            ])
            if pd.api.types.is_datetime64_any_dtype(raw_df[col]):
                score += 100
            else:
                try:
                    sample = raw_df[col].dropna().head(20)
                    if len(sample) > 0:
                        parsed = pd.to_datetime(sample, errors='coerce')
                        if parsed.notna().sum() / len(sample) >= 0.8:
                            score += 80
                except Exception:
                    pass
            if score > 0:
                ts_candidates.append((col, score))
        ts_candidates.sort(key=lambda x: -x[1])
        if ts_candidates and ts_candidates[0][1] >= 80:
            results['timestamp'] = ts_candidates[0][0]
        elif ts_candidates:
            ambiguous['timestamp'] = [c[0] for c in ts_candidates[:5]]

        # 3. Amount (numeric, positive, not binary, not an ID)
        exclude_for_amount = {results.get('entity_id')}
        amount_candidates = []
        for col in raw_df.columns:
            if col.startswith('_') or col in exclude_for_amount:
                continue
            if not pd.api.types.is_numeric_dtype(raw_df[col]):
                continue
            nuniq = raw_df[col].nunique()
            ratio = nuniq / max(n_rows, 1)
            if ratio >= 0.95:
                continue
            s = raw_df[col].dropna()
            if len(s) == 0 or s.nunique() <= 2:
                continue
            score = _score_name(col, [
                'amount', 'value', 'price', 'total', 'amt',
                'cost', 'revenue', 'qty', 'quantity', 'score',
                'duration', 'size', 'weight', 'measurement', 'reading', 'level',
            ])
            if s.min() >= 0:
                score += 30
            if s.max() > s.min() * 10:
                score += 20
            if score > 0:
                amount_candidates.append((col, score))
        amount_candidates.sort(key=lambda x: -x[1])
        if amount_candidates and amount_candidates[0][1] >= 50:
            results['amount'] = amount_candidates[0][0]
        elif amount_candidates:
            ambiguous['amount'] = [c[0] for c in amount_candidates[:5]]

        # 4. Category (low-cardinality, object/categorical)
        exclude_for_category = {results.get('entity_id'), results.get('timestamp')}
        category_candidates = []
        for col in raw_df.columns:
            if col.startswith('_') or col in exclude_for_category:
                continue
            nuniq = raw_df[col].nunique()
            if nuniq < 2 or nuniq > 100:
                continue
            score = _score_name(col, [
                'category', 'cat', 'type', 'kind', 'label',
                'class', 'group', 'segment', 'tag', 'genre',
                'department', 'sector', 'channel', 'mode',
                'eventtype', 'itemtype', 'producttype', 'productcategory',
                'itemcategory', 'subcategory', 'transactiontype',
            ])
            if pd.api.types.is_categorical_dtype(raw_df[col]) or raw_df[col].dtype == object:
                score += 30
            if 2 <= nuniq <= 50:
                score += 20
            if score > 0:
                category_candidates.append((col, score))
        category_candidates.sort(key=lambda x: -x[1])
        if category_candidates and category_candidates[0][1] >= 40:
            results['category'] = category_candidates[0][0]
        elif category_candidates:
            ambiguous['category'] = [c[0] for c in category_candidates[:5]]

        # ── No ambiguity → return immediately ──────────────────────────────────
        if not ambiguous:
            print(
                f"\n[Orchestrator] Column resolution (auto-detect):\n"
                f"  entity_id  = {results.get('entity_id', 'N/A')}\n"
                f"  timestamp  = {results.get('timestamp', 'N/A')}\n"
                f"  amount     = {results.get('amount', 'N/A')}\n"
                f"  category   = {results.get('category', 'N/A')}"
            )
            self.bus.emit('columns_resolved', mode='auto', source='heuristics', **results)
            return results

        # ── Build schema context for LLM / user ────────────────────────────────
        schema_lines = []
        for col in raw_df.columns[:40]:
            if pd.api.types.is_datetime64_any_dtype(raw_df[col]):
                ctype = 'datetime'
            elif pd.api.types.is_numeric_dtype(raw_df[col]):
                ctype = 'numeric'
            elif raw_df[col].dtype == object or pd.api.types.is_categorical_dtype(raw_df[col]):
                ctype = 'categorical'
            else:
                ctype = 'other'
            nuniq = raw_df[col].nunique()
            example = str(raw_df[col].dropna().iloc[0]) if len(raw_df[col].dropna()) > 0 else ''
            if len(example) > 30:
                example = example[:30] + '...'
            schema_lines.append(f"  {col}: {ctype}, {nuniq} unique, e.g. {example!r}")
        schema_str = '\n'.join(schema_lines)

        purpose = (user_intent.business_purpose if user_intent else '') or ''
        target = (user_intent.target_entity if user_intent else 'entities') or 'entities'

        from skills.orchestrator_bus import read_pipeline_mode
        mode = read_pipeline_mode()

        # ── BYPASS MODE: ask the LLM Decision Maker ────────────────────────────
        if mode == 'bypass':
            ambig_desc = []
            for role, cands in ambiguous.items():
                ambig_desc.append(f"{role}: candidates are {', '.join(cands)}")
            prompt = f"""You are resolving ambiguous column roles for a clustering pipeline.

Dataset target: {target}
Business purpose: {purpose}

Schema:
{schema_str}

The following column roles are ambiguous:
{'\n'.join(ambig_desc)}

For each ambiguous role, pick the BEST column from the candidates and explain why.
If none of the candidates fit, respond with null.

Return ONLY a valid JSON object (no markdown, no prose):
{{
  "entity_id": "<column_name or null>",
  "timestamp": "<column_name or null>",
  "amount": "<column_name or null>",
  "category": "<column_name or null>",
  "reasoning": "<1 sentence per decision>"
}}"""
            try:
                raw = self.bus.ask(
                    agent='Orchestrator',
                    purpose='resolve ambiguous columns from schema (bypass mode)',
                    prompt=prompt,
                    max_tokens=512,
                    category='pipeline',
                ).strip()
                if '```' in raw:
                    for part in raw.split('```'):
                        p = part.strip()
                        if p.startswith('json'):
                            p = p[4:].strip()
                        if p.startswith('{'):
                            raw = p
                            break
                parsed = json.loads(raw)
                for key in ['entity_id', 'timestamp', 'amount', 'category']:
                    val = parsed.get(key)
                    if val and val in raw_df.columns:
                        results[key] = val
                    elif key in ambiguous:
                        results[key] = ambiguous[key][0]
                if 'entity_id' not in results:
                    results['entity_id'] = '_row_id'
                print(
                    f"\n[Orchestrator] Column resolution (Decision Maker — bypass):\n"
                    f"  entity_id  = {results.get('entity_id', 'N/A')}\n"
                    f"  timestamp  = {results.get('timestamp', 'N/A')}\n"
                    f"  amount     = {results.get('amount', 'N/A')}\n"
                    f"  category   = {results.get('category', 'N/A')}\n"
                    f"  Reasoning: {parsed.get('reasoning', 'N/A')}"
                )
                self.bus.emit(
                    'columns_resolved',
                    mode='bypass',
                    source='llm',
                    ambiguous_roles=list(ambiguous.keys()),
                    **results,
                    reasoning=parsed.get('reasoning', ''),
                )
                return results
            except Exception as exc:
                print(f'  [Orchestrator] Column-resolution LLM call failed ({exc}) — using heuristic fallbacks.')
                for key, cands in ambiguous.items():
                    results[key] = cands[0]
                if 'entity_id' not in results:
                    results['entity_id'] = '_row_id'
                self.bus.emit('columns_resolved', mode='bypass', source='heuristic_fallback', **results)
                return results

        # ── INTERACTIVE MODE: ask the user via UI modal ────────────────────────
        pending = pathlib.Path('outputs/pending_column_resolution.json')
        try:
            pending.unlink(missing_ok=True)
        except OSError:
            pass

        self.bus.emit(
            'awaiting_column_resolution',
            ambiguous_roles=ambiguous,
            heuristics=results,
            schema=schema_str,
            dataset_stats={
                'n_rows': n_rows,
                'n_cols': len(raw_df.columns),
                'target_entity': target,
                'business_purpose': purpose,
            },
            timeout_s=300,
        )
        print(
            f'\n  [INTERACTIVE MODE] Column resolution — pipeline paused.\n'
            f'  Ambiguous roles: {list(ambiguous.keys())}.\n'
            f'  Please confirm or override column choices in the UI modal.'
        )

        deadline = _time.time() + 300
        try:
            while _time.time() < deadline:
                if pending.exists():
                    try:
                        payload = json.loads(pending.read_text(encoding='utf-8'))
                        for key in ['entity_id', 'timestamp', 'amount', 'category']:
                            val = payload.get(key)
                            if val and val in raw_df.columns:
                                results[key] = val
                            elif key in ambiguous:
                                results[key] = ambiguous[key][0]
                        if 'entity_id' not in results:
                            results['entity_id'] = '_row_id'
                        try:
                            pending.unlink(missing_ok=True)
                        except OSError:
                            pass
                        print(
                            f'\n[Orchestrator] Column resolution set by user:\n'
                            f"  entity_id  = {results.get('entity_id', 'N/A')}\n"
                            f"  timestamp  = {results.get('timestamp', 'N/A')}\n"
                            f"  amount     = {results.get('amount', 'N/A')}\n"
                            f"  category   = {results.get('category', 'N/A')}"
                        )
                        self.bus.emit('columns_resolved', mode='interactive', source='user', **results)
                        return results
                    except (OSError, ValueError, TypeError, json.JSONDecodeError):
                        pass
                _time.sleep(0.6)
        except KeyboardInterrupt:
            pass
        print('  [INTERACTIVE MODE] Column-resolution timed out — using heuristic fallbacks.')
        for key, cands in ambiguous.items():
            results[key] = cands[0]
        if 'entity_id' not in results:
            results['entity_id'] = '_row_id'
        self.bus.emit('columns_resolved', mode='interactive', source='heuristic_fallback', **results)
        return results

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
        _modality_now = getattr(state, 'modality', 'tabular')
        _modality_banner = (
            "\n>>> MODALITY = TEXT / UNSTRUCTURED — read the TEXT-mode section\n"
            ">>> below FIRST. The TABULAR thresholds (silhouette 0.5, F1 0.70,\n"
            ">>> VIF, log-skew) DO NOT apply. Cosine silhouette of 0.10–0.30 is\n"
            ">>> the realistic band for usable topic clusters.\n"
            if _modality_now == 'text' else ""
        )
        prompt = prefs_preamble + case_preamble + _modality_banner + f"""You are orchestrating a customer-clustering pipeline. Iteration {iteration} just failed.

Current parameters:
  vif_threshold  : {cur.get('vif_threshold', 10.0)}   (higher = keep more correlated features; range 5–25) — IGNORE for text
  algorithm      : {cur.get('algorithm') or 'auto'}
  k_range        : {cur.get('k_range') or 'default [3,4,5,6,7,8,10,12,15]'}
  min_silhouette : {cur.get('min_silhouette', 0.05)}  (hard-block; range 0.02–0.12)
  feature_focus  : "{cur.get('feature_focus', '')}"   — IGNORE for text
  modality       : {_modality_now}
  text_vectorizer: {cur.get('text_vectorizer') or 'auto'}   (text mode only)

Best silhouette achieved so far: {best_sil_str}{k_scores_str}

Pipeline history (recent):
{chr(10).join(history_lines)}

Dataset shape: {self._describe_dataset_for_tuning(state)}

Available clustering algorithms (geometric — both modalities):
  kmeans, hierarchical, dbscan, gmm, fuzzy_cmeans, or null (auto-select)

Text-only algorithms (only valid when modality=text — Clusterer rejects them on tabular):
  lda          — Latent Dirichlet Allocation. Probabilistic topic model on TF-IDF.
                 Interpretable topics, soft assignment (we take argmax for hard labels).
                 Good when natural topics exist; needs ≥ ~30 docs for stability.
  nmf          — Non-negative Matrix Factorization on TF-IDF. Sharper / more
                 deterministic topics than LDA on short corpora. Try this first
                 for short-text rescue.
  llm_cluster  — LAST-RESORT rescue: sends every doc to Claude and asks for cluster
                 assignment. Use ONLY when geometric methods AND topic models have
                 BOTH failed. Capped at 200 docs by the clusterer (prompt cost grows
                 linearly). Expensive — burns several thousand tokens per run.

Tuning guidelines (TABULAR mode):
- If VIF gate removes >60 features or hits max_iterations: raise vif_threshold (try 12–18)
- If silhouette consistently <0.10 with hierarchical: switch to kmeans or gmm
- If silhouette is low across all k values: narrow k_range to [3,4,5,6] or [4,5,6,7]
- If many features selected but silhouette stays low: add feature_focus to guide selector
- Lower min_silhouette only if data genuinely resists clustering (e.g. keep at 0.03)
- Do NOT lower min_silhouette below 0.02
- dbscan is good when you suspect outlier-heavy data or irregular cluster shapes

Tuning guidelines (TEXT / UNSTRUCTURED mode — only applies when modality=text):
- Embedding spaces are high-dim and cosine-shaped; numeric expectations differ:
  * silhouette 0.10–0.30 cosine = usable topic clusters (NOT a failure signal)
  * classifier F1 ~0.55–0.70 is the realistic band, not the 0.70 tabular bar
  * VIF / log-skew / feature_focus do NOT apply — leave them as-is
  * max_cluster_size_pct is loosened (0.60) because text topics naturally vary in size
- ALGORITHM PREFERENCE for text — try in this order:
  1. GEOMETRIC FIRST: kmeans (= spherical k-means on L2-normalised embeddings,
     cosine-friendly) or hierarchical (Ward linkage). These are cheap and
     usually adequate.
  2. TOPIC MODELS if geometric silhouette stays below ~0.10 after ≥2 retries
     AND the corpus has clear lexical themes: try `nmf` (sharper topics on
     short text, deterministic) first, then `lda` (probabilistic, interpretable).
     Topic models can find structure that geometric clustering on SVD misses.
  3. LLM-AS-CLUSTERER as LAST RESORT: pick `llm_cluster` ONLY after both
     geometric AND topic-model algos have failed, and only when the corpus is
     small (clusterer caps it at 200 docs). It is expensive; do not pick it
     before iteration 4+.
  * AVOID: gmm (Gaussian assumption fails for text embeddings), dbscan
    (distance saturation in high-dim), fuzzy_cmeans (inherits GMM weakness).
  * If the current algorithm is one of {{gmm, dbscan, fuzzy_cmeans}} and
    silhouette is low, your first move should be to switch to kmeans or
    hierarchical, NOT to tune k_range.
- If silhouette stays low AND current text_vectorizer is "tfidf_svd": switch to
  "transformer" — semantic embeddings often separate topics that share vocabulary.
- If switching to "transformer" failed (it's unavailable / no improvement) and
  the corpus is short/keyword-heavy: stay on "tfidf_svd" and instead narrow
  k_range to [3,4,5,6].
- min_silhouette default for text is 0.01; don't raise it without a clear reason.
- If the user wrote `text_columns=col_a,col_b` in their constraints, RESPECT
  that — don't propose changing the text column. The first listed column is
  weighted 2× in the embedding, which is intentional.

Return ONLY a valid JSON object — no markdown fences, no extra text:
{{
  "vif_threshold": <float 5–25>,
  "k_range": [<int>, ...],
  "algorithm": "kmeans" | "hierarchical" | "dbscan" | "gmm" | "fuzzy_cmeans" | "lda" | "nmf" | "llm_cluster" | null,
  "min_silhouette": <float 0.02–0.12>,
  "feature_focus": "<short hint for FeatureSelector, or empty string>",
  "text_vectorizer": "tfidf_svd" | "transformer" | null,
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
        if params.get('algorithm') in (
            'kmeans', 'hierarchical', 'dbscan', 'gmm', 'fuzzy_cmeans',
            'lda', 'nmf', 'llm_cluster', None,
        ):
            result['algorithm'] = params['algorithm']
        if 'min_silhouette' in params:
            result['min_silhouette'] = float(max(0.02, min(0.12, params['min_silhouette'])))
        if 'feature_focus' in params:
            result['feature_focus'] = str(params.get('feature_focus', ''))
        if 'text_vectorizer' in params and params['text_vectorizer'] in (
                'tfidf_svd', 'transformer', None):
            result['text_vectorizer'] = params['text_vectorizer']

        reasoning = params.get('reasoning', '')
        print(
            f'  [Orchestrator] Tuned → vif={result.get("vif_threshold","—")}, '
            f'algo={result.get("algorithm","—")}, '
            f'k_range={result.get("k_range","—")}, '
            f'min_sil={result.get("min_silhouette","—")}, '
            f'text_vec={result.get("text_vectorizer","—")}'
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

    def _describe_dataset_for_tuning(self, state) -> str:
        """One-line dataset summary for the failure-tuning LLM prompt.

        Was previously hardcoded to a banking-dataset description, which
        confused the LLM on every other domain. Now derived from state.
        """
        modality = getattr(state, 'modality', 'tabular')
        prof = getattr(state, 'dataset_profile', None)
        n_rows = getattr(prof, 'n_rows', None)
        n_cols = getattr(prof, 'n_cols', None)
        if modality == 'text':
            tp = getattr(state, 'text_prep', None)
            method = state.text_artifacts.get('method', 'tfidf_svd') if state.text_artifacts else 'tfidf_svd'
            return (
                f"TEXT corpus, ~{n_rows or '?'} docs, vectorised as "
                f"{tp.n_dims if tp else '?'} dims via {method}. "
                "Cosine silhouettes on text are typically 0.03–0.12 even on clean topical clusters."
            )
        return (
            f"TABULAR data, ~{n_rows or '?'} rows × {n_cols or '?'} columns. "
            "Tabular silhouettes typically land 0.08–0.30 even in good segmentations."
        )

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
