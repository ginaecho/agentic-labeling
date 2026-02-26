"""
Orchestrator

Coordinates four agents in a feedback loop:
  (1) FeatureSelectionAgent  — PCA + AE scoring → Claude picks feature subset
  (2) ClusteringAgent        — hierarchical/kmeans + deepening loop
  (3) PersonaNamingAgent     — Claude names clusters; Clarity Gate validates
  (4) ClassifierAgent        — Random Forest CV validates cluster separability;
                               Claude routes back to (1) or (2) if F1 is low
  ↓
  Human Checkpoint           — user approves, requests re-run, or quits
"""
from __future__ import annotations

import json
import pathlib
import time
from collections import defaultdict

import anthropic
import pandas as pd

from agents.state import HumanDecision, PipelineState
from agents.feature_selector import FeatureSelectionAgent
from agents.clusterer import ClusteringAgent
from agents.persona_namer import PersonaNamingAgent
from agents.classifier import ClassifierAgent


def human_checkpoint(personas: dict, cluster_result, classifier_result) -> HumanDecision:
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


def save_outputs(cluster_result, naming_result, classifier_result) -> None:
    """Save cluster_profiles.json, cluster_lineage.json, personas.json, and classifier_metrics.json."""
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

    # Save classifier validation metrics
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


class Orchestrator:
    """
    Main pipeline coordinator.

    Loop order each iteration:
      1. Feature selection (first run, or when flagged for re-selection)
      2. Clustering
         → if oversized cluster + Claude says reselect: loop to 1
      3. Persona naming (Clarity Gate)
         → if gate fails: loop to 2
      4. Classifier validation (CV F1 gate)
         → if poor: Claude routes to 1 or 2
      5. Human checkpoint
         → approve  : save + exit
         → recluster: loop to 2
         → reselect : loop to 1
         → quit     : exit without saving
    """

    def __init__(self, config: dict):
        self.config = config
        self.client = anthropic.Anthropic()

        # ── Claude API usage tracking (monkeypatch) ────────────────────────
        # Records every messages.create() call across all agents with:
        #   agent, purpose, input_tokens, output_tokens, time_s
        self._claude_calls: list[dict] = []
        self._active_agent: str = 'Orchestrator'

        _orig_create = self.client.messages.create
        _tracker     = self._claude_calls
        _self_ref    = self          # capture for closure

        def _tracked_create(**kwargs):
            t0   = time.perf_counter()
            resp = _orig_create(**kwargs)
            _tracker.append({
                'agent':         _self_ref._active_agent,
                'input_tokens':  resp.usage.input_tokens,
                'output_tokens': resp.usage.output_tokens,
                'time_s':        round(time.perf_counter() - t0, 2),
            })
            return resp

        self.client.messages.create = _tracked_create
        # ──────────────────────────────────────────────────────────────────

        self.feature_agent    = FeatureSelectionAgent(self.client)
        self.cluster_agent    = ClusteringAgent(self.client, config)
        self.naming_agent     = PersonaNamingAgent(self.client)
        self.classifier_agent = ClassifierAgent(self.client)

        # Telemetry: {agent_name: [elapsed_seconds, ...]}
        self._timings: dict[str, list[float]] = defaultdict(list)
        self._pipeline_start: float = 0.0

    def run(
        self,
        features_path: str = 'data/processed/customer_features.parquet',
        max_total_iterations: int = 10,
    ) -> dict:
        """
        Run the full multi-agent pipeline.

        Returns
        -------
        dict with keys:
          'status'      : 'success' | 'quit' | 'max_iterations_reached'
          'personas'    : dict or None
          'run_history' : list of dicts summarising each stage/iteration
        """
        print('\n' + '=' * 65)
        print('Multi-Agent Persona Discovery Pipeline  (4 agents)')
        print('=' * 65)
        print(f'Config       : {self.config}')
        print(f'Features     : {features_path}')
        print(f'Max iters    : {max_total_iterations}')
        print(f'Classifier F1 threshold: {ClassifierAgent.F1_THRESHOLD}')

        features_df = pd.read_parquet(features_path)
        if 'cluster' in features_df.columns:
            features_df = features_df.drop(columns=['cluster'])

        print(f'Loaded {len(features_df)} customers × {len(features_df.columns)} features')

        state = PipelineState(config=self.config)
        run_history = []
        self._timings = defaultdict(list)
        self._pipeline_start = time.perf_counter()

        while state.total_iterations < max_total_iterations:
            state.total_iterations += 1
            iteration = state.total_iterations
            print(f'\n{"─"*65}')
            print(f'ITERATION {iteration} / {max_total_iterations}')
            print(f'{"─"*65}')

            # ── (1) Feature Selection ──────────────────────────────────────────
            if state.needs_feature_selection:
                _t0 = time.perf_counter()
                self._active_agent = 'FeatureSelector'
                fs = self.feature_agent.run(
                    features_df,
                    feedback=state.fs_feedback,
                    iteration=iteration,
                )
                self._timings['FeatureSelector'].append(time.perf_counter() - _t0)
                state.update_features(fs)
                run_history.append({
                    'iteration': iteration,
                    'stage': 'feature_selection',
                    'n_features': fs.n_features,
                    'elapsed_s': round(self._timings['FeatureSelector'][-1], 1),
                    'reasoning': fs.reasoning,
                })

            # ── (2) Clustering ─────────────────────────────────────────────────
            _t0 = time.perf_counter()
            self._active_agent = 'Clusterer'
            cr = self.cluster_agent.run(
                features_df,
                selected_features=state.selected_features,
                history=state.clustering_history,
                feedback=state.cluster_feedback,
                iteration=iteration,
            )
            self._timings['Clusterer'].append(time.perf_counter() - _t0)
            state.clustering_history.append(cr)

            if cr.action == 'reselect_features':
                print(f'\n[Orchestrator] Clustering → reselect features: {cr.reasoning}')
                state.request_feature_reselection(cr.reasoning)
                run_history.append({
                    'iteration': iteration,
                    'stage': 'clustering',
                    'action': 'reselect_features',
                    'elapsed_s': round(self._timings['Clusterer'][-1], 1),
                    'reasoning': cr.reasoning,
                })
                continue   # ← back to feature selection

            # ── (3) Persona Naming ─────────────────────────────────────────────
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
                'elapsed_s': round(self._timings['PersonaNamer'][-1], 1),
                'issues': nr.issues,
            })

            if not nr.passed:
                state.cluster_feedback = f'Clarity Gate failed: {"; ".join(nr.issues)}'
                state.naming_feedback = ''
                print(f'\n[Orchestrator] Clarity Gate failed → re-clustering.')
                continue   # ← back to clustering

            # ── (4) Classifier Validation ──────────────────────────────────────
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
                state.request_feature_reselection(clf.reasoning)
                state.classifier_feedback = clf.reasoning
                continue   # ← back to feature selection

            elif clf.action == 'recluster':
                print(f'\n[Orchestrator] Classifier → re-cluster: {clf.reasoning}')
                state.cluster_feedback = f'Classifier CV F1={clf.cv_f1_macro:.3f} too low: {clf.reasoning}'
                state.classifier_feedback = clf.reasoning
                continue   # ← back to clustering

            # clf.action == 'proceed' — track best result
            state.update_best(nr, cr, clf)

            # ── Human Checkpoint ───────────────────────────────────────────────
            decision = human_checkpoint(nr.personas, cr, clf)

            run_history.append({
                'iteration': iteration,
                'stage': 'human_checkpoint',
                'decision': decision.action,
                'feedback': decision.feedback,
            })

            if decision.action == 'approve':
                print('\n[Orchestrator] Approved! Saving outputs...')
                save_outputs(cr, nr, clf)
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
        best_personas = None
        if state.best_naming_result is not None:
            best_personas = state.best_naming_result.personas
            best_clf = state.best_classifier_result
            print(
                f'  Best result: avg_confidence={state.best_naming_result.avg_confidence:.1f}'
                + (f'  cv_f1={best_clf.cv_f1_macro:.3f}' if best_clf else '')
            )
            print('  Saving best result...')
            save_outputs(state.best_clustering_result, state.best_naming_result, best_clf)

        return {
            'status': 'max_iterations_reached',
            'personas': best_personas,
            'run_history': run_history,
            'timing': self._timing_dict(),
            'claude_usage': self._claude_usage_dict(),
        }

    # ── Telemetry helpers ──────────────────────────────────────────────────────

    def _timing_dict(self) -> dict:
        """Return a structured timing summary for inclusion in the result dict."""
        total_s = time.perf_counter() - self._pipeline_start
        agent_order = ['FeatureSelector', 'Clusterer', 'PersonaNamer', 'Classifier']
        agents = {}
        for name in agent_order:
            runs = self._timings.get(name, [])
            agents[name] = {
                'calls':        len(runs),
                'total_s':      round(sum(runs), 1),
                'per_call_s':   [round(r, 1) for r in runs],
            }
        return {
            'total_s': round(total_s, 1),
            'agents':  agents,
        }

    def _claude_usage_dict(self) -> dict:
        """Return per-agent Claude API usage summary."""
        from collections import defaultdict
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
            'by_agent':     dict(by_agent),
            'total_calls':  len(self._claude_calls),
            'total_input_tokens':  total_in,
            'total_output_tokens': total_out,
            'total_claude_time_s': round(total_t, 1),
            'raw_calls':    list(self._claude_calls),
        }

    def _print_timing_summary(self) -> None:
        """Print a human-readable timing table to stdout."""
        td = self._timing_dict()
        total_s = td['total_s']

        print('\n' + '=' * 65)
        print('PIPELINE TIMING SUMMARY')
        print('=' * 65)
        print(f'{"Agent":<22}  {"Calls":>5}  {"Total":>8}  {"Per call":>10}  {"% of total":>10}')
        print('-' * 65)

        AGENT_LABELS = {
            'FeatureSelector': '(1) FeatureSelector',
            'Clusterer':       '(2) Clusterer',
            'PersonaNamer':    '(3) PersonaNamer',
            'Classifier':      '(4) Classifier',
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
