"""
PersonaNamingAgent

Contract: docs/agents/persona_namer.md. Skills: docs/skills/orchestrator_bus.md.

Sends cluster profiles to the LLM and applies the Clarity Gate.
Cluster profiles are fully generic (feature_means, top_above_average,
top_below_average) — no domain-specific field names.

Reports structured status to OrchestratorBus.
"""
from __future__ import annotations

import json
import numpy as np
from collections import defaultdict

from agents.state import NamingResult
from skills.orchestrator_bus import OrchestratorBus, OrchestratorMessage

TONE_INSTRUCTIONS = {
    'easy': (
        'Use plain, everyday language. Avoid jargon. Use simple analogies a '
        'non-technical person would understand. Keep sentences short and friendly.'
    ),
    'professional': (
        'Use formal business language. Frame insights as actionable recommendations '
        'suitable for executive presentations. Be concise and authoritative.'
    ),
    'data-driven': (
        'Emphasise specific numbers and statistical ratios throughout. Quantify '
        'everything you can. Use technical language and cite exact metric values.'
    ),
    'creative': (
        'Use vivid metaphors, storytelling, and imaginative analogies. Make each '
        'persona feel like a character with a story and personality.'
    ),
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_cluster_block(cid: str, profile: dict, context_note: str = '') -> str:
    """
    Format one cluster's stats as a readable block for the LLM prompt.
    Uses the generic profile structure from ClusteringAgent:
      n_entities, pct_total, top_above_average, top_below_average, feature_means.
    """
    n    = profile.get('n_entities', profile.get('n_customers', '?'))
    pct  = profile.get('pct_total', profile.get('pct_of_total', 0))
    algo = profile.get('algo_detail', profile.get('algorithm', 'clustering'))

    top_above  = profile.get('top_above_average', {})
    top_below  = profile.get('top_below_average', {})
    feat_means = profile.get('feature_means', {})

    header = (
        f"{'='*65}\n"
        f"CLUSTER {cid}  ({n} entities, {pct:.1f}% of all entities)"
    )
    if context_note:
        header += f"\n{context_note}"
    header += f"\nAlgorithm: {algo}\n"

    above_lines = []
    for feat, rel in list(top_above.items())[:10]:
        mean_val = feat_means.get(feat, '?')
        flag = ' ◀◀' if rel >= 2.0 else (' ◀' if rel >= 1.4 else '')
        above_lines.append(
            f"    {feat:<45} mean={mean_val!s:>10}  vs_avg={rel:.2f}x{flag}"
        )

    below_lines = []
    for feat, rel in list(top_below.items())[:5]:
        mean_val = feat_means.get(feat, '?')
        below_lines.append(
            f"    {feat:<45} mean={mean_val!s:>10}  vs_avg={rel:.2f}x ▼"
        )

    above_section = (
        "  ABOVE AVERAGE (strongest signals):\n" + "\n".join(above_lines)
        if above_lines else ""
    )
    below_section = (
        "\n  BELOW AVERAGE:\n" + "\n".join(below_lines)
        if below_lines else ""
    )

    return header + above_section + below_section


def build_all_clusters_prompt(profiles: dict, cluster_lineage: dict,
                               tone_instructions: str) -> str:
    """
    Build a prompt that groups clusters hierarchically.
    Works with any domain — no hard-coded field names or domain vocabulary.
    """
    groups: dict = defaultdict(list)
    for cid, p in profiles.items():
        parent = p['lineage']['parent']
        groups[parent].append(cid)

    sections = []

    # Top-level clusters (no parent)
    top_level = sorted(groups.get(None, []), key=lambda x: int(x))
    if top_level:
        tl_blocks = [_format_cluster_block(cid, profiles[cid]) for cid in top_level]
        sections.append(
            "── TOP-LEVEL CLUSTERS ──────────────────────────────────────────\n"
            + '\n\n'.join(tl_blocks)
        )

    # Sub-cluster groups (grouped by parent)
    sub_parents = sorted([p for p in groups if p is not None])
    for parent in sub_parents:
        children = sorted(groups[parent], key=lambda x: int(x))
        parent_n   = sum(profiles[c].get('n_entities', 0) for c in children)
        parent_pct = sum(profiles[c].get('pct_total', 0) for c in children)
        group_header = (
            f"── SUB-CLUSTERS OF CLUSTER {parent} ────────────────────────────────\n"
            f"   Context: Cluster {parent} originally contained ~{parent_n} entities "
            f"({parent_pct:.1f}% of total), which exceeded the size threshold for a "
            f"single persona. It was automatically split into {len(children)} sub-clusters.\n"
            f"   Your job: name each sub-cluster to reflect HOW it differs from its "
            f"siblings — be specific about the behavioral signal that sets it apart."
        )
        child_blocks = []
        for cid in children:
            sibling_ids = [s for s in children if s != cid]
            note = (
                f"  [Sub-cluster {cid} — sibling of clusters "
                f"{', '.join(sibling_ids)}. Name it to show what is DIFFERENT "
                f"about it vs. those siblings.]"
            )
            child_blocks.append(_format_cluster_block(cid, profiles[cid], note))
        sections.append(group_header + '\n\n' + '\n\n'.join(child_blocks))

    all_sections = '\n\n\n'.join(sections)

    naming_rules = """
CRITICAL NAMING RULES — read carefully before writing any name:

1. SPECIFICITY — names must describe what the entity ACTUALLY DOES or IS, grounded
   in the specific features shown above (especially those marked ◀ or ◀◀).
   Bad: "The Average One", "The Regular Entity", "The Moderate Group".
   Good: names referencing the concrete feature patterns that stand out.
   If a name could apply to multiple clusters, it is too vague — rewrite it.

2. FOR SUB-CLUSTERS — the name must explain HOW this sub-cluster differs from its
   siblings. A sub-cluster name that ignores its siblings is not acceptable.

3. NO DUPLICATES — every name must be unique across all clusters.

4. CONFIDENCE — if a cluster's signals are very mixed and hard to name specifically,
   lower the confidence score and say so in the description.
"""

    return f"""You are a behavioral analyst interpreting entity clusters produced by a machine-learning pipeline.
Each cluster is described by its most distinguishing features:
  vs_avg: ratio of cluster mean to overall mean (1.0 = typical; ◀ = 40%+ above average; ◀◀ = 100%+ above; ▼ = 50%+ below)
  mean: the cluster's average value for that feature

{all_sections}

{'='*65}
{naming_rules}

TONE REQUIREMENT: {tone_instructions}

Return ONLY a valid JSON object (no markdown, no extra text) with this structure:
{{
  "0": {{
    "name": "...",
    "tagline": "...",
    "description": "2-3 sentences grounded in specific feature values from the data above",
    "dominant_features": ["feature1", "feature2", "feature3"],
    "traits": ["specific trait 1", "specific trait 2", "specific trait 3",
               "specific trait 4", "specific trait 5"],
    "confidence": <1-10>
  }},
  ...
}}"""


class PersonaNamingAgent:
    """
    Calls the LLM to name each cluster, then applies the Clarity Gate.
    Reports structured status to OrchestratorBus.
    """

    def __init__(self, bus: OrchestratorBus):
        # PersonaNamingAgent builds prompts from cluster stats (its own skill).
        # It asks the Orchestrator for LLM reasoning to name and describe clusters.
        self.bus = bus

    def run(
        self,
        profiles: dict,
        lineage: dict,
        tone: str = 'easy',
        feedback: str = '',
        iteration: int = 1,
        force_proceed: bool = False,
        user_intent=None,
    ) -> NamingResult:
        """
        Parameters
        ----------
        profiles : dict
            cluster_profiles output from ClusteringAgent.
        lineage : dict
            cluster_lineage output from ClusteringAgent.
        tone : str
            One of 'easy', 'professional', 'data-driven', 'creative'.
        feedback : str
            Free-text feedback from user or previous round.
        iteration : int
        """
        print(f'\n[PersonaNamer] Iteration {iteration}  (tone: {tone!r})')
        if feedback:
            print(f'  Feedback: {feedback}')

        tone_instr = TONE_INSTRUCTIONS.get(tone.lower(), TONE_INSTRUCTIONS['easy'])
        if feedback:
            tone_instr += f'\n\nAdditional guidance: {feedback}'

        # Collect must-have cluster constraints from user_intent
        must_have = []
        if user_intent and getattr(user_intent, 'must_have_clusters', None):
            must_have = list(user_intent.must_have_clusters)

        n_leaf = len(profiles)
        print(f'  Prepared cluster prompt for {n_leaf} clusters — asking Orchestrator for naming...')
        if must_have:
            print(f'  Must-have cluster types: {must_have}')

        prompt = build_all_clusters_prompt(profiles, lineage, tone_instr)

        # ── Adaptive learning: prepend persistent user feedback ────────────
        # Every change the user made in the interactive UI was logged to
        # outputs/user_feedback_log.jsonl. We surface high-/medium-priority
        # entries here so the Decision Maker honours the user's past
        # preferences on every future run.
        try:
            # Experiment opt-in: when EXPERIMENT_DEDUP_PREFS=1 is set
            # (the convergence loop sets it), route through the LLM
            # compaction layer that merges near-duplicates and drops
            # stale rules BEFORE injection. Live UI runs (env var unset)
            # use the original verbatim path.
            import os as _os
            if _os.environ.get('EXPERIMENT_DEDUP_PREFS') == '1':
                from experiments.dedup_prefs import build_deduplicated_preferences_block
                current_names = [
                    (p.get('persona') or {}).get('name')
                    for p in profiles.values()
                ] if isinstance(profiles, dict) else None
                # profiles here is cluster_profiles (no persona names yet),
                # so current_names will be all None — that's fine, dedup
                # just won't apply the stale-target filter on first naming.
                current_names = [n for n in (current_names or []) if n]
                prefs_block = build_deduplicated_preferences_block(
                    current_cluster_names=current_names,
                )
                source_label = 'deduplicated UI feedback'
            else:
                from ui.feedback_store import build_preferences_block
                prefs_block = build_preferences_block(
                    types=('manual_override', 'naming_hint',
                            'global_rule', 'merge'),
                )
                source_label = 'UI feedback'
            if prefs_block:
                prompt = prefs_block + '\n' + prompt
                print(f'  [PersonaNamer] Injected '
                      f'{prefs_block.count(chr(10))} lines of '
                      f'{source_label}.')
        except Exception as _exc:  # noqa: BLE001
            # Memory injection is best-effort; never block the pipeline on it
            print(f'  [PersonaNamer] (no UI feedback memory loaded: {_exc})')

        # Append must-have constraint to prompt if set
        if must_have:
            must_have_str = ', '.join(f'"{t}"' for t in must_have)
            prompt += f"""

MANDATORY CLUSTER REQUIREMENT:
The user has specified that the following cluster type(s) MUST be represented in your output:
  {must_have_str}

For each required type, at least one cluster name or its description must clearly capture that concept.
If none of the clusters naturally fit a required type, assign it to the closest matching cluster
and note in the description why this cluster represents that type.
Do NOT omit any required cluster type from the output."""

        # PersonaNamingAgent has built the full cluster data table itself.
        # It asks the Orchestrator for LLM reasoning to name and describe each cluster.
        raw = self.bus.ask(
            agent="PersonaNamer",
            purpose=f"name and describe {n_leaf} clusters (tone={tone!r})",
            prompt=prompt,
            max_tokens=4096,
        ).strip()
        if '```' in raw:
            for part in raw.split('```'):
                p = part.strip()
                if p.startswith('json'):
                    p = p[4:].strip()
                if p.startswith('{'):
                    raw = p
                    break

        try:
            personas = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f'  JSON parse error: {e}')
            return NamingResult(
                action='recluster',
                personas=None,
                passed=False,
                issues=[f'JSON parse error: {e}'],
                avg_confidence=0.0,
                reasoning='LLM response could not be parsed as JSON.',
                iteration=iteration,
            )

        # ── Clarity Gate (mirrors notebook 04 cell c2e5b2fc) ──────────────────
        confidences = []
        names = []
        for cid, p in personas.items():
            if 'name' in p:
                confidences.append(p.get('confidence', 0))
                names.append(p['name'])

        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        names_unique = len(names) == len(set(names))

        # Silhouette is checked in the orchestrator (we don't have X_scaled here),
        # so the gate here checks confidence + uniqueness only.
        issues = []
        if avg_conf < 6.0:
            issues.append(f'Avg LLM confidence {avg_conf:.1f} < 6.0')
        if not names_unique:
            issues.append('Duplicate persona names detected')

        # Check that every must-have cluster type is covered
        if must_have:
            all_text = ' '.join(
                (p.get('name', '') + ' ' + p.get('description', '')).lower()
                for p in personas.values()
            )
            missing_types = [
                t for t in must_have
                if t.lower().replace('-', ' ') not in all_text
                and t.lower().replace(' ', '-') not in all_text
                and t.lower() not in all_text
            ]
            if missing_types:
                issues.append(
                    f'Must-have cluster type(s) not found in any persona name/description: '
                    f'{missing_types}'
                )

        passed = len(issues) == 0
        if force_proceed and not passed:
            print(f'  [best-effort] Clarity Gate bypassed — delivering best available personas.')
            issues = [f'[forced] {i}' for i in issues]
            passed = True
        action = 'proceed' if passed else 'recluster'

        for cid, p in personas.items():
            lin = lineage.get(int(cid), {})
            depth_str = f'  [depth {lin.get("depth", 0)}'
            if lin.get('parent') is not None:
                depth_str += f', sub of {lin["parent"]}]'
            else:
                depth_str += ']'
            conf = p.get('confidence', '?')
            print(f'  Cluster {cid}{depth_str}: "{p.get("name", "?")}"  conf={conf}/10')

        if passed:
            print(f'  Clarity Gate PASSED  (avg_conf={avg_conf:.1f}, unique={names_unique})')
        else:
            print(f'  Clarity Gate FAILED: {issues}')

        # ── Report to orchestrator ─────────────────────────────────────────────
        if self.bus:
            self.bus.report(OrchestratorMessage(
                agent="PersonaNamer",
                iteration=iteration,
                status="success" if passed else ("warning" if avg_conf >= 4.0 else "blocked"),
                what_was_done=(
                    f"Named {n_leaf} clusters using LLM (tone={tone!r}). "
                    f"Clarity Gate {'PASSED' if passed else 'FAILED'}. "
                    f"Avg confidence={avg_conf:.1f}/10."
                ),
                what_was_not_done=(
                    "Did not validate that description text references specific numbers "
                    "(only checked confidence and name uniqueness)."
                ),
                doubts=(
                    f"Low-confidence clusters: "
                    + ", ".join(
                        f'C{cid}(conf={p.get("confidence","?")})'
                        for cid, p in personas.items()
                        if isinstance(p.get("confidence"), (int, float)) and p["confidence"] < 6
                    )
                    if personas and not passed else ""
                ),
                issues=issues,
                metrics={
                    "n_clusters": n_leaf,
                    "avg_confidence": round(avg_conf, 2),
                    "gate_passed": passed,
                    "names_unique": names_unique,
                    "must_have_clusters": must_have,
                },
                recommendation="proceed" if passed else "retry",
                context={
                    "persona_names": {cid: p.get("name") for cid, p in personas.items()} if personas else {},
                },
            ))

        return NamingResult(
            action=action,
            personas=personas if passed else None,
            passed=passed,
            issues=issues,
            avg_confidence=avg_conf,
            reasoning='Gate passed.' if passed else f'Gate failed: {"; ".join(issues)}',
            iteration=iteration,
        )
