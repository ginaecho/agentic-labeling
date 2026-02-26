"""
PersonaNamingAgent

Sends cluster profiles to Claude and applies the Clarity Gate.
Reuses build_all_clusters_prompt() and _format_cluster_block() logic
verbatim from notebook 04 cell 5eace074 and Clarity Gate from cell c2e5b2fc.
"""
from __future__ import annotations

import json
import numpy as np
import anthropic
from collections import defaultdict

from agents.state import NamingResult

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


# ── Helpers (verbatim from notebook 04 cell 5eace074) ─────────────────────────

def _format_cluster_block(cid: str, profile: dict, context_note: str = '') -> str:
    """Format one cluster's stats as a readable block for the LLM prompt."""
    n   = profile['n_customers']
    pct = profile['pct_of_total']
    ov  = profile['overall']
    cat_stats = profile['category_stats']
    algo = profile.get('algorithm_detail', 'clustering')

    cat_lines = []
    for cat, s in list(cat_stats.items()):
        if s['n_txn_12m'] > 0:
            flag = ' ◀◀' if s['rel_n_txn'] >= 2.0 else (' ◀' if s['rel_n_txn'] >= 1.4 else '')
            low  = ' ▼' if s['rel_n_txn'] <= 0.5 else ''
            cat_lines.append(
                f"    {cat:<20} txns/yr={s['n_txn_12m']:>6}  "
                f"spend/yr=${s['total_amt_12m']:>8}  "
                f"avg/txn=${s['avg_spend_12m']:>7}  "
                f"consec={s['consec_months']:>4}mo  "
                f"vs_avg={s['rel_n_txn']:>5}x{flag}{low}"
            )

    header = (
        f"{'='*65}\n"
        f"CLUSTER {cid}  ({n} customers, {pct:.1%} of all customers)"
    )
    if context_note:
        header += f"\n{context_note}"
    header += f"\nAlgorithm: {algo}\n"

    return (
        header + '\n'.join(cat_lines) +
        f"\n  Overall: avg_txn=${ov['avg_txn_amt']}  "
        f"total_spend=${ov['total_spend']}  "
        f"txns_per_yr={ov['total_txn_count']}  "
        f"pct_high_value={ov['pct_high_value']}%  "
        f"avg_days_between={ov['avg_days_between_txn']}"
    )


def build_all_clusters_prompt(profiles: dict, cluster_lineage: dict,
                               tone_instructions: str) -> str:
    """
    Build a prompt that groups clusters hierarchically.
    Verbatim from notebook 04 cell 5eace074.
    """
    groups: dict = defaultdict(list)
    for cid, p in profiles.items():
        parent = p['lineage']['parent']
        groups[parent].append(cid)

    sections = []

    # Top-level clusters (no parent)
    top_level = sorted(groups.get(None, []), key=lambda x: int(x))
    if top_level:
        tl_blocks = []
        for cid in top_level:
            tl_blocks.append(_format_cluster_block(cid, profiles[cid]))
        sections.append(
            "── TOP-LEVEL CLUSTERS ──────────────────────────────────────────\n"
            + '\n\n'.join(tl_blocks)
        )

    # Sub-cluster groups (grouped by parent)
    sub_parents = sorted([p for p in groups if p is not None])
    for parent in sub_parents:
        children = sorted(groups[parent], key=lambda x: int(x))
        parent_n   = sum(profiles[c]['n_customers'] for c in children)
        parent_pct = sum(profiles[c]['pct_of_total'] for c in children)
        group_header = (
            f"── SUB-CLUSTERS OF CLUSTER {parent} ────────────────────────────────\n"
            f"   Context: Cluster {parent} originally contained ~{parent_n} customers "
            f"({parent_pct:.1%} of total), which exceeded the size threshold for a "
            f"single persona. It was automatically split into {len(children)} sub-clusters.\n"
            f"   Your job: name each sub-cluster to reflect HOW it differs from its "
            f"siblings — not just that it 'spends moderately'. Be specific about the "
            f"behavioral signal that sets it apart from the other sub-clusters below."
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
    n_clusters = len(profiles)

    naming_rules = """
CRITICAL NAMING RULES — read carefully before writing any name:

1. SPECIFICITY — names must describe what the customer ACTUALLY DOES, not a vague
   lifestyle label. Bad: "The Steady Homebody", "The Regular Spender", "The Average User".
   Good: "The Daily Gas & Grocery Commuter", "The Online Grocery Subscriber",
         "The Gym-and-Travel Regular", "The Occasional Big-Ticket Online Shopper".
   If a name could apply to multiple clusters, it is too vague — rewrite it.

2. FOR SUB-CLUSTERS — the name must explain HOW this sub-cluster differs from its
   siblings. A sub-cluster name that ignores its siblings is not acceptable.
   Example: if siblings are split between "high travel" and "high grocery", one should
   be named "The Weekend Traveller" and the other "The Heavy Grocery & Household Buyer".

3. NO DUPLICATES — every name must be unique across all clusters.

4. CONFIDENCE — if a cluster's signals are very mixed and hard to name specifically,
   lower the confidence score (1-5) and say so in the description.
"""

    return f"""You are a consumer behavior analyst interpreting bank transaction clusters.
Each row shows: annual transactions, annual spend, avg per transaction, consecutive loyal
months, and vs_avg (1.0 = typical; ◀ = 40%+ above average; ◀◀ = 100%+ above; ▼ = 50%+ below).

{all_sections}

{'='*65}
{naming_rules}

TONE REQUIREMENT: {tone_instructions}

Return ONLY a valid JSON object (no markdown, no extra text) with this structure:
{{
  "0": {{
    "name": "...",
    "tagline": "...",
    "description": "2-3 sentences grounded in specific numbers from the data above",
    "dominant_categories": ["cat1", "cat2", "cat3"],
    "traits": ["specific trait 1", "specific trait 2", "specific trait 3",
               "specific trait 4", "specific trait 5"],
    "confidence": <1-10>
  }},
  ...
}}"""


class PersonaNamingAgent:
    """
    Calls Claude to name each cluster, then applies the Clarity Gate.
    """

    def __init__(self, client: anthropic.Anthropic):
        self.client = client

    def run(
        self,
        profiles: dict,
        lineage: dict,
        tone: str = 'easy',
        feedback: str = '',
        iteration: int = 1,
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

        n_leaf = len(profiles)
        print(f'  Calling Claude with {n_leaf} clusters...')

        prompt = build_all_clusters_prompt(profiles, lineage, tone_instr)

        response = self.client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=4096,
            messages=[{'role': 'user', 'content': prompt}],
        )

        raw = response.content[0].text.strip()
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
                reasoning='Claude response could not be parsed as JSON.',
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

        passed = len(issues) == 0
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

        return NamingResult(
            action=action,
            personas=personas if passed else None,
            passed=passed,
            issues=issues,
            avg_confidence=avg_conf,
            reasoning='Gate passed.' if passed else f'Gate failed: {"; ".join(issues)}',
            iteration=iteration,
        )
