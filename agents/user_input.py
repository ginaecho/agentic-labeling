"""
UserInputAgent — Pipeline Entry Point

Contract: docs/agents/user_input.md (role, I/O, protocol, retry behaviour).

Collects and validates the user's clustering intent before any computation.
Asks six questions (two required, four optional):
  1. What entity are you clustering? (target_entity)             [required]
  2. What is the business purpose?   (business_purpose)          [required]
  3. Where is your dataset?          (dataset_path — defaults to config)
  4. Any constraints?                (constraints — free text)
  5. How many clusters?              (n_clusters_requested — blank = data-driven)
  6. Must-have cluster types?        (must_have_clusters — comma-separated)

If the business_purpose answer is too vague (< 20 chars), the agent asks
one clarifying follow-up. If running non-interactively (EOFError), it falls
back to defaults from config and reports doubts="running with defaults".
"""
from __future__ import annotations

import json
import pathlib
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from skills.orchestrator_bus import OrchestratorBus, OrchestratorMessage


# Regex patterns to extract a "max cluster size" constraint from free-text
# intent (business_purpose + constraints). Each pattern captures the percentage
# in group 1. Patterns are tried in order; first match wins.
_MAX_PCT_PATTERNS = [
    # "max cluster size 25%", "maximum cluster share of 30%",
    # "the maximum cluster shall be lower than 20%"
    re.compile(
        r"max(?:imum)?\s+cluster\s+(?:size|share|fraction)?\s*"
        r"(?:of|to|≤|<=?|under|below|less than|lower than|no (?:more|larger|bigger) than|"
        r"(?:shall|should|must|will|cannot|can't|may not)\s+(?:be\s+)?"
        r"(?:lower|less|below|under|smaller|no more|no larger|no bigger)"
        r"(?:\s+than)?|be)?\s*"
        r"(\d+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    ),
    # "no cluster larger/bigger/greater than 30%"
    re.compile(
        r"no\s+(?:single\s+)?cluster\s+(?:larger|bigger|greater|more)\s+than\s+"
        r"(\d+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    ),
    # "cluster size must/should/shall be below 30%"
    re.compile(
        r"cluster\s+(?:size|share)\s+(?:must|should|shall|will|cannot|can't)\s+(?:be\s+)?"
        r"(?:lower\s+than|less\s+than|below|under|≤|<=?)\s*(\d+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    ),
    # "no cluster over 30%" / "no cluster above 30%"
    re.compile(
        r"no\s+cluster\s+(?:over|above|exceeds?|exceeding)\s+(\d+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    ),
]


def _parse_max_cluster_pct(*texts: str) -> Optional[float]:
    """Scan one or more free-text fields for a 'max cluster size X%' constraint.

    Returns a fraction in (0, 1] (e.g. 25% → 0.25) or None if no match.
    """
    blob = " ".join(t for t in texts if t)
    if not blob:
        return None
    for pat in _MAX_PCT_PATTERNS:
        m = pat.search(blob)
        if m:
            try:
                pct = float(m.group(1))
            except (TypeError, ValueError):
                continue
            if 1.0 <= pct <= 100.0:
                frac = pct / 100.0
                if 0.0 < frac <= 1.0:
                    return round(frac, 4)
    return None

DEFAULT_DATASET_PATH = "data/raw/fraudTrain.csv"
PENDING_INTENT_PATH = pathlib.Path("outputs/pending_intent.json")
UI_INTENT_TIMEOUT_S = 600   # 10 min — UI users have plenty of time to fill the form


@dataclass
class UserIntent:
    """Captured clustering intent from the user."""
    target_entity: str
    """What is being clustered, e.g. 'customers', 'products'."""
    business_purpose: str
    """Why we are clustering, e.g. 'understand shopping behaviour to personalise offers'."""
    dataset_path: str
    """Path to the feature parquet/CSV file."""
    constraints: str = ""
    """Optional free-text constraints, e.g. 'ignore fraud transactions'."""
    n_clusters_requested: Optional[int] = None
    """User-specified fixed cluster count. If set, ClusteringAgent uses it directly and skips k-optimisation."""
    must_have_clusters: list = field(default_factory=list)
    """Cluster types that MUST appear in the final result, e.g. ['traveller', 'high-value-product'].
    PersonaNamingAgent will enforce this via the Clarity Gate."""
    max_cluster_size_pct: Optional[float] = None
    """Parsed from intent text: 'max cluster size 25%' → 0.25. If set, overrides
    config['max_cluster_size_pct'] (default 0.40) — Clusterer treats any cluster
    larger than this as oversized and either sub-clusters or reselects features."""


class UserInputAgent:
    """
    Collects clustering intent from the user via interactive prompts.

    The agent is instantiated once and its `run()` method called at the
    start of the pipeline. It reports to the orchestrator bus so the full
    pipeline log records the captured intent.
    """

    def __init__(self, bus: OrchestratorBus, default_dataset_path: str = DEFAULT_DATASET_PATH):
        self.bus = bus
        self.default_dataset_path = default_dataset_path

    def run(self, iteration: int = 1) -> UserIntent:
        """
        Collect the user's clustering intent.

        Resolution order:
        1. If `outputs/pending_intent.json` exists, consume it (UI-driven flow).
        2. Otherwise emit `awaiting_intent` event and wait up to UI_INTENT_TIMEOUT_S
           for the UI to write that file.
        3. While waiting, also accept stdin (terminal fallback). First source wins.
        """
        print("\n" + "=" * 65)
        print("AGENTIC CLUSTERING PIPELINE — Intent Collection")
        print("=" * 65)

        # ── UI-first path: poll for pending_intent.json from the browser ─────
        ui_intent = self._wait_for_ui_intent()
        if ui_intent is not None:
            self._announce(ui_intent)
            return ui_intent

        # ── Terminal fallback (original interactive prompts) ─────────────────
        print("Before we begin, please answer a few questions.")
        print("(Press Enter to skip optional questions and use defaults.)\n")

        using_defaults = False
        doubts = ""
        issues = []

        def _prompt_safe(question: str, hint: str, default: str) -> str:
            try:
                return self._prompt(question=question, hint=hint, default=default)
            except (EOFError, KeyboardInterrupt):
                nonlocal using_defaults
                using_defaults = True
                return default

        # ── Q1: Target entity ─────────────────────────────────────────────────
        target_entity = _prompt_safe(
            "1. What entity are you clustering?",
            "  Examples: customers, products, employees, merchants",
            "customers",
        )

        # ── Q2: Business purpose ──────────────────────────────────────────────
        business_purpose = _prompt_safe(
            "2. What is the business purpose of this clustering?",
            (
                "  Be specific — this shapes which features are built and "
                "how clusters are interpreted.\n"
                "  Example: 'understand customer shopping behaviour "
                "to personalise product recommendations'"
            ),
            "",
        )

        # Follow-up if purpose is too vague (per docs/agents/user_input.md: < 20 chars)
        if len(business_purpose.strip()) < 20 and not using_defaults:
            print("\n  [UserInput] Your answer is quite short — a more specific purpose")
            print("  leads to better features and cluster labels.")
            followup = _prompt_safe(
                "  Can you elaborate? (or press Enter to continue with what you gave)",
                "  Tip: mention what decision or action the clusters will support",
                business_purpose,
            )
            if len(followup.strip()) >= len(business_purpose.strip()):
                business_purpose = followup

        if len(business_purpose.strip()) < 20:
            doubts = "Business purpose may still be too vague."
        if using_defaults:
            doubts = "running with defaults" if not doubts else f"{doubts}; running with defaults"

        # ── Q3: Dataset path ──────────────────────────────────────────────────
        print(f"\n  Default dataset path: {self.default_dataset_path}")
        dataset_path_raw = _prompt_safe(
            "3. Dataset path? (press Enter to use default)",
            "",
            "",
        )
        dataset_path = dataset_path_raw.strip() if dataset_path_raw.strip() else self.default_dataset_path

        # ── Q4: Constraints ───────────────────────────────────────────────────
        constraints = _prompt_safe(
            "4. Any constraints or filters? (optional — press Enter to skip)",
            "  Example: 'only use last 12 months of transactions', 'exclude VIP customers'",
            "",
        )

        # ── Q5: Desired cluster count ──────────────────────────────────────────
        n_clusters_requested: Optional[int] = None
        n_clusters_raw = _prompt_safe(
            "5. How many clusters would you like? (press Enter to let the pipeline decide)",
            "  Example: '5' for exactly 5 clusters. Leave blank for data-driven selection.",
            "",
        )
        if n_clusters_raw.strip().isdigit():
            n_clusters_requested = int(n_clusters_raw.strip())
            if n_clusters_requested < 2:
                print("  [UserInput] Minimum 2 clusters required — ignoring, using data-driven selection.")
                n_clusters_requested = None
            else:
                print(f"  [UserInput] Will target exactly {n_clusters_requested} clusters.")

        # ── Q6: Must-have cluster types ────────────────────────────────────────
        must_have_raw = _prompt_safe(
            "6. Must any specific types appear as clusters? (optional — press Enter to skip)",
            (
                "  List types separated by commas — these are semantic labels the pipeline\n"
                "  MUST produce as distinct personas.\n"
                "  Example: 'traveller, high-value-customer, weekend-shopper'"
            ),
            "",
        )
        must_have_clusters = (
            [t.strip() for t in must_have_raw.split(",") if t.strip()]
            if must_have_raw.strip() else []
        )
        if must_have_clusters:
            print(f"  [UserInput] Must-have clusters: {must_have_clusters}")

        # ── Summary ───────────────────────────────────────────────────────────
        max_pct = _parse_max_cluster_pct(business_purpose, constraints)
        intent = UserIntent(
            target_entity=target_entity.strip() or "customers",
            business_purpose=business_purpose.strip(),
            dataset_path=dataset_path,
            constraints=constraints.strip(),
            n_clusters_requested=n_clusters_requested,
            must_have_clusters=must_have_clusters,
            max_cluster_size_pct=max_pct,
        )

        print("\n" + "─" * 65)
        print("Captured intent:")
        print(f"  Target entity    : {intent.target_entity}")
        print(f"  Business purpose : {intent.business_purpose}")
        print(f"  Dataset path     : {intent.dataset_path}")
        if intent.constraints:
            print(f"  Constraints      : {intent.constraints}")
        if intent.n_clusters_requested is not None:
            print(f"  Clusters wanted  : {intent.n_clusters_requested} (fixed)")
        else:
            print(f"  Clusters wanted  : data-driven (auto-select)")
        if intent.must_have_clusters:
            print(f"  Must-have types  : {', '.join(intent.must_have_clusters)}")
        if intent.max_cluster_size_pct is not None:
            print(
                f"  Max cluster size : {intent.max_cluster_size_pct:.0%} "
                f"(parsed from intent — overrides default 40%)"
            )
        print("─" * 65)

        # ── Report to orchestrator ────────────────────────────────────────────
        self.bus.report(OrchestratorMessage(
            agent="UserInput",
            iteration=iteration,
            status="success" if not issues else "warning",
            what_was_done=(
                f"Collected intent: target='{intent.target_entity}', "
                f"purpose='{intent.business_purpose[:60]}'"
            ),
            what_was_not_done="Did not validate that the dataset file actually exists.",
            doubts=doubts,
            issues=issues,
            metrics={
                "target_entity": intent.target_entity,
                "purpose_length": len(intent.business_purpose),
                "has_constraints": bool(intent.constraints),
                "n_clusters_requested": intent.n_clusters_requested,
                "must_have_clusters": intent.must_have_clusters,
            },
            recommendation="proceed",
            context={"user_intent": {
                "target_entity": intent.target_entity,
                "business_purpose": intent.business_purpose,
                "dataset_path": intent.dataset_path,
                "constraints": intent.constraints,
                "n_clusters_requested": intent.n_clusters_requested,
                "must_have_clusters": intent.must_have_clusters,
            }},
        ))

        return intent

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _wait_for_ui_intent(self) -> Optional[UserIntent]:
        """
        Wait for the UI to write outputs/pending_intent.json. Returns the parsed
        UserIntent on success, or None if the user appears to be running headless
        (no UI server reachable, or timeout / interrupt while waiting).

        The strategy:
          - If the file already exists at start, consume it immediately.
          - Else announce 'awaiting_intent' on the bus and poll for up to
            UI_INTENT_TIMEOUT_S, sleeping in short ticks so Ctrl-C is responsive.
        """
        # If a fresh intent file already exists (e.g. saved by the user before
        # the pipeline reached this point), consume it.
        if PENDING_INTENT_PATH.exists():
            return self._consume_intent_file()

        # No live UI server is detectable from the agent's side; do a soft check
        # by simply emitting the event and waiting. If no file ever appears, we
        # fall through to the terminal prompts.
        try:
            self.bus.emit(
                "awaiting_intent",
                fields=[
                    "target_entity", "business_purpose", "dataset_path",
                    "constraints", "n_clusters_requested", "must_have_clusters",
                ],
                default_dataset_path=self.default_dataset_path,
                timeout_s=UI_INTENT_TIMEOUT_S,
            )
        except Exception:
            return None

        print(f"\n  [UserInput] Waiting up to {UI_INTENT_TIMEOUT_S}s for intent from the UI form")
        print(f"  [UserInput] (open http://127.0.0.1:5057/ and submit the intent form,")
        print(f"  [UserInput]  or press Ctrl-C to fall back to terminal prompts)")

        deadline = time.time() + UI_INTENT_TIMEOUT_S
        try:
            while time.time() < deadline:
                if PENDING_INTENT_PATH.exists():
                    return self._consume_intent_file()
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n  [UserInput] Interrupted — falling back to terminal prompts.")
            return None

        print(f"\n  [UserInput] No UI intent received within {UI_INTENT_TIMEOUT_S}s — terminal prompts.")
        return None

    def _consume_intent_file(self) -> Optional[UserIntent]:
        try:
            payload = json.loads(PENDING_INTENT_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  [UserInput] Could not read pending_intent.json: {exc}")
            try:
                PENDING_INTENT_PATH.unlink(missing_ok=True)
            except OSError:
                pass
            return None

        # Best-effort: delete the file so a stale intent isn't re-consumed next run
        try:
            PENDING_INTENT_PATH.unlink(missing_ok=True)
        except OSError:
            pass

        target = str(payload.get("target_entity") or "").strip() or "customers"
        purpose = str(payload.get("business_purpose") or "").strip()
        dataset = str(payload.get("dataset_path") or "").strip() or self.default_dataset_path
        constraints = str(payload.get("constraints") or "").strip()
        n_req = payload.get("n_clusters_requested")
        try:
            n_req = int(n_req) if n_req not in (None, "", "null") else None
            if n_req is not None and n_req < 2:
                n_req = None
        except (TypeError, ValueError):
            n_req = None
        must_have_raw = payload.get("must_have_clusters") or []
        if isinstance(must_have_raw, str):
            must_have = [t.strip() for t in must_have_raw.split(",") if t.strip()]
        elif isinstance(must_have_raw, list):
            must_have = [str(t).strip() for t in must_have_raw if str(t).strip()]
        else:
            must_have = []

        max_pct = _parse_max_cluster_pct(purpose, constraints)
        return UserIntent(
            target_entity=target,
            business_purpose=purpose,
            dataset_path=dataset,
            constraints=constraints,
            n_clusters_requested=n_req,
            must_have_clusters=must_have,
            max_cluster_size_pct=max_pct,
        )

    def _announce(self, intent: "UserIntent", iteration: int = 0) -> None:
        """Print the captured intent and report success to the orchestrator bus."""
        print("\n" + "─" * 65)
        print("Captured intent (from UI):")
        print(f"  Target entity    : {intent.target_entity}")
        print(f"  Business purpose : {intent.business_purpose}")
        print(f"  Dataset path     : {intent.dataset_path}")
        if intent.constraints:
            print(f"  Constraints      : {intent.constraints}")
        if intent.n_clusters_requested is not None:
            print(f"  Clusters wanted  : {intent.n_clusters_requested} (fixed)")
        else:
            print(f"  Clusters wanted  : data-driven (auto-select)")
        if intent.must_have_clusters:
            print(f"  Must-have types  : {', '.join(intent.must_have_clusters)}")
        if intent.max_cluster_size_pct is not None:
            print(
                f"  Max cluster size : {intent.max_cluster_size_pct:.0%} "
                f"(parsed from intent — overrides default 40%)"
            )
        print("─" * 65)

        self.bus.report(OrchestratorMessage(
            agent="UserInput",
            iteration=iteration,
            status="success",
            what_was_done=(
                f"Collected intent from UI: target='{intent.target_entity}', "
                f"purpose='{intent.business_purpose[:60]}'"
            ),
            what_was_not_done="Did not validate that the dataset file actually exists.",
            doubts="",
            issues=[],
            metrics={
                "target_entity": intent.target_entity,
                "purpose_length": len(intent.business_purpose),
                "has_constraints": bool(intent.constraints),
                "n_clusters_requested": intent.n_clusters_requested,
                "must_have_clusters": intent.must_have_clusters,
            },
            recommendation="proceed",
            context={"user_intent": {
                "target_entity": intent.target_entity,
                "business_purpose": intent.business_purpose,
                "dataset_path": intent.dataset_path,
                "constraints": intent.constraints,
                "n_clusters_requested": intent.n_clusters_requested,
                "must_have_clusters": intent.must_have_clusters,
            }},
        ))

    def _prompt(self, question: str, hint: str, default: str) -> str:
        """
        Display a question with optional hint and capture input.
        Falls back to `default` on EOFError (non-interactive environments).
        """
        print(f"\n{question}")
        if hint:
            print(hint)

        try:
            answer = input("  > ").strip()
            return answer if answer else default
        except (EOFError, KeyboardInterrupt):
            print(f"  [Non-interactive] Using default: {default!r}")
            return default
