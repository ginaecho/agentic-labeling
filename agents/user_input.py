"""
UserInputAgent — Pipeline Entry Point

Contract: docs/agents/user_input.md (role, I/O, protocol, retry behaviour).

Collects and validates the user's clustering intent before any computation.
Asks two required questions:
  1. What entity are you clustering? (target_entity)
  2. What is the business purpose?   (business_purpose)

And two optional ones:
  3. Where is your dataset?           (dataset_path — defaults to config)
  4. Any constraints?                 (constraints — free text)

If the business_purpose answer is too vague (< 20 chars), the agent asks
one clarifying follow-up. If running non-interactively (EOFError), it falls
back to defaults from config and reports doubts="running with defaults".
"""
from __future__ import annotations

from dataclasses import dataclass

from skills.orchestrator_bus import OrchestratorBus, OrchestratorMessage

DEFAULT_DATASET_PATH = "data/processed/customer_features.parquet"


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
        Interactively collect the user's clustering intent.

        Returns
        -------
        UserIntent dataclass with captured values.
        """
        print("\n" + "=" * 65)
        print("AGENTIC CLUSTERING PIPELINE — Intent Collection")
        print("=" * 65)
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

        # ── Summary ───────────────────────────────────────────────────────────
        intent = UserIntent(
            target_entity=target_entity.strip() or "customers",
            business_purpose=business_purpose.strip(),
            dataset_path=dataset_path,
            constraints=constraints.strip(),
        )

        print("\n" + "─" * 65)
        print("Captured intent:")
        print(f"  Target entity    : {intent.target_entity}")
        print(f"  Business purpose : {intent.business_purpose}")
        print(f"  Dataset path     : {intent.dataset_path}")
        if intent.constraints:
            print(f"  Constraints      : {intent.constraints}")
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
            },
            recommendation="proceed",
            context={"user_intent": {
                "target_entity": intent.target_entity,
                "business_purpose": intent.business_purpose,
                "dataset_path": intent.dataset_path,
                "constraints": intent.constraints,
            }},
        ))

        return intent

    # ── Helpers ───────────────────────────────────────────────────────────────

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
