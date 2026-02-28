"""
OrchestratorBus — Agent ↔ Orchestrator Communication Protocol

Contract: docs/skills/orchestrator_bus.md (status/recommendation meanings, API).

Two communication directions:

1. AGENT → ORCHESTRATOR (status reports)
   Every agent calls bus.report(OrchestratorMessage(...)) at the end of its run.
   The orchestrator logs all messages and uses them for routing decisions.

2. AGENT → ORCHESTRATOR → CLAUDE → AGENT (LLM queries)
   When an agent needs LLM reasoning, it calls bus.ask(agent, purpose, prompt).
   The orchestrator intercepts the request, calls Claude, and returns the answer.
   Agents never hold a Claude client — all LLM access is mediated by the Orchestrator.

   This means:
   - All Claude API calls are in one place (Orchestrator)
   - Agents focus on their domain expertise (sklearn, pandas, stats)
   - The Orchestrator can log, rate-limit, and route every LLM request
   - Agents ask only when genuinely stuck — they think first with their own skills
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Literal


@dataclass
class OrchestratorMessage:
    """Structured status report sent from an agent to the orchestrator."""

    agent: str
    """Name of the agent sending this message."""

    iteration: int
    """Pipeline iteration number."""

    status: Literal["success", "warning", "blocked", "failure"]
    """
    success  — agent completed its task, no issues.
    warning  — agent completed, but with caveats worth noting.
    blocked  — agent cannot proceed; orchestrator must reroute.
    failure  — unexpected exception; pipeline should halt or retry.
    """

    what_was_done: str
    """Concise description of what the agent accomplished."""

    what_was_not_done: str = ""
    """What was skipped, partial, or outside scope."""

    doubts: str = ""
    """Uncertainties the agent has about the quality of its output."""

    issues: list[str] = field(default_factory=list)
    """Specific problems found (list of strings)."""

    metrics: dict[str, Any] = field(default_factory=dict)
    """Key numeric or categorical results (e.g. n_features, silhouette)."""

    recommendation: Literal["proceed", "retry", "escalate"] = "proceed"
    """
    proceed  — orchestrator should move to the next agent.
    retry    — orchestrator should re-run this agent with adjusted params.
    escalate — trigger human checkpoint immediately.
    """

    context: dict[str, Any] = field(default_factory=dict)
    """Agent-specific payload for the orchestrator (e.g. VIF table, k scores)."""


class OrchestratorBus:
    """
    Shared message bus with two capabilities:

    1. Status reporting (report / get_log / summary_for_claude)
    2. LLM mediation  (set_llm_handler / ask)

    The Orchestrator:
      - Instantiates the bus
      - Registers the LLM handler via set_llm_handler()
      - Passes the bus to every agent constructor

    Agents:
      - Call bus.report() to send status updates
      - Call bus.ask()    to request LLM reasoning from the Orchestrator
    """

    def __init__(self) -> None:
        self._log: list[OrchestratorMessage] = []
        self._llm_handler: Callable | None = None
        self._query_log: list[dict] = []  # log of all LLM queries

    # ── LLM mediation ──────────────────────────────────────────────────────────

    def set_llm_handler(self, handler: Callable) -> None:
        """
        Register the Orchestrator's LLM handler.

        The handler signature must be:
            handler(agent: str, purpose: str, prompt: str, max_tokens: int) -> str

        Called once by the Orchestrator during __init__.
        """
        self._llm_handler = handler

    def ask(
        self,
        agent: str,
        purpose: str,
        prompt: str,
        max_tokens: int = 1024,
    ) -> str:
        """
        Request LLM reasoning from the Orchestrator.

        Parameters
        ----------
        agent : str
            Name of the calling agent (for logging).
        purpose : str
            Short description of what the LLM is being asked to do.
            Displayed in the terminal so the user can follow along.
        prompt : str
            Full prompt to send to Claude.
        max_tokens : int
            Maximum tokens in the response.

        Returns
        -------
        str — Claude's raw text response.

        Raises
        ------
        RuntimeError if no LLM handler has been registered.
        """
        if self._llm_handler is None:
            raise RuntimeError(
                "No LLM handler registered on OrchestratorBus. "
                "Call bus.set_llm_handler() before agents run."
            )

        print(f"  [{agent} → Orchestrator] Requesting LLM: {purpose}")
        result = self._llm_handler(
            agent=agent,
            purpose=purpose,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        self._query_log.append({
            "agent": agent,
            "purpose": purpose,
            "prompt_chars": len(prompt),
            "response_chars": len(result),
        })
        print(f"  [Orchestrator → {agent}] LLM response received ({len(result)} chars)")
        return result

    # ── Status reporting ───────────────────────────────────────────────────────

    def report(self, message: OrchestratorMessage) -> None:
        """Append a message to the log and print a one-line summary."""
        self._log.append(message)
        icon = {
            "success": "✓",
            "warning": "⚠",
            "blocked": "✗",
            "failure": "!!",
        }.get(message.status, "?")
        print(
            f"  [Bus] {icon} {message.agent} (iter {message.iteration}): "
            f"{message.status.upper()} — {message.what_was_done[:80]}"
        )
        if message.issues:
            for issue in message.issues:
                print(f"         Issue: {issue}")
        if message.doubts:
            print(f"         Doubt: {message.doubts}")

    def get_log(self) -> list[OrchestratorMessage]:
        """Return all status messages in chronological order."""
        return list(self._log)

    def get_log_for_agent(self, agent: str) -> list[OrchestratorMessage]:
        """Return all messages from a specific agent."""
        return [m for m in self._log if m.agent == agent]

    def last_message(self, agent: str | None = None) -> OrchestratorMessage | None:
        """Return the most recent message (optionally filtered by agent)."""
        msgs = self.get_log_for_agent(agent) if agent else self._log
        return msgs[-1] if msgs else None

    def summary_for_claude(self, last_n: int = 20) -> str:
        """Format recent messages as a readable string for Claude prompts."""
        msgs = self._log[-last_n:]
        lines = []
        for m in msgs:
            lines.append(
                f"[Iter {m.iteration}] {m.agent} → {m.status.upper()}"
                f" | done: {m.what_was_done}"
            )
            if m.issues:
                lines.append(f"   Issues: {'; '.join(m.issues)}")
            if m.doubts:
                lines.append(f"   Doubts: {m.doubts}")
            if m.metrics:
                lines.append(f"   Metrics: {m.metrics}")
            lines.append(f"   Recommendation: {m.recommendation}")
        return "\n".join(lines) if lines else "No messages yet."

    def has_hard_block(self) -> bool:
        """True if any message has status=blocked and recommendation=escalate."""
        return any(
            m.status == "blocked" and m.recommendation == "escalate"
            for m in self._log
        )

    def save_log(self, path: str = "outputs/pipeline_log.json") -> None:
        """Persist the full message log and LLM query log to JSON."""
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        serialisable = {
            "status_messages": [asdict(m) for m in self._log],
            "llm_queries": self._query_log,
        }
        with open(path, "w") as f:
            json.dump(serialisable, f, indent=2, default=str)
        print(
            f"  [Bus] Pipeline log saved → {path}  "
            f"({len(self._log)} status messages, {len(self._query_log)} LLM queries)"
        )
