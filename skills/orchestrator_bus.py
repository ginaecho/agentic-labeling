"""
OrchestratorBus — Agent ↔ Orchestrator Communication Protocol

Contract: docs/skills/orchestrator_bus.md (status/recommendation meanings, API).

Two communication directions:

1. AGENT → ORCHESTRATOR (status reports)
   Every agent calls bus.report(OrchestratorMessage(...)) at the end of its run.
   The orchestrator logs all messages and uses them for routing decisions.

2. AGENT → ORCHESTRATOR → LLM → AGENT (LLM queries)
   When an agent needs LLM reasoning, it calls bus.ask(agent, purpose, prompt).
   The orchestrator intercepts the request, calls the LLM, and returns the answer.
   Agents never hold an LLM client — all LLM access is mediated by the Orchestrator.

   This means:
   - All LLM API calls are in one place (Orchestrator)
   - Agents focus on their domain expertise (sklearn, pandas, stats)
   - The Orchestrator can log, rate-limit, and route every LLM request
   - Agents ask only when genuinely stuck — they think first with their own skills
"""
from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Literal


DEFAULT_EVENT_LOG = pathlib.Path("outputs/pipeline_events.jsonl")
MODE_FILE = pathlib.Path("outputs/pipeline_mode.json")
PENDING_DECISION = pathlib.Path("outputs/pending_decision.json")
DECISION_TIMEOUT_S = 300   # 5 min: how long to wait for user in interactive mode


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_pipeline_mode() -> str:
    """Returns 'interactive' or 'bypass' (the default)."""
    try:
        if MODE_FILE.exists():
            data = json.loads(MODE_FILE.read_text(encoding="utf-8"))
            return 'interactive' if str(data.get('mode')) == 'interactive' else 'bypass'
    except Exception:
        pass
    return 'bypass'


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

    1. Status reporting (report / get_log / summary_for_llm)
    2. LLM mediation  (set_llm_handler / ask)

    The Orchestrator:
      - Instantiates the bus
      - Registers the LLM handler via set_llm_handler()
      - Passes the bus to every agent constructor

    Agents:
      - Call bus.report() to send status updates
      - Call bus.ask()    to request LLM reasoning from the Orchestrator
    """

    def __init__(self, event_log_path: pathlib.Path | str | None = DEFAULT_EVENT_LOG) -> None:
        self._log: list[OrchestratorMessage] = []
        self._llm_handler: Callable | None = None
        self._query_log: list[dict] = []  # log of all LLM queries

        # Incremental event stream consumed by the UI (Server-Sent Events).
        # Truncated on every new bus so each pipeline run starts clean.
        self._event_log_path: pathlib.Path | None = (
            pathlib.Path(event_log_path) if event_log_path else None
        )
        self.run_id: str = _now_iso()
        if self._event_log_path is not None:
            try:
                self._event_log_path.parent.mkdir(parents=True, exist_ok=True)
                # Truncate to mark a fresh run; the UI keys off this.
                self._event_log_path.write_text("", encoding="utf-8")
                # Also clear stale per-run outputs from any prior session — these
                # would show up in the UI's cluster grid + Evidence tab as
                # outdated info. The pipeline writes fresh copies on completion.
                for stale in (
                    self._event_log_path.parent / "last_upload_preview.json",
                    self._event_log_path.parent / "pending_intent.json",
                    self._event_log_path.parent / "pending_decision.json",
                    self._event_log_path.parent / "pending_target_change.json",
                    # Per-run cluster + evidence outputs
                    self._event_log_path.parent / "personas.json",
                    self._event_log_path.parent / "cluster_profiles.json",
                    self._event_log_path.parent / "cluster_lineage.json",
                    self._event_log_path.parent / "classifier_metrics.json",
                    self._event_log_path.parent / "silhouette_curve.json",
                    self._event_log_path.parent / "persona_summary.txt",
                    self._event_log_path.parent / "persona_metrics.csv",
                    self._event_log_path.parent / "agents_conversation.txt",
                    self._event_log_path.parent / "pipeline_log.json",
                    self._event_log_path.parent / "pca_iterations.json",
                ):
                    try:
                        stale.unlink(missing_ok=True)
                    except OSError:
                        pass
                self.emit("run_started", run_id=self.run_id)
            except OSError as exc:
                print(f"  [Bus] WARNING: could not initialise event log: {exc}")
                self._event_log_path = None

    def reset_for_new_run(self) -> None:
        """Reset bus state so a subsequent .run() in the same process starts
        clean (no stale events, log messages, or LLM queries from the prior
        run leaking into the UI / cost tallies).

        Called by Orchestrator.run() at the top of each invocation. Replays
        the same setup as __init__ for the per-run state, including emitting
        a fresh run_started event with a new run_id (the UI keys off this to
        wipe its in-browser accumulators)."""
        self._log.clear()
        self._query_log.clear()
        self.run_id = _now_iso()
        if self._event_log_path is not None:
            try:
                self._event_log_path.write_text("", encoding="utf-8")
                for stale in (
                    self._event_log_path.parent / "last_upload_preview.json",
                    self._event_log_path.parent / "pending_intent.json",
                    self._event_log_path.parent / "pending_decision.json",
                    self._event_log_path.parent / "pending_target_change.json",
                    self._event_log_path.parent / "personas.json",
                    self._event_log_path.parent / "cluster_profiles.json",
                    self._event_log_path.parent / "cluster_lineage.json",
                    self._event_log_path.parent / "classifier_metrics.json",
                    self._event_log_path.parent / "silhouette_curve.json",
                    self._event_log_path.parent / "persona_summary.txt",
                    self._event_log_path.parent / "persona_metrics.csv",
                    self._event_log_path.parent / "agents_conversation.txt",
                    self._event_log_path.parent / "pipeline_log.json",
                    self._event_log_path.parent / "pca_iterations.json",
                ):
                    try:
                        stale.unlink(missing_ok=True)
                    except OSError:
                        pass
                self.emit("run_started", run_id=self.run_id)
            except OSError as exc:
                print(f"  [Bus] WARNING: reset_for_new_run failed: {exc}")

    # ── Event streaming (consumed by the UI) ───────────────────────────────────

    def emit(self, event_type: str, **payload: Any) -> None:
        """Append a single event line to the event stream JSONL.

        Used for non-agent milestones (pipeline_started, iteration_started,
        pipeline_complete, etc). Failures are swallowed so a missing/locked
        log file never breaks the pipeline.
        """
        if self._event_log_path is None:
            return
        record = {
            "event": event_type,
            "ts": _now_iso(),
            "run_id": self.run_id,
            **payload,
        }
        try:
            with self._event_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            print(f"  [Bus] WARNING: event emit failed ({event_type}): {exc}")

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
        category: str = 'pipeline',
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
            Full prompt to send to the LLM.
        max_tokens : int
            Maximum tokens in the response.

        Returns
        -------
        str — The LLM's raw text response.

        Raises
        ------
        RuntimeError if no LLM handler has been registered.
        """
        if self._llm_handler is None:
            raise RuntimeError(
                "No LLM handler registered on OrchestratorBus. "
                "Call bus.set_llm_handler() before agents run."
            )

        print(f"  [{agent} → Orchestrator] Requesting LLM: {purpose} ({category})")
        # Try to pass category; tolerate older handlers that don't accept it
        try:
            result = self._llm_handler(
                agent=agent,
                purpose=purpose,
                prompt=prompt,
                max_tokens=max_tokens,
                category=category,
            )
        except TypeError:
            result = self._llm_handler(
                agent=agent,
                purpose=purpose,
                prompt=prompt,
                max_tokens=max_tokens,
            )
        self._query_log.append({
            "agent": agent,
            "purpose": purpose,
            "prompt": prompt,
            "response": result,
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

        # Stream the same status to the UI event log
        self.emit(
            "agent_report",
            agent=message.agent,
            iteration=message.iteration,
            status=message.status,
            what_was_done=message.what_was_done,
            what_was_not_done=message.what_was_not_done,
            doubts=message.doubts,
            issues=list(message.issues),
            metrics=dict(message.metrics),
            recommendation=message.recommendation,
            context=dict(message.context),   # full evidence payload for the UI
        )

        # ── INTERACTIVE MODE: pause on warnings/blocks until the user decides ──
        # Only triggers when:
        #   1. Mode is 'interactive' (set via UI toggle, persisted to disk)
        #   2. The report has at least one issue
        # The UI shows a modal; user submits a decision via /api/decision which
        # writes outputs/pending_decision.json. We poll until that arrives, then
        # save the decision as a high-priority global_rule so subsequent agents
        # see it in their build_preferences_block() prompts.
        if message.issues and read_pipeline_mode() == 'interactive':
            self._wait_for_user_decision(message)

    def _wait_for_user_decision(self, message: 'OrchestratorMessage') -> None:
        """Pause until the user submits a decision via the UI."""
        # Make sure any prior decision file doesn't auto-resolve us
        try:
            PENDING_DECISION.unlink(missing_ok=True)
        except OSError:
            pass

        self.emit(
            "awaiting_user_decision",
            agent=message.agent,
            iteration=message.iteration,
            issues=list(message.issues),
            doubts=message.doubts,
            what_was_done=message.what_was_done,
            timeout_s=DECISION_TIMEOUT_S,
        )
        print(f"\n  [INTERACTIVE MODE] Paused after {message.agent} reported warnings.")
        print(f"  [INTERACTIVE MODE] Open the UI, choose how to handle it, and click Apply.")
        print(f"  [INTERACTIVE MODE] Timeout: {DECISION_TIMEOUT_S}s (then auto-bypass).")

        deadline = time.time() + DECISION_TIMEOUT_S
        decision = None
        try:
            while time.time() < deadline:
                if PENDING_DECISION.exists():
                    try:
                        decision = json.loads(PENDING_DECISION.read_text(encoding='utf-8'))
                    except (OSError, json.JSONDecodeError) as exc:
                        print(f"  [INTERACTIVE MODE] Bad decision file: {exc}")
                    try:
                        PENDING_DECISION.unlink(missing_ok=True)
                    except OSError:
                        pass
                    break
                time.sleep(0.6)
        except KeyboardInterrupt:
            print("\n  [INTERACTIVE MODE] Interrupted — proceeding without decision.")
            return

        if decision is None:
            print(f"  [INTERACTIVE MODE] Timed out — bypassing.")
            self.emit("user_decision_received",
                      agent=message.agent, response="(timeout — auto-bypassed)",
                      action="bypass", source="timeout")
            return

        action = decision.get('action') or 'apply'
        response = (decision.get('response') or '').strip()
        priority = decision.get('priority') or 'high'

        # Always log the decision in events for transparency
        self.emit(
            "user_decision_received",
            agent=message.agent,
            response=response,
            action=action,
            priority=priority,
        )
        print(f"  [INTERACTIVE MODE] User decision ({action}): {response[:120]}")

        # Persist as a high-priority memory rule so subsequent agent prompts see it
        if response and action != 'ignore':
            try:
                from ui.feedback_store import append as fb_append
                rule = (
                    f'Guidance for {message.agent} warning '
                    f'("{"; ".join(message.issues) if message.issues else "warning"}"): '
                    f'{response}'
                )
                fb_append({
                    'type': 'global_rule',
                    'rule': rule,
                    'priority': priority,
                })
            except Exception as _exc:  # noqa: BLE001
                print(f"  [INTERACTIVE MODE] Could not save rule: {_exc}")

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

    def summary_for_llm(self, last_n: int = 20) -> str:
        """Format recent messages as a readable string for LLM prompts."""
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
        # JSON omits full prompt/response to keep file size down; full text is in .txt log
        serialisable = {
            "status_messages": [asdict(m) for m in self._log],
            "llm_queries": [
                {k: v for k, v in q.items() if k in ("agent", "purpose", "prompt_chars", "response_chars")}
                for q in self._query_log
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2, default=str)
        print(
            f"  [Bus] Pipeline log saved → {path}  "
            f"({len(self._log)} status messages, {len(self._query_log)} LLM queries)"
        )

    def save_log_txt(self, path: str = "outputs/agents_conversation.txt") -> None:
        """Write a human-readable .txt log of agent status messages and full LLM conversations."""
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        lines = []
        lines.append("=" * 80)
        lines.append("AGENT STATUS MESSAGES")
        lines.append("=" * 80)
        for m in self._log:
            lines.append(f"\n[{m.agent}] Iteration {m.iteration}  Status: {m.status.upper()}  Recommendation: {m.recommendation}")
            lines.append(f"  Done:     {m.what_was_done}")
            if m.what_was_not_done:
                lines.append(f"  Not done: {m.what_was_not_done}")
            if m.doubts:
                lines.append(f"  Doubts:   {m.doubts}")
            for issue in m.issues:
                lines.append(f"  Issue:    {issue}")
            if m.metrics:
                lines.append(f"  Metrics:  {m.metrics}")
            lines.append("")
        lines.append("=" * 80)
        lines.append("LLM CONVERSATIONS (Agent ↔ Orchestrator)")
        lines.append("=" * 80)
        for i, q in enumerate(self._query_log, 1):
            lines.append(f"\n--- LLM call {i}: {q.get('agent', '?')} — {q.get('purpose', '?')} ---")
            lines.append("\n[PROMPT]")
            lines.append(q.get("prompt", "(none)"))
            lines.append("\n[RESPONSE]")
            lines.append(q.get("response", "(none)"))
            lines.append("")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"  [Bus] Agents conversation log saved → {path}")
