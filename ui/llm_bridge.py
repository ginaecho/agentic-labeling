"""Minimal LLM bridge for the UI's live-regenerate endpoints.

The full Orchestrator is too heavy to spin up just to ask the Decision
Maker to re-name a single cluster. This helper wires the same
OrchestratorBus -> Anthropic handler that agents/orchestrator.py uses,
and constructs a PersonaNamingAgent on top of it.

All heavyweight imports (anthropic, numpy via skills/agents) are
deferred to call-time so the Flask app can boot in any environment
that has just Flask installed — only "Regenerate with hint" and
"Merge clusters" need the LLM stack.
"""
from __future__ import annotations

import json
import os
import sys
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_MODEL = 'claude-sonnet-4-6'


def _load_env_file() -> None:
    env_path = _ROOT / '.env'
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, _, v = line.partition('=')
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def make_persona_agent():
    """Construct a PersonaNamingAgent with a working LLM handler."""
    _load_env_file()
    if not os.environ.get('ANTHROPIC_API_KEY'):
        raise RuntimeError(
            'ANTHROPIC_API_KEY not set. Add it to .env or export it before '
            'using regenerate / merge in the UI.'
        )

    # Deferred imports — keep the Flask boot path lightweight
    import anthropic
    from skills.orchestrator_bus import OrchestratorBus
    from agents.persona_namer import PersonaNamingAgent

    # CRITICAL: pass event_log_path=None so this transient UI-side bus does NOT
    # truncate outputs/pipeline_events.jsonl or wipe last_upload_preview.json
    # on init. Those belong to the running pipeline; this bus is just a thin
    # LLM dispatcher for the UI's regenerate / merge / explain endpoints.
    bus = OrchestratorBus(event_log_path=None)
    client = anthropic.Anthropic()

    # Append events to the pipeline's live log so the UI's cost panel + chat
    # bubbles still see the LLM calls this UI bus makes (e.g. explain calls).
    _live_event_log = pathlib.Path('outputs/pipeline_events.jsonl')

    def _emit_event(event: str, **payload):
        if not _live_event_log.exists():
            return
        from datetime import datetime, timezone
        rec = {'event': event,
               'ts': datetime.now(timezone.utc).isoformat(timespec='seconds'),
               **payload}
        try:
            with _live_event_log.open('a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
        except OSError:
            pass

    def _handler(agent: str, purpose: str, prompt: str, max_tokens: int = 1024,
                 category: str = 'pipeline') -> str:
        import time as _time
        _emit_event('llm_call_started', agent=agent, purpose=purpose,
                    prompt_chars=len(prompt), prompt=prompt, category=category)
        t0 = _time.perf_counter()
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=max_tokens,
            system=(
                'You are the Decision Maker for a multi-agent customer '
                'segmentation pipeline. The user is fine-tuning persona '
                'names in an interactive UI; your job is to honour their '
                'hint while staying faithful to the cluster statistics shown.'
            ),
            messages=[{'role': 'user', 'content': prompt}],
        )
        elapsed = round(_time.perf_counter() - t0, 2)
        _emit_event('llm_call_finished', agent=agent, purpose=purpose,
                    input_tokens=resp.usage.input_tokens,
                    output_tokens=resp.usage.output_tokens,
                    time_s=elapsed,
                    response=resp.content[0].text,
                    category=category)
        return resp.content[0].text

    bus.set_llm_handler(_handler)
    return PersonaNamingAgent(bus), bus
