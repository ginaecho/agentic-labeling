"""Interactive web UI for fine-tuning cluster personas.

The UI is launched after the multi-agent pipeline has produced
outputs/personas.json. It lets the user edit, regenerate (with an LLM
hint), and merge clusters. Every action is logged to
outputs/user_feedback_log.jsonl so the agent system can learn from
prior user preferences on subsequent runs.
"""
