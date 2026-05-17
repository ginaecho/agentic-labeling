#!/usr/bin/env bash
# Launch the convergence loop fully detached so it survives terminal close
# on macOS. setsid is not available on macOS — we use `nohup` + `&` plus
# `disown` to drop the job from the shell's job table.
#
# Usage:
#   experiments/run_detached.sh                 # 3h budget, default intent
#   experiments/run_detached.sh --max-hours 6   # any convergence_loop.py flag
#
# Monitor:
#   tail -f experiments/runs/run.log
#   cat experiments/runs/run_history.jsonl
#
# Stop:
#   kill "$(cat experiments/runs/current.pid)"

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p experiments/runs
LOG="experiments/runs/run.log"

if [[ -f experiments/runs/current.pid ]] && \
   kill -0 "$(cat experiments/runs/current.pid)" 2>/dev/null; then
    echo "Already running: pid=$(cat experiments/runs/current.pid)"
    echo "Stop it first:  kill \$(cat experiments/runs/current.pid)"
    exit 1
fi

echo "[detach] launching convergence_loop.py — logging to $LOG"
nohup python -m experiments.convergence_loop "$@" \
    >> "$LOG" 2>&1 &
disown $!

sleep 1
if [[ -f experiments/runs/current.pid ]]; then
    echo "[detach] started — pid=$(cat experiments/runs/current.pid)"
else
    echo "[detach] launched (pid file not yet written — check $LOG)"
fi
echo "[detach] tail -f $LOG    to follow progress"
