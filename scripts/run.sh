#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$PROJECT_ROOT/logs/run_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"
source .venv/bin/activate

PID_FILE="$PROJECT_ROOT/logs/run.pid"

nohup python scripts/htm_runner.py "$@" > "$LOG_FILE" 2>&1 &
PID=$!
echo $PID > "$PID_FILE"
(wait $PID 2>/dev/null; rm -f "$PID_FILE") &
echo "PID: $PID  |  tail -f $LOG_FILE"
