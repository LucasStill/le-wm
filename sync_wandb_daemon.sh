#!/bin/bash
# sync_wandb_daemon.sh — background loop that auto-syncs finished wandb runs.
#
# Polls every INTERVAL seconds (default 300 = 5 min). Picks up:
#   - newly finished offline runs
#   - in-progress runs that have at least one flush of new data
#
# Already-synced runs are skipped (wandb tracks state via --mark-synced).
# Safe to run alongside active training.
#
# Logs to logs/wandb_sync.log
#
# Launch:
#   tmux new -d -s wandb_sync "./sync_wandb_daemon.sh"
# Stop:
#   tmux kill-session -t wandb_sync
# =============================================================================
set -u
cd "$(dirname "$0")"
source .venv/bin/activate

mkdir -p logs
LOG=logs/wandb_sync.log
INTERVAL=${WANDB_SYNC_INTERVAL:-300}

ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

echo "[$(ts)] daemon starting (interval=${INTERVAL}s)" | tee -a "$LOG"

while true; do
    echo "[$(ts)] tick — checking for unsynced runs" >> "$LOG"
    wandb sync --sync-all --include-offline --no-include-synced --mark-synced 2>&1 \
        | grep -vE '^$|Find logs|debug-cli' >> "$LOG"
    sleep "$INTERVAL"
done
