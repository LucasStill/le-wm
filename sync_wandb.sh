#!/bin/bash
# sync_wandb.sh — one-shot: sync all unsynced offline wandb runs to the cloud.
#
# Idempotent. Safe to run while training is in progress (in-flight runs get
# their current snapshot synced; a subsequent call picks up new data).
# Already-synced runs are skipped via wandb's --no-include-synced + --mark-synced.
#
# Usage:
#   ./sync_wandb.sh
# =============================================================================
set -u
cd "$(dirname "$0")"
source .venv/bin/activate

ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
echo "[$(ts)] sync starting"

wandb sync --sync-all --include-offline --no-include-synced --mark-synced 2>&1 \
    | grep -vE '^$|Find logs|debug-cli'

echo "[$(ts)] sync complete"
