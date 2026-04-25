#!/bin/bash
# sync_peer.sh — bidirectional coordination sync with OrailixTower.
#
# Pulls coordination/orailixtower/ from the peer, then pushes our
# coordination/dragon/ to the peer. Tolerates missing SSH key; just logs
# and tries again next tick.
#
# Launch:
#   tmux new -d -s peer_sync "./sync_peer.sh"
# Stop:
#   tmux kill-session -t peer_sync
#
# Env vars:
#   PEER_USER     default: lucas
#   PEER_HOST     default: 192.168.112.35
#   PEER_REPO     default: /home/lucas/thesis/le-wm
#   SYNC_INTERVAL default: 600 (10 min)
# =============================================================================
set -u
cd "$(dirname "$0")"

PEER_USER=${PEER_USER:-lucas}
PEER_HOST=${PEER_HOST:-192.168.112.35}
PEER_REPO=${PEER_REPO:-/home/lucas/thesis/le-wm}
SYNC_INTERVAL=${SYNC_INTERVAL:-600}
LOG=logs/peer_sync.log

mkdir -p logs coordination/dragon coordination/orailixtower

ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*" >> "$LOG"; }

log "daemon starting (peer=$PEER_USER@$PEER_HOST:$PEER_REPO, interval=${SYNC_INTERVAL}s)"

while true; do
    # Pull peer's coordination/orailixtower/ → ours
    if rsync -az --mkpath --timeout=20 \
            -e "ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new" \
            "${PEER_USER}@${PEER_HOST}:${PEER_REPO}/coordination/orailixtower/" \
            coordination/orailixtower/ \
            >> "$LOG" 2>&1; then
        log "PULL ok"
    else
        log "PULL failed (peer probably not yet reachable / no SSH key)"
    fi

    # Push our coordination/dragon/ → peer
    if rsync -az --mkpath --timeout=20 \
            -e "ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new" \
            coordination/dragon/ \
            "${PEER_USER}@${PEER_HOST}:${PEER_REPO}/coordination/dragon/" \
            >> "$LOG" 2>&1; then
        log "PUSH ok"
    else
        log "PUSH failed (peer probably not yet reachable / no SSH key)"
    fi

    # After sync, refresh RESULTS.md and publish to origin/tracking if anything changed
    if [ -x ./publish_tracking.sh ]; then
        ./publish_tracking.sh >> "$LOG" 2>&1
    fi

    sleep "$SYNC_INTERVAL"
done
