#!/bin/bash
# guard_after_e3.sh — backup safety: kills campaign tmux when E3 completes IF
# /tmp/le-wm-skip-e4 exists. Belt-and-suspenders with the in-script flag check.
#
# Watches for "max_epochs=10 reached" in E3's log. The instant that appears,
# checks the pause flag. If set, kills the campaign tmux session before E4
# can launch.
#
# To disable the guard (i.e., to allow E4 to run):
#   rm /tmp/le-wm-skip-e4
#
# Then re-launch the orchestrator if needed.
# =============================================================================
set -u
cd "$(dirname "$0")"

LOG=logs/guard_after_e3.log
E3_LOG=logs/campaign/E3.log

ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "guard starting; watching $E3_LOG for E3 completion"

# Tail the E3 log; exit the loop the first time we see the completion signal
( tail -F "$E3_LOG" 2>/dev/null & echo $! > /tmp/guard_tail_pid ) | \
    grep -m1 -E "max_epochs=10.*reached|Training complete" > /dev/null

log "E3 completion detected"

# stop the tail
kill $(cat /tmp/guard_tail_pid) 2>/dev/null || true

if [ -f /tmp/le-wm-skip-e4 ]; then
    log "pause flag /tmp/le-wm-skip-e4 IS set — killing tmux campaign before E4 launches"
    sleep 2  # give the orchestrator's in-script flag check a chance to fire first
    if tmux has-session -t campaign 2>/dev/null; then
        tmux kill-session -t campaign
        log "campaign tmux killed; E4 prevented"
    else
        log "campaign tmux already gone (orchestrator self-exited via in-script check, good)"
    fi
else
    log "pause flag NOT set — letting E4 proceed normally"
fi

log "guard exiting"
