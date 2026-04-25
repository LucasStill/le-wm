#!/bin/bash
# =============================================================================
# run_experiments.sh — sequential H/S sweep campaign for Scenario 4 (dragon)
#
# Runs inside tmux session `campaign`. Waits for current `s4` session to end,
# then executes E2 → E3 → E4 in order (see logs/experiment_plan.md).
#
# Each experiment writes:
#   logs/campaign/<run_name>.log       full tee'd output
#   logs/campaign/<run_name>.status    one-line status (started/done/oom/nan)
#
# Launch with:
#   tmux new -d -s campaign "cd /home/lthil/thesis/le-wm && ./run_experiments.sh \
#       2>&1 | tee logs/campaign/orchestrator.log"
# =============================================================================
set -u  # NOT set -e: we want to continue if one run fails

cd "$(dirname "$0")"
source .venv/bin/activate

CAMPAIGN_DIR="logs/campaign"
mkdir -p "$CAMPAIGN_DIR"

log()    { printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }
status() { printf '%s\n' "$2" > "$CAMPAIGN_DIR/$1.status"; }

# ── 1. Wait for current s4 session to finish ────────────────────────────────
log "campaign orchestrator starting; waiting for tmux session 's4' to end"
while tmux has-session -t s4 2>/dev/null; do
    sleep 30
done
log "s4 session ended — E1 is done; beginning campaign"

# sanity: GPU should be idle for ~60s before we touch it
sleep 60
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

# ── helper: run one experiment in the current shell, tee logs ───────────────
#   args: tag   H   S   P   BS   ACCUM   MAX_EPOCHS
# Sets LAST_EXIT, writes .status based on pattern matching in the log.
run_one() {
    local tag="$1" H="$2" S="$3" P="$4" bs="$5" accum="$6" me="$7"
    local logfile="$CAMPAIGN_DIR/${tag}.log"

    log "==== START $tag  H=$H S=$S P=$P bs=$bs accum=$accum epochs=$me ===="
    status "$tag" "started $(date -u +%H:%MZ) H=$H S=$S P=$P bs=$bs accum=$accum"

    WIN_SIZE=1 HISTORY_LEN="$H" H_STEP="$S" NUM_PREDS="$P" \
        BATCH_SIZE="$bs" NUM_WORKERS=4 MAX_EPOCHS="$me" ACCUM_GRAD="$accum" \
        WANDB_MODE=offline \
        ./train_lewm_scenario4_dragon.sh 2>&1 | tee "$logfile"
    LAST_EXIT=${PIPESTATUS[0]}

    if grep -qE "OutOfMemoryError|CUDA out of memory" "$logfile"; then
        log "XX $tag OOM'd"
        status "$tag" "OOM"
        return 1
    elif grep -qE "NaN|nan.*loss|loss.*nan" "$logfile" | head -1; then
        # loose check — only set if truly bad. hi_probe can log NaN Pearson legitimately.
        if grep -qE "Loss is NaN|training_step returned NaN" "$logfile"; then
            log "XX $tag NaN training loss"
            status "$tag" "NaN"
            return 1
        fi
    fi

    if [ "$LAST_EXIT" -ne 0 ]; then
        log "XX $tag exited with code $LAST_EXIT"
        status "$tag" "failed exit=$LAST_EXIT"
        return 1
    fi

    log "OK $tag finished cleanly"
    status "$tag" "done $(date -u +%H:%MZ)"
    return 0
}

# ── 2. E2 probe: H=32 S=1 P=4 bs=256 accum=2 (1 epoch, catches OOM) ─────────
if ! run_one "E2_probe" 32 1 4 256 2 1; then
    log "E2 probe failed; trying smaller batch (bs=128 accum=4)"
    if ! run_one "E2_probe_fallback" 32 1 4 128 4 1; then
        log "!! E2 cannot fit; skipping E2"
        status "E2" "skipped (can't fit)"
    else
        log "E2 fallback probe OK — using bs=128 accum=4 for E2 training"
        run_one "E2" 32 1 4 128 4 10
    fi
else
    log "E2 probe OK — launching 10-epoch E2 training"
    run_one "E2" 32 1 4 256 2 10
fi

# ── 3. E3: H=16 S=2 P=4 bs=512 (probe + full, in one 10-epoch run) ──────────
# Memory profile close to E1 (same H=16 tokens, slightly larger seq due to stride).
# If worried, could add a probe step, but E1 already validated H=16 with bs=512.
# Adding a 1-epoch probe step anyway to get the epoch-time measurement.
if ! run_one "E3_probe" 16 5 4 512 1 1; then
    log "E3 probe failed (unexpected); trying bs=256 accum=2"
    if ! run_one "E3_probe_fallback" 16 5 4 256 2 1; then
        log "!! E3 cannot fit; skipping"
        status "E3" "skipped (can't fit)"
    else
        run_one "E3" 16 5 4 256 2 10
    fi
else
    run_one "E3" 16 5 4 512 1 10
fi

# ── 4. E4: H=32 S=2 P=4, same resource profile as E2 ────────────────────────
# Reuse whatever bs/accum worked for E2.
if [ -f "$CAMPAIGN_DIR/E2.status" ] && grep -q "^done" "$CAMPAIGN_DIR/E2.status"; then
    # Detect E2's actual bs from its log
    E2_BS=$(grep -m1 "loader.batch_size=" "$CAMPAIGN_DIR/E2.log" | grep -oP 'batch_size=\K[0-9]+' || echo 256)
    E2_ACCUM=$(grep -m1 "accumulate_grad_batches=" "$CAMPAIGN_DIR/E2.log" | grep -oP 'accumulate_grad_batches=\K[0-9]+' || echo 2)
    log "E4 will use bs=$E2_BS accum=$E2_ACCUM (same as E2)"
    run_one "E4" 32 5 4 "$E2_BS" "$E2_ACCUM" 10
else
    log "E2 did not complete cleanly — skipping E4 to be safe"
    status "E4" "skipped (E2 did not complete)"
fi

log "==== CAMPAIGN COMPLETE ===="
log "Status summary:"
for f in "$CAMPAIGN_DIR"/*.status; do
    printf '  %-30s %s\n' "$(basename "${f%.status}")" "$(cat "$f")"
done
