#!/bin/bash
# =============================================================================
# run_experiments_v2.sh — re-runs E3 and E4 with corrected batch sizes after
# the v1 orchestrator hit OOMs. Root cause: VRAM scales with num_steps (raw
# frames per sample), not just H tokens. Stride S=5 quintuples num_steps so
# the effective sequence the model processes is much larger than H suggests.
#
# Empirical model from E1/E2 measurements:
#   VRAM_GB ≈ 0.00146 * batch_size * num_steps
#   28 GB safety target → batch_size * num_steps < 19,178
#
# E3: H=16 S=5 P=4 → num_steps=96 → bs=128 accum=4 (≈18 GB)
# E4: H=32 S=5 P=4 → num_steps=176 → bs=64 accum=8 (≈16 GB)
#
# Both keep effective batch = 512 via gradient accumulation, matching E1/E2.
# =============================================================================
set -u

cd "$(dirname "$0")"
source .venv/bin/activate

CAMPAIGN_DIR="logs/campaign"
mkdir -p "$CAMPAIGN_DIR"

log()    { printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }
status() { printf '%s\n' "$2" > "$CAMPAIGN_DIR/$1.status"; }

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
    fi
    if grep -qE "Loss is NaN|training_step returned NaN" "$logfile"; then
        log "XX $tag NaN training loss"
        status "$tag" "NaN"
        return 1
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

# Wait for any current python training to finish (defensive)
while pgrep -f "train.py" > /dev/null 2>&1; do
    sleep 30
done
log "no training running; GPU should be free"
sleep 30
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

# ── E3: H=16 S=5 — probe first, then 10 epochs ───────────────────────────
if ! run_one "E3_probe_v2" 16 5 4 128 4 1; then
    log "E3 probe still failing; trying bs=64 accum=8"
    if ! run_one "E3_probe_v2_fallback" 16 5 4 64 8 1; then
        log "!! E3 cannot fit at any bs; skipping"
        status "E3" "skipped (can't fit)"
    else
        run_one "E3" 16 5 4 64 8 10
    fi
else
    run_one "E3" 16 5 4 128 4 10
fi

# ── E4: H=32 S=5 — bs=64 accum=8 (predicted to fit at ~16 GB) ─────────────
if ! run_one "E4_probe_v2" 32 5 4 64 8 1; then
    log "E4 probe still failing; trying bs=32 accum=16"
    if ! run_one "E4_probe_v2_fallback" 32 5 4 32 16 1; then
        log "!! E4 cannot fit at any bs; skipping"
        status "E4" "skipped (can't fit)"
    else
        run_one "E4" 32 5 4 32 16 10
    fi
else
    run_one "E4" 32 5 4 64 8 10
fi

log "==== CAMPAIGN v2 COMPLETE ===="
log "Status summary:"
for f in "$CAMPAIGN_DIR"/*.status; do
    printf '  %-30s %s\n' "$(basename "${f%.status}")" "$(cat "$f")"
done
