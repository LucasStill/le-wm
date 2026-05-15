#!/bin/bash
# =============================================================================
# run_overnight_chain.sh — auto-fires after L_big finishes overnight.
#
# Sequence (all GPU-bound, run sequentially to avoid contention):
#   1. eval_sweep --tasks 1 on L_big × {test, test_hard}  (~20 min)
#   2. T3 per-archetype on L1 × test_hard                  (~15 min)
#   3. T1 multi-task eval_sweep --tasks 1 2 on L1+L2 × {test, test_hard} (~1.5h)
#      (skipping task 3 unless cheap — forecasting is slow)
#   4. T4 multi-seed sanity (raw_sensors + random_encoder) × {test, test_hard}
#      × {sl 1, 10, 50}, 5 seeds                           (~2.5h)
#   5. T4 multi-seed L1 + L2 × {test, test_hard} × sl=1, 5 seeds (~30 min)
#
# Total ~5h after L_big finishes. So if L_big lands ~09:30 UTC, this chain
# finishes ~14:30 UTC → before the user wakes up if they sleep in.
#
# Usage:
#   tmux new -d -s overnight "./run_overnight_chain.sh"
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*"; }

LBIG_RUN_DIR=/home/lucas/.stable_worldmodel/ar_lstm_scenario4_w4_H32_S1_P4_h256l2
LBIG_NAME=ar_lstm_s4_w4_H32_S1_P4_h256l2
L1_CKPT=/home/lucas/.stable_worldmodel/ar_lstm_scenario4_w1_H16_S1_P4_h256l2/ar_lstm_s4_w1_H16_S1_P4_h256l2_epoch_10_object.ckpt
L2_CKPT=/home/lucas/.stable_worldmodel/ar_lstm_scenario4_w1_H32_S1_P4_h256l2/ar_lstm_s4_w1_H32_S1_P4_h256l2_epoch_10_object.ckpt
TEST_H5=/home/lucas/.stable_worldmodel/turbosens2_test.h5
HARD_H5=/home/lucas/.stable_worldmodel/turbosens2_test_hard.h5

CONFIG=/home/lucas/.stable_worldmodel/ar_lstm_scenario4_w1_H16_S1_P4_h256l2/config.yaml

# ── 0. wait for L_big ────────────────────────────────────────────────────────
log "waiting for L_big to finish (poll for train_ar_lstm_scenario4_orailix.sh process to exit)"
while pgrep -f 'train_ar_lstm_scenario4_orailix.sh' > /dev/null; do
    sleep 60
done
log "L_big process gone."

# Extra grace period in case checkpoint saving is still flushing
sleep 30

# Sanity check the L_big ckpt actually exists
LBIG_CKPT=$LBIG_RUN_DIR/${LBIG_NAME}_epoch_10_object.ckpt
if [ ! -f "$LBIG_CKPT" ]; then
    log "WARNING — expected L_big ckpt not found at $LBIG_CKPT. Listing run dir:"
    ls -la $LBIG_RUN_DIR/ 2>&1 | tail -10
    log "Skipping step 1; continuing with the rest of the chain."
    LBIG_CKPT=""
fi

# ── 1. eval_sweep on L_big × {test, test_hard} ──────────────────────────────
if [ -n "$LBIG_CKPT" ]; then
    log "=== STEP 1: eval_sweep on L_big × test ==="
    python eval_sweep.py --tasks 1 --no_parallel \
        --hdf5 $TEST_H5 --out_dir eval_results/all_ckpts_test_Lbig \
        Lbig_te:$LBIG_CKPT 2>&1 || log "step 1 (Lbig test) failed but continuing"

    log "=== STEP 1b: eval_sweep on L_big × test_hard ==="
    python eval_sweep.py --tasks 1 --no_parallel \
        --hdf5 $HARD_H5 --out_dir eval_results/all_ckpts_test_hard_Lbig \
        Lbig_th:$LBIG_CKPT 2>&1 || log "step 1b (Lbig test_hard) failed but continuing"
fi

# ── 2. T3 per-archetype on L1 × test_hard ───────────────────────────────────
log "=== STEP 2: T3 per-archetype on L1 × test_hard ==="
python -u baselines/ar_lstm/per_archetype_diag.py \
    --ckpt $L1_CKPT --hdf5 $HARD_H5 --label L1_th \
    --seq-lens 1 10 \
    --out-csv logs/per_archetype_results.csv 2>&1 || log "step 2 failed but continuing"

# ── 3. T1 multi-task eval_sweep on L1 + L2 × {test, test_hard} ──────────────
# Only tasks 1 + 2 (delta-HI). Skip task 3 (forecasting) — slow + low marginal
# value for the dataset paper.
log "=== STEP 3: T1 multi-task eval_sweep (tasks 1, 2) — L1+L2 × test ==="
python eval_sweep.py --tasks 1 2 --no_parallel \
    --hdf5 $TEST_H5 --out_dir eval_results/L_multi_test \
    L1_te_multi:$L1_CKPT  L2_te_multi:$L2_CKPT 2>&1 || log "step 3 (test) failed but continuing"

log "=== STEP 3b: T1 multi-task eval_sweep — L1+L2 × test_hard ==="
python eval_sweep.py --tasks 1 2 --no_parallel \
    --hdf5 $HARD_H5 --out_dir eval_results/L_multi_test_hard \
    L1_th_multi:$L1_CKPT  L2_th_multi:$L2_CKPT 2>&1 || log "step 3b (test_hard) failed but continuing"

# ── 4. T4 multi-seed sanity baselines ────────────────────────────────────────
log "=== STEP 4: T4 multi-seed sanity baselines (5 seeds × 2 modes × 2 datasets × sl 1/10/50) ==="
log "  (seed=1 produces NaN per dragon's bug report — using {0, 2, 3, 4, 5} instead)"
rm -f logs/sanity_baselines_multiseed.csv
for SEED in 0 2 3 4 5; do
    for HDF5 in $TEST_H5 $HARD_H5; do
        if [ "$HDF5" = "$TEST_H5" ]; then SUFFIX="te_s${SEED}"; else SUFFIX="th_s${SEED}"; fi
        log "  -- seed=$SEED hdf5=$(basename $HDF5) --"
        python -u baselines/ar_lstm/sanity_baselines.py \
            --hdf5 $HDF5 --config $CONFIG \
            --modes raw_sensors random_encoder \
            --seq-lens 1 10 50 \
            --seed $SEED --label-suffix $SUFFIX \
            --out-csv logs/sanity_baselines_multiseed.csv 2>&1 || log "  seed=$SEED hdf5=$(basename $HDF5) failed, continuing"
    done
done

# ── 5. T4 multi-seed L1 + L2 calibrated (sl=1) ──────────────────────────────
log "=== STEP 5: T4 multi-seed L1 + L2 × {test, test_hard} × sl=1 (5 seeds) ==="
rm -f logs/multiseed_l1l2_results.csv
for CKPT_LABEL in "L1_te:$L1_CKPT:$TEST_H5" "L1_th:$L1_CKPT:$HARD_H5" \
                  "L2_te:$L2_CKPT:$TEST_H5" "L2_th:$L2_CKPT:$HARD_H5"; do
    LABEL=$(echo $CKPT_LABEL | cut -d: -f1)
    CKPT=$(echo $CKPT_LABEL | cut -d: -f2)
    HDF5=$(echo $CKPT_LABEL | cut -d: -f3)
    log "  -- $LABEL --"
    python -u baselines/ar_lstm/multiseed_l1l2.py \
        --ckpt $CKPT --hdf5 $HDF5 --label $LABEL \
        --seeds 0 2 3 4 5 \
        --seq-lens 1 \
        --out-csv logs/multiseed_l1l2_results.csv 2>&1 || log "  $LABEL failed, continuing"
done

log "=== OVERNIGHT CHAIN ALL DONE ==="
exec bash
