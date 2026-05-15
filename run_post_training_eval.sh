#!/bin/bash
# =============================================================================
# run_post_training_eval.sh — RSSM post-training eval pipeline
#
# Triggered automatically once "Training complete." appears in the live tee'd
# log. Runs the full RSSM eval suite to match what AR-LSTM/JEPA have:
#   1. Multi-seed Task 1 (TransformerProbe) on test + test_hard, sl=1 and 10.
#   2. Custom Task 3 (latent forecasting) on test + test_hard.
#   3. Per-archetype Task 1 breakdown on test + test_hard.
#   4. Counterfactual fidelity vs the simulator (RSSM rollout).
#   5. Render RESULTS_RSSM.md with side-by-side comparisons.
#
# Manual usage:
#   ./run_post_training_eval.sh                        # autodetect last epoch
#   CKPT_EPOCH=10 ./run_post_training_eval.sh          # force a specific epoch
#   SKIP_CF=1 ./run_post_training_eval.sh              # skip slow CF pass
# =============================================================================
set -uo pipefail

REPO=/home/lthil/thesis/le-wm
cd "$REPO"
source .venv/bin/activate
export STABLEWM_HOME=${STABLEWM_HOME:-/home/lthil/.stable_worldmodel}
export PYTHONUNBUFFERED=1

CKPT_DIR="${CKPT_DIR_OVERRIDE:-$STABLEWM_HOME/rssm_scenario4_T64_S32x32_D512}"
CKPT_PREFIX="${CKPT_PREFIX_OVERRIDE:-rssm_s4_T64_S32x32_D512}"
OUT_DIR="${OUT_DIR_OVERRIDE:-$REPO/eval_results/rssm_s4}"
CF_DIR="${CF_DIR_OVERRIDE:-$REPO/eval_results/counterfactual_rssm}"
RESULTS_MD="${RESULTS_OVERRIDE:-$REPO/RESULTS_RSSM.md}"
LOG="$REPO/logs/post_training_eval_$(date -u +%Y%m%d_%H%M).log"

mkdir -p "$OUT_DIR" "$CF_DIR"

if [[ -n "${CKPT_EPOCH:-}" ]]; then
    CKPT="$CKPT_DIR/${CKPT_PREFIX}_epoch_${CKPT_EPOCH}_object.ckpt"
else
    # Pick the highest epoch number available.
    CKPT=$(ls "$CKPT_DIR"/${CKPT_PREFIX}_epoch_*_object.ckpt 2>/dev/null \
           | sed 's/.*epoch_\([0-9]*\)_object.*/\1\t&/' \
           | sort -n | tail -1 | cut -f2-)
fi

if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
    echo "[FAIL] No RSSM checkpoint found under $CKPT_DIR" | tee -a "$LOG"
    ls -la "$CKPT_DIR" 2>&1 | tee -a "$LOG" || true
    exit 1
fi

run_step () {
    local name="$1"; shift
    echo "" | tee -a "$LOG"
    echo "──────────────────────────────────────────────────" | tee -a "$LOG"
    echo "[$name] $(date -u)" | tee -a "$LOG"
    echo "──────────────────────────────────────────────────" | tee -a "$LOG"
    "$@" 2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        echo "[$name] FAILED (rc=$rc) — continuing" | tee -a "$LOG"
    fi
    return 0
}

echo "==========================================================" | tee -a "$LOG"
echo "RSSM post-training evaluation"  | tee -a "$LOG"
echo "Time     : $(date -u)"          | tee -a "$LOG"
echo "Checkpoint: $CKPT"              | tee -a "$LOG"
echo "Output dir: $OUT_DIR"           | tee -a "$LOG"
echo "==========================================================" | tee -a "$LOG"

# ── (1) Multi-seed Task 1 — headline metric ──────────────────────────────
# Match the AR-LSTM/JEPA multi-seed pipeline: 6 probe seeds × {sl=1, sl=10}
# on each of test and test_hard.
rm -f "$OUT_DIR/multiseed_results.csv"   # fresh run
run_step "task1-multiseed/test" python baselines/rssm/multiseed_eval_rssm.py \
    --ckpt "$CKPT" \
    --hdf5 "$STABLEWM_HOME/turbosens2_test.h5" \
    --label "RSSM_te" \
    --seeds 0 1 2 3 4 5 \
    --seq-lens 1 \
    --out-csv "$OUT_DIR/multiseed_results.csv"

run_step "task1-multiseed/test_hard" python baselines/rssm/multiseed_eval_rssm.py \
    --ckpt "$CKPT" \
    --hdf5 "$STABLEWM_HOME/turbosens2_test_hard.h5" \
    --label "RSSM_th" \
    --seeds 0 1 2 3 4 5 \
    --seq-lens 1 \
    --out-csv "$OUT_DIR/multiseed_results.csv"

# ── (2) Custom Task 3 — latent forecasting via RSSM imagination ──────────
run_step "task3/test" python baselines/rssm/eval_rssm.py \
    --ckpt "$CKPT" \
    --data "$STABLEWM_HOME/turbosens2_test.h5" \
    --out  "$OUT_DIR/task_results_test.json" \
    --history_size 16 --horizon 50 --n_samples 500

run_step "task3/test_hard" python baselines/rssm/eval_rssm.py \
    --ckpt "$CKPT" \
    --data "$STABLEWM_HOME/turbosens2_test_hard.h5" \
    --out  "$OUT_DIR/task_results_test_hard.json" \
    --history_size 16 --horizon 50 --n_samples 500

# ── (3) Per-archetype Task 1 breakdown ────────────────────────────────────
rm -f "$OUT_DIR/per_archetype.csv"
run_step "per-archetype/test" python baselines/rssm/per_archetype_rssm.py \
    --ckpt "$CKPT" \
    --hdf5 "$STABLEWM_HOME/turbosens2_test.h5" \
    --label "RSSM_te" --seq-lens 1 \
    --out-csv "$OUT_DIR/per_archetype.csv"

run_step "per-archetype/test_hard" python baselines/rssm/per_archetype_rssm.py \
    --ckpt "$CKPT" \
    --hdf5 "$STABLEWM_HOME/turbosens2_test_hard.h5" \
    --label "RSSM_th" --seq-lens 1 \
    --out-csv "$OUT_DIR/per_archetype.csv"

# ── (4) Counterfactual fidelity vs the simulator ─────────────────────────
if [[ -z "${SKIP_CF:-}" ]]; then
    run_step "counterfactual" python counterfactual_fidelity_rssm.py \
        --ckpts "RSSM:$CKPT" \
        --sim_h5 "$STABLEWM_HOME/turbosens2_train_sensors.h5" \
        --eval_h5 "$STABLEWM_HOME/turbosens2_test.h5" \
        --out_dir "$CF_DIR" \
        --n_episodes 30 --horizon 100 \
        --actions 0 1 2 3 4 5 6 \
        --branch_fracs 0.25 0.5 0.75
fi

# ── (5) Render the comparison report ──────────────────────────────────────
run_step "render-report" python "$REPO/baselines/rssm/render_results.py" \
    --rssm_test       "$OUT_DIR/task_results_test.json" \
    --rssm_test_hard  "$OUT_DIR/task_results_test_hard.json" \
    --multiseed_csv   "$OUT_DIR/multiseed_results.csv" \
    --per_arch_csv    "$OUT_DIR/per_archetype.csv" \
    --cf_summary      "$CF_DIR/summary.json" \
    --out             "$RESULTS_MD"

echo "" | tee -a "$LOG"
echo "==========================================================" | tee -a "$LOG"
echo "Eval pipeline complete: $LOG"   | tee -a "$LOG"
echo "Report: $REPO/RESULTS_RSSM.md"  | tee -a "$LOG"
echo "==========================================================" | tee -a "$LOG"
