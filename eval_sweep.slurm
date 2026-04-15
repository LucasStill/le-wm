#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
#  Downstream Benchmark Sweep — Tasks 1, 2, 3
#
#  Runs eval_sweep.py on all checkpoints listed in CKPTS.
#  Produces per-checkpoint JSONs + metrics_flat.csv + task3_curves.csv
#  under $OUT_DIR.  Pass these CSVs to plot_results.py to generate figures.
#
#  Task selection (default: all):
#    --tasks 1        HI state probing only  (fastest, ~10 min/ckpt)
#    --tasks 1 2      HI probing + velocity  (~20 min/ckpt)
#    --tasks 1 2 3    All incl. forecasting  (~60 min/ckpt, default)
#
#  Submit: sbatch eval_sweep.slurm
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --account=yil@h100
#SBATCH --job-name=lewm_eval_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_sweep_%j.out
#SBATCH --error=logs/eval_sweep_%j.out
#SBATCH -C h100

set -e

source ~/.bashrc

PROJECT_DIR=$WORK/thesis/le-wm-vanilla
cd "$PROJECT_DIR"
source .venv/bin/activate

mkdir -p logs

echo "========================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "GPU      : $CUDA_VISIBLE_DEVICES"
echo "Dir      : $(pwd)"
echo "========================================"

export PYTHONUNBUFFERED=1
export STABLEWM_HOME=/lustre/fswork/projects/rech/yil/ugy35qd/.stable_worldmodel

# ── Checkpoints ───────────────────────────────────────────────────────────────
# Format: "display_name:/absolute/path/to/checkpoint.ckpt"
# AR-LSTM checkpoints are at epoch 20 (training still in progress).
# Re-run this script pointing at later epoch files once training completes.
CKPTS=(
    "ar_lstm_h1:${STABLEWM_HOME}/turbofan_ar_lstm_h1_hd256_l2/ar_lstm_h1_hd256_l2_epoch_20_object.ckpt"
    "ar_lstm_h10:${STABLEWM_HOME}/turbofan_ar_lstm_h10_hd256_l2/ar_lstm_h10_hd256_l2_epoch_20_object.ckpt"
    "ar_lstm_h50:${STABLEWM_HOME}/turbofan_ar_lstm_h50_hd256_l2/ar_lstm_h50_hd256_l2_epoch_21_object.ckpt"
    "jepa_w1_h50:${STABLEWM_HOME}/turbofan_v6_sensor_sl1_w1_b512_pl3_H50/lewm_turbo_v6_sensor_sl1_w1_H50_epoch_100_object.ckpt"
    "jepa_w1_h100:${STABLEWM_HOME}/turbofan_v6_sensor_sl1_w1_b512_pl3_H100/lewm_turbo_v6_sensor_sl1_w1_H100_epoch_94_object.ckpt"
    "jepa_w50_h1:${STABLEWM_HOME}/turbofan_v6_sensor_sl1_w50_b512_pl3_H1/lewm_turbo_v6_sensor_sl1_w50_H1_epoch_100_object.ckpt"
    "jepa_w50_h50:${STABLEWM_HOME}/turbofan_v6_sensor_sl1_w50_b512_pl3_H50/lewm_turbo_v6_sensor_sl1_w50_H50_epoch_100_object.ckpt"
)

# ── Output ────────────────────────────────────────────────────────────────────
OUT_DIR=${STABLEWM_HOME}/eval/${SLURM_JOB_ID}

# ── Task selection ────────────────────────────────────────────────────────────
# Uncomment one line to restrict which tasks run:
TASKS=""                  # all tasks (1, 2, 3)
# TASKS="--tasks 1"       # Task 1 only  — fastest, ~10 min/ckpt
# TASKS="--tasks 1 2"     # Tasks 1+2    — skip slow forecasting

# ── Misc flags ────────────────────────────────────────────────────────────────
EXTRA="--forecast_horizon 50 --step_size 10"

echo "Checkpoints (${#CKPTS[@]}):"
for c in "${CKPTS[@]}"; do echo "  $c"; done
echo "Output : ${OUT_DIR}"
echo "Tasks  : ${TASKS:-all}"
echo "========================================"

srun python -u eval_sweep.py \
    "${CKPTS[@]}" \
    --out_dir  "${OUT_DIR}" \
    --no_parallel \
    ${TASKS} \
    ${EXTRA}

echo "========================================"
echo "Sweep complete."
echo ""
echo "Generate figures:"
echo "  python plot_results.py \\"
echo "      --csv    ${OUT_DIR}/metrics_flat.csv \\"
echo "      --curves ${OUT_DIR}/task3_curves.csv \\"
echo "      --out_dir figures/sweep_${SLURM_JOB_ID}"
echo "========================================"
