#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
#  Downstream Benchmark Sweep — job array (one GPU per checkpoint)
#
#  Each array task evaluates one checkpoint independently on its own GPU.
#  All results land in the same OUT_DIR, named by checkpoint.
#
#  Submit:
#    sbatch eval_sweep.slurm
#
#  After all tasks complete, aggregate into CSV:
#    python aggregate_eval.py --out_dir $STABLEWM_HOME/eval/<ARRAY_JOB_ID>
#
#  ── Updating the checkpoint list ──────────────────────────────────────────
#  1. Edit the CKPTS array below.
#  2. Update --array=0-N  where N = len(CKPTS) - 1.
#
#  Task selection:
#    TASKS=""           all tasks (1, 2, 3)  — default
#    TASKS="--tasks 1"  HI probing only      — fastest
#    TASKS="--tasks 1 2"                     — skip slow forecasting
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --account=yil@h100
#SBATCH --job-name=lewm_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --array=0-2          # ← update upper bound when CKPTS changes (0-indexed)
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.out
#SBATCH -C h100

set -e

source ~/.bashrc

PROJECT_DIR=$WORK/thesis/le-wm-vanilla
cd "$PROJECT_DIR"
source .venv/bin/activate

mkdir -p logs

export PYTHONUNBUFFERED=1
export STABLEWM_HOME=/lustre/fswork/projects/rech/yil/ugy35qd/.stable_worldmodel

# ── Checkpoints ───────────────────────────────────────────────────────────────
# Format: "display_name:path/to/checkpoint.ckpt"
# Keep this list in sync with --array=0-N above.
# To switch batches: comment out the active CKPTS=(...) block and uncomment another.
# NEVER use names starting with a digit — bash forbids it.

# -- Batch: JEPA baselines (v3 / S5 variants) --
# CKPTS=(
#     "scenario3_sensor_w1_H50_v3:${STABLEWM_HOME}/scenario3_sensor_w1_H50_v3/lewm_s3_sensor_w1_H50_v3_epoch_10_object.ckpt"
#     "scenario3_sensor_w1_H50_S5:${STABLEWM_HOME}/scenario3_sensor_w1_H50_S5/lewm_s3_sensor_w1_H50_S5_epoch_10_object.ckpt"
#     "scenario3_sensor_w5_H50_S5:${STABLEWM_HOME}/scenario3_sensor_w5_H50_S5/lewm_s3_sensor_w5_H50_S5_epoch_10_object.ckpt"
# )

# -- Batch: JEPA large-stride variants --
# CKPTS=(
#     "scenario3_sensor_w10_H10_S10:${STABLEWM_HOME}/scenario3_sensor_w10_H10_S10/lewm_s3_sensor_w10_H10_S10_epoch_10_object.ckpt"
#     "scenario3_sensor_w1_H10_S10:${STABLEWM_HOME}/scenario3_sensor_w1_H10_S10/lewm_s3_sensor_w1_H10_S10_epoch_10_object.ckpt"
#     "scenario3_sensor_w1_H100_v3:${STABLEWM_HOME}/scenario3_sensor_w1_H100_v3/lewm_s3_sensor_w1_H100_v3_epoch_10_object.ckpt"
# )

# -- Batch: AE baselines --
CKPTS=(
    "ae_s3_sensor_w1:${STABLEWM_HOME}/ae_s3_sensor_w1_epoch_91_object.ckpt"
    "ae_s3_sensor_w100:${STABLEWM_HOME}/ae_s3_sensor_w100_epoch_23_object.ckpt"
    "ae_s3_sensor_w50:${STABLEWM_HOME}/ae_s3_sensor_w50_epoch_43_object.ckpt"
)

CKPT="${CKPTS[$SLURM_ARRAY_TASK_ID]}"

# ── Output — shared dir, one JSON per checkpoint ──────────────────────────────
# SLURM_ARRAY_JOB_ID is the same for all tasks, so all JSONs land together.
OUT_DIR=${STABLEWM_HOME}/eval/${SLURM_ARRAY_JOB_ID}
mkdir -p "${OUT_DIR}"

echo "========================================"
echo "Array job : $SLURM_ARRAY_JOB_ID  task $SLURM_ARRAY_TASK_ID"
echo "Node      : $SLURM_NODELIST"
echo "GPU       : $CUDA_VISIBLE_DEVICES"
echo "Checkpoint: $CKPT"
echo "Output    : $OUT_DIR"
echo "========================================"

# ── Task selection ────────────────────────────────────────────────────────────
TASKS=""                  # all tasks (1, 2, 3)
# TASKS="--tasks 1"       # Task 1 only  — fastest
# TASKS="--tasks 1 2"     # Tasks 1+2    — skip slow forecasting

# ── Probe seq_lens ────────────────────────────────────────────────────────────
# Each value trains a separate probe. "1" = fast; "1 10 50" = full sweep.
TASK1_SEQ_LENS="1 10 50"
DELTA_SEQ_LENS="1 10 50"

# ── Probe batch size ──────────────────────────────────────────────────────────
PROBE_BATCH=2048

# ── Misc ──────────────────────────────────────────────────────────────────────
EXTRA="--forecast_horizon 50 --step_size 10"

srun python -u eval_sweep.py \
    "${CKPT}" \
    --out_dir            "${OUT_DIR}" \
    --probe_batch        ${PROBE_BATCH} \
    --task1_seq_lens     ${TASK1_SEQ_LENS} \
    --delta_hi_seq_lens  ${DELTA_SEQ_LENS} \
    --no_parallel \
    --no_aggregate \
    ${TASKS} \
    ${EXTRA}

echo "========================================"
echo "Task $SLURM_ARRAY_TASK_ID complete → ${OUT_DIR}"
echo ""
echo "Once all tasks finish, aggregate with:"
echo "  python aggregate_eval.py --out_dir ${OUT_DIR}"
echo "========================================"
