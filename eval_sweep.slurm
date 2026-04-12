#!/bin/bash
#SBATCH --account=yil@h100
#SBATCH --job-name=lewm_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.out
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

# ── Checkpoints to evaluate ───────────────────────────────────────────────────
# Add or remove lines here; format is   name:/path/to/checkpoint.ckpt
CKPTS=(
    "v6_sensor_sl1_w1:${STABLEWM_HOME}/turbofan_v6_sensor_sl1_w1/lewm_turbo_v6_sensor_sl1_w1_epoch_100_object.ckpt"
    "v6_sensor_sl1_w10:${STABLEWM_HOME}/turbofan_v6_sensor_sl1_w10/lewm_turbo_v6_sensor_sl1_w10_epoch_100_object.ckpt"
)

# ── Output dir (one sub-folder per job) ───────────────────────────────────────
OUT_DIR=${STABLEWM_HOME}/eval/${SLURM_JOB_ID}

# ── Optional flags ────────────────────────────────────────────────────────────
# --skip_task3        skip the slow autoregressive rollout
# --forecast_horizon  max τ for Task 3 (default 50)
EXTRA_FLAGS="--forecast_horizon 50"

echo "Checkpoints:"
for c in "${CKPTS[@]}"; do echo "  $c"; done
echo "Output: ${OUT_DIR}"
echo "========================================"

srun python -u eval_sweep.py \
    "${CKPTS[@]}" \
    --out_dir "${OUT_DIR}" \
    --no_parallel \
    ${EXTRA_FLAGS}

echo "========================================"
echo "Eval complete."
echo "Results : ${OUT_DIR}"
echo "Logs    : logs/eval_${SLURM_JOB_ID}.out"
echo "========================================"
