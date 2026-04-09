#!/bin/bash
#SBATCH --account=yil@h100
#SBATCH --job-name=lewm_turbofan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --output=logs/lewm_%j.out
#SBATCH --error=logs/lewm_%j.out
#SBATCH -C h100
# NOTE: bf16 precision requires A100 or H100.
#       For V100 (yil@v100 / -C v100) change precision to fp16 in lewm.yaml.

set -e

export STABLEWM_HOME=/lustre/fswork/projects/rech/yil/ugy35qd/.stable_worldmodel

source ~/.bashrc

# Activate environment
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
export CUDA_LAUNCH_BLOCKING=0
export WANDB_MODE=offline

# ── Training ──────────────────────────────────────────────────────────────────
# num_workers=8  — overrides the yaml default of 0; use ~CPUs/GPUs on the node.
# loader.persistent_workers=True — required when num_workers > 0 in Lightning.
# loader.pin_memory=True — speeds up host→GPU transfers.
DATA_DIR=/lustre/fswork/projects/rech/yil/ugy35qd/thesis/rl_opendeck_simulator/data

# -----------------------------------------------------------------------
# SEQ_LEN controls the downstream HI probe window size:
#   1  = single-snapshot probe (fast, baseline)
#   10 = 10-step trajectory window probe (slower setup, richer context)
# Change SEQ_LEN and RUN_NAME to switch between the two variants.
# -----------------------------------------------------------------------
SEQ_LEN=10
RUN_NAME=lewm_turbo_v4_sl${SEQ_LEN}

srun python -u train.py \
    data=turbofan \
    num_workers=16 \
    loader.persistent_workers=True \
    loader.pin_memory=True \
    subdir=turbofan_v3_sl${SEQ_LEN} \
    output_model_name=${RUN_NAME} \
    data.dataset.cache_dir=${DATA_DIR} \
    hi_probe.enabled=true \
    hi_probe.probe_seq_len=${SEQ_LEN}

echo "========================================"
echo "Training complete."
echo "Logs : logs/lewm_${SLURM_JOB_ID}.out"
echo "========================================"
