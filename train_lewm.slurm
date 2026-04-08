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
# NOTE: bf16 precision (set in lewm.yaml) requires A100 or H100.
#       If switching to V100 (yil@v100 / -C v100), change precision to fp16
#       in config/train/lewm.yaml first.

set -e

# and set up this path:
export STABLEWM_HOME=/lustre/fswork/projects/rech/yil/ugy35qd/.stable_worldmodel

source ~/.bashrc

# ── Activate environment ──────────────────────────────────────────────────────
# The project ships with a .venv — activate it, or swap for your conda env.
PROJECT_DIR=$WORK/thesis/le-wm-vanilla          # ← update if your checkout lives elsewhere
cd "$PROJECT_DIR"
source .venv/bin/activate
# conda activate lucasenv         # ← alternative if using conda

# ── Logging / output dirs ─────────────────────────────────────────────────────
mkdir -p logs

echo "========================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "GPU      : $CUDA_VISIBLE_DEVICES"
echo "Dir      : $(pwd)"
echo "========================================"

# ── Environment variables ─────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0
# Uncomment to run wandb offline and sync later (useful if the compute nodes
# have no internet access on Jean Zay):
export WANDB_MODE=offline

# ── Training ──────────────────────────────────────────────────────────────────
# num_workers=8  — overrides the yaml default of 0; use ~CPUs/GPUs on the node.
# loader.persistent_workers=True — required when num_workers > 0 in Lightning.
# loader.pin_memory=True — speeds up host→GPU transfers.
srun python -u train.py \
    data=turbofan \
    img_size=28 \
    patch_size=14 \
    num_workers=16 \
    loader.persistent_workers=True \
    loader.pin_memory=True \
    subdir=turbofan1_2 \
    output_model_name=lewm_turbo1_3_1 \
    data.dataset.cache_dir=/lustre/fswork/projects/rech/yil/ugy35qd/thesis/rl_opendeck_simulator/data \
    hi_probe.enabled=true \
    hi_probe.eval_interval=1


echo "========================================"
echo "Training complete."
echo "Logs : logs/lewm_${SLURM_JOB_ID}.out"
echo "========================================"
