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
# SEQ_LEN  — downstream HI probe temporal window (probe head only).
#            Does NOT affect the encoder. The encoder always produces one
#            z_t per timestep regardless of SEQ_LEN.
#              1  = single-snapshot probe (baseline)
#              10 = 10-step trajectory window probe (richer context)
#
# WIN_SIZE — encoder input window (architecture-level temporal depth).
#            obs_window_size=w: the ViT encodes w consecutive frames with
#            shared weights, then a TemporalAggregator compresses the window
#            of embeddings into a single trajectory-aware embedding.
#              1  = snapshot encoder (baseline, no temporal aggregation)
#              5  = 5-step encoder window
#              10 = 10-step encoder window
# -----------------------------------------------------------------------
SEQ_LEN=10
WIN_SIZE=1
RUN_NAME=lewm_turbo_v5b_sl${SEQ_LEN}_w${WIN_SIZE}

srun python -u train.py \
    data=turbofan \
    num_workers=16 \
    loader.persistent_workers=True \
    loader.pin_memory=True \
    subdir=turbofan_v5b_sl${SEQ_LEN}_w${WIN_SIZE} \
    output_model_name=${RUN_NAME} \
    data.dataset.cache_dir=${DATA_DIR} \
    obs_window_size=${WIN_SIZE} \
    hi_probe.enabled=true \
    hi_probe.probe_seq_len=${SEQ_LEN}

echo "========================================"
echo "Training complete."
echo "Logs : logs/lewm_${SLURM_JOB_ID}.out"
echo "========================================"
