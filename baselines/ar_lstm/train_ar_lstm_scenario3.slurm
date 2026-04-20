#!/bin/bash
#SBATCH --account=yil@h100
#SBATCH --job-name=ar_lstm_s3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --output=logs/ar_lstm_s3_%j.out
#SBATCH --error=logs/ar_lstm_s3_%j.out
#SBATCH -C h100

set -e

export STABLEWM_HOME=/lustre/fswork/projects/rech/yil/ugy35qd/.stable_worldmodel

source ~/.bashrc

PROJECT_DIR=$WORK/thesis/le-wm-vanilla
cd "$PROJECT_DIR"
source .venv/bin/activate

mkdir -p logs

echo "========================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "GPU      : $CUDA_VISIBLE_DEVICES"
echo "========================================"

export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export WANDB_INIT_TIMEOUT=5
export WANDB_HTTP_TIMEOUT=5
# NOTE: no WANDB_SILENT — see train_lewm_scenario3.slurm comment

# ── Data: copy HDF5 to fast local NVMe ($SCRATCH) ────────────────────────────
HDF5_NAME=scenario3_train_400_lewm.h5
echo "Copying ${HDF5_NAME} to \$SCRATCH ..."
cp $STABLEWM_HOME/${HDF5_NAME} $SCRATCH/
echo "Copy done."
DATA_DIR=$SCRATCH

# Vary history_size / h_step to ablate temporal context (matches JEPA experiments)
HISTORY_LEN=50   # number of latent steps in context
H_STEP=1         # stride between latents; 1=dense, 5=every 5th obs

# Zero-pad prefix augmentation probability (0.0 = off, 0.3 = 30% chance per batch)
ZERO_PAD_PROB=0.0

RUN_NAME=ar_lstm_s3_h${HISTORY_LEN}_S${H_STEP}_hd256_l2

srun python -u baselines/ar_lstm/train_ar_lstm.py \
    --config-name train_ar_lstm_scenario3 \
    wm.history_size=${HISTORY_LEN} \
    wm.h_step=${H_STEP} \
    wm.zero_pad_prob=${ZERO_PAD_PROB} \
    data.dataset.cache_dir=${DATA_DIR} \
    subdir=scenario3_ar_lstm_h${HISTORY_LEN}_S${H_STEP}_hd256_l2 \
    output_model_name=${RUN_NAME} \
    hi_probe.enabled=true \
    hi_probe.probe_seq_len=${HISTORY_LEN}

echo "========================================"
echo "Training complete."
echo "========================================"
