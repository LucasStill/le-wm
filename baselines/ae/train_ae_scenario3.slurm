#!/bin/bash
#SBATCH --account=yil@h100
#SBATCH --job-name=ae_scenario3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --output=logs/ae_s3_%j.out
#SBATCH --error=logs/ae_s3_%j.out
#SBATCH -C h100

# ── Sensor Autoencoder baseline — Scenario-3 ─────────────────────────────────
#
# Reconstruction baseline: SensorEncoder + MLP decoder, trained with MSE loss.
# No temporal modelling — each frame is encoded/decoded independently.
#
# GPU: H100 (same venv as JEPA/AR-LSTM — V100 is incompatible with the compiled kernels)
#   Estimated: ~4 GB peak at batch_size=512, num_steps=10
#
# Compare against:
#   JEPA     : train_lewm_scenario3.slurm    (H100)
#   AR-LSTM  : train_ar_lstm_scenario3.slurm (H100)
# ─────────────────────────────────────────────────────────────────────────────

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
echo "Dir      : $(pwd)"
echo "========================================"

export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0
export WANDB_MODE=offline
export WANDB_INIT_TIMEOUT=5
export WANDB_HTTP_TIMEOUT=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Data: copy HDF5 to $SCRATCH ───────────────────────────────────────────────
HDF5_NAME=turbosens1_train.h5
echo "Copying ${HDF5_NAME} to \$SCRATCH ..."
cp $STABLEWM_HOME/${HDF5_NAME} $SCRATCH/
echo "Copy done."
DATA_DIR=$SCRATCH

# ── Hyperparameters ───────────────────────────────────────────────────────────
# WIN_SIZE : encoder input window (obs_window_size)
#   1  = snapshot AE — each frame encoded independently
#   >1 = TemporalAggregator fuses w consecutive frames; decoder reconstructs last frame
WIN_SIZE=50

RUN_NAME=ae_s3_sensor_w${WIN_SIZE}

srun python -u baselines/ae/train_ae.py \
    --config-name train_ae_scenario3 \
    obs_window_size=${WIN_SIZE} \
    data.dataset.cache_dir=${DATA_DIR} \
    subdir=${RUN_NAME} \
    output_model_name=${RUN_NAME} \
    hi_probe.enabled=true \
    hi_probe.probe_seq_len=1

echo "========================================"
echo "Training complete."
echo "========================================"
