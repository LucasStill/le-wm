#!/bin/bash
#SBATCH --account=yil@h100
#SBATCH --job-name=lewm_scenario3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --output=logs/lewm_s3_%j.out
#SBATCH --error=logs/lewm_s3_%j.out
#SBATCH -C h100

# ── Scenario-3 JEPA training ──────────────────────────────────────────────────
#
# Dataset changes vs turbofan:
#   n_sensors = 84  (7 sensors × 12 contexts, vs 28 = 7×4)
#   action    = 0–4 scalar (5 discrete overhaul types, vs binary 0/1)
#   ep_len    = 10 000 steps (all episodes same length)
#
# Key override: n_sensors=84  (config default is 28 for turbofan)
#
# Prerequisite: run prepare_scenario3_dataset.py --all_splits first.
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

# ── Data location ─────────────────────────────────────────────────────────────
# The prepared *_lewm.h5 files live alongside checkpoints in STABLEWM_HOME.
# Raw *_sensors.h5 source files stay in the simulator data dir.
DATA_DIR=$STABLEWM_HOME   # scenario3_train_400_lewm.h5 is here after prepare step

# ── Hyperparameters ───────────────────────────────────────────────────────────
# WIN_SIZE : encoder input window (obs_window_size)
#   1  = snapshot encoder (no TemporalAggregator)
#   >1 = window of w consecutive frames fused by TemporalAggregator
#
# HISTORY_LEN : predictor context (transformer history)
#   Reasonable range: 10–100 steps given ep_len=10000
#
# N_SENSORS : MUST be 84 for scenario3 (7 sensors × 12 contexts)

WIN_SIZE=1
HISTORY_LEN=50
N_SENSORS=84

RUN_NAME=lewm_s3_sensor_w${WIN_SIZE}_H${HISTORY_LEN}

srun python -u train.py \
    data=scenario3 \
    encoder_type=sensor \
    n_sensors=${N_SENSORS} \
    obs_window_size=${WIN_SIZE} \
    wm.history_size=${HISTORY_LEN} \
    num_workers=16 \
    loader.persistent_workers=True \
    loader.pin_memory=True \
    data.dataset.cache_dir=${DATA_DIR} \
    subdir=scenario3_sensor_w${WIN_SIZE}_H${HISTORY_LEN} \
    output_model_name=${RUN_NAME} \
    hi_probe.enabled=true \
    hi_probe.probe_seq_len=${HISTORY_LEN}

echo "========================================"
echo "Training complete."
echo "Logs : logs/lewm_s3_${SLURM_JOB_ID}.out"
echo "========================================"
