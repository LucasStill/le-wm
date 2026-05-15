#!/bin/bash
#SBATCH --account=yil@h100
#SBATCH --job-name=lewm_scenario4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --output=logs/lewm_s4_%j.out
#SBATCH --error=logs/lewm_s4_%j.out
#SBATCH -C h100

# ── Scenario-4 JEPA training ──────────────────────────────────────────────────
#
# Dataset vs scenario-3:
#   pixels      : (N, 11, 16)   — 7 sensors + 4 context_params per context
#   n_sensors   : 176  (11 × 16, vs 84 = 7 × 12 in s3)
#   n_actions   : 7    (0=do_nothing, 1=fan, 2=hpc, 3=turbine, 4=full, 5=patch, 6=wash)
#   ep_len      : variable (mean ≈ 13 900 flights)
#
# Key overrides:
#   n_sensors=176
#   sensor_encoder.max_sensors=200  (default 128 is too small for 176 tokens)
#
# Prerequisite: run prepare_scenario4_dataset.py --all_splits first.
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

# ── Data: copy HDF5 to fast local NVMe ($SCRATCH) ────────────────────────────
HDF5_NAME=turbosens2_train.h5
echo "Copying ${HDF5_NAME} to \$SCRATCH ..."
cp $STABLEWM_HOME/${HDF5_NAME} $SCRATCH/
echo "Copy done. Data dir: $SCRATCH"
DATA_DIR=$SCRATCH

# ── Hyperparameters ───────────────────────────────────────────────────────────
# Defaults: w=1 H=1 P=1 S=1 (override via sbatch --export or CLI).
# WIN_SIZE    : encoder input window (obs_window_size).  1 = snapshot encoder.
# HISTORY_LEN : predictor context length (number of latent steps).
# H_STEP      : temporal stride between history latents. 1 = dense.
# NUM_PREDS   : autoregressive rollout steps during training. 1 = 1-step only.

WIN_SIZE=${WIN_SIZE:-1}
HISTORY_LEN=${HISTORY_LEN:-1}
H_STEP=${H_STEP:-1}
NUM_PREDS=${NUM_PREDS:-1}
N_SENSORS=176
MAX_SENSORS=200
ZERO_PAD_PROB=0.0

RUN_NAME=lewm_s4_sensor_w${WIN_SIZE}_H${HISTORY_LEN}_S${H_STEP}_P${NUM_PREDS}

srun python -u train.py \
    data=turbosens2 \
    encoder_type=sensor \
    n_sensors=${N_SENSORS} \
    +sensor_encoder.max_sensors=${MAX_SENSORS} \
    obs_window_size=${WIN_SIZE} \
    wm.history_size=${HISTORY_LEN} \
    wm.h_step=${H_STEP} \
    wm.num_preds=${NUM_PREDS} \
    wm.zero_pad_prob=${ZERO_PAD_PROB} \
    num_workers=16 \
    loader.batch_size=512 \
    loader.persistent_workers=True \
    loader.pin_memory=True \
    trainer.precision=bf16-mixed \
    data.dataset.cache_dir=${DATA_DIR} \
    wandb.config.project=turbofan_S4 \
    subdir=turbosens2_sensor_w${WIN_SIZE}_H${HISTORY_LEN}_S${H_STEP}_P${NUM_PREDS} \
    output_model_name=${RUN_NAME} \
    hi_probe.enabled=true \
    hi_probe.probe_seq_len=${HISTORY_LEN}

echo "========================================"
echo "Training complete."
echo "Logs : logs/lewm_s4_${SLURM_JOB_ID}.out"
echo "========================================"
