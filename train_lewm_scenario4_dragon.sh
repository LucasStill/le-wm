#!/bin/bash
# =============================================================================
# train_lewm_scenario4_dragon.sh — single-GPU launch on dragon workstation
#
# Not a Slurm job. Run inside a tmux session so it survives SSH disconnects:
#
#   ssh dragon-lan
#   cd ~/thesis/le-wm
#   tmux new -s s4train
#   ./train_lewm_scenario4_dragon.sh 2>&1 | tee logs/s4_$(date +%Y%m%d_%H%M).log
#
# Override any hyperparameter via env var, e.g.:
#   WIN_SIZE=4 HISTORY_LEN=50 NUM_PREDS=4 ./train_lewm_scenario4_dragon.sh
# =============================================================================
set -e

cd "$(dirname "$0")"
export STABLEWM_HOME=${STABLEWM_HOME:-/home/lthil/.stable_worldmodel}
source .venv/bin/activate

mkdir -p logs

echo "========================================"
echo "Host     : $(hostname)"
echo "GPU      : $(nvidia-smi -L | head -1)"
echo "Dir      : $(pwd)"
echo "STABLEWM : $STABLEWM_HOME"
echo "========================================"

export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_INIT_TIMEOUT=5
export WANDB_HTTP_TIMEOUT=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Hyperparameters (override via env vars) ──────────────────────────────────
WIN_SIZE=${WIN_SIZE:-1}
HISTORY_LEN=${HISTORY_LEN:-1}
H_STEP=${H_STEP:-1}
NUM_PREDS=${NUM_PREDS:-1}
N_SENSORS=176
MAX_SENSORS=200
ZERO_PAD_PROB=${ZERO_PAD_PROB:-0.0}
BATCH_SIZE=${BATCH_SIZE:-256}      # RTX 5090: 32 GB VRAM
NUM_WORKERS=${NUM_WORKERS:-8}
PRECISION=${PRECISION:-bf16-mixed} # RTX 5090 supports bf16
MAX_EPOCHS=${MAX_EPOCHS:-100}
ACCUM_GRAD=${ACCUM_GRAD:-1}    # gradient accumulation; use 2 to keep effective bs=512 when bs=256

RUN_NAME=lewm_s4_sensor_w${WIN_SIZE}_H${HISTORY_LEN}_S${H_STEP}_P${NUM_PREDS}

echo "Run name : ${RUN_NAME}"
echo "Config   : w=${WIN_SIZE} H=${HISTORY_LEN} s=${H_STEP} P=${NUM_PREDS} bs=${BATCH_SIZE}"
echo "========================================"

python -u train.py \
    data=scenario4 \
    encoder_type=sensor \
    n_sensors=${N_SENSORS} \
    +sensor_encoder.max_sensors=${MAX_SENSORS} \
    obs_window_size=${WIN_SIZE} \
    wm.history_size=${HISTORY_LEN} \
    wm.h_step=${H_STEP} \
    wm.num_preds=${NUM_PREDS} \
    wm.zero_pad_prob=${ZERO_PAD_PROB} \
    num_workers=${NUM_WORKERS} \
    loader.batch_size=${BATCH_SIZE} \
    loader.persistent_workers=True \
    loader.pin_memory=True \
    trainer.precision=${PRECISION} \
    data.dataset.cache_dir=${STABLEWM_HOME} \
    wandb.config.project=turbofan_S4 \
    subdir=scenario4_sensor_w${WIN_SIZE}_H${HISTORY_LEN}_S${H_STEP}_P${NUM_PREDS} \
    output_model_name=${RUN_NAME} \
    hi_probe.enabled=true \
    hi_probe.probe_seq_len=${HISTORY_LEN} \
    trainer.max_epochs=${MAX_EPOCHS} \
    +trainer.accumulate_grad_batches=${ACCUM_GRAD}

echo "========================================"
echo "Training complete."
echo "========================================"
