#!/bin/bash
# =============================================================================
# train_ar_lstm_scenario4_orailix.sh — single-GPU AR-LSTM baseline launch
#
# Mirrors train_lewm_scenario4_dragon.sh (which trains JEPA) so the AR-LSTM
# baseline is launched with the *same* hyperparameter knobs and is therefore
# directly comparable run-for-run.
#
# Run inside tmux so it survives SSH disconnects:
#
#   cd ~/thesis/le-wm
#   tmux new -s s4_arlstm
#   ./train_ar_lstm_scenario4_orailix.sh 2>&1 \
#       | tee logs/s4_arlstm_$(date +%Y%m%d_%H%M).log
#
# Override any hyperparameter via env var, e.g.:
#   WIN_SIZE=4 HISTORY_LEN=16 NUM_PREDS=4 ./train_ar_lstm_scenario4_orailix.sh
# =============================================================================
set -e

cd "$(dirname "$0")"
export STABLEWM_HOME=${STABLEWM_HOME:-/home/lucas/.stable_worldmodel}
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
HISTORY_LEN=${HISTORY_LEN:-16}
H_STEP=${H_STEP:-1}
NUM_PREDS=${NUM_PREDS:-4}
N_SENSORS=176
MAX_SENSORS=200
ZERO_PAD_PROB=${ZERO_PAD_PROB:-0.0}

# LSTM-specific knobs
LSTM_HIDDEN=${LSTM_HIDDEN:-256}
LSTM_LAYERS=${LSTM_LAYERS:-2}
LSTM_DROPOUT=${LSTM_DROPOUT:-0.1}

BATCH_SIZE=${BATCH_SIZE:-512}      # A6000: 48 GB VRAM. Halve + double ACCUM_GRAD for larger num_steps.
NUM_WORKERS=${NUM_WORKERS:-8}
PRECISION=${PRECISION:-bf16-mixed} # A6000 supports bf16
MAX_EPOCHS=${MAX_EPOCHS:-10}       # matches dragon's per-config budget
ACCUM_GRAD=${ACCUM_GRAD:-1}        # set to 2/4/8 for L2/L3/L4 to keep effective bs=512

RUN_NAME=ar_lstm_s4_w${WIN_SIZE}_H${HISTORY_LEN}_S${H_STEP}_P${NUM_PREDS}_h${LSTM_HIDDEN}l${LSTM_LAYERS}

echo "Run name : ${RUN_NAME}"
echo "Config   : w=${WIN_SIZE} H=${HISTORY_LEN} s=${H_STEP} P=${NUM_PREDS} bs=${BATCH_SIZE}"
echo "LSTM     : hidden=${LSTM_HIDDEN} layers=${LSTM_LAYERS} dropout=${LSTM_DROPOUT}"
echo "========================================"

python -u baselines/ar_lstm/train_ar_lstm.py \
    --config-path "$(pwd)/baselines/ar_lstm/config" \
    --config-name train_ar_lstm_scenario4 \
    n_sensors=${N_SENSORS} \
    sensor_encoder.max_sensors=${MAX_SENSORS} \
    compile_encoder=${COMPILE_ENCODER:-true} \
    obs_window_size=${WIN_SIZE} \
    wm.history_size=${HISTORY_LEN} \
    wm.h_step=${H_STEP} \
    wm.num_preds=${NUM_PREDS} \
    wm.zero_pad_prob=${ZERO_PAD_PROB} \
    lstm.hidden_dim=${LSTM_HIDDEN} \
    lstm.num_layers=${LSTM_LAYERS} \
    lstm.dropout=${LSTM_DROPOUT} \
    num_workers=${NUM_WORKERS} \
    loader.batch_size=${BATCH_SIZE} \
    loader.persistent_workers=True \
    loader.pin_memory=True \
    trainer.precision=${PRECISION} \
    data.dataset.cache_dir=${STABLEWM_HOME} \
    wandb.config.project=turbofan_S4 \
    subdir=ar_lstm_scenario4_w${WIN_SIZE}_H${HISTORY_LEN}_S${H_STEP}_P${NUM_PREDS}_h${LSTM_HIDDEN}l${LSTM_LAYERS} \
    output_model_name=${RUN_NAME} \
    hi_probe.enabled=true \
    hi_probe.probe_seq_len=${HISTORY_LEN} \
    trainer.max_epochs=${MAX_EPOCHS} \
    +trainer.accumulate_grad_batches=${ACCUM_GRAD}

echo "========================================"
echo "Training complete."
echo "========================================"
