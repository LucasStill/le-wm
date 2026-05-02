#!/bin/bash
# =============================================================================
# train_rssm_scenario4_orailix.sh — DreamerV3 RSSM baseline launcher
#
# Mirrors train_ar_lstm_scenario4_orailix.sh so the comparison stays one
# command apart. Run inside tmux:
#
#   cd ~/thesis/le-wm
#   tmux new -s s4_rssm
#   ./train_rssm_scenario4_orailix.sh 2>&1 \
#       | tee logs/s4_rssm_$(date +%Y%m%d_%H%M).log
#
# Override hyperparameters via env vars, e.g.:
#   NUM_STEPS=128 BATCH_SIZE=8 ./train_rssm_scenario4_orailix.sh
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
NUM_STEPS=${NUM_STEPS:-64}              # contiguous chunk length
N_SENSORS=176
NUM_ACTIONS=7

# RSSM-specific knobs
STOCH=${STOCH:-32}
DISCRETE=${DISCRETE:-32}
DETER=${DETER:-512}
HIDDEN=${HIDDEN:-512}
EMBED_SIZE=${EMBED_SIZE:-512}

# Loss weights
KL_FREE=${KL_FREE:-1.0}
DYN_SCALE=${DYN_SCALE:-0.5}
REP_SCALE=${REP_SCALE:-0.1}

BATCH_SIZE=${BATCH_SIZE:-16}            # 16 x 64 = 1024 timesteps/batch
NUM_WORKERS=${NUM_WORKERS:-8}
PRECISION=${PRECISION:-bf16-mixed}
MAX_EPOCHS=${MAX_EPOCHS:-10}
ACCUM_GRAD=${ACCUM_GRAD:-1}             # bump to 2/4 to push effective batch size
LR=${LR:-1e-4}

RUN_NAME=rssm_s4_T${NUM_STEPS}_S${STOCH}x${DISCRETE}_D${DETER}

echo "Run name : ${RUN_NAME}"
echo "Config   : T=${NUM_STEPS} stoch=${STOCH}x${DISCRETE} deter=${DETER} bs=${BATCH_SIZE}"
echo "========================================"

python -u baselines/rssm/train_rssm.py \
    --config-path "$(pwd)/baselines/rssm/config" \
    --config-name train_rssm_scenario4 \
    n_sensors=${N_SENSORS} \
    num_actions=${NUM_ACTIONS} \
    data.dataset.num_steps=${NUM_STEPS} \
    rssm.stoch=${STOCH} \
    rssm.discrete=${DISCRETE} \
    rssm.deter=${DETER} \
    rssm.hidden=${HIDDEN} \
    rssm.embed_size=${EMBED_SIZE} \
    loss.kl_free=${KL_FREE} \
    loss.dyn_scale=${DYN_SCALE} \
    loss.rep_scale=${REP_SCALE} \
    optimizer.lr=${LR} \
    num_workers=${NUM_WORKERS} \
    loader.batch_size=${BATCH_SIZE} \
    loader.persistent_workers=True \
    loader.pin_memory=True \
    trainer.precision=${PRECISION} \
    data.dataset.cache_dir=${STABLEWM_HOME} \
    wandb.config.project=turbofan_S4 \
    subdir=rssm_scenario4_T${NUM_STEPS}_S${STOCH}x${DISCRETE}_D${DETER} \
    output_model_name=${RUN_NAME} \
    hi_probe.enabled=true \
    hi_probe.probe_seq_len=1 \
    trainer.max_epochs=${MAX_EPOCHS} \
    +trainer.accumulate_grad_batches=${ACCUM_GRAD}

echo "========================================"
echo "Training complete."
echo "========================================"
