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
# wandb offline config — compute nodes have no internet access
export WANDB_MODE=offline
export WANDB_INIT_TIMEOUT=10
export WANDB_HTTP_TIMEOUT=10
export WANDB_SILENT=true

# ── Training ──────────────────────────────────────────────────────────────────
DATA_DIR=/lustre/fswork/projects/rech/yil/ugy35qd/thesis/rl_opendeck_simulator/data

# -----------------------------------------------------------------------
# ENC_TYPE — encoder architecture.
#   vit    = ViT-on-fake-image (legacy, for ablation comparison)
#   sensor = SensorEncoder: 28 scalar tokens, no image pipeline (v6+)
#
# SEQ_LEN  — downstream HI probe temporal window (probe head only).
#            Does NOT affect the encoder. The encoder always produces one
#            z_t per timestep regardless of SEQ_LEN.
#              1  = single-snapshot probe (baseline)
#              10 = 10-step trajectory window probe (richer context)
#
# WIN_SIZE — encoder input window (architecture-level temporal depth).
#            obs_window_size=w: the encoder processes w consecutive frames
#            with shared weights, then TemporalAggregator compresses the
#            window of embeddings into one trajectory-aware embedding.
#              1  = snapshot encoder (baseline, no temporal aggregation)
#              5  = 5-step encoder window
#              10 = 10-step encoder window
# -----------------------------------------------------------------------
# Recommended: SEQ_LEN=1 to measure what the *encoder* encodes directly.
ENC_TYPE=sensor
SEQ_LEN=1
WIN_SIZE=1
HISTORY_LEN=100
RUN_NAME=lewm_turbo_v6_${ENC_TYPE}_sl${SEQ_LEN}_w${WIN_SIZE}_H${HISTORY_LEN}

srun python -u train.py \
    data=turbofan \
    num_workers=16 \
    loader.persistent_workers=True \
    loader.pin_memory=True \
    encoder_type=${ENC_TYPE} \
    subdir=turbofan_v6_${ENC_TYPE}_sl${SEQ_LEN}_w${WIN_SIZE}_b512_pl3_H${HISTORY_LEN} \
    output_model_name=${RUN_NAME} \
    data.dataset.cache_dir=${DATA_DIR} \
    obs_window_size=${WIN_SIZE} \
    hi_probe.enabled=true \
    hi_probe.probe_seq_len=${SEQ_LEN}

echo "========================================"
echo "Training complete."
echo "Logs : logs/lewm_${SLURM_JOB_ID}.out"
echo "========================================"
