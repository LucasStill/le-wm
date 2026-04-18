#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
#  AR-LSTM Baseline — TurboSens (Jean-Zay, H100)
#
#  Trains the AR-LSTM world model on the TurboSens dataset.
#  Uses the same data, HI probe, and WandB setup as the JEPA baseline.
#  The only difference: the predictor is an LSTM instead of a transformer.
#
#  NOTE: V100 (SM 7.0) is not supported by the installed PyTorch build
#  (requires SM >= 7.5). Using H100 which is confirmed working with this env.
#
#  Partition: H100 (yil@h100)
#  Estimated runtime: ~2h for 100 epochs
#
#  Submit: sbatch baselines/ar_lstm/train_ar_lstm.slurm
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --account=yil@h100
#SBATCH --job-name=ar_lstm_turbofan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=logs/ar_lstm_%j.out
#SBATCH --error=logs/ar_lstm_%j.out
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
echo "Dir      : $(pwd)"
echo "========================================"

export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0
# wandb offline config — compute nodes have no internet access
export WANDB_MODE=offline
export WANDB_INIT_TIMEOUT=10
export WANDB_HTTP_TIMEOUT=10
export WANDB_SILENT=true

# ── Config ────────────────────────────────────────────────────────────────────
# DATA_DIR points to the same HDF5 used by the JEPA runs — no new data needed.
DATA_DIR=/lustre/fswork/projects/rech/yil/ugy35qd/thesis/rl_opendeck_simulator/data

# ── LSTM hyperparameters ──────────────────────────────────────────────────────
# HIDDEN : LSTM hidden state size.  256 gives ~comparable param count to the
#          JEPA transformer predictor (depth=4, mlp_dim=512).
# LAYERS : stacked LSTM layers. 2 matches the sensor encoder depth.
# H      : context history fed to the LSTM. Same default as JEPA (H=3).
#          Increase to 50 for the long-history ablation.
HIDDEN=256
LAYERS=2
H=50
RUN_NAME=ar_lstm_h${H}_hd${HIDDEN}_l${LAYERS}

echo "Predictor: LSTM(hidden=${HIDDEN}, layers=${LAYERS})"
echo "History  : H=${H}"
echo "Run name : ${RUN_NAME}"
echo "========================================"

srun python -u baselines/ar_lstm/train_ar_lstm.py \
    output_model_name=${RUN_NAME} \
    subdir=turbofan_ar_lstm_h${H}_hd${HIDDEN}_l${LAYERS} \
    data.dataset.cache_dir=${DATA_DIR} \
    wm.history_size=${H} \
    lstm.hidden_dim=${HIDDEN} \
    lstm.num_layers=${LAYERS} \
    wandb.config.name=${RUN_NAME} \
    hi_probe.enabled=true \
    hi_probe.probe_seq_len=1

echo "========================================"
echo "Training complete."
echo "Checkpoint : \$STABLEWM_HOME/*/ar_lstm_*.ckpt"
echo "WandB sync : wandb sync --sync-all"
echo "Evaluate   :"
echo "  python baselines/ar_lstm/eval_ar_lstm.py \\"
echo "    --ckpt \$STABLEWM_HOME/<subdir>/${RUN_NAME}_epoch_100_object.ckpt \\"
echo "    --data ${DATA_DIR}/opendeck_2000_lewm.h5 \\"
echo "    --jepa_ckpt \$STABLEWM_HOME/<jepa_subdir>/lewm_turbo_v6_sensor_sl1_w1_epoch_100_object.ckpt \\"
echo "    --out results/ar_lstm_vs_jepa.json"
echo "========================================"
