#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
#  Nonlinear raw baseline (MLP on raw sensors) — 3 seeds, train + export.
#  Fast: ~10 to 20 min per seed on H100.
#      sbatch train_mlp_raw_turbosens21.slurm
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --account=yil@h100
#SBATCH --job-name=ts21_mlpraw
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/ts21_mlpraw_%A_%a.out
#SBATCH --error=logs/ts21_mlpraw_%A_%a.out
#SBATCH -C h100

set -e
source ~/.bashrc

export STABLEWM_HOME=/lustre/fswork/projects/rech/yil/ugy35qd/.stable_worldmodel
TS2_REPO=${TS2_REPO:-$WORK/thesis/turbosens2}
LEWM_DATA=$STABLEWM_HOME/ts21
EMB_OUT=$STABLEWM_HOME/embeddings_ts21

PROJECT_DIR=$WORK/thesis/le-wm-vanilla
cd "$PROJECT_DIR"
source .venv/bin/activate
mkdir -p logs "$EMB_OUT"
export PYTHONUNBUFFERED=1

SEED=$SLURM_ARRAY_TASK_ID
RUN_NAME=ts21_MLPRAW_seed${SEED}
OUT=$STABLEWM_HOME/${RUN_NAME}

srun python -u baselines/oracle/train_mlp_raw.py \
    --data "$LEWM_DATA/turbosens2_train_lewm.h5" \
    --seed ${SEED} --epochs 10 \
    --out-dir "$OUT"

CKPT=$(ls -t "$OUT"/*object.ckpt* | head -1)
echo "exporting embeddings from: $CKPT"
python -u "$TS2_REPO/evaluation/export_embeddings_lewm.py" \
    --ckpt "$CKPT" \
    --name "$RUN_NAME" \
    --h5   "$LEWM_DATA/turbosens2_train_lewm.h5" \
           "$LEWM_DATA/turbosens2_test_lewm.h5" \
           "$LEWM_DATA/turbosens2_test_shift_lewm.h5" \
           "$LEWM_DATA/turbosens2_test_region_lewm.h5" \
    --out  "$EMB_OUT" \
    --train-stride 10

echo "done: $RUN_NAME"
