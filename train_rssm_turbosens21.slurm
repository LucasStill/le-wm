#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
#  RSSM (DreamerV3 style) on TurboSens2.1 — 3 seeds, train + export.
#  Retrain of the May run with the diagnosed fix: kl_free=0.0 (the 1.0
#  free bits floor let the posterior collapse onto the prior; see
#  RESULTS_RSSM.md). T=64 contiguous chunks, 10 epoch budget as usual.
#
#      sbatch train_rssm_turbosens21.slurm
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --account=yil@h100
#SBATCH --job-name=ts21_rssm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/ts21_rssm_%A_%a.out
#SBATCH --error=logs/ts21_rssm_%A_%a.out
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
export PYTHONUNBUFFERED=1 WANDB_MODE=offline WANDB_SILENT=true

SEED=$SLURM_ARRAY_TASK_ID
RUN_NAME=ts21_RSSM_seed${SEED}

cp -n "$LEWM_DATA/turbosens2_train_lewm.h5" "$SCRATCH/" || true

srun python -u baselines/rssm/train_rssm.py \
    data.dataset.cache_dir=$SCRATCH \
    data.dataset.name=turbosens2_train_lewm \
    loss.kl_free=0.0 \
    seed=${SEED} trainer.max_epochs=10 \
    loader.batch_size=512 num_workers=16 \
    wandb.config.project=turbosens21 \
    subdir=${RUN_NAME} output_model_name=${RUN_NAME} \
    hi_probe.enabled=false

CKPT=$(ls -t "$STABLEWM_HOME/${RUN_NAME}"/*object.ckpt* | head -1)
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
