#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
#  TurboSens2.1 training sweep — 6 configs x 3 seeds = 18 H100 jobs.
#
#  Grid (receptive field in RAW steps straddles the wash interval ~70-130):
#    idx/3  ID    family   H   S   W   native_bs  accum   RF
#    0      J16   jepa    16   1   1   512        1        16   (below)
#    1      J32   jepa    32   1   1   512        1        32   (below)
#    2      J160  jepa    16  10   1   256        2       160   (1-2 cycles)
#    3      J320  jepa    16  20   1   128        4       320   (2-4 cycles)
#    4      L16   lstm    16   1   1   512        1       recurrent
#    5      LBIG  lstm    32   1   4   256        2       recurrent
#  seed = 0/1/2 via idx % 3. Fixed budget: 10 epochs, effective batch 512,
#  P=4, bf16. Each job trains, then exports embeddings for the four splits
#  (train at stride 10) in the same allocation.
#
#  Prerequisites:
#    1. python prepare_scenario4_dataset.py --all_splits \
#         --data_dir $WORK/thesis/turbosens2/data --out_dir $STABLEWM_HOME/ts21
#    2. turbosens2 repo checkout at $TS2_REPO (eval-stats branch)
#
#  Submit:  mkdir -p logs && sbatch train_sweep_turbosens21.slurm
#  Chain the CPU eval per run afterwards (see turbosens2
#  slurm/jz_eval_embeddings_cpu.slurm).
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --account=yil@h100
#SBATCH --job-name=ts21_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/ts21_sweep_%A_%a.out
#SBATCH --error=logs/ts21_sweep_%A_%a.out
#SBATCH -C h100

set -e
source ~/.bashrc

export STABLEWM_HOME=/lustre/fswork/projects/rech/yil/ugy35qd/.stable_worldmodel
TS2_REPO=${TS2_REPO:-$WORK/thesis/turbosens2}
LEWM_DATA=$STABLEWM_HOME/ts21               # lewm-prepared H5s (prereq 1)
EMB_OUT=$STABLEWM_HOME/embeddings_ts21
EPOCHS=10

PROJECT_DIR=$WORK/thesis/le-wm-vanilla
cd "$PROJECT_DIR"
source .venv/bin/activate
mkdir -p logs "$EMB_OUT"

export PYTHONUNBUFFERED=1 WANDB_MODE=offline WANDB_SILENT=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── config table ──────────────────────────────────────────────────────────────
CFG=$((SLURM_ARRAY_TASK_ID / 3))
SEED=$((SLURM_ARRAY_TASK_ID % 3))
IDS=(J16 J32 J160 J320 L16 LBIG)
FAMS=(jepa jepa jepa jepa lstm lstm)
HS=(16 32 16 16 16 32)
SS=(1 1 10 20 1 1)
WS=(1 1 1 1 1 4)
BS=(512 512 256 128 512 256)
ACC=(1 1 2 4 1 2)

ID=${IDS[$CFG]}; FAM=${FAMS[$CFG]}; H=${HS[$CFG]}; S=${SS[$CFG]}
W=${WS[$CFG]}; NBS=${BS[$CFG]}; NACC=${ACC[$CFG]}
RUN_NAME=ts21_${ID}_seed${SEED}
SUBDIR=ts21_${ID}_seed${SEED}

echo "== $RUN_NAME  fam=$FAM H=$H S=$S W=$W bs=$NBS acc=$NACC epochs=$EPOCHS =="

# ── data to node-local scratch ────────────────────────────────────────────────
if [ ! -f "$LEWM_DATA/turbosens2_train_lewm.h5" ]; then
  echo "FATAL: $LEWM_DATA/turbosens2_train_lewm.h5 missing — run the lewm"
  echo "prepare step first (see header)."
  exit 1
fi
cp -n "$LEWM_DATA/turbosens2_train_lewm.h5" "$SCRATCH/" || true
DATA_DIR=$SCRATCH

# ── train ─────────────────────────────────────────────────────────────────────
if [ "$FAM" = "jepa" ]; then
  # Main le-wm config tree: data is a config GROUP (data=scenario4).
  srun python -u train.py \
      data=scenario4 encoder_type=sensor n_sensors=176 \
      +sensor_encoder.max_sensors=200 \
      data.dataset.cache_dir=${DATA_DIR} \
      data.dataset.name=turbosens2_train_lewm \
      seed=${SEED} trainer.max_epochs=${EPOCHS} \
      obs_window_size=${W} \
      wm.history_size=${H} wm.h_step=${S} wm.num_preds=4 \
      loader.batch_size=${NBS} +trainer.accumulate_grad_batches=${NACC} \
      num_workers=16 loader.persistent_workers=True loader.pin_memory=True \
      trainer.precision=bf16-mixed \
      wandb.config.project=turbosens21 \
      subdir=${SUBDIR} output_model_name=${RUN_NAME} \
      hi_probe.enabled=false
else
  # Self contained ar_lstm config: data is an INLINED dict (no group),
  # n_sensors defaults to 28, precision stays fp16 (bf16 overflows
  # SIGReg early per the config's own note). num_steps must cover the
  # P=4 rollout that its formula (H*S + W) does not include.
  NSTEPS=$(( (H + 4 - 1) * S + W ))
  srun python -u baselines/ar_lstm/train_ar_lstm.py \
      n_sensors=176 sensor_encoder.max_sensors=200 \
      data.dataset.cache_dir=${DATA_DIR} \
      data.dataset.name=turbosens2_train_lewm \
      data.dataset.num_steps=${NSTEPS} \
      'data.dataset.keys_to_cache=[action,pixels]' \
      seed=${SEED} trainer.max_epochs=${EPOCHS} \
      obs_window_size=${W} \
      wm.history_size=${H} wm.num_preds=4 \
      lstm.hidden_dim=256 lstm.num_layers=2 \
      loader.batch_size=${NBS} +trainer.accumulate_grad_batches=${NACC} \
      wandb.config.project=turbosens21 \
      subdir=${SUBDIR} output_model_name=${RUN_NAME} \
      hi_probe.enabled=false
fi

# ── export embeddings (same allocation, forward passes only) ─────────────────
CKPT=$(ls -t "$STABLEWM_HOME/$SUBDIR"/*object.ckpt* | head -1)
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
