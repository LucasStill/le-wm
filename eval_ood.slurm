#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  eval_ood.slurm — Task 4 OOD Detection (GPU-only job)
#
#  The turbofan simulator runs as a SEPARATE standalone CPU job submitted
#  via:  sbatch $SIMULATOR_DIR/hpc/slurm/launch_cpu_worker.sh
#  That job writes its ZMQ address to $WORK/.simulator_addr.
#  This script reads that file and connects automatically.
#
#  Full workflow:
#    1.  sbatch $WORK/thesis/rl_opendeck_simulator/hpc/slurm/launch_cpu_worker.sh
#    2.  Wait until:  cat $WORK/.simulator_addr   # shows tcp://rXiYnZ:5555
#    3.  sbatch eval_ood.slurm
#
#  The eval job will wait up to 5 minutes for the address file to appear,
#  so steps 1-3 can be submitted back-to-back without manual waiting.
# ══════════════════════════════════════════════════════════════════════════════

#SBATCH --account=yil@h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH -C h100
#SBATCH --job-name=lewm_ood
#SBATCH --time=06:00:00
#SBATCH --output=logs/ood_%j.out
#SBATCH --error=logs/ood_%j.out

set -eo pipefail

source ~/.bashrc

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR=$WORK/thesis/le-wm-vanilla
SIMULATOR_DIR=$WORK/thesis/rl_opendeck_simulator
export STABLEWM_HOME=/lustre/fswork/projects/rech/yil/ugy35qd/.stable_worldmodel
export SIMULATOR_REPO=${SIMULATOR_DIR}
export PYTHONPATH=${SIMULATOR_DIR}:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

mkdir -p logs

# ── Locate the simulator worker address ───────────────────────────────────────
ADDR_FILE="${WORK}/.simulator_addr"

echo "============================================================"
echo "  Task 4 OOD Detection"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  GPU node   : $(hostname -s)"
echo "  Addr file  : $ADDR_FILE"
echo "  Start time : $(date)"
echo "============================================================"

# Wait up to 5 minutes for the CPU worker to write its address
echo "[$(date +%H:%M:%S)] Waiting for simulator worker address …"
for i in $(seq 1 60); do
    if [[ -f "$ADDR_FILE" ]]; then
        break
    fi
    sleep 5
done

if [[ ! -f "$ADDR_FILE" ]]; then
    echo "[$(date +%H:%M:%S)] No simulator address file found — will use dataset proxy."
fi

# ── Checkpoints to evaluate ───────────────────────────────────────────────────
CKPTS=(
    "v6_sensor_sl1_w1:${STABLEWM_HOME}/turbofan_v6_sensor_sl1_w1/lewm_turbo_v6_sensor_sl1_w1_epoch_100_object.ckpt"
    "v6_sensor_sl1_w10:${STABLEWM_HOME}/turbofan_v6_sensor_sl1_w10/lewm_turbo_v6_sensor_sl1_w10_epoch_100_object.ckpt"
)

OUT_DIR=${STABLEWM_HOME}/eval_ood/${SLURM_JOB_ID}
mkdir -p "${OUT_DIR}"

# ── Decide simulator mode ─────────────────────────────────────────────────────
# If USE_SIMULATOR=1 and addr file exists, use the live physics engine.
# Otherwise fall back to the fast-episode dataset proxy (no external service needed).
USE_SIMULATOR="${USE_SIMULATOR:-0}"
SIM_FLAGS="--no_simulator"

if [[ "$USE_SIMULATOR" == "1" && -f "$ADDR_FILE" ]]; then
    WORKER_ADDR=$(cat "$ADDR_FILE")
    echo "[$(date +%H:%M:%S)] Simulator worker found at: ${WORKER_ADDR}"
    SIM_FLAGS="--sim_addr ${WORKER_ADDR}"
else
    echo "[$(date +%H:%M:%S)] Running in no-simulator mode (dataset proxy)."
fi

# ── Run OOD evaluation ────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Starting OOD evaluation …"

cd "${PROJECT_DIR}"
source .venv/bin/activate

python -u eval_ood.py \
    "${CKPTS[@]}" \
    --hdf5    "${STABLEWM_HOME}/opendeck_2000_lewm.h5" \
    ${SIM_FLAGS} \
    --out_dir "${OUT_DIR}" \
    --n_ood_eps 50 \
    --seq_len   500

echo "============================================================"
echo "  OOD results : ${OUT_DIR}"
echo "============================================================"
