#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  eval_ood.slurm — Task 4 OOD Detection on Jean Zay (heterogeneous job)
#
#  Two-pack heterogeneous job:
#    Pack 0 (gpu_p5): GPU node  -- runs eval_ood.py (JEPA encoder + metrics)
#    Pack 1 (cpu_p1): CPU node  -- runs simulator_worker.py (ZMQ server)
#
#  Communication: Pack 0 connects to Pack 1 via ZMQ over InfiniBand.
#  The job ends when the GPU-side evaluation finishes; the worker is then
#  cleaned up automatically.
#
#  Usage
#  -----
#    sbatch eval_ood.slurm
#    sbatch eval_ood.slurm --no_simulator   (dataset-proxy mode, single-node)
# ══════════════════════════════════════════════════════════════════════════════

# --------------------------------------------------------------------------- #
# Pack 0: GPU evaluation node (H100)
# --------------------------------------------------------------------------- #
#SBATCH --account=yil@h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH -C h100

# --------------------------------------------------------------------------- #
# Pack 1: CPU simulator node (many-core, no GPU required)
# --------------------------------------------------------------------------- #
#SBATCH --partition=cpu_p1
#SBATCH --account=yil@cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread

# --------------------------------------------------------------------------- #
# Common options
# --------------------------------------------------------------------------- #
#SBATCH --job-name=lewm_ood
#SBATCH --time=04:00:00
#SBATCH --output=logs/ood_%j.out
#SBATCH --error=logs/ood_%j.out

set -euo pipefail

source ~/.bashrc

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR=$WORK/thesis/le-wm-vanilla
SIMULATOR_DIR=$WORK/thesis/rl_opendeck_simulator
export STABLEWM_HOME=/lustre/fswork/projects/rech/yil/ugy35qd/.stable_worldmodel
export SIMULATOR_REPO=${SIMULATOR_DIR}
export ODSMR_ROOT=${STABLEWM_HOME}/odsmr_bin
export PYTHONUNBUFFERED=1

mkdir -p logs

# ── Resolve node hostnames from packjob environment ───────────────────────────
GPU_NODE=$(scontrol show hostnames "$SLURM_HET_GROUP_0_NODELIST" | head -1)
CPU_NODE=$(scontrol show hostnames "$SLURM_HET_GROUP_1_NODELIST" | head -1)

WORKER_PORT=5555
WORKER_ADDR="tcp://${CPU_NODE}:${WORKER_PORT}"
N_PROCESSES=$(( SLURM_HET_GROUP_1_CPUS_PER_TASK - 2 ))

echo "============================================================"
echo "  Task 4 OOD Detection"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  GPU node   : $GPU_NODE"
echo "  CPU node   : $CPU_NODE"
echo "  Worker addr: $WORKER_ADDR"
echo "  Processes  : $N_PROCESSES"
echo "  Start time : $(date)"
echo "============================================================"

# ── Checkpoints to evaluate ───────────────────────────────────────────────────
CKPTS=(
    "v6_sensor_sl1_w1:${STABLEWM_HOME}/turbofan_v6_sensor_sl1_w1/lewm_turbo_v6_sensor_sl1_w1_epoch_100_object.ckpt"
    "v6_sensor_sl1_w10:${STABLEWM_HOME}/turbofan_v6_sensor_sl1_w10/lewm_turbo_v6_sensor_sl1_w10_epoch_100_object.ckpt"
)

OUT_DIR=${STABLEWM_HOME}/eval_ood/${SLURM_JOB_ID}

# ── Start SimulatorWorker on the CPU pack (background) ────────────────────────
echo "[$(date +%H:%M:%S)] Starting SimulatorWorker on ${CPU_NODE} …"

srun \
    --het-group=1 \
    --nodes=1 \
    --ntasks=1 \
    bash -c "
        cd ${SIMULATOR_DIR}
        source .venv/bin/activate
        python -m hpc.simulator_worker \
            --port ${WORKER_PORT} \
            --sim-root ${ODSMR_ROOT} \
            --n-processes ${N_PROCESSES}
    " &
WORKER_PID=$!

echo "[$(date +%H:%M:%S)] Worker PID: ${WORKER_PID}"

# Give the ZMQ socket time to bind.
# eval_ood.py also calls client.wait_until_ready() which retries up to
# 20 times with 3 s intervals as an additional safety net.
sleep 10

# ── OOD Evaluation on the GPU pack (foreground) ───────────────────────────────
echo "[$(date +%H:%M:%S)] Starting OOD evaluation on ${GPU_NODE} …"

srun \
    --het-group=0 \
    --nodes=1 \
    --ntasks=1 \
    bash -c "
        cd ${PROJECT_DIR}
        source .venv/bin/activate
        python -u eval_ood.py \
            ${CKPTS[*]} \
            --hdf5     ${STABLEWM_HOME}/opendeck_2000_lewm.h5 \
            --sim_addr ${WORKER_ADDR} \
            --out_dir  ${OUT_DIR} \
            --n_ood_eps 50 \
            --seq_len   500
    "

EXIT_CODE=$?

# ── Cleanup ───────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Evaluation finished (exit code ${EXIT_CODE}). Stopping worker …"
kill "${WORKER_PID}" 2>/dev/null || true
wait "${WORKER_PID}" 2>/dev/null || true

echo "============================================================"
echo "  OOD results : ${OUT_DIR}"
echo "  Logs        : logs/ood_${SLURM_JOB_ID}.out"
echo "============================================================"

exit "${EXIT_CODE}"
