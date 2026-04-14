#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  eval_ood.slurm — Task 4 OOD Detection (single H100 node, self-contained)
#
#  The turbofan simulator worker runs as a BACKGROUND PROCESS on the same H100
#  node as the GPU eval. This sidesteps the Jean-Zay cross-partition network
#  isolation (gpu_p6 cannot reach cpu_p1 over TCP).
#
#  H100 nodes have 48 physical CPUs. We request 24 for the simulator worker
#  and leave the rest for the GPU eval / data loading.
#
#  Workflow (single sbatch, no manual steps):
#    sbatch eval_ood.slurm
#
#  To skip the simulator (dataset-proxy mode):
#    USE_SIMULATOR=0 sbatch eval_ood.slurm
# ══════════════════════════════════════════════════════════════════════════════

#SBATCH --account=yil@h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
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

echo "============================================================"
echo "  Task 4 OOD Detection"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $(hostname -s)  (CPUs: $SLURM_CPUS_PER_TASK)"
echo "  Start time : $(date)"
echo "============================================================"

# ── Checkpoints to evaluate ───────────────────────────────────────────────────
CKPTS=(
    "v6_sensor_sl1_w1:${STABLEWM_HOME}/turbofan_v6_sensor_sl1_w1/lewm_turbo_v6_sensor_sl1_w1_epoch_100_object.ckpt"
    "v6_sensor_sl1_w10:${STABLEWM_HOME}/turbofan_v6_sensor_sl1_w10/lewm_turbo_v6_sensor_sl1_w10_epoch_100_object.ckpt"
)

OUT_DIR=${STABLEWM_HOME}/eval_ood/${SLURM_JOB_ID}
mkdir -p "${OUT_DIR}"

# ── Decide simulator mode ─────────────────────────────────────────────────────
USE_SIMULATOR="${USE_SIMULATOR:-1}"   # default ON — worker runs on same node

SIM_FLAGS="--no_simulator"
WORKER_PID=""

if [[ "$USE_SIMULATOR" == "1" ]]; then
    # ── Start the simulator worker as a background process on this node ───────
    # Port scoped to this job to avoid collisions if two OOD jobs overlap.
    WORKER_PORT=$(( 5555 + (SLURM_JOB_ID % 1000) ))
    WORKER_ADDR="tcp://localhost:${WORKER_PORT}"
    ODSMR_ROOT="${SIMULATOR_DIR}/OpenDeckSMR/src/odsmr/bin/DeckSMR"

    # Leave 8 CPUs for the GPU eval process; give the rest to DeckSMR workers.
    N_SIM_PROCS=$(( SLURM_CPUS_PER_TASK - 8 ))
    [[ $N_SIM_PROCS -lt 1 ]] && N_SIM_PROCS=1

    echo "[$(date +%H:%M:%S)] Starting simulator worker (port=${WORKER_PORT}, n_proc=${N_SIM_PROCS}) …"

    cd "${SIMULATOR_DIR}"
    uv run --no-dev --offline python -m hpc.simulator_worker \
        --port        "${WORKER_PORT}"  \
        --sim-root    "${ODSMR_ROOT}"   \
        --n-processes "${N_SIM_PROCS}"  \
        &
    WORKER_PID=$!

    echo "[$(date +%H:%M:%S)] Worker PID=${WORKER_PID}, waiting up to 60s for it to be ready …"

    # ── Wait for worker to become healthy (poll localhost) ────────────────────
    cd "${PROJECT_DIR}"
    source .venv/bin/activate

    python - <<PYEOF
import pickle, sys, time
try:
    import zmq
except ImportError:
    print("ERROR: zmq not installed in le-wm venv. Run: pip install pyzmq")
    sys.exit(1)

addr = "${WORKER_ADDR}"
ctx  = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.setsockopt(zmq.LINGER, 0)
sock.connect(addr)

for attempt in range(1, 21):        # up to 60 s (20 × 3s)
    sock.send(pickle.dumps({"type": "health", "request_id": "startup"}))
    if sock.poll(3_000):
        resp = pickle.loads(sock.recv())
        if resp.get("status") == "ok":
            print(f"[$(date +%H:%M:%S)] Worker ready after {attempt} attempts")
            sock.close(); ctx.term()
            sys.exit(0)
    else:
        print(f"  [startup] attempt {attempt}/20 — not ready yet …")
        # Must recreate socket after a missed REP
        sock.close()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(addr)

print("ERROR: Worker did not become ready after 60s")
sock.close(); ctx.term()
sys.exit(1)
PYEOF

    SIM_FLAGS="--sim_addr ${WORKER_ADDR}"
    echo "[$(date +%H:%M:%S)] Simulator worker ready at ${WORKER_ADDR}"
else
    echo "[$(date +%H:%M:%S)] USE_SIMULATOR=0 — running in dataset-proxy mode."
    cd "${PROJECT_DIR}"
    source .venv/bin/activate
fi

# ── Cleanup: kill the worker when this script exits ───────────────────────────
cleanup() {
    if [[ -n "$WORKER_PID" ]]; then
        echo "[$(date +%H:%M:%S)] Stopping simulator worker (PID=${WORKER_PID}) …"
        kill "${WORKER_PID}" 2>/dev/null || true
        wait "${WORKER_PID}" 2>/dev/null || true
        echo "[$(date +%H:%M:%S)] Worker stopped."
    fi
}
trap cleanup EXIT

# ── Run OOD evaluation ────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Starting OOD evaluation …"
echo ""

cd "${PROJECT_DIR}"

python -u eval_ood.py \
    "${CKPTS[@]}" \
    --hdf5    "${STABLEWM_HOME}/opendeck_2000_lewm.h5" \
    ${SIM_FLAGS} \
    --out_dir "${OUT_DIR}" \
    --n_ood_eps 50 \
    --seq_len   500

echo ""
echo "============================================================"
echo "  OOD results : ${OUT_DIR}"
echo "  End time    : $(date)"
echo "============================================================"
