#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  test_zmq_connectivity.slurm — Diagnose GPU→CPU ZMQ connectivity on Jean-Zay
#
#  Prerequisites:
#    1.  The CPU simulator worker must be running:
#        sbatch $WORK/thesis/rl_opendeck_simulator/hpc/slurm/launch_cpu_worker.sh
#    2.  Wait until:  cat $WORK/.simulator_addr   # shows tcp://rXiYnZ:5555
#    3.  sbatch test_zmq_connectivity.slurm
#
#  The script runs from an H100 GPU node and probes the CPU worker with
#  increasingly heavy tests:
#    Step 1 — Parse address file
#    Step 2 — Raw TCP connect (bash /dev/tcp)
#    Step 3 — Python socket connect
#    Step 4 — ZMQ health check (pickle protocol)
#    Step 5 — Minimal simulate request (1 state vector)
# ══════════════════════════════════════════════════════════════════════════════

#SBATCH --account=yil@h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH -C h100
#SBATCH --job-name=zmq_diag
#SBATCH --time=00:15:00
#SBATCH --output=logs/zmq_diag_%j.out
#SBATCH --error=logs/zmq_diag_%j.out

set -eo pipefail
source ~/.bashrc

PROJECT_DIR=$WORK/thesis/le-wm-vanilla
ADDR_FILE="${WORK}/.simulator_addr"

mkdir -p logs

echo "════════════════════════════════════════════════════════════"
echo "  ZMQ Connectivity Diagnostic"
echo "  Job ID    : $SLURM_JOB_ID"
echo "  This node : $(hostname -s)  ($(hostname -i 2>/dev/null || echo unknown))"
echo "  Time      : $(date)"
echo "════════════════════════════════════════════════════════════"
echo ""

# ── Step 1: Parse address file ────────────────────────────────────────────────
echo "── Step 1: Address file ─────────────────────────────────────"
if [[ ! -f "$ADDR_FILE" ]]; then
    echo "  FAIL  $ADDR_FILE does not exist."
    echo "  → Make sure the CPU worker is running and has written its address."
    exit 1
fi

WORKER_ADDR=$(cat "$ADDR_FILE")
echo "  OK    Address: ${WORKER_ADDR}"

# Parse host and port from tcp://hostname:port
WORKER_HOST=$(echo "$WORKER_ADDR" | sed 's|tcp://||' | cut -d: -f1)
WORKER_PORT=$(echo "$WORKER_ADDR" | sed 's|tcp://||' | cut -d: -f2)
echo "  OK    Host=${WORKER_HOST}  Port=${WORKER_PORT}"
echo ""

# ── Step 2: Raw TCP connect via bash /dev/tcp ─────────────────────────────────
echo "── Step 2: Raw TCP connect (bash /dev/tcp) ──────────────────"
if timeout 5 bash -c "echo '' > /dev/tcp/${WORKER_HOST}/${WORKER_PORT}" 2>/dev/null; then
    echo "  OK    TCP port ${WORKER_PORT} is reachable on ${WORKER_HOST}"
else
    echo "  FAIL  Cannot open /dev/tcp/${WORKER_HOST}/${WORKER_PORT}"
    echo "        Possible causes:"
    echo "          • CPU worker not running (check: squeue -u \$USER)"
    echo "          • Firewall blocking inter-partition traffic (gpu_p6 → cpu_p1)"
    echo "          • Wrong hostname in $ADDR_FILE"
    echo ""
    echo "        Trying alternative ports / checks …"
    for alt_port in 5556 5557 6000; do
        if timeout 3 bash -c "echo '' > /dev/tcp/${WORKER_HOST}/${alt_port}" 2>/dev/null; then
            echo "  INFO  Port ${alt_port} IS reachable (unexpected)"
        fi
    done
    # Don't exit — continue with Python diagnostics for more info
fi
echo ""

# ── Step 3: Python socket connect ────────────────────────────────────────────
echo "── Step 3: Python socket connect ────────────────────────────"
cd "${PROJECT_DIR}"
source .venv/bin/activate

python - <<PYEOF
import socket, sys, time

host = "${WORKER_HOST}"
port = int("${WORKER_PORT}")

print(f"  Attempting TCP connection to {host}:{port} …")
try:
    s = socket.create_connection((host, port), timeout=5)
    s.close()
    print(f"  OK    Python socket connected successfully")
except socket.timeout:
    print(f"  FAIL  Connection timed out after 5s")
    print(f"        This strongly suggests a firewall between partitions.")
    sys.exit(0)
except ConnectionRefusedError:
    print(f"  FAIL  Connection refused — worker not listening on {host}:{port}")
    sys.exit(0)
except OSError as e:
    print(f"  FAIL  OSError: {e}")
    sys.exit(0)
PYEOF
echo ""

# ── Step 4: ZMQ health check ──────────────────────────────────────────────────
echo "── Step 4: ZMQ health check (5s timeout) ────────────────────"
python - <<PYEOF
import sys, pickle, time

addr = "${WORKER_ADDR}"
print(f"  Connecting ZMQ REQ to {addr} …")

try:
    import zmq
except ImportError:
    print("  FAIL  zmq not installed in this venv. Run: pip install pyzmq")
    sys.exit(0)

ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.setsockopt(zmq.LINGER, 0)
sock.connect(addr)

request = {"type": "health", "request_id": "diag-hc"}
t0 = time.time()
sock.send(pickle.dumps(request))

if sock.poll(5_000):   # 5 second timeout
    response = pickle.loads(sock.recv())
    elapsed = time.time() - t0
    if response.get("status") == "ok":
        print(f"  OK    Worker healthy! Round-trip: {elapsed*1000:.1f} ms")
    else:
        print(f"  WARN  Unexpected response: {response}")
else:
    elapsed = time.time() - t0
    print(f"  FAIL  No response after {elapsed:.1f}s")
    print(f"        ZMQ can connect at TCP level but worker did not respond.")
    print(f"        → Worker may be busy, crashed, or the binary failed to start.")

sock.close()
ctx.term()
PYEOF
echo ""

# ── Step 5: Minimal simulate request ─────────────────────────────────────────
echo "── Step 5: Minimal simulate (1 state, 1 context) ────────────"
python - <<PYEOF
import sys, pickle, time, uuid
import numpy as np

addr = "${WORKER_ADDR}"

try:
    import zmq
except ImportError:
    print("  SKIP  zmq not installed")
    sys.exit(0)

# Nominal healthy state — all degradation deltas at 0.0
state = np.zeros((1, 10), dtype=np.float64)
context = [{
    "PHASE_TYPE": "CR",
    "DTAMB": 10.0,
    "ALT": 35000.0,
    "MACH": 0.78,
    "COMMAND": 25000.0,
}]

request = {
    "type": "simulate",
    "request_id": uuid.uuid4().hex[:8],
    "states": state,
    "contexts": context,
}

ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.setsockopt(zmq.LINGER, 0)
sock.connect(addr)

print(f"  Sending simulate request (1 state) to {addr} …")
t0 = time.time()
sock.send(pickle.dumps(request))

if sock.poll(30_000):   # 30 second timeout for first simulation (binary startup)
    response = pickle.loads(sock.recv())
    elapsed = time.time() - t0
    if response.get("error"):
        print(f"  FAIL  Worker returned error: {response['error']}")
    else:
        obs = response["observations"]
        print(f"  OK    Got observations shape={obs.shape}  dtype={obs.dtype}")
        print(f"        Values: {obs.flatten()}")
        print(f"        Round-trip: {elapsed*1000:.0f} ms")
else:
    elapsed = time.time() - t0
    print(f"  FAIL  No response after {elapsed:.0f}s (binary may need longer startup)")

sock.close()
ctx.term()
PYEOF
echo ""

echo "════════════════════════════════════════════════════════════"
echo "  Diagnostic complete — $(date)"
echo "════════════════════════════════════════════════════════════"
