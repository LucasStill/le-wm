#!/usr/bin/env python3
"""
test_zmq.py — Standalone ZMQ connectivity probe for the simulator worker.

Can be run interactively on any compute or login node:

    # From le-wm venv:
    python test_zmq.py                          # reads $WORK/.simulator_addr
    python test_zmq.py tcp://r1i3n21:5555       # explicit address
    python test_zmq.py --port 5556              # override port only

Steps performed:
  1. TCP connect   (raw socket, 5s timeout)
  2. ZMQ health    (pickle REQ, 5s timeout)
  3. Simulate x1   (1 state, 30s timeout)
  4. Simulate x100 (throughput benchmark, optional)
"""

from __future__ import annotations

import argparse
import os
import pickle
import socket
import sys
import time
import uuid

import numpy as np


# ── helpers ────────────────────────────────────────────────────────────────────

def ok(msg: str) -> None:
    print(f"  \033[32mOK  \033[0m  {msg}")

def fail(msg: str) -> None:
    print(f"  \033[31mFAIL\033[0m  {msg}")

def info(msg: str) -> None:
    print(f"  \033[33mINFO\033[0m  {msg}")


def parse_addr(addr: str) -> tuple[str, int]:
    """Return (host, port) from 'tcp://host:port'."""
    stripped = addr.replace("tcp://", "")
    host, port_str = stripped.rsplit(":", 1)
    return host, int(port_str)


# ── tests ──────────────────────────────────────────────────────────────────────

def test_tcp(host: str, port: int) -> bool:
    print(f"\n── Step 1: Raw TCP connect to {host}:{port} ──────────────────")
    try:
        t0 = time.time()
        s = socket.create_connection((host, port), timeout=5)
        s.close()
        ok(f"Connected in {(time.time()-t0)*1000:.0f} ms")
        return True
    except socket.timeout:
        fail("Timed out after 5s — firewall likely blocking inter-node traffic")
        return False
    except ConnectionRefusedError:
        fail("Connection refused — worker not listening on this port")
        return False
    except OSError as exc:
        fail(f"OSError: {exc}")
        return False


def test_zmq_health(addr: str) -> bool:
    print(f"\n── Step 2: ZMQ health check ──────────────────────────────────")
    try:
        import zmq
    except ImportError:
        fail("zmq not installed — run: pip install pyzmq")
        return False

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(addr)

    req = {"type": "health", "request_id": "diag-hc"}
    t0 = time.time()
    sock.send(pickle.dumps(req))

    if sock.poll(5_000):
        resp = pickle.loads(sock.recv())
        elapsed_ms = (time.time() - t0) * 1000
        if resp.get("status") == "ok":
            ok(f"Worker healthy — round-trip {elapsed_ms:.1f} ms")
            sock.close(); ctx.term()
            return True
        else:
            fail(f"Unexpected response: {resp}")
    else:
        fail(f"No response after 5s — worker may have crashed or is busy")

    sock.close(); ctx.term()
    return False


def test_simulate_1(addr: str) -> bool:
    print(f"\n── Step 3: Simulate 1 state (30s timeout) ────────────────────")
    try:
        import zmq
    except ImportError:
        fail("zmq not installed"); return False

    # Nominal healthy state
    state = np.ones((1, 10), dtype=np.float64)
    context = [{"PHASE_TYPE": "climb", "DTAMB": 0.0, "ALT": 10000.0,
                 "MACH": 0.5, "COMMAND": 1.0}]
    req = {"type": "simulate", "request_id": uuid.uuid4().hex[:8],
           "states": state, "contexts": context}

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(addr)

    t0 = time.time()
    sock.send(pickle.dumps(req))

    if sock.poll(30_000):
        resp = pickle.loads(sock.recv())
        elapsed_ms = (time.time() - t0) * 1000
        if resp.get("error"):
            fail(f"Worker error: {resp['error']}")
            sock.close(); ctx.term()
            return False
        obs = resp["observations"]
        ok(f"Got shape={obs.shape} dtype={obs.dtype}  ({elapsed_ms:.0f} ms)")
        print(f"         Sensor values: {np.round(obs.flatten(), 4)}")
        sock.close(); ctx.term()
        return True
    else:
        fail(f"No response after 30s")
        sock.close(); ctx.term()
        return False


def test_simulate_batch(addr: str, n: int = 100) -> None:
    print(f"\n── Step 4: Throughput — simulate {n} states ──────────────────")
    try:
        import zmq
    except ImportError:
        fail("zmq not installed"); return

    states = np.ones((n, 10), dtype=np.float64)
    contexts = [{"PHASE_TYPE": "climb", "DTAMB": 0.0, "ALT": 10000.0,
                  "MACH": 0.5, "COMMAND": 1.0}] * n
    req = {"type": "simulate", "request_id": uuid.uuid4().hex[:8],
           "states": states, "contexts": contexts}

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(addr)

    t0 = time.time()
    sock.send(pickle.dumps(req))

    if sock.poll(120_000):   # 2 min for batch
        resp = pickle.loads(sock.recv())
        elapsed = time.time() - t0
        if resp.get("error"):
            fail(f"Worker error: {resp['error']}")
        else:
            obs = resp["observations"]
            rate = n / elapsed
            ok(f"shape={obs.shape}  elapsed={elapsed:.1f}s  ({rate:.1f} states/s)")
    else:
        fail("No response after 120s")

    sock.close(); ctx.term()


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ZMQ simulator connectivity test")
    parser.add_argument("address", nargs="?", default=None,
                        help="ZMQ address e.g. tcp://r1i3n21:5555 "
                             "(default: read from $WORK/.simulator_addr)")
    parser.add_argument("--port", type=int, default=None,
                        help="Override port in address file")
    parser.add_argument("--batch", type=int, default=0,
                        help="Also run throughput test with N states (0=skip)")
    args = parser.parse_args()

    # Resolve address
    if args.address:
        addr = args.address
        if not addr.startswith("tcp://"):
            addr = f"tcp://{addr}"
    else:
        addr_file = os.path.join(os.environ.get("WORK", os.environ["HOME"]),
                                 ".simulator_addr")
        if not os.path.exists(addr_file):
            print(f"ERROR: address file not found: {addr_file}")
            print("Either pass the address explicitly or start the CPU worker first.")
            sys.exit(1)
        addr = open(addr_file).read().strip()
        info(f"Read address from {addr_file}: {addr}")

    if args.port:
        host, _ = parse_addr(addr)
        addr = f"tcp://{host}:{args.port}"
        info(f"Port overridden → {addr}")

    host, port = parse_addr(addr)

    print(f"\n{'═'*60}")
    print(f"  ZMQ Diagnostic   {addr}")
    print(f"  This host : {socket.gethostname()}")
    print(f"  Time      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*60}")

    # Run tests in order; stop early if TCP fails
    if not test_tcp(host, port):
        print("\n  TCP failed — skipping ZMQ tests.")
        print("  Hint: try  nc -vz <host> <port>  or check firewall rules.")
        sys.exit(2)

    if not test_zmq_health(addr):
        print("\n  Health check failed — skipping simulate test.")
        sys.exit(2)

    test_simulate_1(addr)

    if args.batch > 0:
        test_simulate_batch(addr, args.batch)

    print(f"\n{'═'*60}\n  Done — {time.strftime('%H:%M:%S')}\n{'═'*60}\n")


if __name__ == "__main__":
    main()
