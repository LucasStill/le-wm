#!/usr/bin/env python3
"""Supervised oracle for the TurboSens2.1 reference ladder.

Trains the same 1.14M SensorEncoder backbone used by every SSL run,
end to end with a linear head under MSE supervision on the ground truth
state s (per component standardised — the July 2026 audit showed the
old oracle's unstandardised targets let large scale components dominate
and deflated it below the raw linear floor).

The checkpoint is a whole model object exposing the JEPA style
`encode(info) -> info["emb"]` contract and `obs_window_size = 1`, so
evaluation/export_embeddings_lewm.py consumes it unchanged: the probed
representation is the frozen supervised embedding, giving the ladder
its learned ceiling.

Run from the le-wm repo root:
    python baselines/oracle/train_oracle.py \
        --data $STABLEWM_HOME/ts21/turbosens2_train_lewm.h5 \
        --seed 0 --epochs 10 \
        --out-dir $STABLEWM_HOME/ts21_oracle_seed0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jepa import SensorEncoder  # noqa: E402


class OracleModel(nn.Module):
    """SensorEncoder + linear head, JEPA compatible encode() contract."""

    def __init__(self, n_sensors=176, d_model=64, max_sensors=200,
                 n_state=10):
        super().__init__()
        self.encoder = SensorEncoder(
            n_sensors=n_sensors, d_model=d_model, nhead=4,
            num_layers=2, dropout=0.1, max_sensors=max_sensors)
        self.head = nn.Linear(d_model, n_state)
        self.obs_window_size = 1

    def forward(self, x):                      # x: (B, n_sensors)
        return self.head(self.encoder(x))

    @torch.no_grad()
    def encode(self, info):
        """Mirror of JEPA._encode_sensor at W=1 for the exporter."""
        pixels = info["pixels"].float()        # (B, T_raw, ...)
        b, t = pixels.shape[:2]
        sensors = pixels.reshape(b, t, -1)
        emb = self.encoder(sensors.reshape(b * t, -1))
        info["emb"] = emb.reshape(b, t, -1)[:, :1, :]
        return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="lewm train H5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.data} ...", flush=True)
    with h5py.File(args.data, "r") as f:
        X = f["pixels"][:].reshape(len(f["pixels"]), -1).astype(np.float32)
        Y = f["observation.state"][:].astype(np.float32)
    print(f"X={X.shape} Y={Y.shape}", flush=True)

    # Per component target standardisation (the audit's key fix).
    y_mu, y_sd = Y.mean(0), Y.std(0) + 1e-8
    Yn = (Y - y_mu) / y_sd

    model = OracleModel(n_sensors=X.shape[1]).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"OracleModel: {n_par:,} params", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps_per_epoch = len(X) // args.batch_size
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * steps_per_epoch)

    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Yn)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        tot, nb = 0.0, 0
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
            xb = Xt[idx].to(device, non_blocking=True)
            yb = Yt[idx].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                loss = nn.functional.mse_loss(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item(); nb += 1
        print(f"epoch {ep + 1}/{args.epochs}  mse={tot / nb:.5f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    model.eval().cpu()
    # Stash the target scaler on the object for completeness.
    model.y_mu = torch.from_numpy(y_mu)
    model.y_sd = torch.from_numpy(y_sd)
    ckpt = out_dir / f"oracle_seed{args.seed}_epoch_{args.epochs}_object.ckpt"
    torch.save(model, ckpt)
    print(f"saved: {ckpt}")


if __name__ == "__main__":
    main()
