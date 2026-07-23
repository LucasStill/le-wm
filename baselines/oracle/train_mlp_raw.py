#!/usr/bin/env python3
"""Nonlinear raw sensor baseline for the TurboSens2.1 ladder.

Reviewer request (83ZP): the null ladder has a linear raw floor; add a
nonlinear one. This trains a plain MLP on the raw 176 dim observation
under MSE supervision on the standardized true state, then exposes the
penultimate layer through the JEPA style encode() contract so the
standard exporter and ridge probe evaluate it exactly like every other
representation. No attention, no tokens: this measures what nonlinear
regression on the raw observation achieves, nothing architectural.

Run from the le-wm repo root:
    python baselines/oracle/train_mlp_raw.py \
        --data $STABLEWM_HOME/ts21/turbosens2_train_lewm.h5 \
        --seed 0 --epochs 10 --out-dir $STABLEWM_HOME/ts21_MLPRAW_seed0
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn


class MLPRaw(nn.Module):
    """MLP on raw sensors; penultimate layer is the probed embedding."""

    def __init__(self, n_in=176, hidden=512, d_emb=64, n_state=10):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_in, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d_emb), nn.SiLU(),
        )
        self.head = nn.Linear(d_emb, n_state)
        self.obs_window_size = 1

    def forward(self, x):
        return self.head(self.body(x))

    @torch.no_grad()
    def encode(self, info):
        pixels = info["pixels"].float()
        b, t = pixels.shape[:2]
        emb = self.body(pixels.reshape(b * t, -1))
        info["emb"] = emb.reshape(b, t, -1)[:, :1, :]
        return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.data, "r") as f:
        X = f["pixels"][:].reshape(len(f["pixels"]), -1).astype(np.float32)
        Y = f["observation.state"][:].astype(np.float32)
    y_mu, y_sd = Y.mean(0), Y.std(0) + 1e-8
    Yn = (Y - y_mu) / y_sd

    # Standardize inputs too (a plain MLP has no per token norm).
    x_mu, x_sd = X.mean(0), X.std(0) + 1e-8
    Xn = (X - x_mu) / x_sd

    model = MLPRaw(n_in=X.shape[1]).to(device)
    print(f"MLPRaw: {sum(p.numel() for p in model.parameters()):,} params",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps = len(Xn) // args.batch_size
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * steps)

    Xt, Yt = torch.from_numpy(Xn), torch.from_numpy(Yn)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(steps):
            idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
            xb = Xt[idx].to(device, non_blocking=True)
            yb = Yt[idx].to(device, non_blocking=True)
            loss = nn.functional.mse_loss(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        print(f"epoch {ep + 1}/{args.epochs}  mse={tot / steps:.5f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    model.eval().cpu()
    # Bake the input scaler into the first layer so encode() works on
    # raw inputs without carrying the scaler around.
    with torch.no_grad():
        w0 = model.body[0]
        w0.bias.copy_(w0.bias - (w0.weight @ torch.from_numpy(
            (x_mu / x_sd).astype(np.float32))))
        w0.weight.copy_(w0.weight / torch.from_numpy(
            x_sd.astype(np.float32)))
    ckpt = out_dir / f"mlpraw_seed{args.seed}_epoch_{args.epochs}_object.ckpt"
    torch.save(model, ckpt)
    print(f"saved: {ckpt}")


if __name__ == "__main__":
    main()
