"""
Smoke test for the RSSM baseline.

Run me from the le-wm root:

    cd ~/thesis/le-wm
    source .venv/bin/activate
    python baselines/rssm/smoke_test.py

Verifies:
  1. Model builds with synthetic config.
  2. Forward pass returns the expected shapes for post, prior, recon, feat.
  3. JEPA-compatible encode() returns info["emb"] of shape (B, T, feat_dim).
  4. Backward pass produces non-NaN gradients.
  5. KL is positive and finite.

Does not touch HDF5, wandb, Hydra, or Lightning — pure model wiring check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_LE_WM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LE_WM_ROOT))

from baselines.rssm.model import RSSMWorldModel  # noqa: E402


def main():
    torch.manual_seed(0)
    n_sensors  = 176
    num_actions = 7
    B, T = 2, 8

    model = RSSMWorldModel(
        n_sensors      = n_sensors,
        num_actions    = num_actions,
        embed_size     = 64,
        encoder_hidden = 64,
        decoder_hidden = 64,
        rssm_kwargs    = dict(stoch=8, discrete=8, deter=64, hidden=64),
    )
    feat_size = model.dynamics.feat_size
    print(f"feat_size = {feat_size}  (expected {8*8 + 64} = 128)")
    assert feat_size == 128

    obs    = torch.randn(B, T, n_sensors)
    action = torch.randint(0, num_actions, (B, T)).float()

    # ── world_model_step (training path) ───────────────────────────────────
    out = model.world_model_step({"pixels": obs, "action": action})
    print("post['stoch'].shape   =", tuple(out["post"]["stoch"].shape))
    print("post['deter'].shape   =", tuple(out["post"]["deter"].shape))
    print("prior['logit'].shape  =", tuple(out["prior"]["logit"].shape))
    print("feat.shape            =", tuple(out["feat"].shape))
    print("recon.shape           =", tuple(out["recon"].shape))
    assert out["post"]["stoch"].shape  == (B, T, 8, 8)
    assert out["post"]["deter"].shape  == (B, T, 64)
    assert out["prior"]["logit"].shape == (B, T, 8, 8)
    assert out["feat"].shape           == (B, T, feat_size)
    assert out["recon"].shape          == (B, T, n_sensors)

    recon_loss = (out["recon"] - out["target"]).pow(2).mean()
    kl_total, dyn_loss, rep_loss = model.dynamics.kl_loss(
        out["post"], out["prior"], free=1.0, dyn_scale=0.5, rep_scale=0.1,
    )
    print(f"recon_loss = {recon_loss.item():.4f}")
    print(f"kl_total   = {kl_total.mean().item():.4f}  "
          f"(dyn={dyn_loss.mean().item():.4f}  rep={rep_loss.mean().item():.4f})")
    assert torch.isfinite(recon_loss).all()
    assert torch.isfinite(kl_total).all()
    assert kl_total.mean().item() >= 0.0  # KL is non-negative

    loss = recon_loss + kl_total.mean()
    loss.backward()

    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for _ in model.parameters())
    n_nan = sum(1 for p in model.parameters()
                if p.grad is not None and torch.isnan(p.grad).any())
    print(f"params with grad = {n_grad}/{n_total}, NaN grads = {n_nan}")
    assert n_grad > 0
    assert n_nan == 0

    # ── encode() (HI probe path) ───────────────────────────────────────────
    model.zero_grad()
    snap = torch.randn(B, 1, n_sensors)            # snapshot probe input
    info = model.encode({"pixels": snap})
    print("snapshot encode emb.shape =", tuple(info["emb"].shape))
    assert info["emb"].shape == (B, 1, feat_size)

    win = torch.randn(B, 4, n_sensors)             # 4-step trajectory probe
    info = model.encode({"pixels": win})
    print("window encode emb.shape   =", tuple(info["emb"].shape))
    assert info["emb"].shape == (B, 4, feat_size)

    print()
    print("OK — RSSM model wiring is correct.")


if __name__ == "__main__":
    main()
