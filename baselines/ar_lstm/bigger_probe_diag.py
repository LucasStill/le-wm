"""
Bigger-probe diagnostic for AR-LSTM L1 (and any other run).
==========================================================

Loads a frozen world-model checkpoint, encodes the HI-probe data, and trains
a deliberately oversized TransformerProbe head to settle the question:

    Is the modest HI Pearson-r (~0.24) limited by ENCODER capacity, or by
    PROBE capacity?

Two outcomes:
  - bigger probe recovers Pearson > 0.4  → encoder is fine, default probe is
    the bottleneck. Major paper-worthy finding.
  - bigger probe stays around the default Pearson → encoder genuinely lacks
    HI-relevant signal; the JEPA / AR objective is the issue.

Usage
-----
    cd le-wm/
    source .venv/bin/activate
    python baselines/ar_lstm/bigger_probe_diag.py \\
        --ckpt   /home/lucas/.stable_worldmodel/ar_lstm_scenario4_w1_H16_S1_P4_h256l2/ar_lstm_s4_w1_H16_S1_P4_h256l2_epoch_6_object.ckpt \\
        --config /home/lucas/.stable_worldmodel/ar_lstm_scenario4_w1_H16_S1_P4_h256l2/config.yaml \\
        --data   /home/lucas/.stable_worldmodel/scenario4_train_lewm.h5 \\
        --label  L1-epoch5 \\
        --d-model 512 --num-layers 6
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

# ── path setup so `import hi_probe`, `from baselines.ar_lstm.model import …` work
LE_WM_ROOT = Path(__file__).resolve().parents[2]
if str(LE_WM_ROOT) not in sys.path:
    sys.path.insert(0, str(LE_WM_ROOT))

import hi_probe as hp                                   # noqa: E402
from baselines.ar_lstm.model import build_ar_lstm      # noqa: E402


def _strip_compile_prefix(state_dict: dict) -> dict:
    """torch.compile inserts a `_orig_mod.` segment in state_dict keys for any
    wrapped submodule. The segment can appear at the start (whole-model
    compile) OR inside a nested path (per-submodule compile, e.g. when only
    `world_model.encoder` is compiled → keys look like `encoder._orig_mod.X`).
    Strip every occurrence so the dict loads into an uncompiled module."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def load_world_model(ckpt_path: Path, cfg) -> torch.nn.Module:
    """Load checkpoint into a fresh (non-compiled) JEPA + LSTMPredictor instance.

    Robust to two save formats:
      1. torch.save(model)            — pickled module (this codebase's default)
      2. torch.save(model.state_dict) — plain state_dict
    """
    logging.info(f"[ckpt] loading {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    fresh = build_ar_lstm(cfg)

    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    elif hasattr(obj, "state_dict"):
        sd = obj.state_dict()
    elif isinstance(obj, dict):
        sd = obj
    else:
        # Whole-model unpickle worked AND it isn't a dict → use it directly.
        return obj

    sd = _strip_compile_prefix(sd)
    missing, unexpected = fresh.load_state_dict(sd, strict=False)
    if missing:
        logging.warning(f"[ckpt] missing keys: {missing[:5]}{'…' if len(missing) > 5 else ''}")
    if unexpected:
        logging.warning(f"[ckpt] unexpected keys: {unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")
    return fresh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path,
                        help="Hydra config.yaml saved alongside the ckpts.")
    parser.add_argument("--data", required=True, type=Path,
                        help="HDF5 with 'pixels' + 'observation.state' + 'action'.")
    parser.add_argument("--label", required=True, type=str,
                        help="Label for the result row, e.g. 'L1-epoch5'.")
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--n-probe-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--n-subsample", type=int, default=30_000)
    parser.add_argument("--enc-batch-size", type=int, default=2048)
    parser.add_argument("--out-csv", type=Path,
                        default=Path("logs/bigger_probe_results.csv"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    logging.info(f"=== Bigger-probe diagnostic — {args.label} ===")
    logging.info(f"d_model={args.d_model}, num_layers={args.num_layers}, "
                 f"nhead={args.nhead}, max_epochs={args.n_probe_epochs}, "
                 f"patience={args.patience}")

    # ── 1. load the run's config + ckpt ───────────────────────────────────────
    cfg = OmegaConf.load(args.config)
    with open_dict(cfg):
        # build_ar_lstm needs cfg.wm.action_dim (matches train_ar_lstm.py)
        if "action_dim" not in cfg.wm:
            cfg.wm.action_dim = 1

    world_model = load_world_model(args.ckpt, cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_model.to(device).eval()
    for p in world_model.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in world_model.parameters())
    logging.info(f"[model] frozen, {n_params:,} params on {device}")

    # ── 2. instantiate HIProbeCallback purely as a utility (NOT registered to a Trainer) ──
    cb = hp.HIProbeCallback(
        data_path        = str(args.data),
        img_size         = cfg.get("img_size", 28),
        train_split      = cfg.get("train_split", 0.9),
        seed             = cfg.get("seed", 3072),
        eval_interval    = 1,                            # unused in standalone mode
        n_probe_epochs   = args.n_probe_epochs,
        probe_lr         = args.probe_lr,
        probe_patience   = args.patience,
        d_model          = args.d_model,
        nhead            = args.nhead,
        num_layers       = args.num_layers,
        probe_dropout    = 0.1,
        probe_batch_size = args.probe_batch_size,
        probe_seq_len    = int(cfg.wm.history_size),     # match the in-training probe
        n_subsample      = args.n_subsample,
        enc_batch_size   = args.enc_batch_size,
        rul_max_horizon  = 300,
        obs_window_size  = int(cfg.get("obs_window_size", 1)),
        encoder_type     = "sensor",
    )

    # mock the Trainer / pl_module surface that setup() + _encode() touch
    log_dir = LE_WM_ROOT / "logs" / f"bigger_probe_{args.label}"
    log_dir.mkdir(parents=True, exist_ok=True)
    fake_trainer = SimpleNamespace(log_dir=str(log_dir))
    fake_pl      = SimpleNamespace(model=world_model)

    # ── 3. load + window the data ──────────────────────────────────────────────
    cb.setup(fake_trainer, fake_pl, "test")

    # ── 4. encode (frozen forward pass) ───────────────────────────────────────
    logging.info("[encode] train obs …")
    X_tr, idx_tr = cb._encode(fake_pl, cb._obs_tr, cb._ep_ids_tr)
    logging.info(f"[encode] train: {X_tr.shape}  (subset of {len(cb._obs_tr)} timesteps)")
    logging.info("[encode] test  obs …")
    X_te, idx_te = cb._encode(fake_pl, cb._obs_te, cb._ep_ids_te)
    logging.info(f"[encode] test : {X_te.shape}")

    # Align labels to the surviving indices (obs_window_size>1 path drops some)
    hi_tr     = cb._hi_tr[idx_tr]
    hi_te     = cb._hi_te[idx_te]
    ep_ids_tr = cb._ep_ids_tr[idx_tr]
    ep_ids_te = cb._ep_ids_te[idx_te]

    # ── 5. train the BIGGER HI probe ─────────────────────────────────────────
    logging.info(f"[probe] training HI probe (d_model={args.d_model}, "
                 f"num_layers={args.num_layers}) …")
    metrics = cb._train_and_eval_probe(
        X_tr, hi_tr, X_te, hi_te, ep_ids_tr, ep_ids_te, device,
    )

    pearson = np.array(metrics["pearson_r"], dtype=float)
    r2      = np.array(metrics["r2"], dtype=float)
    rmse    = np.array(metrics["rmse"], dtype=float)
    mean_p  = float(np.nanmean(pearson))
    mean_r2 = float(np.mean(r2))
    mean_rm = float(np.mean(rmse))

    # ── 6. RUL probe (small, included for completeness) ──────────────────────
    rul_metrics = None
    if cb._rul_tr is not None:
        logging.info("[probe] training RUL probe …")
        rul_tr = cb._rul_tr[idx_tr]
        rul_te = cb._rul_te[idx_te]
        rul_metrics = cb._train_and_eval_rul_probe(
            X_tr, rul_tr, X_te, rul_te, ep_ids_tr, ep_ids_te, device,
        )

    # ── 7. report ─────────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print(f"  {args.label}   d_model={args.d_model}  num_layers={args.num_layers}")
    print("=" * 64)
    print(f"  HI mean Pearson-r : {mean_p:.4f}")
    print(f"  HI mean R²        : {mean_r2:.4f}")
    print(f"  HI mean RMSE      : {mean_rm:.5f}")
    print()
    print(f"  Per-component (HI_0..{len(cb._hi_names)-1}):")
    for i, name in enumerate(cb._hi_names):
        print(f"    {name:<32s}  Pearson={pearson[i]:+.4f}  "
              f"R²={r2[i]:+.4f}  RMSE={rmse[i]:.5f}")
    if rul_metrics is not None:
        print()
        print(f"  RUL  Pearson={rul_metrics.get('pearson_r', float('nan')):+.4f}  "
              f"R²={rul_metrics.get('r2', float('nan')):+.4f}  "
              f"RMSE={rul_metrics.get('rmse', float('nan')):.4f} steps")
    print("=" * 64)

    # ── 8. append to CSV ──────────────────────────────────────────────────────
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out_csv.exists()
    with open(args.out_csv, "a") as f:
        if write_header:
            f.write("label,d_model,num_layers,task,component,r2,rmse,pearson_r\n")
        for i, name in enumerate(cb._hi_names):
            f.write(f"{args.label},{args.d_model},{args.num_layers},hi,{name},"
                    f"{r2[i]:.6f},{rmse[i]:.6f},{pearson[i]:.6f}\n")
        f.write(f"{args.label},{args.d_model},{args.num_layers},hi,MEAN,"
                f"{mean_r2:.6f},{mean_rm:.6f},{mean_p:.6f}\n")
        if rul_metrics is not None:
            f.write(f"{args.label},{args.d_model},{args.num_layers},rul,RUL,"
                    f"{rul_metrics.get('r2', 'nan'):.6f},"
                    f"{rul_metrics.get('rmse', 'nan'):.6f},"
                    f"{rul_metrics.get('pearson_r', 'nan'):.6f}\n")
    logging.info(f"[csv] appended to {args.out_csv}")


if __name__ == "__main__":
    main()
