"""
Sanity / lower-bound baselines for the dataset paper.
=====================================================

Runs the same eval_sweep task-1 probe pipeline (~770K probe-train windows,
TransformerProbe head with default sizing) on two ENCODER-FREE / RANDOM
baselines:

  raw_sensors      — skip the encoder entirely. Probe inputs are raw
                     176-dim sensor vectors.
  random_encoder   — instantiate a JEPA + LSTMPredictor with build_ar_lstm,
                     keep at random init (no checkpoint loaded). Encode
                     normally → 64-dim embeddings → probe.

These give two scale anchors:
  - raw_sensors Pearson = "how much HI signal is in the raw 176-dim
    feature vector before any encoder pre-processing?"
  - random_encoder Pearson = "what does an architecturally-identical
    random-init encoder achieve, with no training?"

The L1 / L2 calibrated numbers (0.564, 0.495) only become interpretable
once we know what these lower bounds are.

Run on both regular test and test_hard, both seq_len 1/10/50, mirroring
eval_sweep --tasks 1.

Usage
-----
    cd le-wm/
    source .venv/bin/activate
    python baselines/ar_lstm/sanity_baselines.py \\
        --hdf5 /home/lucas/.stable_worldmodel/scenario4_test_lewm.h5 \\
        --config /home/lucas/.stable_worldmodel/ar_lstm_scenario4_w1_H16_S1_P4_h256l2/config.yaml \\
        --modes raw_sensors random_encoder \\
        --label-suffix te \\
        --out-csv logs/sanity_baselines_results.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

LE_WM_ROOT = Path(__file__).resolve().parents[2]
if str(LE_WM_ROOT) not in sys.path:
    sys.path.insert(0, str(LE_WM_ROOT))

# eval_sweep imports JEPA / LSTMPredictor at the module level so torch.load
# can resolve the pickled classes — same trick we use here.
import eval_sweep as es                                # noqa: E402
from baselines.ar_lstm.model import build_ar_lstm     # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True, type=Path,
                        help="Test HDF5 (scenario4_test_lewm.h5 or _test_hard_…).")
    parser.add_argument("--config", required=True, type=Path,
                        help="L1/L2 hydra config.yaml (used to build the random encoder).")
    parser.add_argument("--modes", nargs="+",
                        choices=["raw_sensors", "random_encoder"],
                        default=["raw_sensors", "random_encoder"])
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[1, 10, 50])
    parser.add_argument("--label-suffix", default="te",
                        help="Tag appended to result rows ('te' for regular test, 'th' for test_hard).")
    parser.add_argument("--out-csv", type=Path,
                        default=Path("logs/sanity_baselines_results.csv"))
    parser.add_argument("--seed", type=int, default=42,
                        help="Seeds (a) the random_encoder init and (b) eval_sweep.train_probe "
                             "(probe head + dataloader shuffle). Pass different seeds to get "
                             "mean ± std for paper-grade results.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"=== Sanity baselines ({', '.join(args.modes)}) ===")
    logging.info(f"hdf5={args.hdf5}  device={device}")

    cfg = OmegaConf.load(args.config)
    with open_dict(cfg):
        if "action_dim" not in cfg.wm:
            cfg.wm.action_dim = 1

    # ── Load the dataset using eval_sweep's exact pipeline ────────────────────
    tr_data, te_data, hi_names, _arch_names, _ep_off, _ep_len = es.load_dataset(
        str(args.hdf5), encoder_type="sensor",
    )
    logging.info(f"  loaded {len(tr_data['obs']):,} train / {len(te_data['obs']):,} test timesteps")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out_csv.exists()
    with open(args.out_csv, "a") as f:
        if write_header:
            f.write("label,mode,seed,sl,task,component,r2,rmse,pearson_r\n")

    for mode in args.modes:
        logging.info(f"\n{'#'*60}\n  MODE = {mode}\n{'#'*60}")

        if mode == "raw_sensors":
            # Skip the encoder entirely — Z is the raw observation vector.
            Z_tr = tr_data["obs"].astype(np.float32, copy=False)
            Z_te = te_data["obs"].astype(np.float32, copy=False)
            logging.info(f"  Z_tr={Z_tr.shape}  Z_te={Z_te.shape}  (raw sensors, no encoder)")

        elif mode == "random_encoder":
            # Build a fresh JEPA → keep at random init → encode normally.
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            world_model = build_ar_lstm(cfg).to(device).eval()
            for p in world_model.parameters():
                p.requires_grad_(False)
            n_params = sum(p.numel() for p in world_model.parameters())
            logging.info(f"  random-init JEPA built, {n_params:,} params")

            Z_tr = es.encode_observations(world_model, tr_data["obs"], "sensor", device)
            Z_te = es.encode_observations(world_model, te_data["obs"], "sensor", device)
            logging.info(f"  Z_tr={Z_tr.shape}  Z_te={Z_te.shape}  (random encoder)")
        else:
            raise ValueError(f"unknown mode {mode!r}")

        # Seed before probe training (mirrors dragon's run_one(... seed=) approach)
        import random as _random
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        _random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        # ── Run task1_hi exactly as eval_sweep does ─────────────────────────
        results, _, _ = es.task1_hi(
            Z_tr, Z_te, tr_data, te_data, hi_names, device,
            seq_lens=args.seq_lens,
        )

        # Write per-component + mean rows
        label = f"{mode}_{args.label_suffix}"
        for sl in args.seq_lens:
            sl_key = f"sl{sl}"
            block = results[sl_key]
            for comp_name, m in block.items():
                row_name = "MEAN" if comp_name == "__mean__" else comp_name
                with open(args.out_csv, "a") as f:
                    f.write(
                        f"{label},{mode},{args.seed},{sl},hi,{row_name},"
                        f"{m['r2']:.6f},{m['rmse']:.6f},{m['pearson_r']:.6f}\n"
                    )

    logging.info(f"\n[csv] appended to {args.out_csv}")


if __name__ == "__main__":
    main()
