"""
Multi-seed eval_sweep task-1 on L1 / L2 (and any provided ckpt).

Same path as eval_sweep, but: (a) wraps train_probe with the new `seed=`
kwarg, (b) writes seed into the CSV row so we can mean±std the results
across seeds without re-running the encoder.

Usage
-----
    python baselines/ar_lstm/multiseed_l1l2.py \\
        --ckpt /path/to/L1_epoch_10_object.ckpt \\
        --hdf5 /path/to/turbosens2_test.h5 \\
        --label L1_te \\
        --seeds 0 1 2 3 4 \\
        --seq-lens 1 \\
        --out-csv logs/multiseed_l1l2_results.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

LE_WM_ROOT = Path(__file__).resolve().parents[2]
if str(LE_WM_ROOT) not in sys.path:
    sys.path.insert(0, str(LE_WM_ROOT))

import eval_sweep as es                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--hdf5", required=True, type=Path)
    ap.add_argument("--label", required=True, type=str)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--seq-lens", nargs="+", type=int, default=[1])
    ap.add_argument("--out-csv", type=Path,
                    default=Path("logs/multiseed_l1l2_results.csv"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"=== multi-seed eval_sweep task-1 — {args.label} ===")
    logging.info(f"ckpt={args.ckpt}  hdf5={args.hdf5}  seeds={args.seeds}")

    # ── Load dataset + encode once (encoder is deterministic w/ frozen weights) ─
    tr_data, te_data, hi_names, _arch, _ep_off, _ep_len = es.load_dataset(
        str(args.hdf5), encoder_type="sensor",
    )
    logging.info(f"  loaded {len(tr_data['obs']):,} train / {len(te_data['obs']):,} test")

    model = torch.load(str(args.ckpt), map_location=device, weights_only=False)
    model.eval().to(device)
    logging.info("  encoding train + test (one-time) …")
    Z_tr = es.encode_observations(model, tr_data["obs"], "sensor", device)
    Z_te = es.encode_observations(model, te_data["obs"], "sensor", device)
    logging.info(f"  Z_tr={Z_tr.shape}  Z_te={Z_te.shape}")
    # Free the model — only the embeddings matter from here on
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out_csv.exists()
    with open(args.out_csv, "a") as fh:
        if write_header:
            fh.write("label,seed,sl,task,component,r2,rmse,pearson_r\n")

    # ── For each seed, retrain ONLY the probe head (encoder is frozen) ────────
    import random as _random
    for seed in args.seeds:
        logging.info(f"\n  -- seed={seed} --")
        # Seed manually before each call (matches dragon's run_one(... seed=) pattern;
        # eval_sweep.task1_hi itself doesn't take a seed kwarg upstream).
        torch.manual_seed(seed)
        np.random.seed(seed)
        _random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        results, _, _ = es.task1_hi(
            Z_tr, Z_te, tr_data, te_data, hi_names, device,
            seq_lens=args.seq_lens,
        )
        for sl in args.seq_lens:
            block = results[f"sl{sl}"]
            for comp_name, m in block.items():
                row = "MEAN" if comp_name == "__mean__" else comp_name
                with open(args.out_csv, "a") as fh:
                    fh.write(f"{args.label},{seed},{sl},hi,{row},"
                             f"{m['r2']:.6f},{m['rmse']:.6f},{m['pearson_r']:.6f}\n")

    logging.info(f"\n[csv] appended to {args.out_csv}")


if __name__ == "__main__":
    main()
