"""
Multi-seed eval_sweep Task-1 for the RSSM/Dreamer baseline.

Mirrors `baselines/ar_lstm/multiseed_l1l2.py` but loads an RSSM checkpoint.
Encodes once, retrains the TransformerProbe head per seed, and writes a CSV
that's drop-in compatible with the existing `eval_results/multiseed/`
aggregator.

Usage
-----
    python baselines/rssm/multiseed_eval_rssm.py \\
        --ckpt /path/to/rssm_*_epoch_10_object.ckpt \\
        --hdf5 /home/lthil/.stable_worldmodel/turbosens2_test.h5 \\
        --label RSSM_te \\
        --seeds 0 1 2 3 4 5 \\
        --seq-lens 1 10 \\
        --out-csv eval_results/rssm_s4/multiseed_results.csv
"""
from __future__ import annotations

import argparse
import logging
import random as _random
import sys
from pathlib import Path

import numpy as np
import torch

LE_WM_ROOT = Path(__file__).resolve().parents[2]
if str(LE_WM_ROOT) not in sys.path:
    sys.path.insert(0, str(LE_WM_ROOT))

# Make sure the RSSM model class is importable so torch.load can unpickle the ckpt.
from baselines.rssm.model import RSSMWorldModel  # noqa: F401, E402

import eval_sweep as es                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--hdf5", required=True, type=Path)
    ap.add_argument("--label", required=True, type=str,
                    help="Identifier written to each CSV row (e.g. 'RSSM_te')")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    # NOTE: sl=10 OOMs at probe-load time for RSSM (feat=1536 vs JEPA's 64
    # → windowed tensor is 24× larger and doesn't fit on 32 GB). sl=1 only
    # by default; pass --seq-lens 1 10 explicitly if you have a bigger card.
    ap.add_argument("--seq-lens", nargs="+", type=int, default=[1])
    ap.add_argument("--out-csv", type=Path,
                    default=Path("eval_results/rssm_s4/multiseed_results.csv"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"=== multi-seed eval_sweep task-1 — {args.label} ===")
    logging.info(f"ckpt={args.ckpt}  hdf5={args.hdf5}  seeds={args.seeds}  "
                 f"seq_lens={args.seq_lens}")

    # ── Load dataset + encode once ────────────────────────────────────────────
    tr_data, te_data, hi_names, _arch, _ep_off, _ep_len = es.load_dataset(
        str(args.hdf5), encoder_type="sensor",
    )
    logging.info(f"  loaded {len(tr_data['obs']):,} train / "
                 f"{len(te_data['obs']):,} test")

    model = torch.load(str(args.ckpt), map_location=device, weights_only=False)
    if hasattr(model, "module"):
        model = model.module
    model.eval().to(device)
    logging.info("  encoding train + test (one-time) …")
    Z_tr = es.encode_observations(model, tr_data["obs"], "sensor", device)
    Z_te = es.encode_observations(model, te_data["obs"], "sensor", device)
    logging.info(f"  Z_tr={Z_tr.shape}  Z_te={Z_te.shape}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out_csv.exists()
    with open(args.out_csv, "a") as fh:
        if write_header:
            fh.write("label,seed,sl,task,component,r2,rmse,pearson_r\n")

    for seed in args.seeds:
        logging.info(f"\n  -- seed={seed} --")
        torch.manual_seed(seed)
        np.random.seed(seed)
        _random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Run each seq_len individually so an OOM on sl=10 doesn't lose sl=1.
        for sl in args.seq_lens:
            try:
                results, _, _ = es.task1_hi(
                    Z_tr, Z_te, tr_data, te_data, hi_names, device,
                    seq_lens=[sl],
                )
            except torch.cuda.OutOfMemoryError as e:
                logging.warning(f"  seed={seed} sl={sl}: OOM — skipping. {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            block = results[f"sl{sl}"]
            with open(args.out_csv, "a") as fh:
                for comp_name, m in block.items():
                    row = "MEAN" if comp_name == "__mean__" else comp_name
                    fh.write(f"{args.label},{seed},{sl},hi,{row},"
                             f"{m['r2']:.6f},{m['rmse']:.6f},"
                             f"{m['pearson_r']:.6f}\n")

    logging.info(f"\n[csv] appended to {args.out_csv}")


if __name__ == "__main__":
    main()
