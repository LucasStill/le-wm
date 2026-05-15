"""
Per-archetype HI Pearson breakdown for L1 (or any trained encoder).
==================================================================

Splits the calibrated HI probe's TEST predictions by archetype and reports
Pearson / R² / RMSE per archetype subset.

Scenario-4 has 4 archetypes: A_compressor, B_fan_booster, C_turbine,
D_balanced (names from the H5's ep_meta/archetype + attrs.archetype_names).

This characterises whether some engine-degradation families are intrinsically
harder to learn HI for — useful for the dataset paper's "diversity" section.

Usage
-----
    cd le-wm/
    python baselines/ar_lstm/per_archetype_diag.py \\
        --ckpt   /home/lucas/.stable_worldmodel/ar_lstm_scenario4_w1_H16_S1_P4_h256l2/ar_lstm_s4_w1_H16_S1_P4_h256l2_epoch_10_object.ckpt \\
        --hdf5   /home/lucas/.stable_worldmodel/turbosens2_test.h5 \\
        --label  L1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

LE_WM_ROOT = Path(__file__).resolve().parents[2]
if str(LE_WM_ROOT) not in sys.path:
    sys.path.insert(0, str(LE_WM_ROOT))

import eval_sweep as es                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--hdf5", required=True, type=Path)
    ap.add_argument("--label", required=True, type=str)
    ap.add_argument("--seq-lens", nargs="+", type=int, default=[1, 10])
    ap.add_argument("--out-csv", type=Path, default=Path("logs/per_archetype_results.csv"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"=== Per-archetype diagnostic — {args.label} ===")

    # ── Load dataset (same path as eval_sweep) + archetype labels ─────────────
    tr_data, te_data, hi_names, archetype_names, ep_offset_all, ep_len_all = es.load_dataset(
        str(args.hdf5), encoder_type="sensor",
    )

    # Map archetype labels per test-timestep (need to re-derive ep-id → archetype)
    with h5py.File(args.hdf5, "r") as f:
        if "ep_meta/archetype" not in f:
            logging.error("HDF5 has no ep_meta/archetype field; can't split by archetype.")
            sys.exit(1)
        archetype_per_ep = f["ep_meta/archetype"][:].astype(np.int32)

    # te_data["eps"] is the array of GLOBAL episode indices in the test split.
    # te_data["ep_ids"] is the LOCAL episode index per timestep (0..len(eps)-1).
    eps_test = te_data["eps"]                    # (n_test_eps,)
    arch_per_local_ep = archetype_per_ep[eps_test]   # (n_test_eps,)
    arch_per_ts_te    = arch_per_local_ep[te_data["ep_ids"]]   # (n_test_timesteps,)

    if archetype_names:
        logging.info(f"  archetypes: {archetype_names}")
    else:
        archetype_names = [f"arch_{i}" for i in range(int(arch_per_ts_te.max()) + 1)]

    counts = np.bincount(arch_per_ts_te, minlength=len(archetype_names))
    logging.info("  per-archetype timestep counts:")
    for n, c in zip(archetype_names, counts):
        logging.info(f"    {n:<24}  {c:,}")

    # ── Load encoder + encode train + test ────────────────────────────────────
    logging.info(f"  loading {args.ckpt} …")
    model = torch.load(str(args.ckpt), map_location=device, weights_only=False)
    model.eval().to(device)
    logging.info(f"  encoding train + test …")
    Z_tr = es.encode_observations(model, tr_data["obs"], "sensor", device)
    Z_te = es.encode_observations(model, te_data["obs"], "sensor", device)
    logging.info(f"  Z_tr={Z_tr.shape}  Z_te={Z_te.shape}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out_csv.exists()
    with open(args.out_csv, "a") as fh:
        if write_header:
            fh.write("label,sl,archetype,n,task,component,r2,rmse,pearson_r\n")

    for sl in args.seq_lens:
        logging.info(f"\n  -- seq_len={sl} --")
        preds_te, gt_te, _probe, _scaler = es.train_probe(
            Z_tr, tr_data["hi"], Z_te, te_data["hi"],
            tr_data["ep_ids"], te_data["ep_ids"],
            n_outputs=tr_data["hi"].shape[1],
            device=device, task_name=f"HI_sl{sl}", seq_len=sl,
        )

        # `train_probe` returns predictions in window space. For sl>1,
        # _build_windows produces M < N entries (one per window, indexed by
        # the LAST timestep of the window). Re-derive the per-window archetype
        # the same way HIProbeCallback._build_windows does.
        if sl == 1:
            arch_w = arch_per_ts_te
        else:
            ep_ids = te_data["ep_ids"]
            arch_w = []
            for ep_id in np.unique(ep_ids):
                mask = ep_ids == ep_id
                a_ep = arch_per_ts_te[mask]
                T_ep = len(a_ep)
                if T_ep < sl:
                    continue
                # window i ranges [i-sl+1 .. i], target at i (last token)
                arch_w.extend(a_ep[sl - 1 : T_ep])
            arch_w = np.array(arch_w, dtype=np.int32)
            assert len(arch_w) == len(preds_te), (len(arch_w), len(preds_te))

        # Per-archetype Pearson (averaged over HI dims)
        for arch_idx, arch_name in enumerate(archetype_names):
            mask = arch_w == arch_idx
            n_w  = int(mask.sum())
            if n_w < 50:
                logging.warning(f"    {arch_name}: only {n_w} windows, skipping")
                continue
            preds_a = preds_te[mask]
            gt_a    = gt_te[mask]
            r2s, rmses, prs = [], [], []
            for i in range(gt_a.shape[1]):
                g, p = gt_a[:, i], preds_a[:, i]
                if g.std() < 1e-8 or p.std() < 1e-8:
                    pr = float("nan")
                else:
                    pr = float(pearsonr(g, p)[0])
                r2s.append(float(r2_score(g, p)))
                rmses.append(float(np.sqrt(np.mean((g - p) ** 2))))
                prs.append(pr)
            mean_r2   = float(np.mean(r2s))
            mean_rmse = float(np.mean(rmses))
            mean_pr   = float(np.nanmean(prs))
            logging.info(
                f"    {arch_name:<24}  n={n_w:>7,}  "
                f"Pearson={mean_pr:+.4f}  R²={mean_r2:+.4f}  RMSE={mean_rmse:.5f}"
            )
            with open(args.out_csv, "a") as fh:
                # per-component
                for i, hi_name in enumerate(hi_names):
                    fh.write(f"{args.label},{sl},{arch_name},{n_w},hi,{hi_name},"
                             f"{r2s[i]:.6f},{rmses[i]:.6f},{prs[i]:.6f}\n")
                # mean
                fh.write(f"{args.label},{sl},{arch_name},{n_w},hi,MEAN,"
                         f"{mean_r2:.6f},{mean_rmse:.6f},{mean_pr:.6f}\n")

    logging.info(f"\n[csv] appended to {args.out_csv}")


if __name__ == "__main__":
    main()
