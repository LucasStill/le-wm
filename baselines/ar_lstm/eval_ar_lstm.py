"""
AR-LSTM Evaluation Script
==========================

Evaluates a trained AR-LSTM checkpoint on the two most informative
TurboSens tasks:

  Task 1 — HI State Estimation
    Freeze encoder, train a lightweight TransformerProbe to predict the
    10 ground-truth Health Indicator dimensions from frozen embeddings.
    Reports R², RMSE, Pearson-r per component.  Directly comparable to
    JEPA Task 1 results.

  Task 3 — Latent Forecasting & Action-Divergence Gap
    Autoregressive rollout of ẑ_{t+ς} from a context window of H
    embeddings, decoded to HI space via the Task-1 probe.
    Reports mean HI RMSE vs horizon ς for clean / event trajectory splits.
    The action-divergence gap φ(ς) = RMSE_event − RMSE_clean quantifies
    how much maintenance transitions degrade forecast accuracy.

Usage
-----
    cd le-wm/
    python baselines/ar_lstm/eval_ar_lstm.py \\
        --ckpt  /path/to/ar_lstm_epoch_100_object.ckpt \\
        --data  /path/to/opendeck_2000_lewm.h5 \\
        --out   results/ar_lstm_eval.json

The script also prints a comparison table against the JEPA baseline when
a JEPA checkpoint is supplied via --jepa_ckpt.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

# ── Add le-wm root ──────────────────────────────────────────────────────────
_LE_WM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LE_WM_ROOT))

from hi_probe import HIProbeCallback   # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_episodes(hdf5_path: str, train_split: float = 0.9, seed: int = 3072):
    """Load all sensor episodes and HI states from an HDF5 file.

    Returns
    -------
    obs_train, obs_test  : list of np.ndarray, each (T_ep, 28)
    hi_train, hi_test    : list of np.ndarray, each (T_ep, 10)
    act_train, act_test  : list of np.ndarray, each (T_ep,)
    ep_ids_train/test    : list of int episode indices
    state_labels         : list of str, length 10
    """
    with h5py.File(hdf5_path, "r") as f:
        ep_lens   = f["ep_len"][:]
        ep_offs   = f["ep_offset"][:]
        obs_raw   = f["pixels"][:]              # (N, 3, 7, 4) or (N, 28)
        hi_all    = f["observation.state"][:]   # (N, 10)
        act_all   = f["action"][:]              # (N,)
        labels    = [s.decode() if isinstance(s, bytes) else s
                     for s in f.attrs.get("state_label_names", [])]

    # Flatten to (N, 28) — average over repeated channels if needed
    flat = obs_raw.reshape(len(obs_raw), -1)
    if flat.shape[1] != 28:
        n_ch = flat.shape[1] // 28
        flat = flat.reshape(len(flat), n_ch, 28).mean(axis=1)
    obs_all = flat.astype(np.float32)

    # Split episodes (same RNG as training)
    n_episodes = len(ep_lens)
    rng        = np.random.default_rng(seed)
    idx        = rng.permutation(n_episodes)
    n_train    = int(np.floor(train_split * n_episodes))
    train_idx  = sorted(idx[:n_train])
    test_idx   = sorted(idx[n_train:])

    def gather(ep_indices):
        obs_list, hi_list, act_list = [], [], []
        for i in ep_indices:
            s, l = int(ep_offs[i]), int(ep_lens[i])
            obs_list.append(obs_all[s : s + l])
            hi_list.append(hi_all[s : s + l])
            act_list.append(act_all[s : s + l])
        return obs_list, hi_list, act_list

    obs_tr, hi_tr, act_tr = gather(train_idx)
    obs_te, hi_te, act_te = gather(test_idx)
    return obs_tr, hi_tr, act_tr, obs_te, hi_te, act_te, train_idx, test_idx, labels


# ═══════════════════════════════════════════════════════════════════════════════
# Encoding helpers
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_episodes(model, obs_list: list, device: torch.device,
                    batch_size: int = 2048) -> list[np.ndarray]:
    """Encode a list of sensor episodes to latent embeddings.

    Parameters
    ----------
    model    : JEPA (or AR-LSTM) instance with .encode()
    obs_list : list of (T_ep, 28) arrays
    device   : torch device

    Returns
    -------
    list of (T_ep, D) arrays
    """
    model.eval()
    emb_list = []
    for obs in obs_list:
        T = len(obs)
        all_z = []
        for s in range(0, T, batch_size):
            chunk = torch.from_numpy(obs[s : s + batch_size]).float()
            # Unsqueeze T dim: encode() expects (B, T, 28)
            chunk = chunk.unsqueeze(1).to(device)
            info  = {"pixels": chunk, "action": torch.zeros(len(chunk), 1, 1, device=device)}
            out   = model.encode(info)
            all_z.append(out["emb"].squeeze(1).cpu().numpy())
        emb_list.append(np.concatenate(all_z, axis=0))
    return emb_list


# ═══════════════════════════════════════════════════════════════════════════════
# Task 1 — HI State Estimation
# ═══════════════════════════════════════════════════════════════════════════════

def task1_hi_probing(
    model,
    obs_train, hi_train,
    obs_test,  hi_test,
    device,
    n_epochs: int  = 150,
    lr: float      = 1e-3,
    patience: int  = 20,
    batch_size: int = 256,
) -> dict:
    """Train a linear probe on frozen encoder embeddings → HI prediction.

    Returns dict with per-component and mean R², RMSE, Pearson-r.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr

    log.info("Task 1: encoding training episodes …")
    Z_tr = np.concatenate(encode_episodes(model, obs_train, device), axis=0)
    Y_tr = np.concatenate(hi_train, axis=0)

    log.info("Task 1: encoding test episodes …")
    Z_te = np.concatenate(encode_episodes(model, obs_test, device), axis=0)
    Y_te = np.concatenate(hi_test, axis=0)

    # Simple Ridge regression probe (fast; for full TransformerProbe reuse
    # HIProbeCallback directly from the training script)
    results = {}
    probe = Ridge(alpha=1.0).fit(Z_tr, Y_tr)
    Y_pred = probe.predict(Z_te)

    r2_per_comp    = r2_score(Y_te, Y_pred, multioutput="raw_values").tolist()
    rmse_per_comp  = np.sqrt(np.mean((Y_te - Y_pred) ** 2, axis=0)).tolist()
    pearson_per_comp = [
        float(pearsonr(Y_te[:, i], Y_pred[:, i])[0])
        for i in range(Y_te.shape[1])
    ]

    results["r2_per_component"]     = r2_per_comp
    results["rmse_per_component"]   = rmse_per_comp
    results["pearson_per_component"] = pearson_per_comp
    results["mean_r2"]              = float(np.mean(r2_per_comp))
    results["mean_rmse"]            = float(np.mean(rmse_per_comp))
    results["mean_pearson"]         = float(np.mean(pearson_per_comp))

    log.info(f"Task 1 — mean R²={results['mean_r2']:.4f}  "
             f"RMSE={results['mean_rmse']:.5f}  "
             f"r={results['mean_pearson']:.4f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Task 3 — Latent Forecasting & Action-Divergence Gap
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def rollout_episode(model, z_context: np.ndarray, act_context: np.ndarray,
                    n_steps: int, device: torch.device, history_size: int = 3) -> np.ndarray:
    """Autoregressively roll out the predictor for n_steps from a context window.

    Parameters
    ----------
    model      : JEPA / AR-LSTM with .predict()
    z_context  : (H, D) — initial context embeddings
    act_context: (H,)   — corresponding binary actions (0 or 1)
    n_steps    : rollout horizon
    device     : torch device
    history_size : H

    Returns
    -------
    (n_steps, D)  — predicted embeddings at each rollout step
    """
    model.eval()
    H = history_size
    D = z_context.shape[-1]

    emb = torch.from_numpy(z_context).float().to(device)   # (H, D)
    act = torch.from_numpy(act_context.astype(np.float32)).to(device).unsqueeze(-1)  # (H, 1)
    act_emb_fn = model.action_encoder

    preds = []
    for _ in range(n_steps):
        emb_in  = emb[-H:].unsqueeze(0)          # (1, H, D)
        act_in  = act_emb_fn(act[-H:].unsqueeze(0))  # (1, H, D)
        pred    = model.predict(emb_in, act_in)[:, -1, :]  # (1, D)
        preds.append(pred.squeeze(0).cpu().numpy())
        emb = torch.cat([emb, pred.squeeze(0).unsqueeze(0)], dim=0)
        act = torch.cat([act, torch.zeros(1, 1, device=device)], dim=0)  # no-action

    return np.stack(preds, axis=0)   # (n_steps, D)


def task3_latent_forecasting(
    model, probe,
    obs_test, hi_test, act_test,
    device,
    history_size: int = 3,
    max_horizon:  int = 50,
    n_samples:    int = 500,
) -> dict:
    """Evaluate latent forecasting accuracy (Task 3).

    Returns dict with:
      - rmse_clean[ς]  : mean HI RMSE at horizon ς for no-maintenance trajectories
      - rmse_event[ς]  : mean HI RMSE at horizon ς for trajectories with maintenance
      - action_divergence_gap[ς] = rmse_event[ς] - rmse_clean[ς]
    """
    from sklearn.linear_model import Ridge

    log.info("Task 3: encoding test episodes for forecasting …")
    Z_test = encode_episodes(model, obs_test, device)

    # Fit a HI decoder (Ridge) from Task-1 encodings
    log.info("Task 3: fitting HI decoder probe …")
    Z_all = np.concatenate(Z_test, axis=0)
    Y_all = np.concatenate(hi_test, axis=0)
    probe = Ridge(alpha=1.0).fit(Z_all[:int(0.9 * len(Z_all))],
                                  Y_all[:int(0.9 * len(Y_all))])

    H = history_size
    horizons = list(range(1, max_horizon + 1))
    rmse_clean = {ς: [] for ς in horizons}
    rmse_event = {ς: [] for ς in horizons}

    rng     = np.random.default_rng(42)
    sampled = 0

    for ep_idx, (Z_ep, Y_ep, act_ep) in enumerate(zip(Z_test, hi_test, act_test)):
        T = len(Z_ep)
        if T < H + max_horizon + 1:
            continue

        # Sample a few start positions from this episode
        starts = rng.integers(H, T - max_horizon - 1, size=min(5, T - H - max_horizon - 1))
        for t0 in starts:
            z_ctx  = Z_ep[t0 - H : t0]
            a_ctx  = act_ep[t0 - H : t0]
            has_event = act_ep[t0 : t0 + max_horizon].any()

            pred_z = rollout_episode(
                model, z_ctx, a_ctx, max_horizon, device, history_size=H
            )  # (max_horizon, D)

            pred_hi  = probe.predict(pred_z)        # (max_horizon, 10)
            true_hi  = Y_ep[t0 : t0 + max_horizon]  # (max_horizon, 10)
            rmse_vec = np.sqrt(np.mean((pred_hi - true_hi) ** 2, axis=1))  # (max_horizon,)

            for i, ς in enumerate(horizons):
                if i < len(rmse_vec):
                    (rmse_event[ς] if has_event else rmse_clean[ς]).append(rmse_vec[i])

            sampled += 1
            if sampled >= n_samples:
                break
        if sampled >= n_samples:
            break

    results = {"horizons": horizons, "rmse_clean": [], "rmse_event": [],
               "action_divergence_gap": []}
    for ς in horizons:
        rc = float(np.mean(rmse_clean[ς])) if rmse_clean[ς] else float("nan")
        re = float(np.mean(rmse_event[ς])) if rmse_event[ς] else float("nan")
        results["rmse_clean"].append(rc)
        results["rmse_event"].append(re)
        results["action_divergence_gap"].append(
            re - rc if not (np.isnan(rc) or np.isnan(re)) else float("nan")
        )

    peak_gap = max((g for g in results["action_divergence_gap"] if not np.isnan(g)), default=0)
    log.info(f"Task 3 — peak action-divergence gap φ = {peak_gap:.5f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate AR-LSTM on TurboSens Tasks 1 & 3")
    p.add_argument("--ckpt",        required=True,  help="Path to ar_lstm_epoch_N_object.ckpt")
    p.add_argument("--data",        required=True,  help="Path to opendeck_2000_lewm.h5")
    p.add_argument("--out",         default="results/ar_lstm_eval.json",
                   help="Output JSON path")
    p.add_argument("--jepa_ckpt",   default=None,
                   help="Optional: JEPA checkpoint for comparison table")
    p.add_argument("--history_size",type=int, default=3,
                   help="Context window H (must match training config)")
    p.add_argument("--horizon",     type=int, default=50,
                   help="Maximum rollout horizon for Task 3")
    p.add_argument("--n_samples",   type=int, default=500,
                   help="Number of rollout start points for Task 3")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip_task3",  action="store_true",
                   help="Skip Task 3 (faster, just do HI probing)")
    return p.parse_args()


def evaluate_checkpoint(ckpt_path: str, data_path: str, device_str: str,
                        history_size: int, horizon: int, n_samples: int,
                        skip_task3: bool) -> dict:
    device = torch.device(device_str)
    log.info(f"Loading checkpoint: {ckpt_path}")
    model = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.eval().to(device)

    log.info(f"Loading dataset: {data_path}")
    (obs_tr, hi_tr, act_tr,
     obs_te, hi_te, act_te,
     _, _, labels) = load_episodes(data_path)

    results = {
        "ckpt":         ckpt_path,
        "hi_labels":    labels,
        "history_size": history_size,
    }

    # ── Task 1 ────────────────────────────────────────────────────────────────
    log.info("─" * 50)
    log.info("Running Task 1: HI State Estimation")
    log.info("─" * 50)
    results["task1"] = task1_hi_probing(
        model, obs_tr, hi_tr, obs_te, hi_te, device
    )

    # ── Task 3 ────────────────────────────────────────────────────────────────
    if not skip_task3:
        log.info("─" * 50)
        log.info("Running Task 3: Latent Forecasting")
        log.info("─" * 50)
        results["task3"] = task3_latent_forecasting(
            model, probe=None,
            obs_test=obs_te, hi_test=hi_te, act_test=act_te,
            device=device,
            history_size=history_size,
            max_horizon=horizon,
            n_samples=n_samples,
        )

    return results


def print_comparison_table(ar_lstm_res: dict, jepa_res: dict | None):
    """Print a side-by-side Task 1 comparison table."""
    print("\n" + "═" * 60)
    print("  TASK 1 — HI State Estimation  (mean R²)")
    print("═" * 60)
    print(f"  {'Model':<20}  {'R²':>8}  {'RMSE':>10}  {'Pearson':>10}")
    print("─" * 60)
    t1 = ar_lstm_res["task1"]
    print(f"  {'AR-LSTM':<20}  {t1['mean_r2']:>8.4f}  "
          f"{t1['mean_rmse']:>10.5f}  {t1['mean_pearson']:>10.4f}")
    if jepa_res:
        t1j = jepa_res["task1"]
        print(f"  {'JEPA (sensor)':<20}  {t1j['mean_r2']:>8.4f}  "
              f"{t1j['mean_rmse']:>10.5f}  {t1j['mean_pearson']:>10.4f}")
    print("═" * 60)


def main():
    args = parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    ar_results = evaluate_checkpoint(
        ckpt_path=args.ckpt,
        data_path=args.data,
        device_str=args.device,
        history_size=args.history_size,
        horizon=args.horizon,
        n_samples=args.n_samples,
        skip_task3=args.skip_task3,
    )

    jepa_results = None
    if args.jepa_ckpt:
        log.info("=" * 50)
        log.info("Evaluating JEPA checkpoint for comparison …")
        jepa_results = evaluate_checkpoint(
            ckpt_path=args.jepa_ckpt,
            data_path=args.data,
            device_str=args.device,
            history_size=args.history_size,
            horizon=args.horizon,
            n_samples=args.n_samples,
            skip_task3=args.skip_task3,
        )
        ar_results["jepa_comparison"] = jepa_results

    print_comparison_table(ar_results, jepa_results)

    with open(args.out, "w") as f:
        json.dump(ar_results, f, indent=2)
    log.info(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
