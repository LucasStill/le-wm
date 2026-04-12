"""eval_benchmark_sweep.py — batch benchmark across multiple le-wm checkpoints.

Runs all three downstream evaluation tasks for every checkpoint in CHECKPOINTS
and writes results to a timestamped output directory.

Output files
------------
results/<run_id>/summary.json          — full nested metrics per checkpoint
results/<run_id>/metrics_flat.csv      — one row per (model, task, component, metric)
results/<run_id>/task3_curves.csv      — RMSE(τ) curves for Task 3

Usage
-----
  python eval_benchmark_sweep.py                       # uses CHECKPOINTS below
  python eval_benchmark_sweep.py --out_dir /path/out   # custom output dir
  python eval_benchmark_sweep.py --skip_task3          # skip slow forecasting loop
  python eval_benchmark_sweep.py --forecast_horizon 50 # override max τ

Tasks
-----
  1. HI State Estimation   — TransformerProbe on frozen embeddings → R², RMSE, Pearson-r
  2. RUL Prediction        — single-output probe on log1p-scaled steps-to-maintenance
  2b. Maintenance Alarm    — binary classifier: maintenance within K steps? (AUC, F1)
  3. Latent Forecasting    — JEPA rollout τ steps, decode z→HI, measure RMSE(τ)
                             Split into: (a) no-maintenance trajectories,
                                         (b) trajectories that contain a maintenance event
                             to quantify action-divergence of the zero-action forecaster.
"""

import argparse
import copy
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import (
    r2_score, roc_auc_score, average_precision_score,
    f1_score, mean_absolute_error,
)
from sklearn.preprocessing import StandardScaler

# le-wm imports — adjust sys.path if running from a different directory
try:
    from hi_probe import _TransformerProbe, HIProbeCallback
    from jepa import JEPA
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.abspath("."))
    from hi_probe import _TransformerProbe, HIProbeCallback
    from jepa import JEPA

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
warnings.filterwarnings("ignore", category=UserWarning)

# ── Checkpoint list ────────────────────────────────────────────────────────────
# Each entry: {path, name, encoder_type}
# obs_window_size is read from the model object itself (model.obs_window_size).
CHECKPOINTS = [
    {
        "path": "/lustre/fswork/projects/rech/yil/ugy35qd/.stable_worldmodel/"
                "turbofan_v6_sensor_sl1_w10/"
                "lewm_turbo_v6_sensor_sl1_w10_epoch_100_object.ckpt",
        "name": "v6_sensor_sl1_w10",
        "encoder_type": "sensor",
    },
    {
        "path": "/lustre/fswork/projects/rech/yil/ugy35qd/.stable_worldmodel/"
                "turbofan_v6_sensor_sl1_w1/"
                "lewm_turbo_v6_sensor_sl1_w1_epoch_100_object.ckpt",
        "name": "v6_sensor_sl1_w1",
        "encoder_type": "sensor",
    },
    # ── add more checkpoints here ──────────────────────────────────────────
    # {
    #     "path": "...",
    #     "name": "v7_sensor_sl10_w1",
    #     "encoder_type": "sensor",
    # },
]

# ── Shared dataset config ──────────────────────────────────────────────────────
HDF5_PATH    = "/lustre/fswork/projects/rech/yil/ugy35qd/.stable_worldmodel/opendeck_2000_lewm.h5"
TRAIN_SPLIT  = 0.9
SEED         = 3072
IMG_SIZE     = 28     # only used for ViT encoder

# ── Probe hyper-params ─────────────────────────────────────────────────────────
PROBE_SEQ_LEN   = 1
N_PROBE_EPOCHS  = 150
PROBE_LR        = 1e-3
PROBE_PATIENCE  = 20
PROBE_BATCH     = 256
D_MODEL         = 64
NHEAD           = 4
NUM_LAYERS      = 2
DROPOUT         = 0.1
N_SUBSAMPLE     = 30_000
ENC_BATCH       = 2048
RUL_MAX_HORIZON = 300

# ── Task 2b: maintenance alarm horizons to evaluate (binary classification) ────
ALARM_HORIZONS  = [10, 20, 50, 100]

# ── Task 3: latent forecasting ─────────────────────────────────────────────────
FORECAST_HORIZONS = list(range(1, 51))   # τ = 1 … 50
FORECAST_N_EPS    = 200                  # max test episodes
HISTORY_SIZE      = 3                    # context frames for predictor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(path, encoder_type, train_split=0.9, seed=3072):
    """Load full HDF5 dataset and return train/test split dicts.

    Returns
    -------
    tr_data, te_data : dict with keys
        obs      : (N, n_sensors) or (N, H, W)
        hi       : (N, n_hi)
        actions  : (N,) float — 1.0 = maintenance event
        ep_ids   : (N,) int   — local episode index per timestep
        eps      : list[int]  — global episode indices in this split
    hi_names : list[str]
    ep_offset_all, ep_len_all : (E,) arrays (full dataset, not split)
    """
    with h5py.File(path, "r") as f:
        ep_len_all    = f["ep_len"][:]
        ep_offset_all = f["ep_offset"][:]
        obs_key       = "pixels" if "pixels" in f else "observation.sensors"
        obs_raw       = f[obs_key][:]
        if encoder_type == "sensor" and obs_raw.ndim > 2:
            obs_raw = obs_raw.reshape(len(obs_raw), -1)
        states  = f["observation.state"][:]
        hi_names = list(f.attrs.get(
            "state_label_names",
            [f"HI_{i}" for i in range(states.shape[1])]
        ))
        actions = f["action"][:]
        if actions.ndim > 1:
            actions = actions[:, 0]

    N_EP = len(ep_len_all)
    rng  = np.random.default_rng(seed)
    perm = rng.permutation(N_EP)
    n_tr = int(N_EP * train_split)
    tr_eps = perm[:n_tr]
    te_eps = perm[n_tr:]

    def gather(eps):
        all_obs, all_hi, all_act, all_ep = [], [], [], []
        for local_i, e in enumerate(eps):
            s, l = int(ep_offset_all[e]), int(ep_len_all[e])
            all_obs.append(obs_raw[s : s + l])
            all_hi.append(states[s : s + l])
            all_act.append(actions[s : s + l])
            all_ep.append(np.full(l, local_i, dtype=np.int32))
        return dict(
            obs     = np.concatenate(all_obs),
            hi      = np.concatenate(all_hi),
            actions = np.concatenate(all_act),
            ep_ids  = np.concatenate(all_ep),
            eps     = eps,
        )

    return gather(tr_eps), gather(te_eps), hi_names, ep_offset_all, ep_len_all


# ══════════════════════════════════════════════════════════════════════════════
#  Encoding
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_observations(model, obs, encoder_type,
                        batch_size=ENC_BATCH, img_size=IMG_SIZE):
    """Encode (N, ...) observations → (N, embed_dim) numpy array.

    When obs_window_size > 1, each frame is tiled w times so the
    TemporalAggregator produces exactly one output per input frame.
    """
    is_sensor = (encoder_type == "sensor")
    w = getattr(model, "obs_window_size", 1)
    if not is_sensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

    all_emb = []
    for start in range(0, len(obs), batch_size):
        chunk = obs[start : start + batch_size]
        if is_sensor:
            t = torch.from_numpy(chunk).float().to(DEVICE).unsqueeze(1)
            if w > 1:
                t = t.expand(-1, w, -1).contiguous()
        else:
            t = torch.from_numpy(chunk).float().to(DEVICE)
            if t.ndim == 3:
                t = t.unsqueeze(1).expand(-1, 3, -1, -1)
            t = F.interpolate(t, size=(img_size, img_size), mode="nearest")
            t = (t - mean) / std
            t = t.unsqueeze(1)
            if w > 1:
                t = t.expand(-1, w, -1, -1, -1).contiguous()
        out = model.encode({"pixels": t})
        all_emb.append(out["emb"][:, 0, :].cpu().numpy())
    return np.concatenate(all_emb, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
#  Probe training helper
# ══════════════════════════════════════════════════════════════════════════════

def train_and_eval_probe(
    X_tr, y_tr, X_te, y_te,
    ep_ids_tr, ep_ids_te,
    n_outputs, task_name="probe",
    seq_len=PROBE_SEQ_LEN,
    n_epochs=N_PROBE_EPOCHS, lr=PROBE_LR,
    patience=PROBE_PATIENCE, batch_size=PROBE_BATCH,
    d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dropout=DROPOUT,
):
    """Train a TransformerProbe; return (preds, gt) in original scale."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    Xw_tr, yw_tr = HIProbeCallback._build_windows(X_tr_s, y_tr, ep_ids_tr, seq_len)
    Xw_te, yw_te = HIProbeCallback._build_windows(X_te_s, y_te, ep_ids_te, seq_len)
    logging.info(f"  [{task_name}] windows train={len(Xw_tr):,} test={len(Xw_te):,}")

    probe = _TransformerProbe(
        input_dim=Xw_tr.shape[2], n_outputs=n_outputs,
        d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout,
    ).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    Xtr_t = torch.from_numpy(Xw_tr).float().to(DEVICE)
    ytr_t = torch.from_numpy(yw_tr).float().to(DEVICE)
    Xte_t = torch.from_numpy(Xw_te).float().to(DEVICE)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr_t, ytr_t),
        batch_size=batch_size, shuffle=True,
    )

    best_rmse, best_state, patience_ctr = float("inf"), None, 0
    for ep in range(n_epochs):
        probe.train()
        for xb, yb in dl:
            opt.zero_grad(); F.mse_loss(probe(xb), yb).backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            pv = probe(Xte_t).cpu().numpy()
        rmse = float(np.mean(np.sqrt(np.mean((yw_te - pv) ** 2, axis=0))))
        if rmse < best_rmse:
            best_rmse = rmse; best_state = copy.deepcopy(probe.state_dict()); patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= patience:
            logging.info(f"  [{task_name}] early stop ep {ep+1}  best_rmse={best_rmse:.5f}")
            break

    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        preds = probe(Xte_t).cpu().numpy()
    return preds, yw_te, probe, scaler


# ══════════════════════════════════════════════════════════════════════════════
#  Task 1 — HI State Estimation
# ══════════════════════════════════════════════════════════════════════════════

def task1_hi_estimation(Z_tr, Z_te, tr_data, te_data, hi_names):
    logging.info("Task 1 — HI State Estimation")
    preds, gt, probe, scaler = train_and_eval_probe(
        Z_tr, tr_data["hi"], Z_te, te_data["hi"],
        tr_data["ep_ids"], te_data["ep_ids"],
        n_outputs=tr_data["hi"].shape[1], task_name="HI",
    )
    results = {}
    r2s, rmses, pearson_rs = [], [], []
    for i, name in enumerate(hi_names):
        g, p = gt[:, i], preds[:, i]
        r2 = float(r2_score(g, p))
        rmse = float(np.sqrt(np.mean((g - p) ** 2)))
        r_val, _ = pearsonr(g, p)
        results[name] = {"r2": r2, "rmse": rmse, "pearson_r": float(r_val)}
        r2s.append(r2); rmses.append(rmse); pearson_rs.append(float(r_val))
    results["__mean__"] = {
        "r2": float(np.mean(r2s)),
        "rmse": float(np.mean(rmses)),
        "pearson_r": float(np.mean(pearson_rs)),
    }
    logging.info(f"  mean R²={np.mean(r2s):.3f}  RMSE={np.mean(rmses):.5f}")
    return results, probe, scaler


# ══════════════════════════════════════════════════════════════════════════════
#  Task 2 — RUL Prediction
# ══════════════════════════════════════════════════════════════════════════════

def task2_rul(Z_tr, Z_te, tr_data, te_data, ep_offset_all, ep_len_all):
    logging.info("Task 2 — RUL Prediction")

    # Recompute RUL for train/test splits
    with h5py.File(HDF5_PATH, "r") as f:
        all_actions = f["action"][:]
        if all_actions.ndim > 1:
            all_actions = all_actions[:, 0]

    rul_full = HIProbeCallback._compute_rul(
        all_actions, ep_offset_all, ep_len_all, RUL_MAX_HORIZON
    )

    def gather_rul(eps):
        return np.concatenate([
            rul_full[int(ep_offset_all[e]) : int(ep_offset_all[e]) + int(ep_len_all[e])]
            for e in eps
        ])

    rul_tr = gather_rul(tr_data["eps"])
    rul_te = gather_rul(te_data["eps"])

    # log1p scale + normalise
    rul_tr_log = np.log1p(rul_tr).reshape(-1, 1).astype(np.float32)
    rul_te_log = np.log1p(rul_te).reshape(-1, 1).astype(np.float32)
    scaler_y = StandardScaler()
    rul_tr_s = scaler_y.fit_transform(rul_tr_log)
    rul_te_s = scaler_y.transform(rul_te_log)

    preds_s, gt_s, _, _ = train_and_eval_probe(
        Z_tr, rul_tr_s, Z_te, rul_te_s,
        tr_data["ep_ids"], te_data["ep_ids"],
        n_outputs=1, task_name="RUL",
    )

    # De-normalise
    preds_raw = np.expm1(scaler_y.inverse_transform(preds_s)).clip(min=0).ravel()
    gt_raw    = np.expm1(scaler_y.inverse_transform(gt_s)).clip(min=0).ravel()

    uncensored  = gt_raw < RUL_MAX_HORIZON
    frac_uncens = float(uncensored.sum()) / max(len(gt_raw), 1)
    logging.info(f"  uncensored: {uncensored.sum()}/{len(gt_raw)} ({frac_uncens:.1%})")

    gt_eval    = gt_raw[uncensored] if uncensored.sum() >= 20 else gt_raw
    preds_eval = preds_raw[uncensored] if uncensored.sum() >= 20 else preds_raw

    results = {
        "rmse":             float(np.sqrt(np.mean((gt_eval - preds_eval) ** 2))),
        "mae":              float(mean_absolute_error(gt_eval, preds_eval)),
        "r2":               float(r2_score(gt_eval, preds_eval)),
        "pearson_r":        float(pearsonr(gt_eval, preds_eval)[0]),
        "frac_uncensored":  frac_uncens,
    }
    logging.info(f"  RMSE={results['rmse']:.2f}  R²={results['r2']:.3f}")

    # ── Task 2b: maintenance alarm (binary classification) ─────────────────
    # "Will maintenance occur within K steps?"
    # Positives: rul < K; Negatives: rul >= K (capped samples excluded)
    alarm_results = {}
    for K in ALARM_HORIZONS:
        mask = gt_raw < RUL_MAX_HORIZON   # exclude censored
        y_true = (gt_raw[mask] <= K).astype(int)
        # Use predicted RUL as score (lower pred RUL = higher alarm probability)
        y_score = -preds_raw[mask]  # negate so high score = imminent maintenance
        if y_true.sum() < 10 or (1 - y_true).sum() < 10:
            alarm_results[f"K{K}"] = {"auc": None, "avg_precision": None, "f1": None}
            continue
        auc = float(roc_auc_score(y_true, y_score))
        ap  = float(average_precision_score(y_true, y_score))
        # F1 at threshold = 0 (pred_rul < K as binary decision)
        y_pred_bin = (preds_raw[mask] <= K).astype(int)
        f1  = float(f1_score(y_true, y_pred_bin, zero_division=0))
        alarm_results[f"K{K}"] = {"auc": auc, "avg_precision": ap, "f1": f1}
        logging.info(f"  Alarm K={K:>3}: AUC={auc:.3f}  AP={ap:.3f}  F1={f1:.3f}")

    results["alarm"] = alarm_results
    return results, rul_tr, rul_te


# ══════════════════════════════════════════════════════════════════════════════
#  Task 3 — Latent Forecasting
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def decode_z_to_hi(probe, scaler, z_np):
    z_s = scaler.transform(z_np)
    z_t = torch.from_numpy(z_s[:, np.newaxis, :]).float().to(DEVICE)
    return probe(z_t).cpu().numpy()


@torch.no_grad()
def task3_latent_forecasting(
    model, probe, scaler_probe,
    te_data, ep_offset_all, ep_len_all,
    encoder_type, max_tau, n_eps, history_size=HISTORY_SIZE,
):
    """Autoregressive latent forecasting over τ = 1…max_tau steps.

    For each test episode, at each starting frame t:
      1. Encode context  [t-H+1 … t]  →  z_context (H, D)
      2. Autoregressively predict z_{t+1} … z_{t+τ}  (zero actions)
      3. Decode each z_{t+τ} to HI via the probe
      4. Compare to ground-truth HI

    Trajectories are split into:
      • "clean" — no maintenance event in [t+1 … t+τ]
      • "event" — at least one maintenance event in [t+1 … t+τ]
    This quantifies how much the zero-action forecaster diverges at
    maintenance transitions (action-divergence).

    Returns
    -------
    dict with keys:
      "all", "clean", "event" → each a (max_tau, n_hi) mean-RMSE array
      "count_all", "count_clean", "count_event" → (max_tau,) int arrays
    """
    logging.info("Task 3 — Latent Forecasting")
    n_hi = te_data["hi"].shape[1]
    H    = history_size

    # Load raw data from HDF5 for episode-level access
    with h5py.File(HDF5_PATH, "r") as f:
        obs_key = "pixels" if "pixels" in f else "observation.sensors"
        all_obs_raw = f[obs_key][:]
        if encoder_type == "sensor" and all_obs_raw.ndim > 2:
            all_obs_raw = all_obs_raw.reshape(len(all_obs_raw), -1)
        all_states  = f["observation.state"][:]
        all_actions = f["action"][:]
        if all_actions.ndim > 1:
            all_actions = all_actions[:, 0]

    # Accumulators: all / clean (no maint) / event (has maint in window)
    def _zeros():
        return np.zeros((max_tau, n_hi), dtype=np.float64)

    sqerr = {"all": _zeros(), "clean": _zeros(), "event": _zeros()}
    count = {"all": np.zeros(max_tau, np.int64),
             "clean": np.zeros(max_tau, np.int64),
             "event": np.zeros(max_tau, np.int64)}

    te_eps  = te_data["eps"]
    n_eval  = min(n_eps, len(te_eps))

    for ep_i, ep_idx in enumerate(te_eps[:n_eval]):
        if ep_i % 20 == 0:
            logging.info(f"  episode {ep_i}/{n_eval}…")
        s = int(ep_offset_all[ep_idx])
        l = int(ep_len_all[ep_idx])
        if l < H + 1:
            continue

        obs_ep  = all_obs_raw[s : s + l]
        hi_ep   = all_states[s : s + l]
        act_ep  = all_actions[s : s + l]   # (T,) binary maintenance flags

        # Encode all frames (with tiling for obs_window > 1)
        z_ep = encode_observations(model, obs_ep, encoder_type, batch_size=512)
        # z_ep : (T, D)

        for t in range(H - 1, l - 1):
            ctx = torch.from_numpy(
                z_ep[t - H + 1 : t + 1]
            ).float().to(DEVICE).unsqueeze(0)   # (1, H, D)
            act_emb = torch.zeros(1, H, ctx.shape[-1], device=DEVICE)

            emb_window = ctx.clone()
            for tau in range(1, max_tau + 1):
                target_t = t + tau
                if target_t >= l:
                    break

                pred_z_seq = model.predict(emb_window[:, -H:], act_emb[:, -H:])
                next_z     = pred_z_seq[:, -1:]   # (1, 1, D)

                hi_pred = decode_z_to_hi(
                    probe, scaler_probe, next_z[:, 0].cpu().numpy()
                )  # (1, n_hi)
                hi_gt = hi_ep[target_t : target_t + 1]  # (1, n_hi)

                se = ((hi_pred - hi_gt) ** 2)[0]

                # Classify trajectory segment
                has_event = bool((act_ep[t + 1 : target_t + 1] > 0.5).any())
                key_extra = "event" if has_event else "clean"

                sqerr["all"][tau - 1]       += se
                sqerr[key_extra][tau - 1]   += se
                count["all"][tau - 1]       += 1
                count[key_extra][tau - 1]   += 1

                emb_window = torch.cat([emb_window, next_z], dim=1)
                act_emb    = torch.cat(
                    [act_emb, torch.zeros(1, 1, ctx.shape[-1], device=DEVICE)],
                    dim=1
                )

    results = {}
    for key in ("all", "clean", "event"):
        cnt = count[key]
        rmse = np.sqrt(sqerr[key] / np.maximum(cnt[:, None], 1))
        results[f"rmse_{key}"]  = rmse.tolist()           # (max_tau, n_hi)
        results[f"mean_rmse_{key}"] = rmse.mean(axis=1).tolist()  # (max_tau,)
        results[f"count_{key}"] = cnt.tolist()

    # Log summary at τ = 1, 5, 10, 20, 50
    for tau in [1, 5, 10, 20, min(50, max_tau)]:
        if tau <= max_tau and count["all"][tau - 1] > 0:
            logging.info(
                f"  τ={tau:>2}  RMSE(all)={results['mean_rmse_all'][tau-1]:.5f}"
                f"  clean={results['mean_rmse_clean'][tau-1]:.5f}"
                f"  event={results['mean_rmse_event'][tau-1]:.5f}"
                f"  [n_all={count['all'][tau-1]:,}"
                f" n_event={count['event'][tau-1]:,}]"
            )
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Main sweep loop
# ══════════════════════════════════════════════════════════════════════════════

def run_checkpoint(cfg, tr_data, te_data, hi_names, ep_offset_all, ep_len_all,
                   skip_task3=False, forecast_horizon=None):
    """Run all tasks for a single checkpoint entry."""
    logging.info(f"\n{'='*70}")
    logging.info(f"  Checkpoint: {cfg['name']}")
    logging.info(f"  Path: {cfg['path']}")
    logging.info(f"{'='*70}")

    model = torch.load(cfg["path"], map_location=DEVICE, weights_only=False)
    model.eval()
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    w = getattr(model, "obs_window_size", 1)
    logging.info(f"  Loaded — {n_params:,} params  obs_window={w}")

    encoder_type = cfg.get("encoder_type", "sensor")

    # Encode observations
    logging.info("  Encoding train observations…")
    Z_tr = encode_observations(model, tr_data["obs"], encoder_type)
    logging.info("  Encoding test observations…")
    Z_te = encode_observations(model, te_data["obs"], encoder_type)
    logging.info(f"  Embeddings: Z_tr={Z_tr.shape}  Z_te={Z_te.shape}")

    result = {
        "name":          cfg["name"],
        "path":          cfg["path"],
        "encoder_type":  encoder_type,
        "obs_window_size": w,
        "n_params":      n_params,
        "embed_dim":     int(Z_tr.shape[1]),
    }

    # Task 1
    t1, probe_hi, scaler_hi = task1_hi_estimation(
        Z_tr, Z_te, tr_data, te_data, hi_names
    )
    result["task1_hi"] = t1

    # Task 2
    t2, rul_tr, rul_te = task2_rul(
        Z_tr, Z_te, tr_data, te_data, ep_offset_all, ep_len_all
    )
    result["task2_rul"] = t2

    # Task 3
    if not skip_task3:
        max_tau = forecast_horizon or max(FORECAST_HORIZONS)
        t3 = task3_latent_forecasting(
            model, probe_hi, scaler_hi,
            te_data, ep_offset_all, ep_len_all,
            encoder_type, max_tau, FORECAST_N_EPS,
        )
        result["task3_forecast"] = t3
    else:
        result["task3_forecast"] = None
        logging.info("  Task 3 skipped.")

    # Free GPU memory before next checkpoint
    del model
    torch.cuda.empty_cache()

    return result


def results_to_flat_csv(all_results, hi_names, out_path):
    """Flatten nested results dict to one row per (model, task, component, metric)."""
    rows = []
    header = ["model", "obs_window", "task", "component", "metric", "value"]

    for res in all_results:
        name = res["name"]
        w    = res["obs_window_size"]

        # Task 1
        for comp, metrics in res.get("task1_hi", {}).items():
            for metric, value in metrics.items():
                rows.append([name, w, "task1_hi", comp, metric, value])

        # Task 2
        t2 = res.get("task2_rul", {})
        for metric in ("rmse", "mae", "r2", "pearson_r", "frac_uncensored"):
            if metric in t2:
                rows.append([name, w, "task2_rul", "overall", metric, t2[metric]])
        for k_str, alarm in t2.get("alarm", {}).items():
            for metric, value in alarm.items():
                if value is not None:
                    rows.append([name, w, "task2_alarm", k_str, metric, value])

        # Task 3 — mean RMSE per tau
        t3 = res.get("task3_forecast")
        if t3:
            for key in ("all", "clean", "event"):
                for tau_i, rmse_val in enumerate(t3.get(f"mean_rmse_{key}", [])):
                    tau = tau_i + 1
                    rows.append([name, w, f"task3_{key}", f"tau_{tau}", "mean_rmse", rmse_val])

    import csv
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    logging.info(f"Flat CSV → {out_path}")


def task3_curves_csv(all_results, out_path):
    """Wide-format CSV: one row per (model, tau), columns = all/clean/event mean RMSE."""
    import csv
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "obs_window", "tau",
                         "mean_rmse_all", "mean_rmse_clean", "mean_rmse_event",
                         "count_all", "count_clean", "count_event"])
        for res in all_results:
            t3 = res.get("task3_forecast")
            if not t3:
                continue
            for tau_i in range(len(t3["mean_rmse_all"])):
                writer.writerow([
                    res["name"],
                    res["obs_window_size"],
                    tau_i + 1,
                    t3["mean_rmse_all"][tau_i],
                    t3["mean_rmse_clean"][tau_i],
                    t3["mean_rmse_event"][tau_i],
                    t3["count_all"][tau_i],
                    t3["count_clean"][tau_i],
                    t3["count_event"][tau_i],
                ])
    logging.info(f"Task-3 curves CSV → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="le-wm benchmark sweep")
    parser.add_argument("--out_dir",          default=None,       help="output directory")
    parser.add_argument("--skip_task3",       action="store_true", help="skip latent forecasting")
    parser.add_argument("--forecast_horizon", type=int, default=None, help="max τ for Task 3")
    parser.add_argument("--checkpoints",      nargs="*", default=None,
                        help="checkpoint paths (positional, overrides CHECKPOINTS list)")
    args = parser.parse_args()

    # Output directory
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output → {out_dir}")

    # Checkpoint list override
    checkpoints = CHECKPOINTS
    if args.checkpoints:
        checkpoints = [
            {"path": p, "name": Path(p).stem, "encoder_type": "sensor"}
            for p in args.checkpoints
        ]

    # Load dataset once (shared across all models)
    logging.info(f"Loading dataset from {HDF5_PATH}")
    tr_data, te_data, hi_names, ep_offset_all, ep_len_all = load_dataset(
        HDF5_PATH, checkpoints[0]["encoder_type"], TRAIN_SPLIT, SEED
    )
    logging.info(
        f"Dataset loaded: train={len(tr_data['obs']):,}  test={len(te_data['obs']):,}  "
        f"HI dims={len(hi_names)}"
    )

    all_results = []
    for cfg in checkpoints:
        try:
            res = run_checkpoint(
                cfg, tr_data, te_data, hi_names, ep_offset_all, ep_len_all,
                skip_task3=args.skip_task3,
                forecast_horizon=args.forecast_horizon,
            )
            all_results.append(res)
        except Exception as e:
            logging.error(f"Failed on {cfg['name']}: {e}", exc_info=True)
            all_results.append({"name": cfg["name"], "error": str(e)})

    # ── Save results ──────────────────────────────────────────────────────────
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"Summary JSON → {summary_path}")

    results_to_flat_csv(all_results, hi_names, out_dir / "metrics_flat.csv")
    task3_curves_csv(all_results, out_dir / "task3_curves.csv")

    # ── Console summary table ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Benchmark sweep — {len(all_results)} checkpoints")
    print(f"{'='*80}")
    print(f"  {'Model':<30}  {'T1 mean R²':>10}  {'T2 RMSE':>10}  "
          f"{'T2 R²':>7}  {'T3 τ=1':>8}  {'T3 τ=10':>8}")
    print("-" * 80)
    for res in all_results:
        if "error" in res:
            print(f"  {res['name']:<30}  ERROR: {res['error'][:40]}")
            continue
        t1_r2   = res.get("task1_hi", {}).get("__mean__", {}).get("r2", float("nan"))
        t2_rmse = res.get("task2_rul", {}).get("rmse", float("nan"))
        t2_r2   = res.get("task2_rul", {}).get("r2", float("nan"))
        t3      = res.get("task3_forecast") or {}
        t3_1    = t3.get("mean_rmse_all", [float("nan")])[0]
        t3_10   = t3.get("mean_rmse_all", [float("nan")] * 10)[9] \
                  if len(t3.get("mean_rmse_all", [])) >= 10 else float("nan")
        print(f"  {res['name']:<30}  {t1_r2:>10.3f}  {t2_rmse:>10.2f}  "
              f"{t2_r2:>7.3f}  {t3_1:>8.5f}  {t3_10:>8.5f}")
    print(f"{'='*80}")
    print(f"  Results saved to: {out_dir}\n")


if __name__ == "__main__":
    main()
