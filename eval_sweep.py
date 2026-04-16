"""eval_sweep.py — le-wm downstream benchmark sweep.

Runs four evaluation tasks for every listed checkpoint and writes results to a
timestamped output directory.  Multiple checkpoints are distributed across
available GPUs and evaluated in parallel.

Tasks
-----
  1. HI State Estimation      — TransformerProbe(z_t) → 10 HI values
                                 Metrics: R², RMSE, Pearson-r per component
  2. Degradation Velocity     — TransformerProbe(z_t) → ΔHI = HI(t)−HI(t−1)
                                 Predicts the instantaneous degradation rate.
                                 Evaluated on non-maintenance steps only so the
                                 target is always a small negative drift.
                                 Metrics: R², RMSE, Pearson-r per component
  2b. Maintenance Alarm       — binary: will maintenance occur within K steps?
                                 Scores derived from ||ΔHI_pred||  (large positive
                                 predicted jump ≈ imminent restoration event)
                                 Metrics: AUC-ROC, Average Precision, F1
  3. Latent Forecasting       — JEPA.predict() rollout for τ = 1…max_tau steps,
                                 decode ẑ_{t+τ} → HI via the Task-1 probe.
                                 Trajectories split into "clean" (no maintenance
                                 in [t+1…t+τ]) and "event" (≥1 maintenance).
                                 Metrics: mean HI RMSE(τ) per split

Usage
-----
  # use the CHECKPOINTS list below
  python eval_sweep.py

  # override with explicit paths
  python eval_sweep.py /path/a.ckpt /path/b.ckpt

  # named checkpoints (name:path syntax)
  python eval_sweep.py v6_w1:/path/a.ckpt v6_w10:/path/b.ckpt

  # skip the slow forecasting loop
  python eval_sweep.py --skip_task3

  # force single-process even with multiple GPUs
  python eval_sweep.py --no_parallel

  # custom output and forecast horizon
  python eval_sweep.py --out_dir /scratch/results --forecast_horizon 100
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── allow running from the repo root or from any directory ────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from scipy.stats import pearsonr
from sklearn.metrics import (
    average_precision_score, f1_score,
    mean_absolute_error, r2_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from hi_probe import _TransformerProbe, HIProbeCallback
from jepa import JEPA  # noqa: F401 — needed for torch.load
from baselines.ar_lstm.model import LSTMPredictor  # noqa: F401 — needed for torch.load AR-LSTM checkpoints

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULT CHECKPOINT LIST
#  Override at the command line:  python eval_sweep.py /path/a.ckpt /path/b.ckpt
# ══════════════════════════════════════════════════════════════════════════════
OLD_CHECKPOINTS: list[dict] = [
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
    # ── add more entries here ──────────────────────────────────────────────
]

CHECKPOINTS: list[dict] = [
    {
        "path": "/lustre/fswork/projects/rech/yil/ugy35qd/.stable_worldmodel/"
                "turbofan_ar_lstm_h1_hd256_l2/"
                "ar_lstm_h1_hd256_l2_epoch_20_object.ckpt.ckpt",
        "name": "ar_lstm_h1_hd256_l2",
        "encoder_type": "sensor",
    },
    {
        "path": "/lustre/fswork/projects/rech/yil/ugy35qd/.stable_worldmodel/turbofan_ar_lstm_h10_hd256_l2"
                "ar_lstm_h10_hd256_l2_epoch_20_object.ckpt.ckpt.ckpt",
        "name": "turbofan_ar_lstm_h10_hd256_l2",
        "encoder_type": "sensor",
    },
    {
        "path": "/lustre/fswork/projects/rech/yil/ugy35qd/.stable_worldmodel/turbofan_ar_lstm_h50_hd256_l2"
                "ar_lstm_h50_hd256_l2_epoch_21_object.ckpt.ckpt",
        "name": "ar_lstm_h50_hd256_l2",
        "encoder_type": "sensor",
    },

    # ── add more entries here ──────────────────────────────────────────────
]

# ══════════════════════════════════════════════════════════════════════════════
#  SHARED CONFIG  (all checkpoints use the same dataset / probe settings)
# ══════════════════════════════════════════════════════════════════════════════
HDF5_PATH       = (
    "/lustre/fswork/projects/rech/yil/ugy35qd/"
    ".stable_worldmodel/opendeck_2000_lewm.h5"
)
TRAIN_SPLIT     = 0.9
SEED            = 3072
IMG_SIZE        = 28         # ViT encoder only

# Probe
TASK1_SEQ_LENS       = [1, 10, 50]  # seq_lens for Task 1 (HI state estimation)
DELTA_HI_SEQ_LENS    = [1, 10, 50]  # seq_lens to evaluate for Task 2 (ΔHI velocity)
N_PROBE_EPOCHS  = 150
PROBE_LR        = 1e-3
PROBE_PATIENCE  = 20
PROBE_BATCH     = 256
D_MODEL         = 64
NHEAD           = 4
NUM_LAYERS      = 2
DROPOUT         = 0.1
ENC_BATCH       = 2048

# Task 2b: alarm classification
ALARM_HORIZONS  = [10, 20, 50, 100]

# Task 3: latent forecasting
FORECAST_HORIZONS = list(range(1, 51))
FORECAST_N_EPS    = 200
HISTORY_SIZE      = 3


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(hdf5_path: str, encoder_type: str,
                 train_split: float = TRAIN_SPLIT,
                 seed: int = SEED) -> tuple:
    """Load full HDF5 dataset and return train/test split dicts.

    Returns  (tr_data, te_data, hi_names, ep_offset_all, ep_len_all)

    Each split dict has keys:
        obs      : (N, n_sensors) or (N, H, W)
        hi       : (N, n_hi)
        actions  : (N,) float — 1.0 = maintenance event
        ep_ids   : (N,) int   — local episode index per timestep
        eps      : ndarray[int] — global episode indices
    """
    with h5py.File(hdf5_path, "r") as f:
        ep_len_all    = f["ep_len"][:]
        ep_offset_all = f["ep_offset"][:]
        obs_key       = "pixels" if "pixels" in f else "observation.sensors"
        obs_raw       = f[obs_key][:]
        if encoder_type == "sensor" and obs_raw.ndim > 2:
            obs_raw = obs_raw.reshape(len(obs_raw), -1)
        states   = f["observation.state"][:]
        hi_names = list(f.attrs.get(
            "state_label_names", [f"HI_{i}" for i in range(states.shape[1])]
        ))
        actions = f["action"][:]
        if actions.ndim > 1:
            actions = actions[:, 0]

    N_EP = len(ep_len_all)
    rng  = np.random.default_rng(seed)
    perm = rng.permutation(N_EP)
    n_tr = int(N_EP * train_split)

    def _gather(eps):
        obs_l, hi_l, act_l, ep_id_l = [], [], [], []
        for local_i, e in enumerate(eps):
            s, l = int(ep_offset_all[e]), int(ep_len_all[e])
            obs_l.append(obs_raw[s : s + l])
            hi_l.append(states[s : s + l])
            act_l.append(actions[s : s + l])
            ep_id_l.append(np.full(l, local_i, dtype=np.int32))
        return dict(
            obs     = np.concatenate(obs_l),
            hi      = np.concatenate(hi_l),
            actions = np.concatenate(act_l),
            ep_ids  = np.concatenate(ep_id_l),
            eps     = eps,
        )

    return (
        _gather(perm[:n_tr]),
        _gather(perm[n_tr:]),
        hi_names,
        ep_offset_all,
        ep_len_all,
    )


def compute_delta_hi(
    hi: np.ndarray,          # (N, n_hi)
    ep_ids: np.ndarray,      # (N,) local episode index
    actions: np.ndarray,     # (N,) binary maintenance flag
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ΔHI = HI(t) − HI(t−1) within episode boundaries.

    Returns
    -------
    delta_hi   : (M, n_hi) float32 — one row per valid (non-boundary) timestep
    valid_mask : (N,) bool          — True where delta was computable and the
                                      timestep is NOT a maintenance event
                                      (so the target is pure degradation drift,
                                      not a restoration jump)
    """
    delta = np.zeros_like(hi, dtype=np.float32)
    boundary = np.ones(len(hi), dtype=bool)   # True = first step of episode

    for ep_id in np.unique(ep_ids):
        mask = ep_ids == ep_id
        idx  = np.where(mask)[0]
        if len(idx) < 2:
            continue
        delta[idx[1:]] = hi[idx[1:]] - hi[idx[:-1]]
        boundary[idx[0]] = True
        boundary[idx[1:]] = False

    # Exclude: (a) episode boundaries (no prior frame), (b) maintenance steps
    # (large positive jumps — predict velocity on degradation steps only)
    valid = (~boundary) & (actions < 0.5)
    return delta[valid].astype(np.float32), valid


# ══════════════════════════════════════════════════════════════════════════════
#  ENCODING
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_observations(
    model, obs: np.ndarray, encoder_type: str,
    device: torch.device,
    batch_size: int = ENC_BATCH,
    img_size: int = IMG_SIZE,
) -> np.ndarray:
    """Encode (N, ...) observations → (N, embed_dim) numpy array.

    When model.obs_window_size > 1, each frame is tiled w times so the
    TemporalAggregator produces exactly one output per input frame.
    T_out = T_raw − w + 1 = w − w + 1 = 1, so index 0 is always valid.
    """
    is_sensor = (encoder_type == "sensor")
    w = getattr(model, "obs_window_size", 1)
    if not is_sensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    all_emb = []
    for start in range(0, len(obs), batch_size):
        chunk = obs[start : start + batch_size]
        if is_sensor:
            t = torch.from_numpy(chunk).float().to(device).unsqueeze(1)
            if w > 1:
                t = t.expand(-1, w, -1).contiguous()
        else:
            t = torch.from_numpy(chunk).float().to(device)
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
#  GENERIC PROBE TRAINER
# ══════════════════════════════════════════════════════════════════════════════

def train_probe(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    ep_ids_tr: np.ndarray, ep_ids_te: np.ndarray,
    n_outputs: int,
    device: torch.device,
    task_name: str = "probe",
    seq_len: int = 1,
    n_epochs: int = N_PROBE_EPOCHS, lr: float = PROBE_LR,
    patience: int = PROBE_PATIENCE, batch_size: int = PROBE_BATCH,
    d_model: int = D_MODEL, nhead: int = NHEAD,
    num_layers: int = NUM_LAYERS, dropout: float = DROPOUT,
) -> tuple[np.ndarray, np.ndarray, _TransformerProbe, StandardScaler]:
    """Fit a TransformerProbe with early stopping.

    Returns (preds_te, gt_te, probe, input_scaler).
    All arrays are in the *original* (unscaled) target space.
    """
    scaler_X = StandardScaler()
    X_tr_s   = scaler_X.fit_transform(X_tr)
    X_te_s   = scaler_X.transform(X_te)

    Xw_tr, yw_tr = HIProbeCallback._build_windows(X_tr_s, y_tr, ep_ids_tr, seq_len)
    Xw_te, yw_te = HIProbeCallback._build_windows(X_te_s, y_te, ep_ids_te, seq_len)
    logging.info(
        f"  [{task_name}] windows  train={len(Xw_tr):,}  test={len(Xw_te):,}"
    )

    probe = _TransformerProbe(
        input_dim=Xw_tr.shape[2], n_outputs=n_outputs,
        d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout,
    ).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=patience // 2, factor=0.5, min_lr=1e-5
    )

    Xtr_t = torch.from_numpy(Xw_tr).float().to(device)
    ytr_t = torch.from_numpy(yw_tr).float().to(device)
    Xte_t = torch.from_numpy(Xw_te).float().to(device)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr_t, ytr_t),
        batch_size=batch_size, shuffle=True,
    )

    best_rmse, best_state, patience_ctr = float("inf"), None, 0
    for ep in range(n_epochs):
        probe.train()
        for xb, yb in dl:
            opt.zero_grad()
            F.mse_loss(probe(xb), yb).backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            pv = probe(Xte_t).cpu().numpy()
        rmse = float(np.mean(np.sqrt(np.mean((yw_te - pv) ** 2, axis=0))))
        scheduler.step(rmse)
        if rmse < best_rmse:
            best_rmse = rmse
            best_state = copy.deepcopy(probe.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= patience:
            logging.info(
                f"  [{task_name}] early stop ep {ep+1}  best_rmse={best_rmse:.6f}"
            )
            break

    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        preds = probe(Xte_t).cpu().numpy()

    return preds, yw_te, probe, scaler_X


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 1 — HI STATE ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════

def task1_hi(Z_tr, Z_te, tr_data, te_data, hi_names, device, seq_lens=None):
    """HI State Estimation across multiple probe seq_lens.

    seq_lens=[1]  (default): single snapshot z_t → directly measures encoder quality.
    seq_lens=[1,10,50]: also evaluates windowed probes, revealing how much temporal
    context helps — key for comparing AR-LSTM vs JEPA encoder richness.

    Returns dict keyed by "sl{n}" for each seq_len, plus the sl=1 probe/scaler
    for use in Task 3 forecasting.
    """
    if seq_lens is None:
        seq_lens = TASK1_SEQ_LENS

    results = {}
    probe_hi, scaler_hi = None, None   # retained from sl=1 for Task 3

    for sl in seq_lens:
        logging.info(f"  Task 1 — HI State Estimation  (seq_len={sl})")
        preds, gt, probe, scaler = train_probe(
            Z_tr, tr_data["hi"], Z_te, te_data["hi"],
            tr_data["ep_ids"], te_data["ep_ids"],
            n_outputs=tr_data["hi"].shape[1],
            device=device, task_name=f"HI_sl{sl}",
            seq_len=sl,
        )
        per_dim, r2s, rmses, prs = {}, [], [], []
        for i, name in enumerate(hi_names):
            g, p = gt[:, i], preds[:, i]
            r2   = float(r2_score(g, p))
            rmse = float(np.sqrt(np.mean((g - p) ** 2)))
            pr   = float(pearsonr(g, p)[0])
            per_dim[name] = {"r2": r2, "rmse": rmse, "pearson_r": pr}
            r2s.append(r2); rmses.append(rmse); prs.append(pr)
        per_dim["__mean__"] = {
            "r2":        float(np.mean(r2s)),
            "rmse":      float(np.mean(rmses)),
            "pearson_r": float(np.mean(prs)),
        }
        logging.info(
            f"  Task 1 sl={sl}  mean R²={np.mean(r2s):.3f}  "
            f"RMSE={np.mean(rmses):.5f}  Pearson={np.mean(prs):.3f}"
        )
        results[f"sl{sl}"] = per_dim
        if sl == 1:
            probe_hi, scaler_hi = probe, scaler   # used by Task 3

    return results, probe_hi, scaler_hi


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 2 — DEGRADATION VELOCITY  ΔHI = HI(t) − HI(t−1)
# ══════════════════════════════════════════════════════════════════════════════

def task2_delta_hi(Z_tr, Z_te, tr_data, te_data, hi_names, device,
                   seq_lens=None):
    """Predict the instantaneous degradation rate from encoder embeddings.

    Target: ΔHI(t) = HI(t) − HI(t−1), computed within each episode and
    restricted to non-maintenance steps (pure degradation drift, no jumps).

    The probe is evaluated for each value of seq_len in seq_lens:
      seq_len=1  : predict rate from a single snapshot z_t.
      seq_len>1  : predict rate from a window [z_{t-w+1},...,z_t], giving the
                   probe access to the trajectory's local slope.  This is the
                   natural setting for a velocity prediction task.

    Returns dict keyed by seq_len string (e.g. "sl1", "sl10") containing
    per-dimension metrics, plus "alarm" from the best (last-trained) probe.
    """
    if seq_lens is None:
        seq_lens = DELTA_HI_SEQ_LENS

    logging.info(f"  Task 2 — Degradation Velocity (ΔHI)  seq_lens={seq_lens}")

    def _prep(data):
        delta, valid = compute_delta_hi(
            data["hi"], data["ep_ids"], data["actions"]
        )
        Z = Z_tr if data is tr_data else Z_te
        return Z[valid], delta, data["ep_ids"][valid]

    Z_tr_v, delta_tr, ep_tr = _prep(tr_data)
    Z_te_v, delta_te, ep_te = _prep(te_data)

    logging.info(
        f"  ΔHI samples  train={len(Z_tr_v):,}  test={len(Z_te_v):,}"
    )

    all_results = {}
    last_probe_v = None
    last_scaler_v = None

    for sl in seq_lens:
        tag = f"sl{sl}"
        logging.info(f"  ΔHI seq_len={sl}")
        preds, gt, probe_v, scaler_v = train_probe(
            Z_tr_v, delta_tr, Z_te_v, delta_te,
            ep_tr, ep_te,
            n_outputs=delta_tr.shape[1],
            device=device, task_name=f"ΔHI(sl={sl})",
            seq_len=sl,
        )

        per_dim, r2s, rmses, prs = {}, [], [], []
        for i, name in enumerate(hi_names):
            g, p = gt[:, i], preds[:, i]
            r2   = float(r2_score(g, p))
            rmse = float(np.sqrt(np.mean((g - p) ** 2)))
            pr   = float(pearsonr(g, p)[0]) if np.std(g) > 0 and np.std(p) > 0 else 0.0
            per_dim[name] = {"r2": r2, "rmse": rmse, "pearson_r": pr}
            r2s.append(r2); rmses.append(rmse); prs.append(pr)

        per_dim["__mean__"] = {
            "r2": float(np.mean(r2s)),
            "rmse": float(np.mean(rmses)),
            "pearson_r": float(np.mean(prs)),
        }
        logging.info(
            f"  Task 2 (sl={sl})  mean R²={np.mean(r2s):.3f}  "
            f"RMSE={np.mean(rmses):.7f}  Pearson={np.mean(prs):.3f}"
        )
        all_results[tag] = per_dim
        last_probe_v  = probe_v
        last_scaler_v = scaler_v

    # ── Task 2b: maintenance alarm via predicted ΔHI magnitude ────────────
    # Use the longest seq_len probe as the alarm scorer (best trajectory context).
    logging.info("  Task 2b — Maintenance Alarm (AUC via ΔHI score)")
    alarm_results = _alarm_from_delta(
        Z_te_full=Z_te,
        te_data=te_data,
        probe_v=last_probe_v,
        scaler_v=last_scaler_v,
        device=device,
        hi_names=hi_names,
    )

    return all_results, alarm_results


def _alarm_from_delta(Z_te_full, te_data, probe_v, scaler_v, device, hi_names):
    """Score alarm quality using ||ΔHI_pred|| as the anomaly signal.

    For every test timestep we predict ΔHI and take the L2-norm as the alarm
    score (large norm = large predicted change = imminent restoration or anomaly).
    """
    # Encode all test embeddings through the velocity probe
    scaler_X = scaler_v
    Z_te_s   = scaler_X.transform(Z_te_full)
    Z_te_t   = torch.from_numpy(Z_te_s[:, np.newaxis, :]).float().to(device)
    with torch.no_grad():
        delta_pred = probe_v(Z_te_t).cpu().numpy()   # (N, n_hi)
    alarm_score = np.linalg.norm(delta_pred, axis=1)  # (N,) — larger = more alarming

    actions = te_data["actions"]
    ep_ids  = te_data["ep_ids"]
    n       = len(actions)

    # Steps-to-next-maintenance for each timestep (within episode)
    # We rebuild this from ep_ids + actions without calling HIProbeCallback
    steps_to_maint = np.full(n, 999_999, dtype=np.int32)
    for ep_id in np.unique(ep_ids):
        mask = ep_ids == ep_id
        idx  = np.where(mask)[0]
        ep_act = actions[idx]
        maint_times = np.where(ep_act > 0.5)[0]
        for t_local, t_global in enumerate(idx):
            future = maint_times[maint_times > t_local]
            if len(future) > 0:
                steps_to_maint[t_global] = int(future[0] - t_local)

    alarm_out = {}
    for K in ALARM_HORIZONS:
        y_true = (steps_to_maint <= K).astype(int)
        # Require balanced-ish classes
        if y_true.sum() < 20 or (1 - y_true).sum() < 20:
            alarm_out[f"K{K}"] = {"auc": None, "avg_precision": None, "f1": None}
            continue
        auc = float(roc_auc_score(y_true, alarm_score))
        ap  = float(average_precision_score(y_true, alarm_score))
        # F1: threshold at top-K% of scores
        thresh   = np.percentile(alarm_score, 100 * (1 - y_true.mean()))
        y_pred   = (alarm_score >= thresh).astype(int)
        f1       = float(f1_score(y_true, y_pred, zero_division=0))
        alarm_out[f"K{K}"] = {"auc": auc, "avg_precision": ap, "f1": f1}
        logging.info(
            f"    Alarm K={K:>3}:  AUC={auc:.3f}  AP={ap:.3f}  F1={f1:.3f}"
        )
    return alarm_out


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 3 — LATENT FORECASTING  (JEPA autoregressive rollout)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _decode_z_to_hi(probe, scaler_X, z_np, device):
    z_s = scaler_X.transform(z_np)
    z_t = torch.from_numpy(z_s[:, np.newaxis, :]).float().to(device)
    return probe(z_t).cpu().numpy()


@torch.no_grad()
def task3_forecast(
    model, probe_hi, scaler_hi,
    te_data, ep_offset_all, ep_len_all,
    hdf5_path, encoder_type, device,
    max_tau: int, n_eps: int, history_size: int = HISTORY_SIZE,
    step_size: int = 10,
):
    """Autoregressive JEPA rollout, decoded to HI via the Task-1 probe.

    Each trajectory starting at frame t is classified:
      "clean" — no maintenance in (t, t+τ]
      "event" — ≥1 maintenance event in (t, t+τ]

    Speed-up strategy
    -----------------
    Instead of a (t, τ) double loop (~N_frames × max_tau sequential forward
    passes), we batch ALL starting frames for a given episode and unroll τ
    steps jointly.  The outer loop is now just max_tau batched predict calls
    per episode regardless of episode length.

    step_size : subsample starting frames every `step_size` timesteps
                (default 10 — 10× fewer starts, negligible metric impact).
    """
    logging.info(
        f"  Task 3 — Latent Forecasting (τ=1…{max_tau}, step_size={step_size})"
    )
    n_hi = te_data["hi"].shape[1]
    H    = history_size

    with h5py.File(hdf5_path, "r") as f:
        obs_key    = "pixels" if "pixels" in f else "observation.sensors"
        all_obs    = f[obs_key][:]
        if encoder_type == "sensor" and all_obs.ndim > 2:
            all_obs = all_obs.reshape(len(all_obs), -1)
        all_states  = f["observation.state"][:]
        all_actions = f["action"][:]
        if all_actions.ndim > 1:
            all_actions = all_actions[:, 0]

    def _zeros_tau():
        return np.zeros((max_tau, n_hi), dtype=np.float64)

    sqerr  = {"all": _zeros_tau(), "clean": _zeros_tau(), "event": _zeros_tau()}
    cnt    = {k: np.zeros(max_tau, np.int64) for k in sqerr}

    te_eps = te_data["eps"]
    n_eval = min(n_eps, len(te_eps))

    for ep_i, ep_idx in enumerate(te_eps[:n_eval]):
        if ep_i % 10 == 0:
            logging.info(f"    episode {ep_i}/{n_eval}…")
        s = int(ep_offset_all[ep_idx])
        l = int(ep_len_all[ep_idx])
        if l < H + max_tau + 1:
            continue

        obs_ep = all_obs[s : s + l]
        hi_ep  = all_states[s : s + l]
        act_ep = all_actions[s : s + l]

        # Encode all frames once  (T, D)
        z_ep = encode_observations(
            model, obs_ep, encoder_type, device, batch_size=512
        )
        D = z_ep.shape[1]

        # ── Subsampled starting frames ────────────────────────────────────
        # Only keep starts where we have at least max_tau future frames.
        starts = np.arange(H - 1, l - max_tau, step_size)
        if len(starts) == 0:
            continue
        N = len(starts)

        # ── Pre-compute event mask: has_event[i, tau] = True if there is a
        # maintenance action in act_ep[starts[i]+1 : starts[i]+tau+1].
        # Use searchsorted for fully vectorised O(N * log(M)) computation.
        maint_times = np.where(act_ep > 0.5)[0]   # (M,)
        # has_event[i, tau-1] = True iff any maint in (starts[i], starts[i]+tau]
        # = (# maints <= starts[i]+tau) > (# maints <= starts[i])
        if len(maint_times) > 0:
            # count_up_to[i] = # maints in [0, starts[i]]
            count_at_start  = np.searchsorted(maint_times, starts,     side="right")
            # for each tau, count_at_target[i, tau] = # maints in [0, starts[i]+tau]
            targets = starts[:, None] + np.arange(1, max_tau + 1)[None, :]  # (N, max_tau)
            count_at_target = np.searchsorted(
                maint_times, targets.ravel(), side="right"
            ).reshape(N, max_tau)
            has_event = count_at_target > count_at_start[:, None]  # (N, max_tau) bool
        else:
            has_event = np.zeros((N, max_tau), dtype=bool)

        # ── Initialise rolling window batch  (N, H, D) ───────────────────
        ctx_np  = np.stack([z_ep[t - H + 1 : t + 1] for t in starts])
        emb_win = torch.from_numpy(ctx_np).float().to(device)   # (N, H, D)
        act_buf = torch.zeros(N, H, D, device=device)

        # ── Unroll τ steps — ONE batched predict call per tau ─────────────
        for tau in range(1, max_tau + 1):
            pred_seq = model.predict(emb_win[:, -H:], act_buf[:, -H:])
            next_z   = pred_seq[:, -1:]                   # (N, 1, D)

            # Decode all N predictions at once
            hi_pred = _decode_z_to_hi(
                probe_hi, scaler_hi,
                next_z[:, 0].cpu().numpy(), device
            )  # (N, n_hi)

            # Ground-truth HI for each starting frame at this horizon
            target_ts = starts + tau                      # (N,)
            hi_gt     = hi_ep[target_ts]                  # (N, n_hi)

            se = (hi_pred - hi_gt) ** 2                   # (N, n_hi)

            evt = has_event[:, tau - 1]                   # (N,) bool

            sqerr["all"][tau - 1]             += se.sum(axis=0)
            sqerr["event"][tau - 1]           += se[evt].sum(axis=0)
            sqerr["clean"][tau - 1]           += se[~evt].sum(axis=0)
            cnt["all"][tau - 1]               += N
            cnt["event"][tau - 1]             += int(evt.sum())
            cnt["clean"][tau - 1]             += int((~evt).sum())

            # Advance rolling window
            emb_win  = torch.cat([emb_win, next_z], dim=1)
            act_buf  = torch.cat(
                [act_buf, torch.zeros(N, 1, D, device=device)], dim=1
            )

    results = {}
    for key in ("all", "clean", "event"):
        safe_cnt = np.maximum(cnt[key][:, None], 1)
        rmse     = np.sqrt(sqerr[key] / safe_cnt)           # (max_tau, n_hi)
        mean_r   = rmse.mean(axis=1)                         # (max_tau,)
        results[f"rmse_{key}"]      = rmse.tolist()
        results[f"mean_rmse_{key}"] = mean_r.tolist()
        results[f"count_{key}"]     = cnt[key].tolist()

    # Action-divergence gap = rmse_event − rmse_clean
    event_arr = np.array(results["mean_rmse_event"])
    clean_arr = np.array(results["mean_rmse_clean"])
    results["mean_rmse_gap"] = (event_arr - clean_arr).tolist()

    for tau in [1, 5, 10, 20, min(50, max_tau)]:
        if tau <= max_tau and cnt["all"][tau - 1] > 0:
            logging.info(
                f"    τ={tau:>2}  all={results['mean_rmse_all'][tau-1]:.5f}"
                f"  clean={results['mean_rmse_clean'][tau-1]:.5f}"
                f"  event={results['mean_rmse_event'][tau-1]:.5f}"
                f"  gap={results['mean_rmse_gap'][tau-1]:+.5f}"
                f"  [n={cnt['all'][tau-1]:,}  evt={cnt['event'][tau-1]:,}]"
            )
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PER-CHECKPOINT RUNNER  (called in each worker process)
# ══════════════════════════════════════════════════════════════════════════════

ALL_TASKS = {1, 2, 3}   # valid task numbers


def run_one(
    cfg: dict,
    tr_data: dict, te_data: dict,
    hi_names: list,
    ep_offset_all: np.ndarray, ep_len_all: np.ndarray,
    device: torch.device,
    tasks: set | None = None,
    forecast_horizon: int | None = None,
    step_size: int = 10,
) -> dict:
    """Run downstream evaluation for one checkpoint.

    Parameters
    ----------
    tasks : set of ints, e.g. {1, 2} to run only Tasks 1 & 2.
            None (default) runs all tasks.
    """
    if tasks is None:
        tasks = ALL_TASKS

    name    = cfg["name"]
    enc_t   = cfg.get("encoder_type", "sensor")
    max_tau = forecast_horizon or max(FORECAST_HORIZONS)

    logging.info(f"\n{'='*70}\n  {name}  [{device}]  tasks={sorted(tasks)}\n{'='*70}")

    model = torch.load(cfg["path"], map_location=device, weights_only=False)
    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    w = getattr(model, "obs_window_size", 1)
    logging.info(f"  Loaded — {n_params:,} params  obs_window={w}")

    logging.info("  Encoding observations…")
    Z_tr = encode_observations(model, tr_data["obs"], enc_t, device)
    Z_te = encode_observations(model, te_data["obs"], enc_t, device)
    logging.info(f"  Z_tr={Z_tr.shape}  Z_te={Z_te.shape}")

    t1 = t2_vel = t2b_alarm = t3 = None
    probe_hi = scaler_hi = None

    if 1 in tasks:
        t1, probe_hi, scaler_hi = task1_hi(
            Z_tr, Z_te, tr_data, te_data, hi_names, device
        )

    if 2 in tasks:
        t2_vel, t2b_alarm = task2_delta_hi(
            Z_tr, Z_te, tr_data, te_data, hi_names, device
        )

    if 3 in tasks:
        if probe_hi is None:
            # Task 3 needs the Task-1 probe to decode latents → HI space.
            # Train sl=1 probe silently even if Task 1 was skipped.
            logging.info("  Task 3 requires Task-1 probe — training sl=1 probe…")
            _, probe_hi, scaler_hi = task1_hi(
                Z_tr, Z_te, tr_data, te_data, hi_names, device, seq_lens=[1]
            )
        t3 = task3_forecast(
            model, probe_hi, scaler_hi,
            te_data, ep_offset_all, ep_len_all,
            HDF5_PATH, enc_t, device, max_tau, FORECAST_N_EPS,
            step_size=step_size,
        )

    del model; torch.cuda.empty_cache()

    return {
        "name":            name,
        "path":            cfg["path"],
        "encoder_type":    enc_t,
        "obs_window_size": w,
        "n_params":        n_params,
        "embed_dim":       int(Z_tr.shape[1]),
        "task1_hi":        t1,
        "task2_delta_hi":  t2_vel,
        "task2b_alarm":    t2b_alarm,
        "task3_forecast":  t3,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MULTIPROCESSING WORKER
# ══════════════════════════════════════════════════════════════════════════════

def _worker(rank, cfg, gpu_id, shared_data, result_path, tasks, forecast_horizon):
    """Subprocess entry point — one checkpoint, one GPU."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"[GPU{gpu_id}] %(levelname)s  %(message)s",
        force=True,
    )
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    tr_data, te_data, hi_names, ep_offset_all, ep_len_all = shared_data
    try:
        result = run_one(
            cfg, tr_data, te_data, hi_names,
            ep_offset_all, ep_len_all, device,
            tasks=tasks,
            forecast_horizon=forecast_horizon,
        )
    except Exception as e:
        import traceback
        result = {"name": cfg["name"], "error": str(e),
                  "traceback": traceback.format_exc()}
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    logging.info(f"  Saved → {result_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def write_flat_csv(all_results, hi_names, path):
    import csv
    rows, header = [], ["model", "obs_window", "task", "seq_len", "component", "metric", "value"]
    for res in all_results:
        if "error" in res:
            continue
        name = res["name"]; w = res["obs_window_size"]
        # task1_hi is keyed by sl_tag ("sl1", "sl10", "sl50") → component → metric
        t1 = res.get("task1_hi") or {}
        for sl_tag, sl_metrics in t1.items():
            for comp, m in sl_metrics.items():
                for metric, val in m.items():
                    rows.append([name, w, "task1_hi", sl_tag, comp, metric, val])
        for sl_tag, sl_metrics in res.get("task2_delta_hi", {}).items():
            for comp, m in sl_metrics.items():
                for metric, val in m.items():
                    rows.append([name, w, "task2_delta_hi", sl_tag, comp, metric, val])
        for k_str, m in res.get("task2b_alarm", {}).items():
            for metric, val in (m or {}).items():
                if val is not None:
                    rows.append([name, w, "task2b_alarm", "-", k_str, metric, val])
        t3 = res.get("task3_forecast") or {}
        for key in ("all", "clean", "event", "gap"):
            for tau_i, val in enumerate(t3.get(f"mean_rmse_{key}", [])):
                rows.append([name, w, f"task3_{key}", "-", f"tau_{tau_i+1}", "mean_rmse", val])
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerows([header] + rows)
    logging.info(f"Flat CSV → {path}")


def write_curves_csv(all_results, path):
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "obs_window", "tau",
                    "rmse_all", "rmse_clean", "rmse_event", "rmse_gap",
                    "n_all", "n_clean", "n_event"])
        for res in all_results:
            t3 = res.get("task3_forecast")
            if not t3:
                continue
            for i in range(len(t3["mean_rmse_all"])):
                w.writerow([
                    res["name"], res["obs_window_size"], i + 1,
                    t3["mean_rmse_all"][i],
                    t3["mean_rmse_clean"][i],
                    t3["mean_rmse_event"][i],
                    t3["mean_rmse_gap"][i],
                    t3["count_all"][i],
                    t3["count_clean"][i],
                    t3["count_event"][i],
                ])
    logging.info(f"Task-3 curves CSV → {path}")


def print_summary(all_results):
    print(f"\n{'='*90}")
    print(f"  {'Model':<32}  {'T1 R²':>6}  {'T2 R²':>6}  "
          f"{'Alarm AUC@K20':>13}  {'T3 τ=1':>7}  {'T3 τ=10':>8}  {'gap@τ=10':>9}")
    print("-" * 90)
    for res in all_results:
        if "error" in res:
            print(f"  {res['name']:<32}  ERROR: {res['error'][:40]}")
            continue
        t1   = res.get("task1_hi", {}).get("__mean__", {})
        # Show best (highest R²) seq_len for Task 2
        t2_all = res.get("task2_delta_hi", {})
        t2 = max(
            (v.get("__mean__", {}) for v in t2_all.values()),
            key=lambda d: d.get("r2", -999),
            default={},
        ) if t2_all else {}
        alrm = res.get("task2b_alarm", {}).get("K20", {}) or {}
        t3   = res.get("task3_forecast") or {}
        mr_all  = t3.get("mean_rmse_all", [float("nan")] * 50)
        gap_arr = t3.get("mean_rmse_gap",  [float("nan")] * 50)
        print(
            f"  {res['name']:<32}  "
            f"{t1.get('r2', float('nan')):>6.3f}  "
            f"{t2.get('r2', float('nan')):>6.3f}  "
            f"{alrm.get('auc', float('nan') if alrm.get('auc') is None else alrm['auc']):>13.3f}  "
            f"{mr_all[0]:>7.5f}  "
            f"{mr_all[9] if len(mr_all) > 9 else float('nan'):>8.5f}  "
            f"{gap_arr[9] if len(gap_arr) > 9 else float('nan'):>+9.5f}"
        )
    print("=" * 90)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_checkpoint_arg(s: str) -> dict:
    """Parse  name:/path/to/ckpt  or just  /path/to/ckpt."""
    if ":" in s and not s.startswith("/"):
        name, path = s.split(":", 1)
    else:
        path = s
        name = Path(s).stem
    return {"path": path, "name": name, "encoder_type": "sensor"}


def main():
    parser = argparse.ArgumentParser(
        description="le-wm downstream benchmark sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "checkpoints", nargs="*",
        help="checkpoint paths, optionally as name:/path. "
             "Omit to use the CHECKPOINTS list at the top of the script.",
    )
    parser.add_argument("--out_dir",          default=None)
    parser.add_argument("--tasks",            type=int, nargs="+", default=None,
                        metavar="N",
                        help="tasks to run, e.g. --tasks 1 2  (default: all = 1 2 3)")
    parser.add_argument("--no_parallel",      action="store_true",
                        help="force sequential execution even with multiple GPUs")
    parser.add_argument("--forecast_horizon", type=int, default=None,
                        help=f"max τ for Task 3 (default {max(FORECAST_HORIZONS)})")
    parser.add_argument("--hdf5",             default=None,
                        help="override HDF5_PATH in the script")
    parser.add_argument("--step_size",        type=int, default=10,
                        help="subsample starting frames every N steps for Task 3 "
                             "(default 10 — 10× faster, negligible metric impact)")
    args = parser.parse_args()
    # --tasks 1 2 → set {1, 2};  omitted → None (all tasks)
    tasks = set(args.tasks) if args.tasks else None

    if args.hdf5:
        global HDF5_PATH
        HDF5_PATH = args.hdf5

    checkpoints = (
        [parse_checkpoint_arg(c) for c in args.checkpoints]
        if args.checkpoints else CHECKPOINTS
    )

    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {out_dir}")

    # ── Load dataset once in the main process ─────────────────────────────
    logging.info(f"Loading dataset from {HDF5_PATH}…")
    shared_data = load_dataset(
        HDF5_PATH, checkpoints[0]["encoder_type"], TRAIN_SPLIT, SEED
    )
    tr_data, te_data, hi_names, ep_offset_all, ep_len_all = shared_data
    logging.info(
        f"Dataset  train={len(tr_data['obs']):,}  test={len(te_data['obs']):,}  "
        f"HI dims={len(hi_names)}  episodes={len(ep_len_all)}"
    )

    n_gpus = torch.cuda.device_count()
    logging.info(f"GPUs available: {n_gpus}  Checkpoints: {len(checkpoints)}")

    result_paths = [out_dir / f"{cfg['name']}.json" for cfg in checkpoints]

    if n_gpus > 1 and len(checkpoints) > 1 and not args.no_parallel:
        # ── Parallel: one subprocess per checkpoint, round-robin GPU assignment
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
        procs = []
        for rank, (cfg, rp) in enumerate(zip(checkpoints, result_paths)):
            gpu_id = rank % n_gpus
            p = mp.Process(
                target=_worker,
                args=(rank, cfg, gpu_id, shared_data,
                      str(rp), tasks, args.forecast_horizon),
            )
            p.start()
            procs.append(p)
            logging.info(f"  Spawned process {rank} → GPU {gpu_id} ({cfg['name']})")
        for p in procs:
            p.join()
        logging.info("All workers finished.")
    else:
        # ── Sequential: single GPU or forced sequential ────────────────────
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for cfg, rp in zip(checkpoints, result_paths):
            result = run_one(
                cfg, tr_data, te_data, hi_names,
                ep_offset_all, ep_len_all, device,
                tasks=tasks,
                forecast_horizon=args.forecast_horizon,
                step_size=args.step_size,
            )
            with open(rp, "w") as f:
                json.dump(result, f, indent=2)

    # ── Collect, aggregate, write outputs ─────────────────────────────────
    all_results = []
    for rp in result_paths:
        if rp.exists():
            with open(rp) as f:
                all_results.append(json.load(f))
        else:
            all_results.append({"name": rp.stem, "error": "result file missing"})

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"Summary JSON → {summary_path}")

    write_flat_csv(all_results, hi_names, out_dir / "metrics_flat.csv")
    write_curves_csv(all_results, out_dir / "task3_curves.csv")
    print_summary(all_results)
    print(f"  Results → {out_dir}\n")


if __name__ == "__main__":
    main()
