"""
RSSM/DreamerV3 Evaluation Script (TurboSens scenario 4)
========================================================

Mirrors `baselines/ar_lstm/eval_ar_lstm.py` for the RSSM baseline. Two
tasks land in `--out` as a JSON, and a comparison table is printed.

  Task 1 — HI State Estimation
    Freeze the encoder, run model.encode() over every test frame,
    fit a Ridge probe → predict the 10-dim HI state. Reports
    R² / RMSE / Pearson-r per HI dim and the means across dims.

  Task 3 — Latent Forecasting & Action-Divergence Gap
    For n_samples random rollout starts, observe an H-step context
    with `dynamics.observe()`, then roll the **prior** forward with
    `dynamics.img_step()` for τ ∈ [1, max_horizon] using the ground
    truth future actions. Decode each predicted feat to HI via the
    Task-1 probe. Reports rmse_clean[τ], rmse_event[τ], and the gap.

Usage
-----
    cd ~/thesis/le-wm
    source .venv/bin/activate
    python baselines/rssm/eval_rssm.py \\
        --ckpt /home/lthil/.stable_worldmodel/rssm_scenario4_T64_S32x32_D512/rssm_s4_T64_S32x32_D512_epoch_10_object.ckpt \\
        --data /home/lthil/.stable_worldmodel/turbosens2_test.h5 \\
        --out  eval_results/rssm_s4/task1_test.json \\
        --history_size 16 --horizon 50 --n_samples 500
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
import torch.nn.functional as F

# ── Add le-wm root ──────────────────────────────────────────────────────────
_LE_WM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LE_WM_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data helpers — turbosens2 layout: pixels (N, 11, 16) → flatten to (N, 176).
# ═══════════════════════════════════════════════════════════════════════════════

def load_episodes(hdf5_path: str, train_split: float = 0.9, seed: int = 3072):
    """Load all sensor episodes and HI states from a turbosens2 HDF5 file.

    Returns
    -------
    obs_train, obs_test  : list of np.ndarray, each (T_ep, 176)  float32
    hi_train,  hi_test   : list of np.ndarray, each (T_ep, 10)   float32
    act_train, act_test  : list of np.ndarray, each (T_ep,)      int64
    train_idx, test_idx  : list[int]
    labels               : list[str], length 10 (HI dim names)
    """
    with h5py.File(hdf5_path, "r") as f:
        ep_lens   = f["ep_len"][:]
        ep_offs   = f["ep_offset"][:]
        obs_raw   = f["pixels"][:]              # (N, 11, 16)
        hi_all    = f["observation.state"][:]   # (N, 10)
        act_all   = f["action"][:]              # (N, 1) or (N,)
        labels    = [s.decode() if isinstance(s, bytes) else s
                     for s in f.attrs.get("state_label_names", [])]

    obs_all = obs_raw.reshape(len(obs_raw), -1).astype(np.float32)  # (N, 176)
    if act_all.ndim > 1:
        act_all = act_all.squeeze(-1)
    act_all = act_all.astype(np.int64)

    n_episodes = len(ep_lens)
    rng        = np.random.default_rng(seed)
    idx        = rng.permutation(n_episodes)
    n_train    = int(np.floor(train_split * n_episodes))
    train_idx  = sorted(idx[:n_train].tolist())
    test_idx   = sorted(idx[n_train:].tolist())

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
# Encoding helper — uses RSSM's JEPA-compatible encode(info)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_episodes(model, obs_list: list, act_list: list,
                    device: torch.device, batch_size: int = 4096) -> list[np.ndarray]:
    """Encode a list of sensor episodes to RSSM feat = [stoch_flat, deter].

    The HIProbeCallback path uses (B, 1, n_sensors) batches with a
    zero-action and `is_first[:, 0] = True`, which is exactly what
    `model.encode()` does internally. We replicate that here.
    """
    model.eval()
    out_list = []
    for obs, act in zip(obs_list, act_list):
        T = len(obs)
        embs = []
        for s in range(0, T, batch_size):
            x = torch.from_numpy(obs[s : s + batch_size]).float().unsqueeze(1).to(device)  # (B, 1, 176)
            a = torch.from_numpy(act[s : s + batch_size]).long().unsqueeze(1).unsqueeze(-1).to(device)  # (B, 1, 1)
            info = {"pixels": x, "action": a}
            out  = model.encode(info)
            embs.append(out["emb"].squeeze(1).cpu().numpy())
        out_list.append(np.concatenate(embs, axis=0))
    return out_list


# ═══════════════════════════════════════════════════════════════════════════════
# Task 1 — HI State Estimation
# ═══════════════════════════════════════════════════════════════════════════════

def task1_hi_probing(
    model,
    obs_train, act_train, hi_train,
    obs_test,  act_test,  hi_test,
    device,
) -> tuple[dict, "Ridge", np.ndarray, np.ndarray]:
    """Train a Ridge probe on frozen RSSM feat → HI prediction.

    Returns (results_dict, fitted_probe, Z_te, Y_te). The probe and the
    encoded test-set arrays are reused by Task 3.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr

    log.info("Task 1: encoding training episodes …")
    Z_tr = np.concatenate(encode_episodes(model, obs_train, act_train, device), axis=0)
    Y_tr = np.concatenate(hi_train, axis=0)

    log.info("Task 1: encoding test episodes …")
    Z_te = np.concatenate(encode_episodes(model, obs_test, act_test, device), axis=0)
    Y_te = np.concatenate(hi_test, axis=0)

    log.info(f"Task 1: fitting Ridge (Z_tr={Z_tr.shape}, Y_tr={Y_tr.shape})")
    probe = Ridge(alpha=1.0).fit(Z_tr, Y_tr)
    Y_pred = probe.predict(Z_te)

    r2_per   = r2_score(Y_te, Y_pred, multioutput="raw_values").tolist()
    rmse_per = np.sqrt(np.mean((Y_te - Y_pred) ** 2, axis=0)).tolist()
    pear_per = [
        float(pearsonr(Y_te[:, i], Y_pred[:, i])[0])
        for i in range(Y_te.shape[1])
    ]

    results = {
        "r2_per_component":      r2_per,
        "rmse_per_component":    rmse_per,
        "pearson_per_component": pear_per,
        "mean_r2":               float(np.mean(r2_per)),
        "mean_rmse":             float(np.mean(rmse_per)),
        "mean_pearson":          float(np.mean(pear_per)),
    }
    log.info(f"Task 1 — mean R²={results['mean_r2']:.4f}  "
             f"RMSE={results['mean_rmse']:.5f}  r={results['mean_pearson']:.4f}")
    return results, probe, Z_te, Y_te


# ═══════════════════════════════════════════════════════════════════════════════
# Task 3 — Latent Forecasting via RSSM imagination
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def rssm_rollout(model, obs_ctx: np.ndarray, act_ctx: np.ndarray,
                 act_future: np.ndarray, device: torch.device) -> np.ndarray:
    """Roll out RSSM from an H-step context, then imagine `n_steps` ahead.

    Parameters
    ----------
    obs_ctx     : (H, 176) — context observations
    act_ctx     : (H,)     — context actions (int)
    act_future  : (n_steps,) — actions taken AT each future step (int)
    device      : torch device

    Returns
    -------
    feat_pred : (n_steps, feat_dim) — RSSM feat at each future step
    """
    model.eval()
    H        = obs_ctx.shape[0]
    n_steps  = act_future.shape[0]

    # ── Observe context → final posterior state ───────────────────────────
    obs = torch.from_numpy(obs_ctx).float().unsqueeze(0).to(device)             # (1, H, 176)
    act = torch.from_numpy(act_ctx).long().unsqueeze(0).unsqueeze(-1).to(device)  # (1, H, 1)
    a_oh = model._one_hot_actions(act)                                          # (1, H, A)

    embed_flat = model.encoder(obs.reshape(-1, obs.shape[-1]))                  # (H, embed)
    embed = embed_flat.reshape(1, H, -1)
    is_first = torch.zeros(1, H, dtype=torch.bool, device=device)
    is_first[0, 0] = True
    post, _ = model.dynamics.observe(embed, a_oh, is_first)
    state   = {k: v[:, -1] for k, v in post.items()}                            # last-step state

    # ── Imagine forward using ground-truth future actions ─────────────────
    feats = []
    a_fut = torch.from_numpy(act_future).long().to(device)                      # (n_steps,)
    a_fut = F.one_hot(a_fut.clamp(0, model.num_actions - 1),
                      num_classes=model.num_actions).float().unsqueeze(0)        # (1, n_steps, A)
    for t in range(n_steps):
        state = model.dynamics.img_step(state, a_fut[:, t])
        feats.append(model.dynamics.get_feat(state).squeeze(0).cpu().numpy())

    return np.stack(feats, axis=0)                                              # (n_steps, feat)


def task3_latent_forecasting(
    model, probe,
    obs_test, act_test, hi_test,
    device,
    history_size: int = 16,
    max_horizon:  int = 50,
    n_samples:    int = 500,
    seed:         int = 3072,
) -> dict:
    """Latent forecasting via RSSM imagination (Task 3)."""

    log.info(f"Task 3: H={history_size}  max_horizon={max_horizon}  n_samples={n_samples}")

    # Build sample pool: (ep_idx, t0) pairs s.t. there are H+max_horizon frames available.
    rng = np.random.default_rng(seed)
    pool = []
    for ep_idx, obs in enumerate(obs_test):
        T = len(obs)
        if T < history_size + max_horizon + 1:
            continue
        starts = np.arange(history_size, T - max_horizon)
        for t0 in starts:
            pool.append((ep_idx, int(t0)))
    if not pool:
        log.warning("Task 3: no eligible samples; skipping")
        return {"skipped": True, "reason": "no samples meet history_size + max_horizon"}
    rng.shuffle(pool)
    pool = pool[:n_samples]

    rmse_clean_sum = np.zeros(max_horizon, dtype=np.float64)
    rmse_clean_n   = 0
    rmse_event_sum = np.zeros(max_horizon, dtype=np.float64)
    rmse_event_n   = 0

    for k, (ep_idx, t0) in enumerate(pool):
        obs_ep = obs_test[ep_idx]
        act_ep = act_test[ep_idx]
        hi_ep  = hi_test[ep_idx]

        ctx_obs = obs_ep[t0 - history_size : t0]                 # (H, 176)
        ctx_act = act_ep[t0 - history_size : t0]                 # (H,)
        fut_act = act_ep[t0 : t0 + max_horizon]                  # (max_horizon,)
        gt_hi   = hi_ep[t0 : t0 + max_horizon]                   # (max_horizon, 10)

        feats = rssm_rollout(model, ctx_obs, ctx_act, fut_act, device)
        pred_hi = probe.predict(feats)                           # (max_horizon, 10)
        per_step_rmse = np.sqrt(np.mean((pred_hi - gt_hi) ** 2, axis=1))  # (max_horizon,)

        # "event" = any non-zero action in the rollout window; else "clean"
        if np.any(fut_act != 0):
            rmse_event_sum += per_step_rmse
            rmse_event_n   += 1
        else:
            rmse_clean_sum += per_step_rmse
            rmse_clean_n   += 1

        if (k + 1) % 100 == 0:
            log.info(f"Task 3: rolled {k+1}/{len(pool)} starts  "
                     f"({rmse_clean_n} clean, {rmse_event_n} event)")

    rmse_clean = (rmse_clean_sum / max(1, rmse_clean_n)).tolist()
    rmse_event = (rmse_event_sum / max(1, rmse_event_n)).tolist()
    gap = [(e - c) for e, c in zip(rmse_event, rmse_clean)]

    results = {
        "history_size":         history_size,
        "max_horizon":          max_horizon,
        "n_clean":              int(rmse_clean_n),
        "n_event":              int(rmse_event_n),
        "rmse_clean":           rmse_clean,
        "rmse_event":           rmse_event,
        "action_divergence_gap": gap,
    }

    # Headline numbers for the comparison table
    if rmse_clean_n:
        log.info(f"Task 3 — clean  RMSE@τ=10={rmse_clean[9]:.4f}  "
                 f"τ=50={rmse_clean[-1]:.4f}")
    if rmse_event_n:
        log.info(f"Task 3 — event  RMSE@τ=10={rmse_event[9]:.4f}  "
                 f"τ=50={rmse_event[-1]:.4f}  "
                 f"gap@50={gap[-1]:+.4f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RSSM on TurboSens turbosens2 Tasks 1+3")
    p.add_argument("--ckpt",          required=True)
    p.add_argument("--data",          required=True,
                   help="Path to turbosens2_test*.h5")
    p.add_argument("--out",           default="eval_results/rssm_s4/task_results.json")
    p.add_argument("--history_size",  type=int, default=16)
    p.add_argument("--horizon",       type=int, default=50)
    p.add_argument("--n_samples",     type=int, default=500)
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip_task3",    action="store_true")
    p.add_argument("--seed",          type=int, default=3072)
    return p.parse_args()


def evaluate(ckpt_path: str, data_path: str, device_str: str,
             history_size: int, horizon: int, n_samples: int,
             skip_task3: bool, seed: int) -> dict:
    device = torch.device(device_str)
    log.info(f"Loading checkpoint: {ckpt_path}")
    model = torch.load(ckpt_path, map_location=device, weights_only=False)
    if hasattr(model, "module"):
        model = model.module
    model.eval().to(device)

    log.info(f"Loading dataset: {data_path}")
    (obs_tr, hi_tr, act_tr,
     obs_te, hi_te, act_te,
     _, _, labels) = load_episodes(data_path, seed=seed)

    log.info(f"  train episodes: {len(obs_tr)}   test episodes: {len(obs_te)}")
    log.info(f"  HI labels: {labels}")

    results = {
        "ckpt":         ckpt_path,
        "data":         data_path,
        "hi_labels":    labels,
        "history_size": history_size,
    }

    log.info("─" * 60)
    log.info("Running Task 1: HI State Estimation")
    log.info("─" * 60)
    t1, probe, _, _ = task1_hi_probing(
        model,
        obs_train=obs_tr, act_train=act_tr, hi_train=hi_tr,
        obs_test=obs_te,  act_test=act_te,  hi_test=hi_te,
        device=device,
    )
    results["task1"] = t1

    if not skip_task3:
        log.info("─" * 60)
        log.info("Running Task 3: Latent Forecasting")
        log.info("─" * 60)
        results["task3"] = task3_latent_forecasting(
            model, probe,
            obs_test=obs_te, act_test=act_te, hi_test=hi_te,
            device=device,
            history_size=history_size,
            max_horizon=horizon,
            n_samples=n_samples,
            seed=seed,
        )

    return results


def main():
    args = parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res = evaluate(
        ckpt_path=args.ckpt, data_path=args.data, device_str=args.device,
        history_size=args.history_size, horizon=args.horizon,
        n_samples=args.n_samples, skip_task3=args.skip_task3, seed=args.seed,
    )
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    log.info(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
