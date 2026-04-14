"""eval_ood.py -- Task 4: Out-of-Distribution Detection for le-wm.

Generates OOD episodes via the turbofan degradation simulator (or from the
static HDF5 dataset using extreme-episode selection when --no_simulator is
given), then evaluates three unsupervised anomaly detectors on frozen
JEPA encoder + predictor representations:

  Detector 1 -- Surprise score
      S(t) = ||z_{t+1} - f_theta(z_t, a_t)||_2
      (autoregressive residual in latent space)

  Detector 2 -- Mahalanobis distance
      d_M(z) based on mu / Sigma fitted on training embeddings

  Detector 3 -- k-Nearest-Neighbour distance
      d_kNN(z) = mean distance to top-k training embeddings

All three detectors are thresholded to produce a binary OOD label and
evaluated against the ground-truth ID/OOD split with AUC-ROC and
average-precision metrics.

OOD Scenarios (simulator required)
------------------------------------
  accel_3x   -- degradation slope 3x faster than training distribution
  accel_5x   -- degradation slope 5x faster (extreme)
  premaint   -- maintenance_interval=(50,150) << training (10000,10001)
                with low effectiveness (maintenance_coeff=0.1)
  correlated -- only high-pressure subsystem (CmpH + TrbH) degrades,
                simulating a localised correlated component failure

Without the simulator (--no_simulator)
---------------------------------------
  ID-extreme -- top-10% fastest-degrading test episodes as near-OOD proxy.
  This uses only the static HDF5 dataset and requires no external service.

Usage
-----
  # With ZMQ simulator worker running on a CPU node:
  python eval_ood.py name:checkpoint.ckpt \\
      --sim_addr tcp://r6n1.jean-zay:5555 \\
      --hdf5 /path/to/opendeck_2000_lewm.h5 \\
      --out_dir /scratch/ood_results

  # Without simulator (static dataset only):
  python eval_ood.py name:checkpoint.ckpt \\
      --no_simulator \\
      --hdf5 /path/to/opendeck_2000_lewm.h5 \\
      --out_dir ./ood_results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

from jepa import JEPA  # noqa  -- required for torch.load to reconstruct the object
from eval_sweep import (
    encode_observations,
    load_dataset,
    ENC_BATCH,
    HDF5_PATH,
    TRAIN_SPLIT,
    SEED,
)

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ── Simulator contexts (same as Scenario2) ────────────────────────────────────
CONTEXTS_PARAMS = [
    {"PHASE_TYPE": "CR",  "DTAMB": 10.0, "ALT": 35000, "MACH": 0.78,  "COMMAND": 25000},
    {"PHASE_TYPE": "MTO", "DTAMB": 15.0, "ALT":     0, "MACH":  0.0,  "COMMAND": 120000},
    {"PHASE_TYPE": "MCL", "DTAMB": 10.0, "ALT": 15000, "MACH":  0.55, "COMMAND": 120000},
    {"PHASE_TYPE": "MCL", "DTAMB": 10.0, "ALT": 35000, "MACH":  0.78, "COMMAND": 120000},
]
N_CONTEXTS = len(CONTEXTS_PARAMS)    # 4
N_SENSORS  = 7

# ── OOD scenario definitions ──────────────────────────────────────────────────
# Each entry: (scenario_name, description, trajectory_kwargs_overrides)
OOD_SCENARIOS = [
    (
        "accel_3x",
        "Accelerated degradation (3x slope)",
        {
            "speed_params": {
                "slow":   {"mean_slope": -1.5e-2, "std_slope": 3e-3},
                "normal": {"mean_slope": -3.0e-2, "std_slope": 6e-3},
                "fast":   {"mean_slope": -1.2e-1, "std_slope": 1.5e-2},
            },
        },
    ),
    (
        "accel_5x",
        "Accelerated degradation (5x slope)",
        {
            "speed_params": {
                "slow":   {"mean_slope": -2.5e-2, "std_slope": 5e-3},
                "normal": {"mean_slope": -5.0e-2, "std_slope": 1e-2},
                "fast":   {"mean_slope": -2.0e-1, "std_slope": 2.5e-2},
            },
        },
    ),
    (
        "premaint",
        "Premature/ineffective maintenance",
        {
            "maintenance_interval": (50, 150),
            "maintenance_coeff":    0.1,
        },
    ),
    (
        "correlated",
        "Correlated HPC+turbine failure",
        {
            "degradation_origins": [
                "deg_CmpH_s_mapEff_in",
                "deg_CmpH_s_mapWc_in",
                "deg_TrbH_s_mapEff_in",
                "deg_TrbH_s_mapWc_in",
            ],
            "speed_params": {
                "slow":   {"mean_slope": -1e-2,  "std_slope": 2e-3},
                "normal": {"mean_slope": -3e-2,  "std_slope": 5e-3},
                "fast":   {"mean_slope": -1.2e-1, "std_slope": 1.5e-2},
            },
            "speed_probability_distribution": {
                # Only the 4 active components degrade (fast-biased)
                "deg_CmpH_s_mapEff_in": [0.1, 0.2, 0.7],
                "deg_CmpH_s_mapWc_in":  [0.1, 0.2, 0.7],
                "deg_TrbH_s_mapEff_in": [0.1, 0.2, 0.7],
                "deg_TrbH_s_mapWc_in":  [0.1, 0.2, 0.7],
                # Remaining components stay near-zero (very slow)
                "deg_CmpFan_s_mapEff_in": [0.95, 0.05, 0.0],
                "deg_CmpFan_s_mapWc_in":  [0.95, 0.05, 0.0],
                "deg_CmpBst_s_mapEff_in": [0.95, 0.05, 0.0],
                "deg_CmpBst_s_mapWc_in":  [0.95, 0.05, 0.0],
                "deg_TrbL_s_mapEff_in":   [0.95, 0.05, 0.0],
                "deg_TrbL_s_mapWc_in":    [0.95, 0.05, 0.0],
            },
        },
    ),
]

# ═════════════════════════════════════════════════════════════════════════════
#  TRAJECTORY GENERATION  (self-contained pure numpy, zero external imports)
#  Mirrors exactly the logic in scenarios/OpenDeckGeneration/trajectory_generation.py
#  but with all constants inlined so we never need to import from the simulator.
# ═════════════════════════════════════════════════════════════════════════════

# Component order matches STATE_LABELS in shared/constants.py
_STATE_LABELS = [
    "deg_CmpBst_s_mapEff_in", "deg_CmpBst_s_mapWc_in",
    "deg_CmpFan_s_mapEff_in", "deg_CmpFan_s_mapWc_in",
    "deg_CmpH_s_mapEff_in",   "deg_CmpH_s_mapWc_in",
    "deg_TrbH_s_mapEff_in",   "deg_TrbH_s_mapWc_in",
    "deg_TrbL_s_mapEff_in",   "deg_TrbL_s_mapWc_in",
]

_STATE_BOUNDS = {
    "deg_CmpBst_s_mapEff_in": (-0.05, 0.0),
    "deg_CmpBst_s_mapWc_in":  (-0.05, 0.03),
    "deg_CmpFan_s_mapEff_in": (-0.05, 0.0),
    "deg_CmpFan_s_mapWc_in":  (-0.05, 0.03),
    "deg_CmpH_s_mapEff_in":   (-0.05, 0.0),
    "deg_CmpH_s_mapWc_in":    (-0.05, 0.03),
    "deg_TrbH_s_mapEff_in":   (-0.05, 0.0),
    "deg_TrbH_s_mapWc_in":    (-0.05, 0.05),
    "deg_TrbL_s_mapEff_in":   (-0.05, 0.0),
    "deg_TrbL_s_mapWc_in":    (-0.05, 0.05),
}

_DEFAULT_SPEED_PARAMS = {
    "slow":   {"mean_slope": -5e-3,  "std_slope": 1e-3},
    "normal": {"mean_slope": -1e-2,  "std_slope": 2e-3},
    "fast":   {"mean_slope": -4e-2,  "std_slope": 5e-3},
}

_DEFAULT_SPEED_PROB = {
    "deg_CmpFan_s_mapEff_in": [0.4, 0.4, 0.2],
    "deg_CmpFan_s_mapWc_in":  [0.4, 0.4, 0.2],
    "deg_CmpBst_s_mapEff_in": [0.4, 0.4, 0.2],
    "deg_CmpBst_s_mapWc_in":  [0.4, 0.4, 0.2],
    "deg_CmpH_s_mapEff_in":   [0.2, 0.2, 0.6],
    "deg_CmpH_s_mapWc_in":    [0.2, 0.2, 0.6],
    "deg_TrbH_s_mapEff_in":   [0.4, 0.4, 0.2],
    "deg_TrbH_s_mapWc_in":    [0.4, 0.4, 0.2],
    "deg_TrbL_s_mapEff_in":   [0.4, 0.4, 0.2],
    "deg_TrbL_s_mapWc_in":    [0.4, 0.4, 0.2],
}

# speed_division mirrors config.yaml.
# The original code computes: division_factor = sequence_length / random_factor
# where factors=[1,2,3,4]. So for seq_len=500: divisors are 500,250,167,125.
# This makes per-step slopes very small (order 1e-5), matching the original data.
_SPEED_DIV_FACTORS = [1, 2, 3, 4]   # denominators for seq_len/factor
_SPEED_DIV_DIST    = [0.4, 0.2, 0.2, 0.2]
_SLOPE_NOISE_STD   = 1.5e-2


def _simulate_one_trajectory(
    speed_params: dict,
    speed_prob: dict,
    sequence_length: int,
    maintenance_interval: tuple,
    maintenance_coeff: float,
    change_speed_occurrence: int,
    seed: int | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure-numpy trajectory generation. Returns (traj, maint_occ).

    traj     : (T, 10) float32 -- clipped to STATE_BOUNDS, truncated at first
                                   bound-crossing (mirrors filter_trajectory)
    maint_occ: (T,) bool       -- True at maintenance timesteps
    """
    import copy as _copy

    rng = np.random.default_rng(seed)

    # Sample division factor: original code uses sequence_length / random_factor
    # so divisors are e.g. 500, 250, 167, 125 for seq_len=500.
    # This keeps per-step slopes in the 1e-5 range, matching training data.
    base_factor = int(rng.choice(_SPEED_DIV_FACTORS, p=_SPEED_DIV_DIST))
    div_factor  = max(1, sequence_length // base_factor)

    # Scale slopes by division factor (same as original)
    sp = _copy.deepcopy(speed_params)
    for cat in sp:
        sp[cat]["mean_slope"] /= div_factor
        sp[cat]["std_slope"]  /= div_factor
    noise_std = _SLOPE_NOISE_STD / div_factor  # also scaled down, same as original

    speed_keys = list(sp.keys())   # ["slow", "normal", "fast"]

    # Maintenance schedule
    lo, hi = maintenance_interval
    if hi <= lo:
        hi = lo + 1
    n_maints = max(1, sequence_length // lo)
    intervals = rng.integers(lo, hi, size=n_maints)
    maint_times = list(np.cumsum(intervals))
    maint_times = [t - 1 for t in maint_times if t < sequence_length]

    maint_occ = np.zeros(sequence_length, dtype=bool)
    for t in maint_times:
        maint_occ[t] = True

    trajectory = []
    for key in _STATE_LABELS:
        bounds = _STATE_BOUNDS[key]
        min_b, max_b = bounds
        prob = speed_prob.get(key, [1/3, 1/3, 1/3])
        current_val = 0.0
        state_maint = list(maint_times)  # per-component copy
        last_maint  = 0
        current_state = []

        speed = rng.choice(speed_keys, p=prob)

        for ts in range(sequence_length):
            # Maintenance recovery
            if state_maint and ((ts + 1) % state_maint[0]) == 0:
                maint_t   = state_maint.pop(0)
                dur       = maint_t - last_maint
                beg_      = (ts + 1) - dur
                last_maint = maint_t - 1
                if len(current_state) > 0 and beg_ >= 0:
                    current_val += maintenance_coeff * abs(
                        current_state[-1] - (current_state[beg_] if beg_ < len(current_state) else 0.0)
                    )

            # Speed change
            if ts % change_speed_occurrence == 0 and ts > 0:
                speed = rng.choice(speed_keys, p=prob)

            slope = float(rng.normal(sp[speed]["mean_slope"], sp[speed]["std_slope"]))
            noise = float(rng.normal(0.0, noise_std))
            current_val += slope + noise
            current_val = float(np.clip(current_val, min_b, max_b))
            current_state.append(current_val)

        trajectory.append(current_state)

    traj = np.array(trajectory, dtype=np.float32).T   # (T, 10)

    # Truncate at first bound crossing (mirrors filter_trajectory)
    cutoff = sequence_length
    for idx, key in enumerate(_STATE_LABELS):
        min_b = _STATE_BOUNDS[key][0]
        hits  = np.where(traj[:, idx] <= min_b)[0]
        if len(hits) > 0:
            cutoff = min(cutoff, int(hits[0]))

    traj      = traj[:cutoff]
    maint_occ = maint_occ[:cutoff]
    return traj, maint_occ


def generate_ood_state_trajectory(
    scenario_overrides: dict,
    n_episodes: int = 50,
    seq_len: int = 500,
    seed: int | None = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Generate OOD state trajectories using scenario-specific parameters.

    Fully self-contained: no imports from the simulator repo.

    Returns
    -------
    states_list : list of (T_i, 10) float32 arrays
    maint_list  : list of (T_i,) bool arrays
    """
    import copy as _copy

    speed_params = _copy.deepcopy(
        scenario_overrides.get("speed_params", _DEFAULT_SPEED_PARAMS)
    )
    speed_prob = _copy.deepcopy(
        scenario_overrides.get("speed_probability_distribution", _DEFAULT_SPEED_PROB)
    )
    maint_interval = scenario_overrides.get("maintenance_interval", (10000, 10001))
    maint_coeff    = scenario_overrides.get("maintenance_coeff", 0.4)
    active_origins = scenario_overrides.get("degradation_origins", None)

    rng_master = np.random.default_rng(seed if seed is not None else 42)

    states_list, maint_list = [], []
    for _ in range(n_episodes):
        ep_seed = int(rng_master.integers(0, 2**31))
        traj, maint_occ = _simulate_one_trajectory(
            speed_params=speed_params,
            speed_prob=speed_prob,
            sequence_length=seq_len,
            maintenance_interval=maint_interval,
            maintenance_coeff=maint_coeff,
            change_speed_occurrence=100,
            seed=ep_seed,
        )

        if len(traj) == 0:
            continue

        # Correlated scenario: zero out non-active components
        if active_origins is not None:
            active_idx = {_STATE_LABELS.index(o) for o in active_origins if o in _STATE_LABELS}
            for j in range(traj.shape[1]):
                if j not in active_idx:
                    traj[:, j] = 0.0

        states_list.append(traj)
        maint_list.append(maint_occ)

    logging.info(f"  Generated {len(states_list)} OOD episodes, "
                 f"mean length {np.mean([len(s) for s in states_list]):.0f}")
    return states_list, maint_list


# ═════════════════════════════════════════════════════════════════════════════
#  SENSOR OBSERVATION GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def states_to_obs_zmq(
    states_list: List[np.ndarray],
    sim_addr: str,
    ch_min: np.ndarray,    # (7,) -- from HDF5 pixels attrs
    ch_range: np.ndarray,  # (7,) -- from HDF5 pixels attrs
    batch_size: int = 512,
) -> List[np.ndarray]:
    """Convert raw degradation states to normalised (T, 28) sensor observations.

    Uses the ZMQ SimulatorClient (nodegpu branch) to call the physics engine.
    The 4 flight contexts are fixed (same as training data generation).

    Returns list of (T_i, 28) float32 arrays.
    """
    # Import SimulatorClient from the simulator repo (added to PYTHONPATH in slurm script)
    try:
        from hpc.simulator_client import SimulatorClient
    except ImportError as e:
        raise RuntimeError(
            "Could not import SimulatorClient. Ensure SIMULATOR_REPO is set and "
            "rl_simulator_safran is on PYTHONPATH."
        ) from e

    client = SimulatorClient(sim_addr)
    client.wait_until_ready(retries=10)
    logging.info(f"  ZMQ simulator connected at {sim_addr}")

    obs_list = []
    for states in states_list:
        T = len(states)
        all_raw = []  # will be (T * 4, 7)

        # Process in batches of states; for each state send all 4 contexts
        for start in range(0, T, batch_size):
            chunk = states[start : start + batch_size]
            B = len(chunk)
            # Tile states for 4 contexts: (B*4, 10)
            states_tiled  = np.repeat(chunk, N_CONTEXTS, axis=0)
            contexts_tiled = CONTEXTS_PARAMS * B

            raw = client.simulate(
                states=states_tiled.astype(np.float64),
                contexts=contexts_tiled,
            )  # (B*4, 7) float32

            # Reshape to (B, 4, 7) then transpose to (B, 7, 4)
            raw = raw.reshape(B, N_CONTEXTS, N_SENSORS)
            raw = raw.transpose(0, 2, 1)  # (B, 7, 4)
            all_raw.append(raw)

        raw_ep = np.concatenate(all_raw, axis=0)  # (T, 7, 4)

        # Apply same channel normalization as prepare_lewm_dataset.py
        normed = (raw_ep - ch_min[None, :, None]) / ch_range[None, :, None]
        normed = np.clip(normed, 0.0, 1.0)
        # Flatten (T, 7, 4) -> (T, 28)
        obs_list.append(normed.reshape(T, -1).astype(np.float32))

    client.close()
    return obs_list


# ═════════════════════════════════════════════════════════════════════════════
#  NO-SIMULATOR FALLBACK: extract extreme episodes from test set
# ═════════════════════════════════════════════════════════════════════════════

def extract_extreme_episodes(
    te_data: dict,
    ep_offset_all: np.ndarray,
    ep_len_all: np.ndarray,
    hdf5_path: str,
    encoder_type: str,
    top_frac: float = 0.15,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Return obs/hi/action lists for the fastest-degrading test episodes.

    These serve as a near-OOD proxy when the live simulator is unavailable.
    'Fastest degrading' = highest mean absolute HI change per step.
    """
    with h5py.File(hdf5_path, "r") as f:
        obs_key = "pixels" if "pixels" in f else "observation.sensors"
        all_obs    = f[obs_key][:]
        if encoder_type == "sensor" and all_obs.ndim > 2:
            all_obs = all_obs.reshape(len(all_obs), -1)
        all_hi     = f["observation.state"][:]
        all_act    = f["action"][:]
        if all_act.ndim > 1:
            all_act = all_act[:, 0]

    # Compute mean degradation speed per episode
    speeds = []
    for ep_idx in te_data["eps"]:
        s, l = int(ep_offset_all[ep_idx]), int(ep_len_all[ep_idx])
        hi_ep = all_hi[s : s + l]
        if l < 2:
            speeds.append(0.0)
            continue
        speeds.append(float(np.mean(np.abs(np.diff(hi_ep, axis=0)))))

    speeds = np.array(speeds)
    n_ood = max(1, int(len(speeds) * top_frac))
    ood_indices = np.argsort(speeds)[-n_ood:]  # fastest episodes

    obs_list, hi_list, act_list = [], [], []
    for local_i in ood_indices:
        ep_idx = te_data["eps"][local_i]
        s, l = int(ep_offset_all[ep_idx]), int(ep_len_all[ep_idx])
        obs_list.append(all_obs[s : s + l])
        hi_list.append(all_hi[s : s + l])
        act_list.append(all_act[s : s + l])

    logging.info(
        f"  Extracted {len(obs_list)} extreme episodes as OOD proxy "
        f"(top {top_frac*100:.0f}% by degradation speed)"
    )
    return obs_list, hi_list, act_list


# ═════════════════════════════════════════════════════════════════════════════
#  ANOMALY DETECTORS
# ═════════════════════════════════════════════════════════════════════════════

def fit_mahalanobis(Z_tr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit mean + regularised precision matrix on training embeddings."""
    mu    = Z_tr.mean(axis=0)
    cov   = np.cov(Z_tr, rowvar=False)
    # Tikhonov regularization for numerical stability
    cov  += 1e-4 * np.eye(cov.shape[0])
    prec  = np.linalg.inv(cov)
    return mu, prec


def mahal_score(Z: np.ndarray, mu: np.ndarray, prec: np.ndarray) -> np.ndarray:
    """Mahalanobis distance for each row in Z."""
    diff = Z - mu[None, :]                  # (N, D)
    return np.sqrt(np.einsum("nd,de,ne->n", diff, prec, diff))


def knn_score(
    Z: np.ndarray,
    Z_tr: np.ndarray,
    k: int = 20,
    chunk: int = 2048,
) -> np.ndarray:
    """Mean distance to k nearest neighbours in the training set."""
    scores = []
    Z_tr_t = torch.from_numpy(Z_tr).float()
    for start in range(0, len(Z), chunk):
        zb = torch.from_numpy(Z[start : start + chunk]).float()  # (B, D)
        # squared distances (B, N_tr)
        dists = torch.cdist(zb, Z_tr_t)   # (B, N_tr)
        topk  = dists.topk(k, largest=False, dim=1).values  # (B, k)
        scores.append(topk.mean(dim=1).numpy())
    return np.concatenate(scores)


@torch.no_grad()
def surprise_scores(
    model,
    obs_list: List[np.ndarray],
    actions_list: List[np.ndarray] | None,
    encoder_type: str,
    device: torch.device,
    history_size: int = 3,
    batch_encode: int = 512,
) -> np.ndarray:
    """Compute the JEPA predictor residual for each valid timestep.

    S(t) = ||z_{t+1} - f_theta(z_{t-H+1:t}, 0)||_2

    We use zero action vectors because maintenance actions are not known in
    advance for OOD trajectories (and the predictor is trained with zero
    actions for most steps).

    Returns flat (M,) array of surprise scores across all episodes.
    """
    H = history_size
    all_scores = []

    for ep_i, obs_ep in enumerate(obs_list):
        if len(obs_ep) < H + 2:
            continue
        # Encode episode
        z_ep = encode_observations(
            model, obs_ep, encoder_type, device, batch_size=batch_encode
        )  # (T, D)
        T, D = z_ep.shape

        # Slide a window of size H over the episode
        # For each t in [H-1, T-2], predict z_{t+1} from z_{t-H+1:t+1}
        all_starts = np.arange(H - 1, T - 1)
        if len(all_starts) == 0:
            continue

        ctx_np  = np.stack([z_ep[t - H + 1 : t + 1] for t in all_starts])  # (N, H, D)
        ctx_t   = torch.from_numpy(ctx_np).float().to(device)
        act_buf = torch.zeros(len(all_starts), H, D, device=device)

        # One batched predict call
        pred = model.predict(ctx_t, act_buf)  # (N, H, D) -- last step is prediction
        z_pred = pred[:, -1, :].cpu().numpy()   # (N, D)

        # Ground-truth next z
        gt_idx  = all_starts + 1                # z_{t+1}
        z_gt    = z_ep[gt_idx]                  # (N, D)

        scores = np.linalg.norm(z_pred - z_gt, axis=1)  # (N,)
        all_scores.append(scores)

    if not all_scores:
        return np.array([], dtype=np.float32)
    return np.concatenate(all_scores).astype(np.float32)


@torch.no_grad()
def embedding_scores(
    model,
    obs_list: List[np.ndarray],
    encoder_type: str,
    device: torch.device,
    batch_encode: int = 512,
) -> np.ndarray:
    """Return flat (M, D) embedding matrix across all OOD episodes."""
    parts = []
    for obs_ep in obs_list:
        if len(obs_ep) == 0:
            continue
        z = encode_observations(model, obs_ep, encoder_type, device, batch_size=batch_encode)
        parts.append(z)
    if not parts:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
#  METRICS
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_detector(
    scores_id: np.ndarray,
    scores_ood: np.ndarray,
    detector_name: str,
) -> dict:
    """Compute AUC-ROC and Average Precision for one detector.

    ID label = 0, OOD label = 1.
    """
    y_true = np.concatenate([
        np.zeros(len(scores_id), dtype=int),
        np.ones(len(scores_ood), dtype=int),
    ])
    y_score = np.concatenate([scores_id, scores_ood])

    if len(np.unique(y_true)) < 2 or len(y_true) < 10:
        logging.warning(f"  [{detector_name}] Skipped -- insufficient data.")
        return {"auc": None, "avg_precision": None, "mean_id": None, "mean_ood": None}

    auc = float(roc_auc_score(y_true, y_score))
    ap  = float(average_precision_score(y_true, y_score))
    logging.info(
        f"  [{detector_name}]  AUC={auc:.3f}  AP={ap:.3f}  "
        f"mu_id={scores_id.mean():.4f}  mu_ood={scores_ood.mean():.4f}"
    )
    return {
        "auc": auc,
        "avg_precision": ap,
        "mean_id":  float(scores_id.mean()),
        "std_id":   float(scores_id.std()),
        "mean_ood": float(scores_ood.mean()),
        "std_ood":  float(scores_ood.std()),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN TASK 4 RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def task4_ood(
    model,
    encoder_type: str,
    device: torch.device,
    tr_data: dict,
    te_data: dict,
    ep_offset_all: np.ndarray,
    ep_len_all: np.ndarray,
    hdf5_path: str,
    hi_names: List[str],
    sim_addr: Optional[str],
    out_dir: Path,
    n_ood_episodes: int = 50,
    seq_len: int = 500,
    knn_k: int = 20,
    history_size: int = 3,
) -> dict:
    """Run Task 4 OOD detection evaluation.

    Steps
    -----
    1.  Build ID reference from test-set embeddings.
    2.  Fit Mahalanobis and pre-index k-NN on training embeddings.
    3.  Compute ID baseline surprise, Mahal, and kNN scores.
    4.  For each OOD scenario:
           a. Generate state trajectories.
           b. Simulate sensor observations (ZMQ client) OR use dataset proxy.
           c. Compute surprise / Mahal / kNN scores.
           d. Evaluate detectors: AUC-ROC, AP.
    5.  Return consolidated results dict.
    """
    logging.info("Task 4 -- OOD Detection")

    # ── Step 1: ID embeddings ──────────────────────────────────────────────────
    logging.info("  Encoding ID (train) embeddings for Mahal / k-NN …")
    Z_tr = embedding_scores(model, _split_to_list(tr_data, ep_offset_all, ep_len_all, hdf5_path, encoder_type), encoder_type, device)
    logging.info(f"  ID train embeddings: {Z_tr.shape}")

    logging.info("  Encoding ID (test) embeddings …")
    Z_te = embedding_scores(model, _split_to_list(te_data, ep_offset_all, ep_len_all, hdf5_path, encoder_type), encoder_type, device)
    logging.info(f"  ID test embeddings: {Z_te.shape}")

    # ── Step 2: Fit detectors on training embeddings ───────────────────────────
    logging.info("  Fitting Mahalanobis model on training embeddings …")
    mu, prec = fit_mahalanobis(Z_tr)

    # Subsample training set for k-NN index to keep memory / compute manageable.
    # 50 K points is enough for a reliable density estimate with D=64.
    KNN_INDEX_SIZE = 50_000
    if len(Z_tr) > KNN_INDEX_SIZE:
        rng_idx = np.random.default_rng(42)
        idx = rng_idx.choice(len(Z_tr), KNN_INDEX_SIZE, replace=False)
        Z_tr_knn = Z_tr[idx]
        logging.info(f"  k-NN index subsampled: {KNN_INDEX_SIZE:,} / {len(Z_tr):,} training points")
    else:
        Z_tr_knn = Z_tr

    # ── Step 3: ID baseline scores (evaluated on test embeddings) ─────────────
    logging.info("  Computing ID baseline scores (test set) …")
    id_mahal   = mahal_score(Z_te, mu, prec)
    id_knn     = knn_score(Z_te, Z_tr_knn, k=knn_k)

    # ID surprise scores (test episodes)
    id_obs_list = _split_to_obs_list(te_data, ep_offset_all, ep_len_all, hdf5_path, encoder_type)
    id_act_list = _split_to_act_list(te_data, ep_offset_all, ep_len_all, hdf5_path)
    logging.info("  Computing ID surprise scores …")
    id_surprise = surprise_scores(
        model, id_obs_list, id_act_list, encoder_type, device, history_size=history_size
    )
    logging.info(
        f"  ID baseline  surprise={id_surprise.mean():.4f}+-{id_surprise.std():.4f}  "
        f"mahal={id_mahal.mean():.2f}  knn={id_knn.mean():.4f}"
    )

    # ── Step 4: OOD scenarios ──────────────────────────────────────────────────
    ch_min, ch_range = _load_norm_stats(hdf5_path)
    results = {"id_baseline": {
        "surprise": {"mean": float(id_surprise.mean()), "std": float(id_surprise.std())},
        "mahal":    {"mean": float(id_mahal.mean()),    "std": float(id_mahal.std())},
        "knn":      {"mean": float(id_knn.mean()),      "std": float(id_knn.std())},
    }}

    scenarios_to_run = OOD_SCENARIOS

    if sim_addr is None:
        # No simulator: use fast-episode proxy
        logging.info("  No simulator -- using extreme test-episode proxy as OOD.")
        ood_obs_list, _, ood_act_list = extract_extreme_episodes(
            te_data, ep_offset_all, ep_len_all, hdf5_path, encoder_type
        )
        proxy_result = _eval_one_scenario(
            name="extreme_proxy",
            ood_obs_list=ood_obs_list,
            model=model,
            encoder_type=encoder_type,
            device=device,
            history_size=history_size,
            mu=mu, prec=prec,
            Z_tr_knn=Z_tr_knn,
            knn_k=knn_k,
            id_surprise=id_surprise,
            id_mahal=id_mahal,
            id_knn=id_knn,
        )
        results["extreme_proxy"] = proxy_result
    else:
        for (sc_name, sc_desc, sc_overrides) in scenarios_to_run:
            logging.info(f"  --- Scenario: {sc_name} ({sc_desc}) ---")
            try:
                states_list, _ = generate_ood_state_trajectory(
                    scenario_overrides=sc_overrides,
                    n_episodes=n_ood_episodes,
                    seq_len=seq_len,
                )
                ood_obs_list = states_to_obs_zmq(
                    states_list, sim_addr, ch_min, ch_range
                )
            except Exception as exc:
                logging.error(f"  [!] Scenario {sc_name} FAILED: {exc}")
                results[sc_name] = {"error": str(exc)}
                continue

            sc_result = _eval_one_scenario(
                name=sc_name,
                ood_obs_list=ood_obs_list,
                model=model,
                encoder_type=encoder_type,
                device=device,
                history_size=history_size,
                mu=mu, prec=prec,
                Z_tr_knn=Z_tr_knn,
                knn_k=knn_k,
                id_surprise=id_surprise,
                id_mahal=id_mahal,
                id_knn=id_knn,
            )
            sc_result["description"] = sc_desc
            results[sc_name] = sc_result

    return results


def _eval_one_scenario(
    name: str,
    ood_obs_list: List[np.ndarray],
    model,
    encoder_type: str,
    device: torch.device,
    history_size: int,
    mu: np.ndarray,
    prec: np.ndarray,
    Z_tr_knn: np.ndarray,
    knn_k: int,
    id_surprise: np.ndarray,
    id_mahal: np.ndarray,
    id_knn: np.ndarray,
) -> dict:
    """Evaluate all three detectors for a single OOD scenario."""
    logging.info(f"  [{name}] Encoding {len(ood_obs_list)} OOD episodes …")

    Z_ood = embedding_scores(model, ood_obs_list, encoder_type, device)
    if len(Z_ood) == 0:
        return {"error": "No valid OOD embeddings"}

    ood_surprise = surprise_scores(
        model, ood_obs_list, None, encoder_type, device, history_size=history_size
    )
    ood_mahal = mahal_score(Z_ood, mu, prec)
    ood_knn   = knn_score(Z_ood, Z_tr_knn, k=knn_k)

    logging.info(
        f"  [{name}]  surprise={ood_surprise.mean():.4f}  "
        f"mahal={ood_mahal.mean():.2f}  knn={ood_knn.mean():.4f}"
    )

    return {
        "n_ood_timesteps": int(len(Z_ood)),
        "surprise":  evaluate_detector(id_surprise, ood_surprise, f"{name}/surprise"),
        "mahal":     evaluate_detector(id_mahal,    ood_mahal,    f"{name}/mahal"),
        "knn":       evaluate_detector(id_knn,      ood_knn,      f"{name}/knn"),
    }


# ── Dataset helpers ────────────────────────────────────────────────────────────

def _split_to_list(split_data, ep_offset_all, ep_len_all, hdf5_path, encoder_type):
    """Build list of per-episode observation arrays from a split dict."""
    with h5py.File(hdf5_path, "r") as f:
        obs_key = "pixels" if "pixels" in f else "observation.sensors"
        all_obs = f[obs_key][:]
        if encoder_type == "sensor" and all_obs.ndim > 2:
            all_obs = all_obs.reshape(len(all_obs), -1)
    obs_list = []
    for ep_idx in split_data["eps"]:
        s, l = int(ep_offset_all[ep_idx]), int(ep_len_all[ep_idx])
        obs_list.append(all_obs[s : s + l])
    return obs_list


def _split_to_obs_list(split_data, ep_offset_all, ep_len_all, hdf5_path, encoder_type):
    return _split_to_list(split_data, ep_offset_all, ep_len_all, hdf5_path, encoder_type)


def _split_to_act_list(split_data, ep_offset_all, ep_len_all, hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        all_act = f["action"][:]
        if all_act.ndim > 1:
            all_act = all_act[:, 0]
    act_list = []
    for ep_idx in split_data["eps"]:
        s, l = int(ep_offset_all[ep_idx]), int(ep_len_all[ep_idx])
        act_list.append(all_act[s : s + l])
    return act_list


def _load_norm_stats(hdf5_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load per-sensor ch_min and ch_range from the lewm HDF5 attrs."""
    with h5py.File(hdf5_path, "r") as f:
        if "pixels" in f and "ch_min" in f["pixels"].attrs:
            ch_min   = np.array(f["pixels"].attrs["ch_min"],   dtype=np.float32)
            ch_range = np.array(f["pixels"].attrs["ch_range"], dtype=np.float32)
            return ch_min, ch_range
    # Fallback: unit normalization (will work only if backend outputs normalised obs)
    logging.warning("ch_min/ch_range not found in HDF5 -- using identity normalization.")
    return np.zeros(N_SENSORS, np.float32), np.ones(N_SENSORS, np.float32)


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+",
                    help="Checkpoints to evaluate, in 'name:path' or 'path' format")
    ap.add_argument("--hdf5",        default=HDF5_PATH)
    ap.add_argument("--out_dir",     default="ood_results")
    ap.add_argument("--sim_addr",    default=None,
                    help="ZMQ address of the simulator worker, e.g. tcp://hostname:5555")
    ap.add_argument("--no_simulator", action="store_true",
                    help="Skip live simulation; use dataset-extreme proxy instead")
    ap.add_argument("--n_ood_eps",   type=int, default=50,
                    help="Number of OOD episodes to generate per scenario")
    ap.add_argument("--seq_len",     type=int, default=500,
                    help="Length of each generated OOD trajectory")
    ap.add_argument("--knn_k",       type=int, default=20)
    ap.add_argument("--history",     type=int, default=3)
    ap.add_argument("--encoder_type", default="sensor",
                    choices=["sensor", "vit"])
    ap.add_argument("--device",      default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logging.info(f"Device: {device}")

    sim_addr = None if args.no_simulator else args.sim_addr

    # Parse checkpoints
    ckpt_list = []
    for entry in args.checkpoints:
        if ":" in entry and not entry.startswith("/"):
            name, path = entry.split(":", 1)
        else:
            path = entry
            name = Path(entry).stem
        ckpt_list.append({"name": name, "path": path})

    # Load dataset once
    logging.info(f"Loading dataset: {args.hdf5}")
    tr_data, te_data, hi_names, ep_offset_all, ep_len_all = load_dataset(
        args.hdf5, args.encoder_type, TRAIN_SPLIT, SEED
    )

    all_results = {}
    for ckpt in ckpt_list:
        name, ckpt_path = ckpt["name"], ckpt["path"]
        logging.info(f"\n{'='*60}\nCheckpoint: {name}\n{'='*60}")

        model = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.eval().to(device)

        results = task4_ood(
            model=model,
            encoder_type=args.encoder_type,
            device=device,
            tr_data=tr_data,
            te_data=te_data,
            ep_offset_all=ep_offset_all,
            ep_len_all=ep_len_all,
            hdf5_path=args.hdf5,
            hi_names=hi_names,
            sim_addr=sim_addr,
            out_dir=out_dir,
            n_ood_episodes=args.n_ood_eps,
            seq_len=args.seq_len,
            knn_k=args.knn_k,
            history_size=args.history,
        )
        all_results[name] = results

        out_path = out_dir / f"{name}_ood.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"  Saved {out_path}")

    # Write combined summary
    summary_path = out_dir / "ood_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"\nOOD results saved to {out_dir}/")


if __name__ == "__main__":
    main()
