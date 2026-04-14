"""
plot_ood.py  — Visualise Task 4 OOD Detection results.

Usage
-----
    python plot_ood.py \
        --results_dir /path/to/eval/2052963 \
        --out_dir     ./figures/ood

Produces four figures:
  1. auc_heatmap.pdf     — AUC-ROC per detector x scenario (both models side-by-side)
  2. score_shift.pdf     — Mean OOD score vs ID baseline per detector (bar chart)
  3. trajectories.pdf    — Example degradation state trajectories: ID vs each OOD scenario
  4. detector_summary.pdf— Compact summary table for the paper
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

# ── Colours ───────────────────────────────────────────────────────────────────
SCENARIO_COLORS = {
    "accel_3x":  "#e07b39",
    "accel_5x":  "#c0392b",
    "premaint":  "#8e44ad",
    "correlated":"#2980b9",
}
DETECTOR_COLORS = {
    "surprise": "#e07b39",
    "mahal":    "#2980b9",
    "knn":      "#27ae60",
    "recon":    "#8e44ad",
}
SCENARIO_LABELS = {
    "accel_3x":  "Accel 3×",
    "accel_5x":  "Accel 5×",
    "premaint":  "Pre-maint",
    "correlated":"Correlated",
}
DETECTOR_LABELS = {
    "surprise": "Surprise",
    "mahal":    "Mahalanobis",
    "knn":      "k-NN",
    "recon":    "Recon. Error",
}
MODEL_LABELS = {
    "v6_sensor_sl1_w1":  "JEPA  w=1",
    "v6_sensor_sl1_w10": "JEPA  w=10",
}

SCENARIOS   = ["accel_3x", "accel_5x", "premaint", "correlated", "spike_fault"]
DETECTORS   = ["surprise", "mahal", "knn", "recon"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Trajectory generation (pure numpy — mirrors eval_ood.py)
# ═══════════════════════════════════════════════════════════════════════════════

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
_DEFAULT_SPEED_PROB = {k: [0.4, 0.4, 0.2] for k in _STATE_LABELS}
_DEFAULT_SPEED_PROB.update({
    "deg_CmpH_s_mapEff_in": [0.2, 0.2, 0.6],
    "deg_CmpH_s_mapWc_in":  [0.2, 0.2, 0.6],
})
_SPEED_DIV_FACTORS = [1, 2, 3, 4]
_SPEED_DIV_DIST    = [0.4, 0.2, 0.2, 0.2]
_SLOPE_NOISE_STD   = 1.5e-2


def simulate_trajectory(
    speed_params=None,
    speed_prob=None,
    sequence_length=500,
    maintenance_interval=(10000, 10001),
    maintenance_coeff=0.8,
    degradation_origins=None,
    speed_probability_distribution=None,
    seed=None,
):
    """Generate a single degradation trajectory.  Returns (T, 10) float32."""
    rng = np.random.default_rng(seed)
    if speed_params is None:
        speed_params = _DEFAULT_SPEED_PARAMS
    if speed_probability_distribution is not None:
        speed_prob = speed_probability_distribution
    if speed_prob is None:
        speed_prob = _DEFAULT_SPEED_PROB

    base_factor = int(rng.choice(_SPEED_DIV_FACTORS, p=_SPEED_DIV_DIST))
    div_factor  = max(1, sequence_length // base_factor)

    sp = copy.deepcopy(speed_params)
    for cat in sp:
        sp[cat]["mean_slope"] /= div_factor
        sp[cat]["std_slope"]  /= div_factor
    noise_std = _SLOPE_NOISE_STD / div_factor

    speed_keys = list(sp.keys())

    lo, hi = maintenance_interval
    if hi <= lo: hi = lo + 1
    n_maints   = max(1, sequence_length // lo)
    intervals  = rng.integers(lo, hi, size=n_maints)
    maint_times = [t - 1 for t in np.cumsum(intervals) if t < sequence_length]

    trajectory = []
    for key in _STATE_LABELS:
        min_b, max_b = _STATE_BOUNDS[key]
        prob = speed_prob.get(key, [1/3, 1/3, 1/3])

        # If this component is not in degradation_origins, keep near zero
        if degradation_origins is not None and key not in degradation_origins:
            trajectory.append([0.0] * sequence_length)
            continue

        current_val = 0.0
        mt = list(maint_times)
        last_maint = 0
        vals = []
        speed = rng.choice(speed_keys, p=prob)

        for ts in range(sequence_length):
            if mt and ((ts + 1) % mt[0]) == 0:
                maint_t = mt.pop(0)
                dur = maint_t - last_maint
                beg = (ts + 1) - dur
                last_maint = maint_t - 1
                if vals and beg >= 0:
                    current_val += maintenance_coeff * abs(
                        vals[-1] - (vals[beg] if beg < len(vals) else 0.0)
                    )
            if ts % 50 == 0 and ts > 0:
                speed = rng.choice(speed_keys, p=prob)
            slope = float(rng.normal(sp[speed]["mean_slope"], sp[speed]["std_slope"]))
            noise = float(rng.normal(0.0, noise_std))
            current_val += slope + noise
            current_val = float(np.clip(current_val, min_b, max_b))
            vals.append(current_val)
        trajectory.append(vals)

    traj = np.array(trajectory, dtype=np.float32).T   # (T, 10)

    # Truncate at first bound crossing
    cutoff = sequence_length
    for idx, key in enumerate(_STATE_LABELS):
        min_b = _STATE_BOUNDS[key][0]
        hits = np.where(traj[:, idx] <= min_b)[0]
        if len(hits):
            cutoff = min(cutoff, int(hits[0]) + 1)

    return traj[:cutoff]


# OOD scenario kwargs (mirrors OOD_SCENARIOS in eval_ood.py)
SCENARIO_KWARGS = {
    "id": {},
    "accel_3x": {
        "speed_params": {
            "slow":   {"mean_slope": -1.5e-2, "std_slope": 3e-3},
            "normal": {"mean_slope": -3.0e-2, "std_slope": 6e-3},
            "fast":   {"mean_slope": -1.2e-1, "std_slope": 1.5e-2},
        },
    },
    "accel_5x": {
        "speed_params": {
            "slow":   {"mean_slope": -2.5e-2, "std_slope": 5e-3},
            "normal": {"mean_slope": -5.0e-2, "std_slope": 1e-2},
            "fast":   {"mean_slope": -2.0e-1, "std_slope": 2.5e-2},
        },
    },
    "premaint": {
        "maintenance_interval": (50, 150),
        "maintenance_coeff":    0.1,
    },
    "correlated": {
        "degradation_origins": [
            "deg_CmpH_s_mapEff_in", "deg_CmpH_s_mapWc_in",
            "deg_TrbH_s_mapEff_in", "deg_TrbH_s_mapWc_in",
        ],
        "speed_params": {
            "slow":   {"mean_slope": -1e-2,  "std_slope": 2e-3},
            "normal": {"mean_slope": -3e-2,  "std_slope": 5e-3},
            "fast":   {"mean_slope": -1.2e-1, "std_slope": 1.5e-2},
        },
        "speed_probability_distribution": {
            "deg_CmpH_s_mapEff_in": [0.1, 0.2, 0.7],
            "deg_CmpH_s_mapWc_in":  [0.1, 0.2, 0.7],
            "deg_TrbH_s_mapEff_in": [0.1, 0.2, 0.7],
            "deg_TrbH_s_mapWc_in":  [0.1, 0.2, 0.7],
            "deg_CmpFan_s_mapEff_in": [0.95, 0.05, 0.0],
            "deg_CmpFan_s_mapWc_in":  [0.95, 0.05, 0.0],
            "deg_CmpBst_s_mapEff_in": [0.95, 0.05, 0.0],
            "deg_CmpBst_s_mapWc_in":  [0.95, 0.05, 0.0],
            "deg_TrbL_s_mapEff_in":   [0.95, 0.05, 0.0],
            "deg_TrbL_s_mapWc_in":    [0.95, 0.05, 0.0],
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 1 — AUC heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def plot_auc_heatmap(results: dict, out_dir: Path):
    models = list(results.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 3.2),
                             constrained_layout=True)
    if n_models == 1:
        axes = [axes]

    for ax, model_key in zip(axes, models):
        # Build matrix: rows=detectors, cols=scenarios
        matrix = np.zeros((len(DETECTORS), len(SCENARIOS)))
        for j, sc in enumerate(SCENARIOS):
            for i, det in enumerate(DETECTORS):
                matrix[i, j] = results[model_key][sc][det]["auc"]

        im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")

        ax.set_xticks(range(len(SCENARIOS)))
        ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS], rotation=30, ha="right")
        ax.set_yticks(range(len(DETECTORS)))
        ax.set_yticklabels([DETECTOR_LABELS[d] for d in DETECTORS])
        ax.set_title(MODEL_LABELS.get(model_key, model_key), fontweight="bold")

        # Annotate cells
        for i in range(len(DETECTORS)):
            for j in range(len(SCENARIOS)):
                val = matrix[i, j]
                color = "white" if val < 0.4 or val > 0.75 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

        # Chance line reference
        ax.axhline(-0.5, color="gray", lw=0.5)

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("AUC-ROC", fontsize=9)
    cbar.ax.axhline(0.5, color="black", lw=1.5, linestyle="--")
    cbar.ax.text(2.8, 0.5, "chance", va="center", fontsize=7, color="black")

    fig.suptitle("Task 4 — OOD Detection AUC-ROC", fontsize=11, fontweight="bold", y=1.01)

    path = out_dir / "auc_heatmap.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 2 — Mean score shift (OOD vs ID baseline)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_score_shift(results: dict, out_dir: Path):
    """Bar chart: mean OOD score normalised by ID baseline (ratio). >1 = detected."""
    models = list(results.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 3.6),
                             constrained_layout=True, sharey=False)
    if n_models == 1:
        axes = [axes]

    for ax, model_key in zip(axes, models):
        model_data = results[model_key]
        id_means = {d: model_data["id_baseline"][d]["mean"] for d in DETECTORS}

        x = np.arange(len(SCENARIOS))
        width = 0.25

        for k, det in enumerate(DETECTORS):
            ratios = []
            for sc in SCENARIOS:
                ood_mean = model_data[sc][det]["mean_ood"]
                id_mean  = id_means[det]
                # Normalised shift: (ood - id) / id_std
                id_std   = model_data["id_baseline"][det]["std"]
                ratios.append((ood_mean - id_mean) / (id_std + 1e-9))

            bars = ax.bar(x + k * width, ratios, width,
                          label=DETECTOR_LABELS[det],
                          color=DETECTOR_COLORS[det], alpha=0.85, edgecolor="white")

        ax.axhline(0, color="black", lw=0.8, linestyle="--", label="ID baseline (z=0)")
        ax.set_xticks(x + width)
        ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS], rotation=20, ha="right")
        ax.set_ylabel("Score shift  (OOD − ID) / σ_ID")
        ax.set_title(MODEL_LABELS.get(model_key, model_key), fontweight="bold")
        ax.legend(frameon=False)

    fig.suptitle("Task 4 — OOD Score Shift per Detector", fontsize=11,
                 fontweight="bold", y=1.01)

    path = out_dir / "score_shift.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 3 — Example trajectories: ID vs OOD scenarios
# ═══════════════════════════════════════════════════════════════════════════════

def plot_trajectories(out_dir: Path, n_traj: int = 5, seq_len: int = 500):
    """Generate example degradation trajectories for each scenario and plot."""
    # Pick a representative subset of components to show
    SHOW_COMPONENTS = [
        ("deg_CmpH_s_mapEff_in",  "HPC eff."),
        ("deg_CmpH_s_mapWc_in",   "HPC flow"),
        ("deg_TrbH_s_mapEff_in",  "HPT eff."),
        ("deg_TrbL_s_mapEff_in",  "LPT eff."),
    ]
    comp_indices = [_STATE_LABELS.index(k) for k, _ in SHOW_COMPONENTS]
    comp_names   = [name for _, name in SHOW_COMPONENTS]

    all_scenarios = ["id"] + SCENARIOS
    scenario_display = {
        "id":         ("ID (training distribution)", "#555555"),
        "accel_3x":   (SCENARIO_LABELS["accel_3x"], SCENARIO_COLORS["accel_3x"]),
        "accel_5x":   (SCENARIO_LABELS["accel_5x"], SCENARIO_COLORS["accel_5x"]),
        "premaint":   (SCENARIO_LABELS["premaint"],  SCENARIO_COLORS["premaint"]),
        "correlated": (SCENARIO_LABELS["correlated"],SCENARIO_COLORS["correlated"]),
    }

    n_comp = len(SHOW_COMPONENTS)
    n_sc   = len(all_scenarios)

    fig, axes = plt.subplots(n_comp, n_sc,
                             figsize=(2.8 * n_sc, 2.0 * n_comp),
                             constrained_layout=True, sharey="row")

    for col, sc_key in enumerate(all_scenarios):
        kwargs = SCENARIO_KWARGS[sc_key]
        label, color = scenario_display[sc_key]

        trajs = [simulate_trajectory(seed=i, sequence_length=seq_len, **kwargs)
                 for i in range(n_traj)]

        for row, (cidx, cname) in enumerate(zip(comp_indices, comp_names)):
            ax = axes[row, col]

            for traj in trajs:
                t = np.arange(len(traj))
                ax.plot(t, traj[:, cidx], color=color, alpha=0.55, lw=0.9)

            # Mark the bound
            min_b = _STATE_BOUNDS[_STATE_LABELS[cidx]][0]
            ax.axhline(min_b, color="red", lw=0.7, linestyle=":", alpha=0.6)

            # Formatting
            if row == 0:
                ax.set_title(label, fontsize=8, fontweight="bold", color=color)
            if col == 0:
                ax.set_ylabel(cname, fontsize=8)
            if row == n_comp - 1:
                ax.set_xlabel("Timestep", fontsize=8)
            ax.set_xlim(0, seq_len)

    fig.suptitle("Degradation Trajectories — ID vs OOD Scenarios",
                 fontsize=11, fontweight="bold")

    path = out_dir / "trajectories.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 4 — Trajectory length distribution
# ═══════════════════════════════════════════════════════════════════════════════

def plot_length_distribution(results: dict, out_dir: Path):
    """
    Show mean episode length per scenario from results metadata.
    Also simulate many trajectories and plot their length histograms.
    """
    # Simulate episode lengths for each scenario
    N_SIM = 200
    SEQ   = 500

    sim_lengths = {}
    for sc_key in ["id"] + SCENARIOS:
        kwargs = SCENARIO_KWARGS[sc_key]
        lengths = [len(simulate_trajectory(seed=i + 1000, sequence_length=SEQ, **kwargs))
                   for i in range(N_SIM)]
        sim_lengths[sc_key] = lengths

    all_scenarios = ["id"] + SCENARIOS
    colors = ["#555555"] + [SCENARIO_COLORS[s] for s in SCENARIOS]
    labels = ["ID"] + [SCENARIO_LABELS[s] for s in SCENARIOS]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)

    # Left: histogram of trajectory lengths
    ax = axes[0]
    bins = np.linspace(0, SEQ + 10, 30)
    for sc_key, color, label in zip(all_scenarios, colors, labels):
        ax.hist(sim_lengths[sc_key], bins=bins, alpha=0.55,
                color=color, label=label, density=True)
    ax.set_xlabel("Episode length (steps to first bound)")
    ax.set_ylabel("Density")
    ax.set_title("Episode Length Distribution")
    ax.legend(frameon=False, ncol=2)

    # Right: mean ± std from the eval results (n_ood_timesteps / n_eps)
    ax2 = axes[1]
    model_key = list(results.keys())[0]
    scenario_means = []
    scenario_stds  = []
    n_eps = 50  # as set in slurm

    for sc in SCENARIOS:
        n_ts = results[model_key][sc]["n_ood_timesteps"]
        mean_len = n_ts / n_eps
        scenario_means.append(mean_len)
        # Get std from simulation
        scenario_stds.append(np.std(sim_lengths[sc]))

    x = np.arange(len(SCENARIOS))
    bars = ax2.bar(x, scenario_means, yerr=scenario_stds,
                   color=[SCENARIO_COLORS[s] for s in SCENARIOS],
                   capsize=4, alpha=0.85, edgecolor="white")
    ax2.axhline(np.mean(sim_lengths["id"]), color="#555555",
                linestyle="--", lw=1.5, label=f"ID mean ({np.mean(sim_lengths['id']):.0f})")
    ax2.set_xticks(x)
    ax2.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS])
    ax2.set_ylabel("Mean episode length (steps)")
    ax2.set_title("Mean OOD Episode Length\n(from eval run, n=50)")
    ax2.legend(frameon=False)

    fig.suptitle("Task 4 — Episode Length Analysis", fontsize=11, fontweight="bold")

    path = out_dir / "episode_lengths.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 5 — Summary for paper (compact)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_paper_summary(results: dict, out_dir: Path):
    """
    Single compact figure for the paper:
    Left:  AUC-ROC grouped bars (both models overlaid)
    Right: Score shift normalised
    """
    models = list(results.keys())

    fig = plt.figure(figsize=(10, 3.8))
    gs  = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[2, 2, 1],
                            wspace=0.35)

    ax_mahal = fig.add_subplot(gs[0])
    ax_knn   = fig.add_subplot(gs[1], sharey=ax_mahal)
    ax_leg   = fig.add_subplot(gs[2])
    ax_leg.axis("off")

    model_hatches = ["", "///"]
    model_alphas  = [0.9, 0.65]

    for ax, det in [(ax_mahal, "mahal"), (ax_knn, "knn")]:
        x = np.arange(len(SCENARIOS))
        width = 0.35

        for k, (model_key, hatch, alpha) in enumerate(zip(models, model_hatches, model_alphas)):
            aucs = [results[model_key][sc][det]["auc"] for sc in SCENARIOS]
            ax.bar(x + k * width - width / 2, aucs, width,
                   color=[SCENARIO_COLORS[s] for s in SCENARIOS],
                   hatch=hatch, alpha=alpha, edgecolor="white")

        ax.axhline(0.5, color="black", lw=1, linestyle="--", alpha=0.6)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS], rotation=25, ha="right")
        ax.set_ylabel("AUC-ROC" if det == "mahal" else "")
        ax.set_title(f"{DETECTOR_LABELS[det]} detector", fontweight="bold")

        # y=0.5 annotation
        ax.text(len(SCENARIOS) - 0.3, 0.52, "chance", fontsize=7, color="gray", va="bottom")

    plt.setp(ax_knn.get_yticklabels(), visible=False)

    # Legend panel — use Rectangle patches as handles (Patch has no path)
    import matplotlib.patches as mp
    legend_handles = [
        *[mp.Rectangle((0, 0), 1, 1, color=SCENARIO_COLORS[sc],
                        label=SCENARIO_LABELS[sc]) for sc in SCENARIOS],
        *[mp.Rectangle((0, 0), 1, 1, facecolor="lightgray", hatch=h,
                        edgecolor="gray",
                        label=MODEL_LABELS.get(mk, mk))
          for mk, h in zip(models, model_hatches)],
    ]
    ax_leg.legend(handles=legend_handles, loc="center", frameon=False, fontsize=8)

    fig.suptitle("Task 4 — OOD Detection Performance",
                 fontsize=12, fontweight="bold")

    path = out_dir / "paper_summary.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 6 — Correlated failure pattern (highlight component asymmetry)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_correlated_pattern(out_dir: Path, n_traj: int = 8, seq_len: int = 500):
    """Show the 'correlated' OOD scenario — only HPC+HPT degrade, rest stay flat."""
    GROUPS = [
        ("Fan / Booster", ["deg_CmpFan_s_mapEff_in", "deg_CmpBst_s_mapEff_in"]),
        ("HPC (active)",  ["deg_CmpH_s_mapEff_in",   "deg_CmpH_s_mapWc_in"]),
        ("HPT (active)",  ["deg_TrbH_s_mapEff_in",   "deg_TrbH_s_mapWc_in"]),
        ("LPT",           ["deg_TrbL_s_mapEff_in"]),
    ]
    GROUP_COLORS = ["#95a5a6", "#e07b39", "#c0392b", "#95a5a6"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)

    for col, (scenario_key, title_suffix) in enumerate([("id", "ID"), ("correlated", "Correlated OOD")]):
        ax = axes[col]
        kwargs = SCENARIO_KWARGS[scenario_key]
        trajs  = [simulate_trajectory(seed=i, sequence_length=seq_len, **kwargs)
                  for i in range(n_traj)]

        for group_label, comp_keys, color in zip(
            [g[0] for g in GROUPS],
            [g[1] for g in GROUPS],
            GROUP_COLORS
        ):
            indices = [_STATE_LABELS.index(k) for k in comp_keys if k in _STATE_LABELS]
            for traj in trajs:
                for idx in indices:
                    t = np.arange(len(traj))
                    ax.plot(t, traj[:, idx], color=color, alpha=0.45, lw=0.9)

            # Add legend entry once
            ax.plot([], [], color=color, lw=2, label=group_label)

        ax.axhline(-0.05, color="red", lw=0.7, ls=":", alpha=0.5, label="Bound (−0.05)")
        ax.set_xlim(0, seq_len)
        ax.set_ylim(-0.065, 0.04)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Degradation delta")
        ax.set_title(title_suffix, fontweight="bold")
        ax.legend(frameon=False, fontsize=7, loc="lower left")

    fig.suptitle("Correlated Failure: Only HPC + HPT Degrade",
                 fontsize=11, fontweight="bold")

    path = out_dir / "correlated_pattern.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 7 — Episode-level AUC vs per-timestep AUC comparison
# ═══════════════════════════════════════════════════════════════════════════════

def plot_episode_vs_timestep_auc(results: dict, out_dir: Path):
    """Side-by-side: timestep AUC vs episode-mean AUC for each detector."""
    models = list(results.keys())
    n_models = len(models)

    fig, axes = plt.subplots(len(DETECTORS), n_models,
                             figsize=(4.5 * n_models, 2.5 * len(DETECTORS)),
                             constrained_layout=True, sharey="row")
    if n_models == 1:
        axes = axes[:, np.newaxis]

    for col, model_key in enumerate(models):
        model_data = results[model_key]
        for row, det in enumerate(DETECTORS):
            ax = axes[row, col]
            sc_avail = [s for s in SCENARIOS if s in model_data and det in model_data[s]]
            if not sc_avail:
                ax.axis("off"); continue

            ts_aucs  = [model_data[sc][det]["auc"]                              for sc in sc_avail]
            ep_aucs  = [model_data[sc].get("episode", {}).get(det, {}).get("auc") for sc in sc_avail]
            ep_aucs  = [v if v is not None else 0.0 for v in ep_aucs]

            x = np.arange(len(sc_avail))
            w = 0.35
            ax.bar(x - w/2, ts_aucs, w, color="#aac4e0", label="Per-timestep", edgecolor="white")
            ax.bar(x + w/2, ep_aucs, w, color="#2980b9", label="Per-episode",  edgecolor="white")
            ax.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in sc_avail],
                               rotation=25, ha="right", fontsize=7)
            ax.set_ylim(0, 1.05)
            if col == 0:
                ax.set_ylabel(f"{DETECTOR_LABELS.get(det, det)}\nAUC-ROC", fontsize=8)
            if row == 0:
                ax.set_title(MODEL_LABELS.get(model_key, model_key), fontweight="bold")
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=7)

    fig.suptitle("Episode-level vs Per-timestep AUC", fontsize=11, fontweight="bold")
    path = out_dir / "episode_auc.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 8 — Reconstruction error profile over episode lifetime
# ═══════════════════════════════════════════════════════════════════════════════

def plot_recon_profile(results: dict, out_dir: Path):
    """Plot mean reconstruction error at each fractional position in the episode.

    Shows whether error grows as the episode progresses — the key hypothesis
    for the reconstruction-based OOD detector.
    """
    models = list(results.keys())
    sc_avail = [s for s in SCENARIOS
                if "recon_profile_ood" in list(results.values())[0].get(s, {})]
    if not sc_avail:
        print("  (skip recon_profile — no recon_profile_ood in results)")
        return

    n_sc = len(sc_avail)
    n_models = len(models)
    fig, axes = plt.subplots(n_models, n_sc,
                             figsize=(3.0 * n_sc, 2.8 * n_models),
                             constrained_layout=True, sharey=False)
    if n_models == 1:
        axes = axes[np.newaxis, :]
    if n_sc == 1:
        axes = axes[:, np.newaxis]

    x_pct = np.linspace(0, 100, 10)

    for row, model_key in enumerate(models):
        model_data = results[model_key]
        for col, sc in enumerate(sc_avail):
            ax = axes[row, col]
            sc_data = model_data.get(sc, {})
            profile_ood = sc_data.get("recon_profile_ood")
            profile_id  = sc_data.get("recon_profile_id")

            if profile_ood:
                ax.plot(x_pct, profile_ood, color=SCENARIO_COLORS.get(sc, "tab:blue"),
                        lw=2, label="OOD")
            if profile_id:
                ax.plot(x_pct, profile_id, color="#555555", lw=1.5, ls="--", label="ID")

            ax.set_xlabel("Episode progress (%)", fontsize=8)
            if col == 0:
                ax.set_ylabel("Mean recon. error", fontsize=8)
            if row == 0:
                ax.set_title(SCENARIO_LABELS.get(sc, sc), fontsize=8,
                             fontweight="bold", color=SCENARIO_COLORS.get(sc, "black"))
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=7)

    # Row labels (model names)
    for row, model_key in enumerate(models):
        axes[row, 0].annotate(
            MODEL_LABELS.get(model_key, model_key),
            xy=(-0.45, 0.5), xycoords="axes fraction",
            fontsize=8, fontweight="bold", rotation=90, va="center",
        )

    fig.suptitle("Reconstruction Error over Episode Lifetime\n"
                 "(rising error = OOD regime reached)",
                 fontsize=11, fontweight="bold")
    path = out_dir / "recon_profile.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=None,
                        help="Path to eval output directory (containing *_ood.json files)")
    parser.add_argument("--summary_json", default=None,
                        help="Path to ood_summary.json directly")
    parser.add_argument("--out_dir", default="./figures/ood",
                        help="Output directory for figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    if args.summary_json:
        summary_path = Path(args.summary_json)
    elif args.results_dir:
        summary_path = Path(args.results_dir) / "ood_summary.json"
    else:
        # Try default
        summary_path = Path("/Users/lucas-andreithil/.stable_worldmodel/eval/2052963/ood_summary.json")

    with open(summary_path) as f:
        results = json.load(f)

    # Filter SCENARIOS to only those present in results
    first_model = list(results.values())[0]
    available_scenarios = [s for s in SCENARIOS if s in first_model]
    global SCENARIOS
    SCENARIOS = available_scenarios

    print(f"\nLoaded results for models: {list(results.keys())}")
    print(f"Scenarios: {SCENARIOS}")
    print(f"Output directory: {out_dir}\n")

    # ── Quick textual analysis ─────────────────────────────────────────────
    print("─" * 60)
    print("  ANALYSIS SUMMARY")
    print("─" * 60)
    for model_key, model_data in results.items():
        print(f"\n  Model: {MODEL_LABELS.get(model_key, model_key)}")
        id_surprise = model_data["id_baseline"]["surprise"]["mean"]
        id_mahal    = model_data["id_baseline"]["mahal"]["mean"]
        id_knn      = model_data["id_baseline"]["knn"]["mean"]

        for sc in SCENARIOS:
            sc_data = model_data[sc]
            n_ts = sc_data["n_ood_timesteps"]
            print(f"\n    Scenario: {SCENARIO_LABELS[sc]}  "
                  f"(n_timesteps={n_ts}, mean_len~{n_ts//50:.0f})")
            for det in DETECTORS:
                d = sc_data[det]
                shift = d["mean_ood"] - d[f"mean_id"]
                sign  = "↑" if shift > 0 else "↓"
                print(f"      {DETECTOR_LABELS[det]:14s}  "
                      f"AUC={d['auc']:.3f}  "
                      f"mean_ood={d['mean_ood']:.3f} vs id={d['mean_id']:.3f}  "
                      f"({sign}{abs(shift):.3f})")

    print("\n─" * 60)
    print("  INTERPRETATION")
    print("─" * 60)
    print("""
  Surprise (prediction error in latent space):
    AUC ≈ 0.1–0.3  → BELOW chance. OOD episodes actually have LOWER
    surprise scores than ID episodes. Why? OOD trajectories are shorter
    (faster degradation hits bounds earlier) and smoother — the world
    model predicts them well because they degrade monotonically.
    The world model doesn't 'know' that fast degradation is unusual.

  Mahalanobis distance:
    AUC ≈ 0.69–0.77 → BEST detector. OOD embeddings sit slightly
    further from the training distribution mean (mean_ood > mean_id).
    The shift is small (+0.3–0.4) but consistent across all scenarios.
    This suggests the encoder does learn some distributional structure
    that separates OOD from ID, but the margin is narrow.

  k-NN distance:
    AUC ≈ 0.54–0.74 → MODERATE. Best for 'correlated' scenario (w10
    gets 0.74 AUC), where only 4/10 components degrade — this creates
    a genuinely novel region in embedding space with no nearby neighbors.
    Weaker for acceleration scenarios where states are within the
    training manifold, just visited more quickly.

  Overall: The JEPA world model representations weakly detect OOD via
  distribution-based detectors (Mahal, k-NN) but the surprise score
  is not a reliable detector. This makes sense — the model is trained
  to predict the next latent state, not to flag distributional shift.
  """)

    # ── Generate all figures ───────────────────────────────────────────────
    print("\nGenerating figures …")
    plot_auc_heatmap(results, out_dir)
    plot_score_shift(results, out_dir)
    plot_trajectories(out_dir)
    plot_length_distribution(results, out_dir)
    plot_paper_summary(results, out_dir)
    plot_correlated_pattern(out_dir)

    # New figures (only if recon data present)
    first_model_data = list(results.values())[0]
    has_recon    = any("recon" in first_model_data.get(sc, {}) for sc in SCENARIOS)
    has_episode  = any("episode" in first_model_data.get(sc, {}) for sc in SCENARIOS)
    has_profile  = any("recon_profile_ood" in first_model_data.get(sc, {}) for sc in SCENARIOS)

    if has_recon:
        plot_auc_heatmap(results, out_dir)   # will now include recon row
    if has_episode:
        plot_episode_vs_timestep_auc(results, out_dir)
    if has_profile:
        plot_recon_profile(results, out_dir)

    print(f"\nAll figures saved to {out_dir}/")
    print("Files: auc_heatmap, score_shift, trajectories, episode_lengths,")
    print("       paper_summary, correlated_pattern,")
    print("       episode_auc, recon_profile  (.pdf + .png each)")


if __name__ == "__main__":
    main()
