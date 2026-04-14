"""plot_results.py -- generate publication figures from eval_sweep CSV outputs.

Usage
-----
  python plot_results.py --csv /path/to/metrics_flat.csv \
                         --curves /path/to/task3_curves.csv \
                         --out_dir figures/
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.titlesize":     9,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "lines.linewidth":    1.5,
})

MODEL_LABELS = {
    "v6_sensor_sl1_w1":  r"$w{=}1$",
    "v6_sensor_sl1_w10": r"$w{=}10$",
}
COLORS = {
    "v6_sensor_sl1_w1":  "#2166ac",
    "v6_sensor_sl1_w10": "#d6604d",
}
COMPONENT_SHORT = {
    "deg_CmpBst_s_mapEff_in": "Bst/Eff",
    "deg_CmpBst_s_mapWc_in":  "Bst/Wc",
    "deg_CmpFan_s_mapEff_in": "Fan/Eff",
    "deg_CmpFan_s_mapWc_in":  "Fan/Wc",
    "deg_CmpH_s_mapEff_in":   "H/Eff",
    "deg_CmpH_s_mapWc_in":    "H/Wc",
    "deg_TrbH_s_mapEff_in":   "TrbH/Eff",
    "deg_TrbH_s_mapWc_in":    "TrbH/Wc",
    "deg_TrbL_s_mapEff_in":   "TrbL/Eff",
    "deg_TrbL_s_mapWc_in":    "TrbL/Wc",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 1 -- Task 1: per-component R² bar chart
# ══════════════════════════════════════════════════════════════════════════════

def fig_task1_r2(flat_df, out_dir):
    task1 = flat_df[
        (flat_df["task"] == "task1_hi") &
        (flat_df["metric"] == "r2") &
        (~flat_df["component"].str.startswith("__"))
    ].copy()
    task1["short"] = task1["component"].map(COMPONENT_SHORT)
    models = task1["model"].unique()

    components = list(COMPONENT_SHORT.values())
    x = np.arange(len(components))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, model in enumerate(sorted(models)):
        subset = task1[task1["model"] == model].set_index("short")["value"]
        vals = [float(subset.get(c, np.nan)) for c in components]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      label=MODEL_LABELS.get(model, model),
                      color=COLORS.get(model, f"C{i}"),
                      alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.axhline(1.0, color="grey", lw=0.6, ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(components, rotation=35, ha="right")
    ax.set_ylabel(r"$R^2$")
    ax.set_ylim(0.6, 1.02)
    ax.set_title("Task 1: HI State Estimation per Component")
    ax.legend(frameon=False)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    fig.tight_layout()
    path = out_dir / "fig_task1_r2.pdf"
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"))
    print(f"Saved {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 2 -- Task 3: RMSE(tau) curves (clean / event / gap)
# ══════════════════════════════════════════════════════════════════════════════

def fig_task3_curves(curves_df, out_dir):
    models = sorted(curves_df["model"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=False)

    for model in models:
        sub  = curves_df[curves_df["model"] == model].sort_values("tau")
        tau  = sub["tau"].values
        col  = COLORS.get(model, "C0")
        lbl  = MODEL_LABELS.get(model, model)

        axes[0].plot(tau, sub["rmse_clean"],  color=col, label=lbl)
        axes[1].plot(tau, sub["rmse_event"],  color=col, label=lbl)
        axes[2].plot(tau, sub["rmse_gap"],    color=col, label=lbl)

    for ax, title, ylab in zip(
        axes,
        ["Clean trajectories", "Event trajectories", "Action-divergence gap"],
        ["Mean HI RMSE", "Mean HI RMSE", r"$\delta(\tau)$"],
    ):
        ax.set_xlabel(r"Forecast horizon $\tau$ (steps)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(frameon=False)
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))

    # Shade the gap panel to highlight the peak region
    for model in models:
        sub = curves_df[curves_df["model"] == model].sort_values("tau")
        axes[2].fill_between(
            sub["tau"], 0, sub["rmse_gap"],
            color=COLORS.get(model, "C0"), alpha=0.12
        )
    axes[2].axhline(0, color="grey", lw=0.7, ls="--")

    fig.suptitle("Task 3: Latent Forecasting", y=1.01)
    fig.tight_layout()
    path = out_dir / "fig_task3_curves.pdf"
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"))
    print(f"Saved {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 3 -- Task 3: w=1 vs w=10 clean RMSE comparison
# ══════════════════════════════════════════════════════════════════════════════

def fig_task3_comparison(curves_df, out_dir):
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for model in sorted(curves_df["model"].unique()):
        sub = curves_df[curves_df["model"] == model].sort_values("tau")
        ax.plot(sub["tau"], sub["rmse_clean"],
                color=COLORS.get(model, "C0"),
                label=MODEL_LABELS.get(model, model),
                lw=2)
        ax.plot(sub["tau"], sub["rmse_event"],
                color=COLORS.get(model, "C0"),
                ls="--", lw=1.2, alpha=0.7)

    # Manual legend entries
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="grey", lw=1.5, label="clean"),
        Line2D([0], [0], color="grey", lw=1.2, ls="--", alpha=0.7, label="event"),
    ] + [
        Line2D([0], [0], color=COLORS.get(m, "C0"), lw=2,
               label=MODEL_LABELS.get(m, m))
        for m in sorted(curves_df["model"].unique())
    ]
    ax.legend(handles=legend_elements, frameon=False, ncol=2)
    ax.set_xlabel(r"Forecast horizon $\tau$ (steps)")
    ax.set_ylabel("Mean HI RMSE")
    ax.set_title("Task 3: Clean vs Event Trajectories")
    fig.tight_layout()
    path = out_dir / "fig_task3_comparison.pdf"
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"))
    print(f"Saved {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    required=True, help="metrics_flat.csv")
    parser.add_argument("--curves", required=True, help="task3_curves.csv")
    parser.add_argument("--out_dir", default="figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    flat_df   = pd.read_csv(args.csv)
    curves_df = pd.read_csv(args.curves)

    fig_task1_r2(flat_df, out_dir)
    fig_task3_curves(curves_df, out_dir)
    fig_task3_comparison(curves_df, out_dir)

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
