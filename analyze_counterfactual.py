"""analyze_counterfactual.py — read counterfactual_fidelity.py outputs and
produce fig5 (the online-simulator paper figure) + a CSV summary table.

Reads:  eval_results/counterfactual/results_<name>.json
Writes: figures/fig5_counterfactual_fidelity.{pdf,png}
        eval_results/counterfactual/SUMMARY.md  (table)
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
CF_DIR = ROOT / "eval_results/counterfactual"
FIG_DIR = ROOT / "figures"

ACTION_NAMES = {
    0: "do_nothing", 1: "fan_overhaul", 2: "hpc_overhaul",
    3: "turbine_overhaul", 4: "full_overhaul", 5: "patch", 6: "wash",
}


def load_results() -> dict:
    """Return {ckpt_name: list of result dicts}."""
    out = {}
    for f in sorted(CF_DIR.glob("results_*.json")):
        name = f.stem.replace("results_", "")
        blob = json.loads(f.read_text())
        out[name] = blob.get("results", [])
        print(f"  {name}: {len(out[name])} rollouts")
    return out


def aggregate_rmse(results_by_ckpt: dict) -> dict:
    """Per-ckpt → per-action → mean & std of RMSE-per-step across episodes×branches."""
    out = {}
    for ckpt, results in results_by_ckpt.items():
        per_action = defaultdict(list)
        for r in results:
            per_action[r["action"]].append(r["rmse_per_step"])
        # Each action has list of (horizon,) curves; stack and average
        agg = {}
        for action, curves in per_action.items():
            stack = np.array(curves)  # (N, horizon)
            agg[action] = {
                "mean": stack.mean(axis=0).tolist(),
                "std": stack.std(axis=0).tolist(),
                "n": len(curves),
            }
        out[ckpt] = agg
    return out


def aggregate_differentials(results_by_ckpt: dict) -> dict:
    """Per-ckpt: |Δmodel(a, 0) - Δsim(a, 0)| averaged over episodes×branches.

    For each (ep, branch_t), pair action_a with action_0 and compute
      |(model_a - model_0) - (sim_a - sim_0)|  Frobenius per step
    Average these across (ep, branch_t).
    """
    out = {}
    for ckpt, results in results_by_ckpt.items():
        # Group by (ep, branch_t)
        groups = defaultdict(dict)
        for r in results:
            key = (r["ep_idx"], r["branch_t"])
            groups[key][r["action"]] = r
        per_action = defaultdict(list)
        for key, action_results in groups.items():
            if 0 not in action_results:
                continue
            base_sim = np.array(action_results[0]["sim_HI"])    # (horizon, 10)
            base_mod = np.array(action_results[0]["model_HI"])
            for a, r in action_results.items():
                if a == 0:
                    continue
                d_sim = np.array(r["sim_HI"]) - base_sim       # (horizon, 10)
                d_mod = np.array(r["model_HI"]) - base_mod
                diff = np.sqrt(((d_mod - d_sim) ** 2).mean(axis=1))  # (horizon,)
                per_action[a].append(diff)
        agg = {}
        for action, curves in per_action.items():
            stack = np.array(curves)
            agg[action] = {
                "mean": stack.mean(axis=0).tolist(),
                "std": stack.std(axis=0).tolist(),
                "n": len(curves),
            }
        out[ckpt] = agg
    return out


def plot_fig5(rmse_agg: dict, diff_agg: dict, out_path: Path):
    """2-row figure: top = absolute RMSE(τ) per action per ckpt;
    bottom = differential |Δmodel - Δsim|(τ) per action per ckpt."""
    ckpts = sorted(rmse_agg.keys())
    # Filter ckpts with any data
    ckpts = [c for c in ckpts if rmse_agg[c]]
    if not ckpts:
        print("  no ckpts with data, skipping fig5")
        return

    n_ckpts = len(ckpts)
    fig, axes = plt.subplots(2, n_ckpts, figsize=(3.6 * n_ckpts, 5.5),
                             sharex=True, squeeze=False)

    cmap = plt.cm.viridis
    actions = sorted({a for c in ckpts for a in rmse_agg[c].keys()})
    color_for = {a: cmap(i / max(1, len(actions) - 1)) for i, a in enumerate(actions)}

    for col, ckpt in enumerate(ckpts):
        ax_abs = axes[0, col]; ax_diff = axes[1, col]
        # Top: absolute RMSE per action
        for a, blob in rmse_agg[ckpt].items():
            mean = np.array(blob["mean"])
            std = np.array(blob["std"])
            tau = np.arange(len(mean))
            ax_abs.plot(tau, mean, color=color_for[a],
                        label=f"a={a} ({ACTION_NAMES.get(a, '?')})", lw=1.0)
            ax_abs.fill_between(tau, mean - std, mean + std,
                                color=color_for[a], alpha=0.10)
        ax_abs.set_title(f"{ckpt}\nAbsolute HI RMSE(τ)", fontsize=10)
        ax_abs.set_ylabel("RMSE" if col == 0 else "")
        ax_abs.grid(alpha=0.3)

        # Bottom: differential
        for a, blob in diff_agg.get(ckpt, {}).items():
            mean = np.array(blob["mean"])
            std = np.array(blob["std"])
            tau = np.arange(len(mean))
            ax_diff.plot(tau, mean, color=color_for[a], lw=1.0)
            ax_diff.fill_between(tau, mean - std, mean + std,
                                 color=color_for[a], alpha=0.10)
        ax_diff.set_title(f"|Δ_model(a, 0) − Δ_sim(a, 0)|", fontsize=10)
        ax_diff.set_xlabel("rollout step τ")
        ax_diff.set_ylabel("differential" if col == 0 else "")
        ax_diff.grid(alpha=0.3)

    # Shared legend at the right
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8.5, frameon=False,
               bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Counterfactual fidelity: world model vs simulator under "
                 "alternative actions\n"
                 "Top: absolute RMSE — both conditioned on same action.  "
                 "Bottom: differential — does the model RESPOND to action change?",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_path}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path.name}.{{pdf,png}}")


def write_summary(rmse_agg: dict, diff_agg: dict, out_path: Path):
    """Markdown table: per-ckpt × per-action mean RMSE at τ=10, 50, 99 + mean diff."""
    lines = ["# Counterfactual fidelity summary", ""]
    lines.append("Mean across episodes × branch positions. Both metrics are mean across HI dimensions per step.")
    lines.append("")
    for ckpt in sorted(rmse_agg.keys()):
        if not rmse_agg[ckpt]:
            lines.append(f"## {ckpt}: no successful rollouts (skipped)\n")
            continue
        lines.append(f"## {ckpt}")
        lines.append("")
        lines.append("| action | RMSE@τ=10 | RMSE@τ=50 | RMSE@τ=99 | Δ@τ=10 | Δ@τ=50 | Δ@τ=99 |")
        lines.append("|--------|-----------|-----------|-----------|--------|--------|--------|")
        for a in sorted(rmse_agg[ckpt]):
            r = np.array(rmse_agg[ckpt][a]["mean"])
            d_blob = diff_agg.get(ckpt, {}).get(a, {})
            d = np.array(d_blob.get("mean", [np.nan])) if d_blob else np.array([np.nan])
            tau10 = r[10] if len(r) > 10 else np.nan
            tau50 = r[50] if len(r) > 50 else np.nan
            tau99 = r[-1] if len(r) > 0 else np.nan
            d10 = d[10] if len(d) > 10 else np.nan
            d50 = d[50] if len(d) > 50 else np.nan
            d99 = d[-1] if len(d) > 0 else np.nan
            lines.append(
                f"| {ACTION_NAMES.get(a, str(a))} | {tau10:.4f} | {tau50:.4f} | {tau99:.4f} "
                f"| {d10:.4f} | {d50:.4f} | {d99:.4f} |"
            )
        lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"  → {out_path}")


def main():
    print("Loading results…")
    results = load_results()
    print()
    if not results:
        print("No results in eval_results/counterfactual/. Run counterfactual_fidelity.py first.")
        return
    print("Aggregating RMSE…")
    rmse_agg = aggregate_rmse(results)
    print("Aggregating differentials…")
    diff_agg = aggregate_differentials(results)
    print("Plotting…")
    plot_fig5(rmse_agg, diff_agg, FIG_DIR / "fig5_counterfactual_fidelity")
    print("Writing markdown summary…")
    write_summary(rmse_agg, diff_agg, CF_DIR / "SUMMARY.md")
    print("Done.")


if __name__ == "__main__":
    main()
