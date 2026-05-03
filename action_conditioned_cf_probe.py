"""
action_conditioned_cf_probe.py
==============================

Tightens `counterfactual_fidelity.py`'s sanity-check differentials into a
quantitative claim: how well does the world model's action response match
the deterministic simulator's, per action × per HI dim?

Reads `eval_results/counterfactual*/results_*.json` written by
`counterfactual_fidelity.py` (or the RSSM variant). Each rollout has
`sim_HI`(T, 10) and `model_HI`(T, 10) for a given (ckpt, ep, branch_t,
action). For every (ep, branch_t) we compute

    Δ_sim(a)   = sim_HI(a)   - sim_HI(0)        # per-step, per-HI
    Δ_model(a) = model_HI(a) - model_HI(0)

and report, per action a ∈ {1..6}:

  - Pearson(Δ_sim, Δ_model) flattened over (ep, branch, τ, HI)
  - R² of regressing Δ_sim onto Δ_model (1.0 = perfect, 0.0 = trivial,
    negative = worse than predicting the mean)
  - Sign-agreement fraction
  - Magnitude ratio mean(‖Δ_model‖)/mean(‖Δ_sim‖)
  - Same metrics at horizons τ ∈ {10, 50, 99}

Output goes to `eval_results/action_cf_probe/` as both JSON (per ckpt)
and a single Markdown table cross-architecture.

Pure numpy + scipy.stats — runs on CPU in seconds.

Usage
-----
    python action_conditioned_cf_probe.py
    # custom dirs:
    python action_conditioned_cf_probe.py \\
        --in_dirs eval_results/counterfactual eval_results/counterfactual_rssm \\
        --out_dir eval_results/action_cf_probe
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ACTION_NAMES = {
    0: "do_nothing", 1: "fan_overhaul", 2: "hpc_overhaul",
    3: "turbine_overhaul", 4: "full_overhaul", 5: "patch", 6: "wash",
}


def load_rollouts(in_dirs: list[Path]) -> dict:
    """Returns {ckpt_name: list[rollout dict]}. Skips empty results files."""
    out = {}
    for d in in_dirs:
        for f in sorted(Path(d).glob("results_*.json")):
            name = f.stem.replace("results_", "")
            blob = json.loads(f.read_text())
            results = blob.get("results", [])
            if not results:
                continue
            if name in out:
                log.warning(f"  duplicate ckpt name {name} (overwriting from {f})")
            out[name] = results
            log.info(f"  loaded {name}: {len(results)} rollouts ({f.parent.name})")
    return out


def deltas_by_action(results: list) -> dict:
    """Group rollouts by (ep_idx, branch_t), yield per-action deltas vs action 0.

    Returns {action_a: {"sim": list of (T, 10) arrays,
                        "model": list of (T, 10) arrays}}
    where each list element is one (ep, branch) pair where both action 0 and
    action a were rolled out.
    """
    grouped = defaultdict(dict)   # (ep, branch) → action → result
    for r in results:
        grouped[(r["ep_idx"], r["branch_t"])][r["action"]] = r

    out = {a: {"sim": [], "model": []} for a in ACTION_NAMES if a != 0}
    for (_, _), action_results in grouped.items():
        if 0 not in action_results:
            continue
        sim0   = np.asarray(action_results[0]["sim_HI"])    # (T, 10)
        model0 = np.asarray(action_results[0]["model_HI"])  # (T, 10)
        for a in ACTION_NAMES:
            if a == 0 or a not in action_results:
                continue
            T_a = min(len(action_results[a]["sim_HI"]), len(sim0))
            sim_a   = np.asarray(action_results[a]["sim_HI"])[:T_a]
            model_a = np.asarray(action_results[a]["model_HI"])[:T_a]
            out[a]["sim"].append(sim_a - sim0[:T_a])
            out[a]["model"].append(model_a - model0[:T_a])
    return out


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(pearsonr(x, y)[0])


def metrics_for_action(deltas: dict, tau: int | None = None) -> dict:
    """Compute Pearson / R² / sign / magnitude for one action's deltas.

    If tau is None, flatten over all τ in [0, T). Else use only τ = tau.
    """
    if not deltas["sim"]:
        return {"n_pairs": 0}
    sim_all   = []
    model_all = []
    for s, m in zip(deltas["sim"], deltas["model"]):
        if tau is not None:
            if s.shape[0] <= tau:
                continue
            sim_all.append(s[tau])     # (10,)
            model_all.append(m[tau])
        else:
            sim_all.append(s.reshape(-1))
            model_all.append(m.reshape(-1))
    if not sim_all:
        return {"n_pairs": 0}
    sim   = np.concatenate(sim_all)
    model = np.concatenate(model_all)

    # Pearson
    pearson = safe_pearson(sim, model)

    # R² of regressing sim on model (intercept-free, since deltas are signed):
    sse = np.sum((sim - model) ** 2)
    sst = np.sum((sim - sim.mean()) ** 2)
    r2  = float(1 - sse / sst) if sst > 1e-12 else float("nan")

    # Sign agreement (skip ties at zero)
    nz = (np.abs(sim) > 1e-8) & (np.abs(model) > 1e-8)
    if nz.sum() > 0:
        sign_agree = float((np.sign(sim[nz]) == np.sign(model[nz])).mean())
    else:
        sign_agree = float("nan")

    # Magnitude ratio (uses all entries)
    sim_mag   = float(np.sqrt(np.mean(sim ** 2)))
    model_mag = float(np.sqrt(np.mean(model ** 2)))
    mag_ratio = float(model_mag / sim_mag) if sim_mag > 1e-12 else float("nan")

    return {
        "n_pairs":     len(sim_all),
        "pearson":     pearson,
        "r2":          r2,
        "sign_agree":  sign_agree,
        "magnitude_ratio": mag_ratio,
        "rms_sim":     sim_mag,
        "rms_model":   model_mag,
    }


def metrics_for_ckpt(results: list) -> dict:
    """Returns {action_id: {"all_tau": {...}, "tau10": {...}, ...}}."""
    deltas = deltas_by_action(results)
    out = {}
    for a in deltas:
        block = {
            "name": ACTION_NAMES[a],
            "all_tau": metrics_for_action(deltas[a]),
        }
        for tau in (10, 50, 99):
            block[f"tau{tau}"] = metrics_for_action(deltas[a], tau=tau)
        out[a] = block
    return out


# ─────────────────────────────────────────────────────────────────────────
#  Markdown rendering
# ─────────────────────────────────────────────────────────────────────────

def render_md(per_ckpt: dict, out_path: Path):
    lines = ["# Action-conditioned counterfactual probe — quantitative match\n"]
    lines.append(
        "For every (episode, branch_t) we compute "
        "Δ_sim(a) = sim_HI(a) − sim_HI(do_nothing) and "
        "Δ_model(a) = model_HI(a) − model_HI(do_nothing) "
        "across the full horizon. The four metrics below answer "
        "_does the world model react to action a like the simulator does?_:\n"
    )
    lines.append(
        "- **Pearson** between the Δ vectors flattened over (ep, branch, τ, HI dim).\n"
        "- **R²** of regressing Δ_sim on Δ_model (intercept-free; 1.0 = perfect, "
        "0.0 = trivial mean, negative = worse than mean).\n"
        "- **Sign agreement**: fraction of non-zero (ep, branch, τ, HI) tuples where "
        "sign(Δ_sim) = sign(Δ_model). 0.5 = chance.\n"
        "- **Magnitude ratio**: ‖Δ_model‖_RMS / ‖Δ_sim‖_RMS. 1.0 = matched amplitude; "
        "near 0 = model is action-blind; >>1 = model overreacts.\n"
    )

    # ── Headline table: per-ckpt × action, full horizon ─────────────────
    lines.append("## Headline — full-horizon metrics per (architecture, action)\n")
    lines.append("| ckpt | action | n pairs | Pearson | R² | sign agree | "
                 "‖model‖/‖sim‖ |")
    lines.append("|------|--------|---------|---------|-----|-----------|---------------|")
    for ckpt, blocks in sorted(per_ckpt.items()):
        for a, block in sorted(blocks.items()):
            m = block["all_tau"]
            if m.get("n_pairs", 0) == 0:
                continue
            lines.append(
                f"| {ckpt} | {block['name']} | {m['n_pairs']} | "
                f"{m['pearson']:+.3f} | {m['r2']:+.3f} | "
                f"{m['sign_agree']:.3f} | {m['magnitude_ratio']:.3f} |"
            )
        lines.append("|      |        |         |         |     |           |               |")
    lines.append("")

    # ── Cross-architecture summary: mean across actions ─────────────────
    lines.append("## Cross-architecture summary (mean across the 6 non-zero actions)\n")
    lines.append("| ckpt | mean Pearson | mean R² | mean sign agree | "
                 "mean ‖model‖/‖sim‖ |")
    lines.append("|------|--------------|---------|------------------|---------------------|")
    for ckpt, blocks in sorted(per_ckpt.items()):
        prs, r2s, sgs, mrs = [], [], [], []
        for block in blocks.values():
            m = block["all_tau"]
            if m.get("n_pairs", 0) == 0:
                continue
            if not np.isnan(m["pearson"]):  prs.append(m["pearson"])
            if not np.isnan(m["r2"]):       r2s.append(m["r2"])
            if not np.isnan(m["sign_agree"]): sgs.append(m["sign_agree"])
            if not np.isnan(m["magnitude_ratio"]): mrs.append(m["magnitude_ratio"])
        def mean_fmt(xs):
            return f"{np.mean(xs):+.3f}" if xs else "—"
        def ratio_fmt(xs):
            return f"{np.mean(xs):.3f}" if xs else "—"
        def frac_fmt(xs):
            return f"{np.mean(xs):.3f}" if xs else "—"
        lines.append(f"| {ckpt} | {mean_fmt(prs)} | {mean_fmt(r2s)} | "
                     f"{frac_fmt(sgs)} | {ratio_fmt(mrs)} |")
    lines.append("")

    # ── Horizon decay (mean across actions, per τ) ──────────────────────
    lines.append("## Horizon decay (mean across actions, per τ)\n")
    lines.append("| ckpt | τ=10 Pearson | τ=50 Pearson | τ=99 Pearson | "
                 "τ=10 sign | τ=50 sign | τ=99 sign |")
    lines.append("|------|--------------|--------------|--------------|"
                 "-----------|-----------|-----------|")
    for ckpt, blocks in sorted(per_ckpt.items()):
        row = [ckpt]
        for metric in ("pearson",) :
            for tau in ("tau10", "tau50", "tau99"):
                xs = [b[tau][metric] for b in blocks.values()
                      if b[tau].get("n_pairs", 0) > 0
                      and not np.isnan(b[tau][metric])]
                row.append(f"{np.mean(xs):+.3f}" if xs else "—")
        for metric in ("sign_agree",):
            for tau in ("tau10", "tau50", "tau99"):
                xs = [b[tau][metric] for b in blocks.values()
                      if b[tau].get("n_pairs", 0) > 0
                      and not np.isnan(b[tau][metric])]
                row.append(f"{np.mean(xs):.3f}" if xs else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    out_path.write_text("\n".join(lines))
    log.info(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dirs", nargs="+",
                   default=["eval_results/counterfactual",
                            "eval_results/counterfactual_rssm"])
    p.add_argument("--out_dir", default="eval_results/action_cf_probe")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    in_dirs = [Path(d) for d in args.in_dirs]

    rollouts = load_rollouts(in_dirs)
    if not rollouts:
        log.error("No rollouts found.")
        return

    per_ckpt = {}
    for ckpt, results in rollouts.items():
        log.info(f"computing metrics for {ckpt} ({len(results)} rollouts) …")
        per_ckpt[ckpt] = metrics_for_ckpt(results)
        (out_dir / f"metrics_{ckpt}.json").write_text(
            json.dumps(per_ckpt[ckpt], indent=2, default=str)
        )

    render_md(per_ckpt, out_dir / "ACTION_CF_PROBE.md")
    log.info("done.")


if __name__ == "__main__":
    main()
