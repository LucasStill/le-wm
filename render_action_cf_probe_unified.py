"""render_action_cf_probe_unified.py — combine all metrics_*.json under
eval_results/action_cf_probe/ into a single ACTION_CF_PROBE.md so JEPA + RSSM +
AR-LSTM rows live in the same headline table.

Mirrors action_conditioned_cf_probe.py's render_md output exactly; just loads
existing per-ckpt metrics from disk instead of recomputing from rollout JSONs.
This keeps parity with dragon's harness — we don't touch the source script.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ACTION_NAMES = {
    1: "fan_overhaul", 2: "hpc_overhaul", 3: "turbine_overhaul",
    4: "full_overhaul", 5: "patch", 6: "wash",
}

# Order ckpts so JEPA appears first, then RSSM, then AR-LSTM (paper-friendly).
DESIRED_ORDER = [
    "E1", "E2", "E5", "E6probe", "E7probe",
    "RSSM",
    "L1", "L2", "Lbig",
]

ROOT = Path(__file__).resolve().parent
METRICS_DIR = ROOT / "eval_results" / "action_cf_probe"
OUT_PATH = METRICS_DIR / "ACTION_CF_PROBE.md"


def load_metrics() -> dict:
    """Load all metrics_*.json under METRICS_DIR. Returns {ckpt: per-action dict}."""
    out = {}
    for f in sorted(METRICS_DIR.glob("metrics_*.json")):
        ckpt = f.stem.replace("metrics_", "")
        out[ckpt] = json.loads(f.read_text())
    return out


def render(per_ckpt: dict) -> str:
    lines = ["# Action-conditioned counterfactual probe — quantitative match\n"]
    lines.append(
        "For every (episode, branch_t) we compute Δ_sim(a) = sim_HI(a) − "
        "sim_HI(do_nothing) and Δ_model(a) = model_HI(a) − model_HI(do_nothing) "
        "across the full horizon. The four metrics below answer _does the world "
        "model react to action a like the simulator does?_:\n"
    )
    lines.append("- **Pearson** between the Δ vectors flattened over (ep, branch, τ, HI dim).")
    lines.append("- **R²** of regressing Δ_sim on Δ_model (intercept-free; 1.0 = perfect, "
                 "0.0 = trivial mean, negative = worse than mean).")
    lines.append("- **Sign agreement**: fraction of non-zero (ep, branch, τ, HI) tuples where "
                 "sign(Δ_sim) = sign(Δ_model). 0.5 = chance.")
    lines.append("- **Magnitude ratio**: ‖Δ_model‖_RMS / ‖Δ_sim‖_RMS. 1.0 = matched amplitude; "
                 "near 0 = model is action-blind; >>1 = model overreacts.\n")

    # Sort ckpts by DESIRED_ORDER, putting any extras at the end
    order = [c for c in DESIRED_ORDER if c in per_ckpt]
    extras = [c for c in per_ckpt if c not in DESIRED_ORDER]
    order.extend(sorted(extras))

    # ── Headline ────────────────────────────────────────────────────────────
    lines.append("## Headline — full-horizon metrics per (architecture, action)\n")
    lines.append("| ckpt | action | n pairs | Pearson | R² | sign agree | ‖model‖/‖sim‖ |")
    lines.append("|------|--------|---------|---------|-----|-----------|---------------|")
    for ckpt in order:
        blocks = per_ckpt[ckpt]
        for a_str, name in [(str(k), v) for k, v in ACTION_NAMES.items()]:
            block = blocks.get(a_str)
            if block is None:
                lines.append(f"| {ckpt} | {name} | — | — | — | — | — |")
                continue
            m = block.get("all_tau", {})
            n = m.get("n_pairs", 0)
            if n == 0:
                lines.append(f"| {ckpt} | {name} | 0 | — | — | — | — |")
                continue
            def fmt(x, sign=False, places=3):
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return "—"
                return f"{x:+.{places}f}" if sign else f"{x:.{places}f}"
            lines.append(
                f"| {ckpt} | {name} | {n} | "
                f"{fmt(m.get('pearson'), sign=True)} | "
                f"{fmt(m.get('r2'), sign=True)} | "
                f"{fmt(m.get('sign_agree'))} | "
                f"{fmt(m.get('magnitude_ratio'))} |"
            )
        lines.append("|      |        |         |         |     |           |               |")
    lines.append("")

    # ── Cross-architecture summary ───────────────────────────────────────────
    lines.append("## Cross-architecture summary (mean across the 6 non-zero actions)\n")
    lines.append("| ckpt | mean Pearson | mean R² | mean sign agree | mean ‖model‖/‖sim‖ |")
    lines.append("|------|--------------|---------|------------------|---------------------|")
    for ckpt in order:
        blocks = per_ckpt[ckpt]
        prs, r2s, sgs, mrs = [], [], [], []
        for k, block in blocks.items():
            m = block.get("all_tau", {})
            if m.get("n_pairs", 0) == 0:
                continue
            for arr, key in [(prs, "pearson"), (r2s, "r2"),
                             (sgs, "sign_agree"), (mrs, "magnitude_ratio")]:
                v = m.get(key)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    arr.append(v)

        def m_fmt(xs, sign=False):
            if not xs: return "—"
            return f"{np.mean(xs):+.3f}" if sign else f"{np.mean(xs):.3f}"

        lines.append(
            f"| {ckpt} | {m_fmt(prs, True)} | {m_fmt(r2s, True)} | "
            f"{m_fmt(sgs)} | {m_fmt(mrs)} |"
        )
    lines.append("")

    # ── Horizon decay ────────────────────────────────────────────────────────
    lines.append("## Horizon decay (mean across actions, per τ)\n")
    lines.append("| ckpt | τ=10 Pearson | τ=50 Pearson | τ=99 Pearson | "
                 "τ=10 sign | τ=50 sign | τ=99 sign |")
    lines.append("|------|--------------|--------------|--------------|"
                 "-----------|-----------|-----------|")
    for ckpt in order:
        blocks = per_ckpt[ckpt]
        row = [ckpt]
        for metric in ("pearson",):
            for tau in ("tau10", "tau50", "tau99"):
                xs = []
                for block in blocks.values():
                    t = block.get(tau, {})
                    v = t.get(metric)
                    if t.get("n_pairs", 0) > 0 and v is not None and \
                       not (isinstance(v, float) and np.isnan(v)):
                        xs.append(v)
                row.append(f"{np.mean(xs):+.3f}" if xs else "—")
        for metric in ("sign_agree",):
            for tau in ("tau10", "tau50", "tau99"):
                xs = []
                for block in blocks.values():
                    t = block.get(tau, {})
                    v = t.get(metric)
                    if t.get("n_pairs", 0) > 0 and v is not None and \
                       not (isinstance(v, float) and np.isnan(v)):
                        xs.append(v)
                row.append(f"{np.mean(xs):.3f}" if xs else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    per_ckpt = load_metrics()
    print(f"Loaded {len(per_ckpt)} ckpts: {sorted(per_ckpt.keys())}")
    md = render(per_ckpt)
    OUT_PATH.write_text(md)
    print(f"Wrote {OUT_PATH}  ({len(md)} chars)")


if __name__ == "__main__":
    main()
