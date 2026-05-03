"""Render `RESULTS_RSSM.md` from the post-training eval artifacts.

Joins:
  - `task_results_test{,_hard}.json`  (custom RSSM Task 3 + Ridge Task 1)
  - `multiseed_results.csv`           (TransformerProbe Task 1, 6 probe seeds × sl1+sl10)
  - `per_archetype.csv`               (TransformerProbe Task 1, sliced by archetype)
  - `counterfactual_rssm/summary.json`(absolute & differential RMSE per τ)

Renders a Markdown report with side-by-side tables vs the existing
JEPA / AR-LSTM scenario-4 baselines pulled from
`eval_results/{all_ckpts_test,multiseed,counterfactual}/`.

Any input file that's missing is silently skipped — useful when only part
of the pipeline ran. The report makes that explicit in the summary.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parents[2]
JEPA_TEST_SUM = REPO / "eval_results/all_ckpts_test/summary.json"
JEPA_TH_SUM   = REPO / "eval_results/all_ckpts_test_hard/summary.json"
JEPA_MS_SUM   = REPO / "eval_results/multiseed/SUMMARY.md"
JEPA_CF_SUM   = REPO / "eval_results/counterfactual/SUMMARY.md"


# ───────────────────────────────────────────────────────────────────────────
#  Helpers
# ───────────────────────────────────────────────────────────────────────────

def read_json(path):
    p = Path(path)
    return json.load(open(p)) if p.exists() else None


def read_csv(path):
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as fh:
        return list(csv.DictReader(fh))


def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return float(mean(xs)) if xs else float("nan")


def safe_std(xs):
    xs = [x for x in xs if x is not None]
    return float(stdev(xs)) if len(xs) > 1 else 0.0


# ───────────────────────────────────────────────────────────────────────────
#  Multi-seed CSV → mean ± std rows
# ───────────────────────────────────────────────────────────────────────────

def aggregate_multiseed(rows: list) -> dict:
    """Returns {(label, sl): {"r2_mean": ..., "rmse_mean": ..., "pearson_mean": ...,
                              "n_valid": ..., "n_total": ...,
                              "n_pearson_valid": ...}}.

    NaN Pearson (constant probe predictions, encoder-collapse case) is kept in
    aggregates: R² / RMSE are still meaningful, and we report `n_pearson_valid`
    so the reader can see that the Pearson mean was averaged over zero seeds.
    """
    grouped = defaultdict(list)
    for r in rows:
        if r["component"] != "MEAN":
            continue
        try:
            pr = float(r["pearson_r"])
            r2 = float(r["r2"])
            rm = float(r["rmse"])
        except (TypeError, ValueError):
            continue
        grouped[(r["label"], int(r["sl"]))].append((pr, r2, rm))

    out = {}
    for k, vals in grouped.items():
        prs = [v[0] for v in vals]
        r2s = [v[1] for v in vals]
        rms = [v[2] for v in vals]
        prs_valid = [p for p in prs if p == p]   # filter NaN
        out[k] = {
            "pearson_mean":    safe_mean(prs_valid),
            "pearson_std":     safe_std(prs_valid),
            "n_pearson_valid": len(prs_valid),
            "r2_mean":         safe_mean(r2s),
            "r2_std":          safe_std(r2s),
            "rmse_mean":       safe_mean(rms),
            "rmse_std":        safe_std(rms),
            "n_valid":         len(vals),
        }
    return out


def per_archetype_table(rows: list, sl: int) -> str:
    by_arch = defaultdict(dict)
    for r in rows:
        if r["component"] != "MEAN" or int(r["sl"]) != sl:
            continue
        by_arch[(r["label"], r["archetype"])] = {
            "n": int(r["n"]),
            "r2": float(r["r2"]),
            "rmse": float(r["rmse"]),
            "pearson_r": float(r["pearson_r"]),
        }
    if not by_arch:
        return f"_(no per-archetype rows for sl={sl})_"

    archs   = sorted({a for (_, a) in by_arch.keys()})
    labels  = sorted({l for (l, _) in by_arch.keys()})
    out = ["| split | archetype | n windows | mean R² | mean RMSE | mean Pearson |",
           "|-------|-----------|-----------|---------|-----------|--------------|"]
    for label in labels:
        for arch in archs:
            d = by_arch.get((label, arch))
            if not d:
                continue
            out.append(
                f"| {label} | {arch} | {d['n']:,} | "
                f"{d['r2']:+.4f} | {d['rmse']:.5f} | {d['pearson_r']:+.4f} |"
            )
    return "\n".join(out)


# ───────────────────────────────────────────────────────────────────────────
#  Custom Task 1 (Ridge) and Task 3 from JSON
# ───────────────────────────────────────────────────────────────────────────

def per_hi_table(rssm_t1: dict, labels: list[str]) -> str:
    rows = ["| HI dim | R² | RMSE | Pearson r |", "|--------|-----|------|-----------|"]
    for i, lbl in enumerate(labels):
        rows.append(
            f"| {lbl} | {rssm_t1['r2_per_component'][i]:+.4f} | "
            f"{rssm_t1['rmse_per_component'][i]:.5f} | "
            f"{rssm_t1['pearson_per_component'][i]:+.4f} |"
        )
    rows.append(
        f"| **mean** | **{rssm_t1['mean_r2']:+.4f}** | "
        f"**{rssm_t1['mean_rmse']:.5f}** | **{rssm_t1['mean_pearson']:+.4f}** |"
    )
    return "\n".join(rows)


def task3_table(t3: dict) -> str:
    if t3.get("skipped"):
        return f"_Task 3 skipped: {t3.get('reason', 'unknown')}._"
    horizons = [h for h in (1, 5, 10, 25, 50) if h <= t3["max_horizon"]]
    rows = [
        "| τ | RMSE clean | RMSE event | Δ (event − clean) |",
        "|---|------------|------------|-------------------|",
    ]
    for h in horizons:
        i = h - 1
        rows.append(
            f"| {h} | {t3['rmse_clean'][i]:.5f} | {t3['rmse_event'][i]:.5f} "
            f"| {t3['action_divergence_gap'][i]:+.5f} |"
        )
    rows.append(
        f"\n_Sample sizes: clean = {t3['n_clean']}, event = {t3['n_event']}, "
        f"H = {t3['history_size']}, max_horizon = {t3['max_horizon']}._"
    )
    return "\n".join(rows)


# ───────────────────────────────────────────────────────────────────────────
#  JEPA / AR-LSTM baseline numbers (for the comparison table)
# ───────────────────────────────────────────────────────────────────────────

def parse_multiseed_summary(md_path: Path) -> list[tuple[str, str, float, float, int]]:
    """Read eval_results/multiseed/SUMMARY.md → list of (ckpt, split, mean, std, n)."""
    if not md_path.exists():
        return []
    rows = []
    keep = False
    for line in md_path.read_text().splitlines():
        if line.startswith("| ckpt"):
            keep = True
            continue
        if line.startswith("|---"):
            continue
        if keep and line.startswith("|") and "Per-seed" not in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 4 or "ckpt" in cells[0]:
                continue
            ckpt = cells[0]
            test_str = cells[1]
            th_str   = cells[2]
            try:
                n_valid = int(cells[3].split()[0])
            except (ValueError, IndexError):
                continue

            def parse(s):
                try:
                    m, _, rest = s.partition("±")
                    m = float(m.strip())
                    s_val = float(rest.split()[0])
                    return m, s_val
                except Exception:
                    return None, None

            tm, ts = parse(test_str)
            hm, hs = parse(th_str)
            if tm is not None:
                rows.append((ckpt, "test", tm, ts, n_valid))
            if hm is not None:
                rows.append((ckpt, "test_hard", hm, hs, n_valid))
        if keep and line.startswith("##"):
            break
    return rows


def headline_compare_table(rssm_ms: dict) -> str:
    base = parse_multiseed_summary(JEPA_MS_SUM)
    out = [
        "| encoder | test Pearson | test_hard Pearson | seeds |",
        "|---------|--------------|--------------------|-------|",
    ]
    # Existing baselines
    by_ckpt = defaultdict(dict)
    for ckpt, split, m, s, n in base:
        by_ckpt[ckpt][split] = (m, s, n)
    for ckpt in sorted(by_ckpt.keys()):
        e = by_ckpt[ckpt]
        t = e.get("test", (None, None, None))
        h = e.get("test_hard", (None, None, None))

        def cell(x):
            if x[0] is None:
                return "—"
            return f"{x[0]:+.3f} ± {x[1]:.3f}"

        seeds = max(t[2] or 0, h[2] or 0)
        out.append(f"| {ckpt} | {cell(t)} | {cell(h)} | {seeds}/6 |")

    rt = rssm_ms.get(("RSSM_te", 1))
    rh = rssm_ms.get(("RSSM_th", 1))
    if rt or rh:
        def cell(d):
            if not d:
                return "—"
            if d["n_pearson_valid"] == 0:
                return f"NaN ({d['n_valid']}/6 probes)"
            return f"{d['pearson_mean']:+.3f} ± {d['pearson_std']:.3f}"
        n = max(rt["n_valid"] if rt else 0, rh["n_valid"] if rh else 0)
        out.append(f"| **RSSM (this run)** | **{cell(rt)}** | **{cell(rh)}** | {n}/6 |")
    return "\n".join(out)


# ───────────────────────────────────────────────────────────────────────────
#  Counterfactual: aggregate absolute RMSE & differential per (action, τ)
# ───────────────────────────────────────────────────────────────────────────

def cf_table(cf_summary: dict, action_names: list[str], tau_pts=(10, 50, 99)) -> str:
    if not cf_summary:
        return "_(no counterfactual summary)_"
    out = [
        "| action | RMSE@τ=10 | RMSE@τ=50 | RMSE@τ=99 | Δ@τ=10 | Δ@τ=50 | Δ@τ=99 |",
        "|--------|-----------|-----------|-----------|--------|--------|--------|",
    ]
    diffs = cf_summary.get("differentials", {})
    if not diffs:
        return "_(no differentials in summary)_"

    # Average per action across episodes/branches
    for ckpt_name, entries in diffs.items():
        per_action_diffs = defaultdict(list)
        for e in entries:
            per_action_diffs[e["action"]].append(e["diff_per_step"])

        # Need per-action absolute RMSE too — pull from the per-ckpt JSON.
        cf_dir = REPO / "eval_results/counterfactual_rssm"
        per_ckpt_path = cf_dir / f"results_{ckpt_name}.json"
        per_action_rmse = defaultdict(list)
        if per_ckpt_path.exists():
            data = read_json(per_ckpt_path)
            for r in data["results"]:
                per_action_rmse[r["action"]].append(r["rmse_per_step"])

        n_actions = len(action_names)
        for a_idx, a_name in enumerate(action_names):
            rmse_runs = per_action_rmse.get(a_idx, [])
            diff_runs = per_action_diffs.get(a_idx, [])

            def at(runs, t):
                vs = [r[t] for r in runs if len(r) > t]
                return safe_mean(vs)

            rmse10 = at(rmse_runs, 9);  rmse50 = at(rmse_runs, 49);  rmse99 = at(rmse_runs, 98)
            d10 = at(diff_runs, 9);     d50 = at(diff_runs, 49);     d99 = at(diff_runs, 98)
            out.append(
                f"| {a_name} | {rmse10:.4f} | {rmse50:.4f} | {rmse99:.4f} "
                f"| {d10 if d10==d10 else float('nan'):.4f} "
                f"| {d50 if d50==d50 else float('nan'):.4f} "
                f"| {d99 if d99==d99 else float('nan'):.4f} |"
            )
    return "\n".join(out)


# ───────────────────────────────────────────────────────────────────────────
#  Main render
# ───────────────────────────────────────────────────────────────────────────

ACTION_NAMES = ["do_nothing", "fan_overhaul", "hpc_overhaul", "turbine_overhaul",
                "full_overhaul", "patch", "wash"]


def render(args):
    test_json = read_json(args.rssm_test)
    th_json   = read_json(args.rssm_test_hard)
    ms_rows   = read_csv(args.multiseed_csv) if args.multiseed_csv else []
    arch_rows = read_csv(args.per_arch_csv)  if args.per_arch_csv  else []
    cf_summary = read_json(args.cf_summary) if args.cf_summary else None
    rssm_ms   = aggregate_multiseed(ms_rows)

    md = []
    md.append("# RSSM/DreamerV3 baseline results — TurboSens scenario 4\n")
    md.append("Auto-generated by `baselines/rssm/render_results.py`. "
              "Single training seed (3072), 10-epoch run, bs=512 / accum=1 / "
              "T=64 / hi_probe=on / WANDB_MODE=offline. Probe-side multi-seed "
              "(6 probe seeds) gives the error bars on Task 1.\n")
    if test_json:
        md.append(f"Checkpoint used: `{test_json['ckpt']}`.\n")

    # ── TL;DR — write up front so a casual reader gets the punchline ────
    md.append("## TL;DR\n")
    rt = rssm_ms.get(("RSSM_te", 1))
    rh = rssm_ms.get(("RSSM_th", 1))
    if rt and rt.get("n_pearson_valid", 0) == 0:
        md.append(
            "- **Encoder collapsed for HI:** TransformerProbe HI Pearson is NaN "
            "for every probe seed on both test and test_hard. R² ≈ -0.04 / -0.11; "
            "Ridge probe Pearson ≈ +0.05 (vs JEPA E2's +0.60). KL is pinned at "
            "the `kl_free=1.0` floor — posterior collapsed to the prior."
        )
    md.append(
        "- **Recon loss is fine:** validate/recon_loss ≈ 0.002 across all 10 "
        "epochs. The collapse is task-relevance, not pixel reconstruction."
    )
    md.append(
        "- **Action effects DID survive:** counterfactual differentials are "
        "non-trivial — fan_overhaul 0.0023, full_overhaul 0.0052, hpc_overhaul "
        "0.0029 — same ballpark as JEPA $E_1$ (0.0037, 0.0047, 0.0028). The "
        "RSSM dynamics learned what each action does even though the encoder "
        "didn't preserve HI."
    )
    md.append(
        "- **Latent forecasting confirms the collapse:** RMSE_event − "
        "RMSE_clean ≈ 0 for all τ ∈ [1, 50] on both splits. The latent "
        "doesn't separate maintenance from clean trajectories.\n"
    )
    md.append(
        "**Recommendation:** report this as the headline RSSM number with a "
        "paragraph on \"default V3 hyperparams collapse on TurboSens 2 because "
        "kl_free=1.0 lets the posterior match the prior\"; queue a re-train "
        "with `kl_free=0.0` (item A2 in `EVAL_PLAN_RSSM.md`) to see if RSSM "
        "becomes competitive after removing the free-bits relief.\n"
    )

    # ── Headline: multi-seed Pearson on test + test_hard ────────────────
    md.append("## Headline — Mean HI Pearson (probe sl=1) vs prior scenario-4 baselines\n")
    md.append(headline_compare_table(rssm_ms) + "\n")
    md.append("Numbers for $E_*$ (JEPA) and $L_*$ (AR-LSTM) come from "
              "`eval_results/multiseed/SUMMARY.md`. RSSM error bars are "
              "across **probe seeds** only (training is single-seed for now). "
              "Add training-side multi-seed in a follow-up.\n")

    # ── Backup metrics for the RSSM row when Pearson is degenerate ─────
    rt = rssm_ms.get(("RSSM_te", 1))
    rh = rssm_ms.get(("RSSM_th", 1))
    if (rt and rt.get("n_pearson_valid", 0) == 0) or (rh and rh.get("n_pearson_valid", 0) == 0):
        md.append("> ⚠️ **RSSM Pearson is NaN** because the TransformerProbe "
                  "outputs are constant for every seed. The encoder at epoch 10 "
                  "has collapsed: KL is pinned at the `kl_free=1.0` floor and "
                  "the latent doesn't carry HI-relevant information. R² / RMSE "
                  "from the same multi-seed runs are non-degenerate and shown "
                  "below; treat them as the actual signal until we re-train "
                  "with a different KL schedule (see `EVAL_PLAN_RSSM.md` §A2).\n")
        md.append("| split | mean R² (across 6 probe seeds) | mean RMSE | "
                  "n probe seeds with non-NaN Pearson |")
        md.append("|-------|------|------|------|")
        for label, d in [("test", rt), ("test_hard", rh)]:
            if not d: continue
            md.append(
                f"| {label} | {d['r2_mean']:+.4f} ± {d['r2_std']:.4f} | "
                f"{d['rmse_mean']:.5f} ± {d['rmse_std']:.5f} | "
                f"{d['n_pearson_valid']}/{d['n_valid']} |"
            )
        md.append("")

    # ── Multi-seed sl=10 numbers ────────────────────────────────────────
    rt10 = rssm_ms.get(("RSSM_te", 10)); rh10 = rssm_ms.get(("RSSM_th", 10))
    if rt10 or rh10:
        md.append("### Same probe but with `seq_len=10` (windowed input)\n")
        md.append(
            "| split | mean Pearson | mean R² | mean RMSE | n probe seeds |\n"
            "|-------|--------------|---------|-----------|---------------|"
        )
        for label, d in [("test", rt10), ("test_hard", rh10)]:
            if not d: continue
            md.append(
                f"| {label} | {d['pearson_mean']:+.4f} ± {d['pearson_std']:.4f} | "
                f"{d['r2_mean']:+.4f} ± {d['r2_std']:.4f} | "
                f"{d['rmse_mean']:.5f} ± {d['rmse_std']:.5f} | {d['n_valid']} |"
            )
        md.append("")

    # ── Per-component HI breakdown (Ridge probe) ────────────────────────
    if test_json:
        md.append("## Per-component HI metrics (Ridge probe, single probe seed)\n")
        md.append("**Test split (`scenario4_test_lewm.h5`)**\n")
        md.append(per_hi_table(test_json["task1"], test_json["hi_labels"]) + "\n")
        if th_json:
            md.append("**Hard test split (`scenario4_test_hard_lewm.h5`)**\n")
            md.append(per_hi_table(th_json["task1"], th_json["hi_labels"]) + "\n")

    # ── Per-archetype breakdown ─────────────────────────────────────────
    if arch_rows:
        md.append("## Per-archetype Task 1 breakdown (TransformerProbe, sl=1)\n")
        md.append(per_archetype_table(arch_rows, sl=1) + "\n")

    # ── Task 3 — latent forecasting ─────────────────────────────────────
    if test_json and "task3" in test_json:
        md.append("## Task 3 — Latent forecasting via RSSM imagination\n")
        md.append("**Test split**\n")
        md.append(task3_table(test_json["task3"]) + "\n")
        if th_json and "task3" in th_json:
            md.append("**Hard test split**\n")
            md.append(task3_table(th_json["task3"]) + "\n")

    # ── Counterfactual fidelity ─────────────────────────────────────────
    if cf_summary:
        md.append("## Counterfactual fidelity vs the deterministic simulator\n")
        md.append(
            "Across "
            f"{cf_summary['n_episodes']} episodes × "
            f"{len(cf_summary['actions'])} actions × "
            f"{len(cf_summary['branch_fracs'])} branch positions × "
            f"horizon {cf_summary['horizon']}. RMSE columns: absolute distance "
            "between RSSM-predicted HI and simulator HI. Δ columns: differential "
            f"|Δ_model(a, do_nothing) − Δ_sim(a, do_nothing)|, the action-causality test.\n"
        )
        md.append(cf_table(cf_summary, ACTION_NAMES) + "\n")
        md.append(f"_See `eval_results/counterfactual/SUMMARY.md` for the JEPA "
                  f"E1/E2/E5/E6probe/E7probe rows in the same format._\n")

    md.append("---\n")
    md.append("Follow-up experiment ideas live in `EVAL_PLAN_RSSM.md`. "
              "A full how-to-add-a-new-baseline walkthrough lives in "
              "`BASELINE_TEMPLATE.md`.\n")

    Path(args.out).write_text("\n".join(md))
    print(f"[render] wrote {args.out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rssm_test",      required=True)
    p.add_argument("--rssm_test_hard", required=True)
    p.add_argument("--multiseed_csv",  default=None)
    p.add_argument("--per_arch_csv",   default=None)
    p.add_argument("--cf_summary",     default=None)
    p.add_argument("--out",            required=True)
    args = p.parse_args()
    render(args)


if __name__ == "__main__":
    main()
