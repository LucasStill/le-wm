"""make_paper_figures.py — regenerate all paper figures from eval results.

Deterministic. Reads:
  - eval_results/*/E*.json         (dragon JEPA runs, eval_sweep output)
  - eval_results/*/L*.json         (orailixtower AR-LSTM runs, eval_sweep output)
  - coordination/{dragon,orailixtower}/results.md  (for any numbers not in JSON)

Writes PDF + PNG into ./figures/ . Idempotent — safe to re-run.

Currently covers four headline figures (extend as more results land):
  fig1_xarch_xconfig_inOOD.{pdf,png}     — cross-arch × cross-config Pearson, in-dist vs OOD
  fig2_perHI_E1.{pdf,png}                — per-component HI Pearson, regular vs test_hard
  fig3_probe_starvation.{pdf,png}        — in-training (28k) vs eval_sweep (770k) Pearson
  fig4_inDist_vs_OOD_scatter.{pdf,png}   — specificity-generality scatter
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── matplotlib styling — paper-ready ──────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.frameon": False,
    "legend.fontsize": 9,
})

# Brand-consistent colors
C_INDIST = "#1f77b4"   # blue   — in-distribution
C_OOD = "#ff7f0e"      # orange — out-of-distribution
C_JEPA = "#2ca02c"     # green  — JEPA architecture
C_LSTM = "#d62728"     # red    — AR-LSTM architecture
C_NEUTRAL = "#7f7f7f"

# ── data sources ──────────────────────────────────────────────────────────────
def hi_mean(json_path: Path) -> dict:
    """Return dict with mean Pearson, R², RMSE across HI_0..HI_9 from a task1 JSON."""
    if not json_path.exists():
        return None
    d = json.loads(json_path.read_text())
    if "task1_hi" not in d or not d["task1_hi"]:
        return None
    sl1 = d["task1_hi"].get("sl1") or next(iter(d["task1_hi"].values()))
    items = [v for k, v in sl1.items() if k.startswith("HI_")]
    if not items:
        return None
    return {
        "pearson": np.mean([v["pearson_r"] for v in items]),
        "r2": np.mean([v["r2"] for v in items]),
        "rmse": np.mean([v["rmse"] for v in items]),
        "per_hi": {k: v for k, v in sl1.items() if k.startswith("HI_")},
    }


def per_hi_dict(json_path: Path) -> dict:
    """Return dict {HI_0: {pearson, r2, rmse}, ...} from a task1 JSON."""
    d = json.loads(json_path.read_text())
    sl1 = d["task1_hi"].get("sl1") or next(iter(d["task1_hi"].values()))
    return {k: v for k, v in sl1.items() if k.startswith("HI_")}


# Map of run name → eval JSONs
JEPA_RUNS = {
    "E1": {
        "regular": ROOT / "eval_results/E1_test_regular/E1_te.json",
        "test_hard": ROOT / "eval_results/E1_test_hard/E1_th.json",
        "H": 16, "S": 1, "params_M": 0.828, "arch": "JEPA",
    },
    "E2": {
        "regular": ROOT / "eval_results/all_ckpts_test/E2_te.json",
        "test_hard": ROOT / "eval_results/all_ckpts_test_hard/E2_th.json",
        "H": 32, "S": 1, "params_M": 0.828, "arch": "JEPA",
    },
    "E3": {
        "regular": ROOT / "eval_results/all_ckpts_test/E3_te.json",
        "test_hard": ROOT / "eval_results/all_ckpts_test_hard/E3_th.json",
        "H": 16, "S": 5, "params_M": 0.828, "arch": "JEPA",
    },
    "E5": {
        "regular": ROOT / "eval_results/E5_test/E5_te.json",
        "test_hard": ROOT / "eval_results/E5_test_hard/E5_th.json",
        "H": 8, "S": 1, "params_M": 0.828, "arch": "JEPA",
    },
    "E6probe": {
        "regular": ROOT / "eval_results/E6probe_test/E6probe_te.json",
        "test_hard": ROOT / "eval_results/E6probe_test_hard/E6probe_th.json",
        "H": 16, "S": 10, "params_M": 0.828, "arch": "JEPA",
    },
}

# AR-LSTM numbers (from OT's pushed results.md — they live on OT machine, hardcoded here)
LSTM_RUNS = {
    "L1": {"regular_pearson": 0.564, "test_hard_pearson": 0.325, "H": 16, "S": 1, "params_M": 1.138, "arch": "AR-LSTM"},
    "L2": {"regular_pearson": 0.495, "test_hard_pearson": -0.014, "H": 32, "S": 1, "params_M": 1.138, "arch": "AR-LSTM"},
}


MULTISEED_SUMMARY = ROOT / "eval_results/multiseed/SUMMARY.json"

# Configs that have been planned/launched but don't have results yet.
# Each entry shows on plots as a hatched / annotated "in progress" placeholder.
IN_PROGRESS = {
    "E7": {"H": 16, "S": 20, "params_M": 0.828, "arch": "JEPA",
           "note": "1-epoch probe queued — wall ~8h"},
    "L_big": {"H": 32, "S": 1, "params_M": 1.138, "arch": "AR-LSTM",
              "note": "W=4 H=32 — running on OT, ETA ~09:30 UTC"},
}


def load_multiseed() -> dict:
    """Return {(ckpt, split): {mean, std, n_valid}} from the multi-seed SUMMARY."""
    if not MULTISEED_SUMMARY.exists():
        return {}
    rows = json.loads(MULTISEED_SUMMARY.read_text())
    out = {}
    for r in rows:
        out[(r["ckpt"], r["split"])] = {
            "mean": r["pearson_mean"],
            "std": r["pearson_std"],
            "n_valid": r["n_seeds_valid"],
        }
    return out


def collect_summary() -> list:
    """Build a unified row list. Prefers multi-seed mean±std if available."""
    multiseed = load_multiseed()
    rows = []
    for name, info in JEPA_RUNS.items():
        ms_reg = multiseed.get((name, "test"))
        ms_th = multiseed.get((name, "test_hard"))
        # Fallback to single-seed JSON if no multi-seed
        reg = hi_mean(info["regular"]) if not ms_reg else None
        th = hi_mean(info["test_hard"]) if not ms_th else None
        rows.append({
            "name": name, "arch": info["arch"], "H": info["H"], "S": info["S"],
            "params_M": info["params_M"],
            "regular": (ms_reg["mean"] if ms_reg else (reg["pearson"] if reg else None)),
            "regular_std": ms_reg["std"] if ms_reg else None,
            "test_hard": (ms_th["mean"] if ms_th else (th["pearson"] if th else None)),
            "test_hard_std": ms_th["std"] if ms_th else None,
            "n_seeds": ms_reg["n_valid"] if ms_reg else 1,
            "is_in_progress": False,
        })
    for name, info in LSTM_RUNS.items():
        rows.append({
            "name": name, "arch": info["arch"], "H": info["H"], "S": info["S"],
            "params_M": info["params_M"],
            "regular": info["regular_pearson"],
            "regular_std": None,
            "test_hard": info["test_hard_pearson"],
            "test_hard_std": None,
            "n_seeds": 1,
            "is_in_progress": False,
        })
    for name, info in IN_PROGRESS.items():
        rows.append({
            "name": name, "arch": info["arch"], "H": info["H"], "S": info["S"],
            "params_M": info["params_M"],
            "regular": None, "regular_std": None,
            "test_hard": None, "test_hard_std": None,
            "n_seeds": 0, "is_in_progress": True,
            "note": info["note"],
        })
    return rows


# ── Figure 1: cross-arch × cross-config Pearson, in-dist vs OOD ──────────────
def fig1():
    rows = collect_summary()
    # Order: real results first (sorted by name), then in-progress
    real = [r for r in rows if not r["is_in_progress"]]
    in_progress = [r for r in rows if r["is_in_progress"]]
    real = sorted(real, key=lambda r: (r["arch"], r["H"], r["S"]))
    rows_ord = real + in_progress

    labels = []
    for r in rows_ord:
        tag = "\n(in progress)" if r["is_in_progress"] else (f"\n(n={r['n_seeds']} seeds)" if r["n_seeds"] > 1 else "")
        labels.append(f"{r['name']}\n{r['arch']}\nH={r['H']} S={r['S']}{tag}")
    in_d = [(r["regular"] if r["regular"] is not None else 0) for r in rows_ord]
    ood = [(r["test_hard"] if r["test_hard"] is not None else 0) for r in rows_ord]
    in_d_err = [(r["regular_std"] or 0) for r in rows_ord]
    ood_err = [(r["test_hard_std"] or 0) for r in rows_ord]
    is_ip = [r["is_in_progress"] for r in rows_ord]

    x = np.arange(len(rows_ord))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.5))
    in_colors = [C_INDIST if not p else "lightgray" for p in is_ip]
    ood_colors = [C_OOD if not p else "lightgray" for p in is_ip]
    ax.bar(x - w/2, in_d, w, color=in_colors,
           hatch=["" if not p else "//" for p in is_ip],
           edgecolor="black", linewidth=0.6,
           label="In-distribution (test_lewm)", yerr=in_d_err, capsize=3)
    ax.bar(x + w/2, ood, w, color=ood_colors,
           hatch=["" if not p else "//" for p in is_ip],
           edgecolor="black", linewidth=0.6,
           label="OOD (test_hard_lewm)", yerr=ood_err, capsize=3)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("HI mean Pearson-r")
    ax.set_title("Encoder representation quality — multi-seed mean ± std (where available)\n"
                 "Hatched bars = config queued, awaiting results")
    ax.legend(loc="upper right")

    valid_ood = [v for v in ood if not np.isnan(v)]
    valid_ind = [v for v in in_d if not np.isnan(v)]
    ymin = min(min(valid_ood + [0]) - 0.1, -0.15)
    ymax = max(valid_ind) + 0.12
    ax.set_ylim(ymin, ymax)

    for i, (v, p) in enumerate(zip(in_d, is_ip)):
        if p:
            ax.text(i - w/2, 0.02, "(in progress)", ha="center",
                    fontsize=8, color="dimgray", rotation=90, va="bottom")
        else:
            ax.text(i - w/2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    for i, (v, p) in enumerate(zip(ood, is_ip)):
        if p:
            continue
        offset = 0.015 if v >= 0 else -0.04
        ax.text(i + w/2, v + offset, f"{v:.2f}", ha="center", fontsize=8)

    out = FIG_DIR / "fig1_xarch_xconfig_inOOD"
    fig.savefig(f"{out}.pdf"); fig.savefig(f"{out}.png")
    plt.close(fig)
    print(f"  → {out.name}.{{pdf,png}}")


# ── Figure 2: per-HI breakdown for E1, regular vs test_hard ──────────────────
def fig2():
    reg = per_hi_dict(JEPA_RUNS["E1"]["regular"])
    th = per_hi_dict(JEPA_RUNS["E1"]["test_hard"])
    his = sorted(reg.keys())  # HI_0 .. HI_9

    reg_p = [reg[h]["pearson_r"] for h in his]
    th_p = [th[h]["pearson_r"] for h in his]

    x = np.arange(len(his))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(x - w/2, reg_p, w, label="In-distribution (test_lewm)", color=C_INDIST)
    ax.bar(x + w/2, th_p, w, label="OOD (test_hard_lewm)", color=C_OOD)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(his, fontsize=9)
    ax.set_ylabel("Pearson-r")
    ax.set_title("E1 (JEPA, H=16): per-HI Pearson — robust vs distribution-shift-sensitive components")
    ax.legend(loc="lower left")

    # Annotate notable cases
    idx_HI3 = his.index("HI_3")
    idx_HI8 = his.index("HI_8")
    ax.annotate("robust", xy=(idx_HI3 + w/2, th_p[idx_HI3]),
                xytext=(idx_HI3 + w/2 + 0.2, th_p[idx_HI3] + 0.15),
                arrowprops=dict(arrowstyle="-", lw=0.5, color=C_NEUTRAL),
                fontsize=9, color=C_NEUTRAL)
    ax.annotate("sign flip on OOD!", xy=(idx_HI8 + w/2, th_p[idx_HI8]),
                xytext=(idx_HI8 + w/2 - 1.5, th_p[idx_HI8] - 0.2),
                arrowprops=dict(arrowstyle="->", lw=0.5, color="darkred"),
                fontsize=9, color="darkred")

    out = FIG_DIR / "fig2_perHI_E1"
    fig.savefig(f"{out}.pdf"); fig.savefig(f"{out}.png")
    plt.close(fig)
    print(f"  → {out.name}.{{pdf,png}}")


# ── Figure 3: in-training vs eval_sweep Pearson (probe-budget effect) ────────
def fig3():
    # Hardcoded — these are findings, not file outputs
    runs = ["E1\n(JEPA H=16)", "L1\n(AR-LSTM H=16)"]
    in_training = [0.250, 0.242]   # 28k probe-train budget
    eval_sweep = [0.563, 0.564]    # 770k probe-train budget

    x = np.arange(len(runs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w/2, in_training, w, label="In-training probe (28k samples)", color=C_NEUTRAL)
    ax.bar(x + w/2, eval_sweep, w, label="eval_sweep probe (770k samples)", color=C_INDIST)
    ax.set_xticks(x); ax.set_xticklabels(runs, fontsize=9.5)
    ax.set_ylabel("HI mean Pearson-r")
    ax.set_title("Default in-training probe was data-starved (~27× less training data)\n"
                 "Same encoders, same head architecture, only probe-data budget differs")

    # Annotate ratio
    for i, (lo, hi) in enumerate(zip(in_training, eval_sweep)):
        ax.text(i - w/2, lo + 0.01, f"{lo:.2f}", ha="center", fontsize=9)
        ax.text(i + w/2, hi + 0.01, f"{hi:.2f}", ha="center", fontsize=9)
        ax.annotate(f"×{hi/lo:.1f}", xy=(i, max(lo, hi) + 0.08),
                    ha="center", fontsize=11, color="darkgreen", weight="bold")

    ax.legend(loc="upper left")
    ax.set_ylim(0, 0.7)
    out = FIG_DIR / "fig3_probe_starvation"
    fig.savefig(f"{out}.pdf"); fig.savefig(f"{out}.png")
    plt.close(fig)
    print(f"  → {out.name}.{{pdf,png}}")


# ── Figure 4: in-distribution vs OOD scatter — the tradeoff visualised ───────
def fig4():
    rows = collect_summary()
    real = [r for r in rows if not r["is_in_progress"] and r["regular"] is not None and r["test_hard"] is not None]
    in_progress = [r for r in rows if r["is_in_progress"]]

    fig, ax = plt.subplots(figsize=(7, 6))
    seen = set()
    for r in real:
        color = C_JEPA if r["arch"] == "JEPA" else C_LSTM
        marker = "o" if r["arch"] == "JEPA" else "s"
        label = r["arch"] if r["arch"] not in seen else None
        seen.add(r["arch"])
        ax.errorbar(r["regular"], r["test_hard"],
                    xerr=r["regular_std"] if r["regular_std"] else 0,
                    yerr=r["test_hard_std"] if r["test_hard_std"] else 0,
                    fmt=marker, ms=10, color=color, mec="black", mew=0.6,
                    elinewidth=0.7, capsize=2, zorder=3, label=label)
        ax.annotate(f" {r['name']}\n (H={r['H']}, S={r['S']})",
                    (r["regular"], r["test_hard"]), fontsize=9, va="center")

    # In-progress placeholders (gray hollow markers along axes)
    for r in in_progress:
        color = C_JEPA if r["arch"] == "JEPA" else C_LSTM
        marker = "o" if r["arch"] == "JEPA" else "s"
        # Place at edge marker; use x=ymin to indicate "unknown"
        ax.scatter([-0.05], [-0.08], s=100, marker=marker, facecolors="none",
                   edgecolors=color, linewidths=1.2, zorder=2)
        ax.annotate(f" {r['name']} (H={r['H']}, S={r['S']}) — in progress",
                    (-0.05, -0.08), fontsize=8.5, va="top", color="dimgray",
                    xytext=(-0.05 + 0.01, -0.08 - 0.025))

    lo, hi = -0.1, 0.75
    ax.plot([lo, hi], [lo, hi], "--", color=C_NEUTRAL, lw=0.7, zorder=1,
            label="Perfect generalization (identity)")
    ax.axhline(0, color="black", lw=0.4, zorder=1)
    ax.axvline(0, color="black", lw=0.4, zorder=1)

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("In-distribution HI Pearson-r (test_lewm)")
    ax.set_ylabel("OOD HI Pearson-r (test_hard_lewm)")
    ax.set_title("Specificity ↔ generality (multi-seed mean, error bars = ±1 std)\n"
                 "Larger S → higher OOD; in-distribution best at H=32 S=1 (E2)")
    ax.legend(loc="lower right")
    ax.set_aspect("equal", "box")

    out = FIG_DIR / "fig4_inDist_vs_OOD_scatter"
    fig.savefig(f"{out}.pdf"); fig.savefig(f"{out}.png")
    plt.close(fig)
    print(f"  → {out.name}.{{pdf,png}}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Generating paper figures into {FIG_DIR}/ ...")
    fig1()
    fig2()
    fig3()
    fig4()
    print("Done. PDFs are vector (paper-ready); PNGs are 300 DPI raster (slides).")


if __name__ == "__main__":
    main()
