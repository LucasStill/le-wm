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
}

# AR-LSTM numbers (from OT's pushed results.md — they live on OT machine, hardcoded here)
LSTM_RUNS = {
    "L1": {"regular_pearson": 0.564, "test_hard_pearson": 0.325, "H": 16, "S": 1, "params_M": 1.138, "arch": "AR-LSTM"},
    "L2": {"regular_pearson": 0.495, "test_hard_pearson": -0.014, "H": 32, "S": 1, "params_M": 1.138, "arch": "AR-LSTM"},
}


def collect_summary() -> list:
    """Build a unified [{name, arch, H, S, regular, test_hard}, ...] table."""
    rows = []
    for name, info in JEPA_RUNS.items():
        reg = hi_mean(info["regular"])
        th = hi_mean(info["test_hard"])
        rows.append({
            "name": name, "arch": info["arch"], "H": info["H"], "S": info["S"],
            "params_M": info["params_M"],
            "regular": reg["pearson"] if reg else None,
            "test_hard": th["pearson"] if th else None,
        })
    for name, info in LSTM_RUNS.items():
        rows.append({
            "name": name, "arch": info["arch"], "H": info["H"], "S": info["S"],
            "params_M": info["params_M"],
            "regular": info["regular_pearson"],
            "test_hard": info["test_hard_pearson"],
        })
    return rows


# ── Figure 1: cross-arch × cross-config Pearson, in-dist vs OOD ──────────────
def fig1():
    rows = collect_summary()
    rows = [r for r in rows if r["regular"] is not None and r["test_hard"] is not None]

    labels = [f"{r['name']}\n{r['arch']}\nH={r['H']} S={r['S']}" for r in rows]
    in_d = [r["regular"] for r in rows]
    ood = [r["test_hard"] for r in rows]

    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w/2, in_d, w, label="In-distribution (test_lewm)", color=C_INDIST)
    ax.bar(x + w/2, ood, w, label="OOD (test_hard_lewm)", color=C_OOD)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("HI mean Pearson-r")
    ax.set_title("Encoder representation quality across configurations and architectures")
    ax.legend(loc="upper right")
    ax.set_ylim(min(min(ood) - 0.05, -0.1), max(in_d) + 0.08)

    # Annotate each bar with its value
    for i, v in enumerate(in_d):
        ax.text(i - w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(ood):
        offset = 0.01 if v >= 0 else -0.04
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
    rows = [r for r in rows if r["regular"] is not None and r["test_hard"] is not None]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for r in rows:
        color = C_JEPA if r["arch"] == "JEPA" else C_LSTM
        marker = "o" if r["arch"] == "JEPA" else "s"
        ax.scatter(r["regular"], r["test_hard"], s=110, color=color, marker=marker,
                   edgecolors="black", linewidths=0.6, zorder=3,
                   label=r["arch"] if r["name"] in ("E1", "L1") else None)
        ax.annotate(f" {r['name']} (H={r['H']}, S={r['S']})",
                    (r["regular"], r["test_hard"]), fontsize=9.5, va="center")

    # Identity line (perfect generalization)
    lo, hi = -0.1, 0.7
    ax.plot([lo, hi], [lo, hi], "--", color=C_NEUTRAL, lw=0.7, zorder=1,
            label="Perfect generalization (identity)")
    ax.axhline(0, color="black", lw=0.4, zorder=1)
    ax.axvline(0, color="black", lw=0.4, zorder=1)

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("In-distribution HI Pearson-r (test_lewm)")
    ax.set_ylabel("OOD HI Pearson-r (test_hard_lewm)")
    ax.set_title("Specificity ↔ generality: in-distribution vs OOD probe quality\n"
                 "JEPA shows the tradeoff (E2 above identity); AR-LSTM does not")
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
