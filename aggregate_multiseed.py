"""aggregate_multiseed.py — turn multi-seed eval JSONs into mean ± std table.

Reads eval_results/multiseed/<tag>/<run_label>.json files where tag is
'<ckpt>_<split>_seed<n>' and run_label embeds the same.

Writes:
  eval_results/multiseed/SUMMARY.md   — pretty markdown table
  eval_results/multiseed/SUMMARY.json — machine-readable

Usage:  python aggregate_multiseed.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
import math
from statistics import mean, stdev

ROOT = Path(__file__).parent / "eval_results" / "multiseed"


def hi_mean_pearson(json_path: Path) -> float | None:
    d = json.loads(json_path.read_text())
    if "task1_hi" not in d or not d["task1_hi"]:
        return None
    sl = d["task1_hi"].get("sl1") or next(iter(d["task1_hi"].values()))
    items = [v for k, v in sl.items() if k.startswith("HI_")]
    return mean(v["pearson_r"] for v in items)


def main():
    # group: (ckpt, split) → list of (seed, pearson)
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)

    for outdir in sorted(ROOT.glob("*")):
        if not outdir.is_dir():
            continue
        m = re.match(r"^(.+)_(test|test_hard)_seed(\d+)$", outdir.name)
        if not m:
            continue
        ckpt, split, seed = m.group(1), m.group(2), int(m.group(3))
        for jf in outdir.glob("*.json"):
            if jf.name == "summary.json":
                continue
            p = hi_mean_pearson(jf)
            if p is not None:
                grouped[(ckpt, split)].append((seed, p))

    if not grouped:
        print("No multiseed results found yet.")
        return

    # Build summary
    summary = []
    for (ckpt, split), seeds_p in sorted(grouped.items()):
        ps_all = [p for _, p in seeds_p]
        # Filter NaN runs (probe degenerate-init produces them)
        valid = [(s, p) for s, p in seeds_p if not math.isnan(p)]
        ps = [p for _, p in valid]
        n_valid, n_nan = len(ps), len(ps_all) - len(ps)
        summary.append({
            "ckpt": ckpt, "split": split,
            "n_seeds_total": len(ps_all), "n_seeds_valid": n_valid, "n_nan": n_nan,
            "seeds_valid": [s for s, _ in valid],
            "pearson_mean": mean(ps) if n_valid else float("nan"),
            "pearson_std": stdev(ps) if n_valid > 1 else 0.0,
            "pearson_values": ps_all,
        })

    # Write JSON
    out_json = ROOT / "SUMMARY.json"
    out_json.write_text(json.dumps(summary, indent=2))

    # Write markdown
    out_md = ROOT / "SUMMARY.md"
    lines = ["# Multi-seed eval_sweep task-1 — HI mean Pearson-r", ""]
    lines.append(f"_Aggregated {sum(s['n_seeds_total'] for s in summary)} runs across {len(summary)} (ckpt, split) cells._")
    lines.append("")

    # Per-ckpt rows with both splits side by side
    ckpts = sorted({s["ckpt"] for s in summary})
    lines.append("| ckpt   | regular test (mean ± std)        | test_hard (mean ± std)           | n_valid |")
    lines.append("|--------|----------------------------------|----------------------------------|---------|")
    for ckpt in ckpts:
        reg = next((s for s in summary if s["ckpt"] == ckpt and s["split"] == "test"), None)
        th = next((s for s in summary if s["ckpt"] == ckpt and s["split"] == "test_hard"), None)
        def cell(s):
            if not s: return "—"
            tag = f" ({s['n_nan']} NaN)" if s['n_nan'] else ""
            return f"{s['pearson_mean']:+.3f} ± {s['pearson_std']:.3f}{tag}"
        n_valid = (reg["n_seeds_valid"] if reg else 0) + (th["n_seeds_valid"] if th else 0)
        lines.append(f"| {ckpt:6s} | {cell(reg):32s} | {cell(th):32s} | {n_valid} |")

    lines.append("")
    lines.append("## Per-seed breakdown")
    lines.append("")
    lines.append("| ckpt   | split      | seed | Pearson |")
    lines.append("|--------|------------|------|---------|")
    # Need original (seed, p) tuples for per-seed dump — re-walk grouped
    for (ckpt, split), seeds_p in sorted(grouped.items()):
        for seed, p in seeds_p:
            tag = " ← NaN" if math.isnan(p) else ""
            lines.append(f"| {ckpt:6s} | {split:10s} | {seed}    | {p:+.3f}{tag} |")

    out_md.write_text("\n".join(lines) + "\n")
    print(out_md.read_text())


if __name__ == "__main__":
    main()
