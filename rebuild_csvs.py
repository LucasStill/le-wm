"""
Rebuild metrics_flat.csv and task3_curves.csv from existing per-checkpoint
JSON files in an eval directory.

Use this when:
  - The job timed out before all checkpoints finished (partial results)
  - You want to regenerate CSVs after adding new checkpoints to an existing dir
  - The CSV writing step at the end of eval_sweep.py was skipped for any reason

Usage:
    python rebuild_csvs.py /path/to/eval/dir/
    python rebuild_csvs.py $STABLEWM_HOME/eval/2111995
"""

import json
import sys
from pathlib import Path

# Import the two CSV writers directly from eval_sweep
sys.path.insert(0, str(Path(__file__).parent))
from eval_sweep import write_flat_csv, write_curves_csv


def main():
    if len(sys.argv) < 2:
        print("Usage: python rebuild_csvs.py /path/to/eval/dir/")
        sys.exit(1)

    out_dir = Path(sys.argv[1])
    if not out_dir.exists():
        print(f"Directory not found: {out_dir}")
        sys.exit(1)

    # Load all per-checkpoint JSONs (skip summary.json itself)
    json_files = sorted(
        p for p in out_dir.glob("*.json")
        if p.name not in ("summary.json",)
    )
    if not json_files:
        print(f"No checkpoint JSON files found in {out_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} checkpoint result(s):")
    all_results = []
    for p in json_files:
        with open(p) as f:
            res = json.load(f)
        status = "ERROR" if "error" in res else f"T1={'✓' if res.get('task1_hi') else '✗'}  T2={'✓' if res.get('task2_delta_hi') else '✗'}  T3={'✓' if res.get('task3_forecast') else '✗'}"
        print(f"  {p.name:<40}  {status}")
        all_results.append(res)

    # Collect HI component names from first successful Task-1 result
    hi_names = []
    for res in all_results:
        t1 = res.get("task1_hi") or {}
        # new structure: sl1 → {component: metrics}
        sl1 = t1.get("sl1") or t1  # fall back to flat structure for old results
        hi_names = [k for k in sl1 if not k.startswith("__")]
        if hi_names:
            break

    # Write summary.json
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {summary_path}")

    # Write CSVs
    write_flat_csv(all_results, hi_names, out_dir / "metrics_flat.csv")
    write_curves_csv(all_results, out_dir / "task3_curves.csv")

    print(f"\nDone. Now run:")
    print(f"  python plot_results.py \\")
    print(f"      --csv    {out_dir}/metrics_flat.csv \\")
    print(f"      --curves {out_dir}/task3_curves.csv \\")
    print(f"      --out_dir figures/sweep")


if __name__ == "__main__":
    main()
