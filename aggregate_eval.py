"""aggregate_eval.py — collect per-checkpoint JSONs and write CSVs + summary.

Run this after all eval_sweep job array tasks have finished:

    python aggregate_eval.py --out_dir /path/to/eval/<ARRAY_JOB_ID>
    python aggregate_eval.py --out_dir /path/to/eval/<ARRAY_JOB_ID> --figures
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_sweep import write_flat_csv, write_curves_csv, print_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True, help="directory containing *.json result files")
    parser.add_argument("--figures", action="store_true", help="also run plot_results.py")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    json_files = sorted(f for f in out_dir.glob("*.json") if f.name != "summary.json")

    if not json_files:
        print(f"No result JSONs found in {out_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} result(s) in {out_dir}:")
    all_results = []
    hi_names = None
    for jf in json_files:
        with open(jf) as f:
            res = json.load(f)
        all_results.append(res)
        print(f"  {jf.name}  {'ERROR' if 'error' in res else 'ok'}")
        if hi_names is None and "task1_hi" in res and res["task1_hi"]:
            first_sl = next(iter(res["task1_hi"].values()), {})
            hi_names = [k for k in first_sl if not k.startswith("__")]

    hi_names = hi_names or []

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary  → {summary_path}")

    write_flat_csv(all_results, hi_names, out_dir / "metrics_flat.csv")
    write_curves_csv(all_results, out_dir / "task3_curves.csv")
    print_summary(all_results)

    if args.figures:
        import subprocess
        subprocess.run([
            sys.executable, "plot_results.py",
            "--csv",    str(out_dir / "metrics_flat.csv"),
            "--curves", str(out_dir / "task3_curves.csv"),
            "--out_dir", str(out_dir / "figures"),
        ], check=True)


if __name__ == "__main__":
    main()
