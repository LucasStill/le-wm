#!/usr/bin/env python
"""
prepare_scenario3_dataset.py
============================
Convert turbosens1 *_sensors.h5 files into pipeline-ready *_lewm.h5 files
for le-wm training.

What this script does
---------------------
  1. Reads ``observation/sensors``  (N, 7, 12) from the source file.
  2. Normalises each sensor-context channel to [0, 1] using training-set
     statistics (computed from the train file, applied to all splits).
  3. Writes the result as ``pixels``  (N, 7, 12)  — intentionally *without*
     channel replication.  The SensorEncoder flattens to (B, T, 84) tokens
     internally, so the redundant ×3 channel trick used for the ViT is not
     needed here.
  4. Copies ``action``  as (N, 1) float32 with integer values 0–4.
     (0=do_nothing, 1-4=overhaul types)
  5. Copies ``observation/state`` → ``observation.state`` (flat key, as
     expected by eval_sweep.py / train.py).
  6. Copies ``ep_len`` and ``ep_offset`` as int64.
  7. Writes normalisation stats and metadata to dataset/file attributes.

Usage
-----
  # Single file
  python prepare_scenario3_dataset.py \\
      --input  /path/to/turbosens1_train_400_sensors.h5 \\
      --output /path/to/turbosens1_train.h5

  # All three splits in one shot (recommended — shares train-set stats)
  python prepare_scenario3_dataset.py \\
      --all_splits \\
      --data_dir /lustre/fswork/projects/rech/yil/ugy35qd/thesis/rl_opendeck_simulator/data

Output files (--all_splits):
  turbosens1_train.h5
  turbosens1_test.h5
  turbosens1_test_hard.h5

After running, point the training config at the lewm file:
  python train.py data=turbosens1 n_sensors=84 \\
      data.dataset.cache_dir=/path/to/data

Notes
-----
- The base turbosens1_*.h5 files have STUB ZEROS in observation/sensors.
  Always use the *_sensors.h5 variants as input.
- Normalisation is computed from the TRAIN file and applied to all splits,
  so test values are on the same scale (out-of-range values are clipped to
  [0, 1] with a warning).
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ── metadata preservation ─────────────────────────────────────────────────────
#
# Source files carry rich metadata that is valuable for downstream analysis
# (archetype labels, events, weather, speed regimes).  These fields are
# copied through unchanged on every conversion.  See patch_scenario3_metadata.py
# for the list and rationale; this script uses the same set so both paths
# produce identical lewm files.

EP_META_KEYS  = ["archetype", "archetype_onset", "eol_triggered", "success_coeff"]
PER_STEP_KEYS = ["event_mask", "event_types", "speed_strategy"]
WEATHER_KEYS  = ["weather/dtamb", "weather/rain"]

PROPAGATED_ATTRS = [
    "archetype_names", "context_names", "event_names", "sensor_names",
    "scenario", "split", "n_episodes", "n_timesteps",
]


def _copy_key(src: h5py.File, dst: h5py.File, key: str, compress: bool) -> None:
    """Copy one dataset from src to dst, preserving dtype. Safe no-op if absent."""
    if key not in src:
        return
    if key in dst:
        return
    data = src[key][:]
    kwargs = {"data": data}
    if compress and data.nbytes > 1024 * 1024:
        kwargs["compression"] = "lzf"
        chunk0 = min(4096, data.shape[0])
        kwargs["chunks"] = (chunk0,) + data.shape[1:]
    dst.create_dataset(key, **kwargs)


# ── helpers ───────────────────────────────────────────────────────────────────

def check_sensors_not_stub(path: Path) -> None:
    """Warn loudly if observation/sensors looks like stub zeros."""
    with h5py.File(path, "r") as f:
        sample = f["observation/sensors"][:1000]
    if sample.std() < 1e-6:
        logging.error(
            f"\n{'='*70}\n"
            f"  STUB SENSORS DETECTED in {path.name}!\n"
            f"  observation/sensors is all zeros — you probably passed the\n"
            f"  base file instead of the *_sensors.h5 variant.\n"
            f"  Run fill_sensors_scenario3.py first, or use the correct file.\n"
            f"{'='*70}"
        )
        sys.exit(1)
    logging.info(
        f"  Sensor check OK — std={sample.std():.4f}  "
        f"range=[{sample.min():.4f}, {sample.max():.4f}]"
    )


def compute_norm_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel min / range over the entire file for [0,1] scaling.

    Returns ch_min and ch_range, each shape (7, 12).
    """
    logging.info(f"  Computing normalisation stats from {path.name} ...")
    with h5py.File(path, "r") as f:
        sensors = f["observation/sensors"][:]   # (N, 7, 12)

    ch_min   = sensors.min(axis=0)             # (7, 12)
    ch_max   = sensors.max(axis=0)             # (7, 12)
    ch_range = ch_max - ch_min
    # Guard against constant channels (shouldn't happen with real sensor data)
    n_const = (ch_range == 0).sum()
    if n_const > 0:
        logging.warning(f"  {n_const} constant sensor channels — setting range=1")
        ch_range = np.where(ch_range == 0, 1.0, ch_range)

    logging.info(
        f"  Stats: min range={ch_range.min():.4f}  max range={ch_range.max():.4f}"
    )
    return ch_min.astype(np.float32), ch_range.astype(np.float32)


# ── main conversion ───────────────────────────────────────────────────────────

def prepare_one(
    input_path: Path,
    output_path: Path,
    ch_min: np.ndarray,       # (7, 12)  training-set normalisation stats
    ch_range: np.ndarray,     # (7, 12)
) -> None:
    logging.info(f"\nProcessing  {input_path.name}")
    logging.info(f"         →  {output_path}")

    check_sensors_not_stub(input_path)

    with h5py.File(input_path, "r") as src:
        N = src["observation/sensors"].shape[0]
        logging.info(f"  Timesteps  : {N:,}")

        # ── Sensors: normalise to [0, 1] ─────────────────────────────────
        sensors = src["observation/sensors"][:].astype(np.float32)  # (N, 7, 12)
        pixels  = (sensors - ch_min) / ch_range                     # (N, 7, 12)

        n_clipped = ((pixels < 0) | (pixels > 1)).sum()
        if n_clipped > 0:
            logging.warning(
                f"  {n_clipped} sensor values outside training range — clipping to [0, 1]"
            )
            pixels = np.clip(pixels, 0.0, 1.0)
        logging.info(
            f"  pixels     : {pixels.shape}  "
            f"[{pixels.min():.4f}, {pixels.max():.4f}]"
        )

        # ── Action: int8 {0..4} → float32 (N, 1) ────────────────────────
        action = src["action"][:].astype(np.float32)   # already (N, 1)
        if action.ndim == 1:
            action = action[:, np.newaxis]
        unique_actions = np.unique(action).astype(int).tolist()
        logging.info(f"  action     : {action.shape}  unique={unique_actions}")

        # ── HI ground truth ───────────────────────────────────────────────
        hi = src["observation/state"][:].astype(np.float32)   # (N, 10)
        logging.info(f"  HI state   : {hi.shape}")

        # ── Episode metadata ──────────────────────────────────────────────
        ep_len    = src["ep_len"][:].astype(np.int64)    # (E,)
        ep_offset = src["ep_offset"][:].astype(np.int64) # (E,)
        logging.info(
            f"  Episodes   : {len(ep_len):,}   "
            f"ep_len={ep_len.min()}-{ep_len.max()}   "
            f"total={ep_len.sum():,}"
        )

        # ── Collect file-level attrs ──────────────────────────────────────
        action_names  = list(src.attrs.get("action_names", []))
        sensor_names  = list(src.attrs.get("sensor_names", []))
        context_names = list(src.attrs.get("context_names", []))

        # Snapshot the source object for metadata copy below (re-opened outside
        # this `with` block, so we capture the path now).
        src_path_for_meta = input_path

    n_hi = hi.shape[1]
    state_label_names = [f"HI_{i}" for i in range(n_hi)]

    # ── Write output ──────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as dst:

        # pixels — (N, 7, 12) normalised float32
        # Shape note: SensorEncoder in jepa.py reshapes (B, T, 7, 12) → (B, T, 84)
        # automatically, so no 3-channel replication is needed.
        ds_pix = dst.create_dataset(
            "pixels", data=pixels,
            compression="lzf",
            chunks=(min(4096, N), 7, 12),
        )
        ds_pix.attrs["source_key"]    = "observation/sensors"
        ds_pix.attrs["ch_min"]        = ch_min          # (7, 12) for denormalisation
        ds_pix.attrs["ch_range"]      = ch_range
        ds_pix.attrs["sensor_names"]  = np.array(sensor_names,  dtype=object)
        ds_pix.attrs["context_names"] = np.array(context_names, dtype=object)

        # action — scalar float32, values 0.0–4.0
        dst.create_dataset(
            "action", data=action,
            compression="lzf",
            chunks=(min(4096, N), 1),
        )

        # observation.state — flat key to match old turbofan HDF5 convention
        # (eval_sweep.py reads f["observation.state"][:], not f["observation"]["state"])
        dst.create_dataset(
            "observation.state", data=hi,
            compression="lzf",
            chunks=(min(4096, N), n_hi),
        )

        # Episode metadata
        dst.create_dataset("ep_len",    data=ep_len)
        dst.create_dataset("ep_offset", data=ep_offset)

        # File-level attributes
        dst.attrs["lewm_ready"]        = True
        dst.attrs["state_label_names"] = np.array(state_label_names, dtype=object)
        dst.attrs["action_names"]      = np.array(action_names,      dtype=object)
        dst.attrs["n_sensors"]         = 7
        dst.attrs["n_contexts"]        = 12
        dst.attrs["n_actions"]         = int(max(unique_actions)) + 1
        dst.attrs["source_file"]       = str(input_path)

        # ── Copy rich metadata from source ────────────────────────────────
        # Episode-level (tiny), per-timestep events, speed regimes, weather.
        # Training ignores these; downstream analysis / archetype-stratified
        # eval uses them.  Kept in sync with patch_scenario3_metadata.py.
        with h5py.File(src_path_for_meta, "r") as meta_src:
            for k in EP_META_KEYS:
                _copy_key(meta_src, dst, f"ep_meta/{k}", compress=False)
            for k in PER_STEP_KEYS:
                _copy_key(meta_src, dst, k, compress=True)
            for k in WEATHER_KEYS:
                _copy_key(meta_src, dst, k, compress=True)
            for a in PROPAGATED_ATTRS:
                if a in meta_src.attrs and a not in dst.attrs:
                    dst.attrs[a] = meta_src.attrs[a]

    logging.info(f"  Written ✓  {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--all_splits", action="store_true",
        help="Prepare train, test, and test_hard splits from --data_dir."
    )
    mode.add_argument(
        "--input", type=Path, metavar="FILE",
        help="Single input *_sensors.h5 file."
    )
    parser.add_argument(
        "--output", type=Path, metavar="FILE",
        help="Output *_lewm.h5 path (required with --input)."
    )
    parser.add_argument(
        "--data_dir", type=Path,
        default=Path(
            "/lustre/fswork/projects/rech/yil/ugy35qd"
            "/thesis/rl_opendeck_simulator/data"
        ),
        help="Directory containing the raw *_sensors.h5 source files.",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None,
        help=(
            "Directory where *_lewm.h5 output files are written "
            "(default: same as --data_dir).  "
            "Set to $STABLEWM_HOME to co-locate with checkpoints."
        ),
    )
    parser.add_argument(
        "--train_file", default="turbosens1_train_400_sensors.h5",
        help="Training file name inside --data_dir (used for norm stats).",
    )
    args = parser.parse_args()

    if args.all_splits:
        data_dir   = args.data_dir
        output_dir = args.output_dir if args.output_dir is not None else data_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        splits = [
            ("turbosens1_train_400_sensors.h5", "turbosens1_train.h5"),
            ("turbosens1_test_sensors.h5",       "turbosens1_test.h5"),
            ("turbosens1_test_hard_sensors.h5",  "turbosens1_test_hard.h5"),
        ]
        # Compute normalisation stats once from the training file only
        train_path = data_dir / args.train_file
        if not train_path.exists():
            logging.error(f"Training file not found: {train_path}")
            sys.exit(1)
        ch_min, ch_range = compute_norm_stats(train_path)

        for inp_name, out_name in splits:
            inp = data_dir   / inp_name
            out = output_dir / out_name
            if not inp.exists():
                logging.warning(f"  Skipping (not found): {inp}")
                continue
            prepare_one(inp, out, ch_min, ch_range)

    else:
        if args.output is None:
            parser.error("--output is required when using --input")
        # Compute stats from the input file itself (single-file mode)
        ch_min, ch_range = compute_norm_stats(args.input)
        prepare_one(args.input, args.output, ch_min, ch_range)

    print(
        "\n" + "=" * 60 + "\n"
        "Done. Next steps:\n\n"
        "  # JEPA training on turbosens1:\n"
        "  python train.py \\\n"
        "      data=turbosens1 \\\n"
        "      n_sensors=84 \\\n"
        "      data.dataset.cache_dir=/path/to/data \\\n"
        "      obs_window_size=1 \\\n"
        "      wm.history_size=50\n\n"
        "  # Or just submit the SLURM script:\n"
        "  sbatch train_lewm_scenario3.slurm\n"
        + "=" * 60
    )


if __name__ == "__main__":
    main()
