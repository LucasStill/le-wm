#!/usr/bin/env python
"""
prepare_scenario4_dataset.py
============================
Convert scenario4 *_sensors.h5 files into pipeline-ready *_lewm.h5 files
for le-wm training.

What this script does
---------------------
  1. Reads ``observation/sensors``   (N, 7, 16)  from the source file.
  2. Reads ``context_params``        (N, 16, 4) — [dtamb, alt, mach, cmd].
  3. Stacks them along the channel axis into ``pixels`` (N, 11, 16),
     so each of the 16 contexts carries its 7 sensors + 4 operating-point
     values side-by-side.  The SensorEncoder flattens (B, T, 11, 16)
     to (B, T, 176) tokens automatically.
  4. Normalises each of the 11 channels × 16 contexts to [0, 1] using
     training-set min / range statistics (applied to all splits).
  5. Copies ``action_idx``  as (N, 1) float32, integer values 0–6.
     (0=do_nothing, 1=fan, 2=hpc, 3=turbine, 4=full, 5=patch, 6=wash)
  6. Copies ``observation/state``    → ``observation.state``  (N, 10).
  7. Copies ``ep_len`` and ``ep_offset`` as int64.
  8. Writes rich metadata (archetype, events, weather, repaired_mask).

Usage
-----
  # All three splits in one shot (recommended — shares train-set stats)
  python prepare_scenario4_dataset.py \\
      --all_splits \\
      --data_dir /path/to/rl_opendeck_simulator/data

  # Single file
  python prepare_scenario4_dataset.py \\
      --input  /path/to/scenario4_train_sensors.h5 \\
      --output /path/to/scenario4_train_lewm.h5

Output files (--all_splits):
  scenario4_train_lewm.h5
  scenario4_test_lewm.h5
  scenario4_test_hard_lewm.h5

After running, point the training config at the lewm file:
  python train.py data=scenario4 n_sensors=176 \\
      sensor_encoder.max_sensors=200 \\
      data.dataset.cache_dir=/path/to/data

Notes
-----
- Input MUST be the *_sensors.h5 variants (base *.h5 have stub zeros in
  observation/sensors).
- Normalisation is computed from the TRAIN file and applied to all splits,
  so test values are on the same scale (out-of-range values clipped with
  a warning).
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


EP_META_KEYS  = ["archetype", "archetype_onset", "eol_triggered",
                 "region", "flights_per_day", "is_extreme"]
PER_STEP_KEYS = ["event_mask", "event_types", "repaired_flight_mask",
                 "valid_flight_mask", "visit_type", "component_service_mask"]
WEATHER_KEYS  = ["weather/dtamb", "weather/summer_factor",
                 "weather/calendar_months"]

PROPAGATED_ATTRS = [
    "archetype_names", "context_names", "context_phases", "event_names",
    "sensor_names", "region_names", "action_names",
    "scenario", "split", "n_episodes", "n_timesteps",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _copy_key(src: h5py.File, dst: h5py.File, key: str, compress: bool) -> None:
    if key not in src or key in dst:
        return
    data = src[key][:]
    kwargs = {"data": data}
    if compress and data.nbytes > 1024 * 1024:
        kwargs["compression"] = "lzf"
        chunk0 = min(4096, data.shape[0])
        kwargs["chunks"] = (chunk0,) + data.shape[1:]
    dst.create_dataset(key, **kwargs)


def check_sensors_not_stub(path: Path) -> None:
    with h5py.File(path, "r") as f:
        sample = f["observation/sensors"][:1000]
    if sample.std() < 1e-6:
        logging.error(
            f"\n{'='*70}\n"
            f"  STUB SENSORS DETECTED in {path.name}!\n"
            f"  Pass the *_sensors.h5 variant, not the base file.\n"
            f"{'='*70}"
        )
        sys.exit(1)
    logging.info(
        f"  Sensor check OK — std={sample.std():.4f}  "
        f"range=[{sample.min():.4f}, {sample.max():.4f}]"
    )


def _build_stacked(sensors: np.ndarray, ctx: np.ndarray) -> np.ndarray:
    """sensors (N,7,16) + ctx (N,16,4) → stacked (N,11,16)."""
    ctx_t = np.transpose(ctx, (0, 2, 1))  # (N, 4, 16)
    return np.concatenate([sensors, ctx_t], axis=1).astype(np.float32)


def compute_norm_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel min / range over the entire file. Shapes: (11, 16)."""
    logging.info(f"  Computing normalisation stats from {path.name} ...")
    with h5py.File(path, "r") as f:
        sensors = f["observation/sensors"][:]    # (N, 7, 16)
        ctx     = f["context_params"][:]         # (N, 16, 4)
    stacked  = _build_stacked(sensors, ctx)      # (N, 11, 16)
    ch_min   = stacked.min(axis=0)
    ch_max   = stacked.max(axis=0)
    ch_range = ch_max - ch_min
    n_const  = (ch_range == 0).sum()
    if n_const > 0:
        logging.warning(f"  {n_const} constant channels — setting range=1")
        ch_range = np.where(ch_range == 0, 1.0, ch_range)
    logging.info(
        f"  Stats: min range={ch_range.min():.4f}  max range={ch_range.max():.4f}"
    )
    return ch_min.astype(np.float32), ch_range.astype(np.float32)


# ── main conversion ───────────────────────────────────────────────────────────

def prepare_one(
    input_path: Path,
    output_path: Path,
    ch_min: np.ndarray,       # (11, 16)
    ch_range: np.ndarray,     # (11, 16)
) -> None:
    logging.info(f"\nProcessing  {input_path.name}")
    logging.info(f"         →  {output_path}")

    check_sensors_not_stub(input_path)

    with h5py.File(input_path, "r") as src:
        N = src["observation/sensors"].shape[0]
        logging.info(f"  Timesteps  : {N:,}")

        sensors = src["observation/sensors"][:].astype(np.float32)
        ctx     = src["context_params"][:].astype(np.float32)
        stacked = _build_stacked(sensors, ctx)             # (N, 11, 16)
        pixels  = (stacked - ch_min) / ch_range

        n_clipped = ((pixels < 0) | (pixels > 1)).sum()
        if n_clipped > 0:
            logging.warning(
                f"  {n_clipped} values outside training range — clipping to [0, 1]"
            )
            pixels = np.clip(pixels, 0.0, 1.0)
        logging.info(
            f"  pixels     : {pixels.shape}  "
            f"[{pixels.min():.4f}, {pixels.max():.4f}]"
        )

        action = src["action_idx"][:].astype(np.float32)
        if action.ndim == 1:
            action = action[:, np.newaxis]
        unique_actions = np.unique(action).astype(int).tolist()
        logging.info(f"  action     : {action.shape}  unique={unique_actions}")

        hi = src["observation/state"][:].astype(np.float32)   # (N, 10)
        logging.info(f"  HI state   : {hi.shape}")

        ep_len    = src["ep_len"][:].astype(np.int64)
        ep_offset = src["ep_offset"][:].astype(np.int64)
        logging.info(
            f"  Episodes   : {len(ep_len):,}   "
            f"ep_len={ep_len.min()}-{ep_len.max()}   "
            f"total={ep_len.sum():,}"
        )

        sensor_names  = list(src.attrs.get("sensor_names",  []))
        context_names = list(src.attrs.get("context_names", []))
        action_names  = list(src.attrs.get("action_names",  []))
        src_path_for_meta = input_path

    n_hi = hi.shape[1]
    state_label_names = [f"HI_{i}" for i in range(n_hi)]
    channel_names = list(sensor_names) + ["dtamb", "alt", "mach", "cmd"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as dst:

        ds_pix = dst.create_dataset(
            "pixels", data=pixels,
            compression="lzf",
            chunks=(min(4096, N), 11, 16),
        )
        ds_pix.attrs["source_key"]     = "observation/sensors + context_params"
        ds_pix.attrs["ch_min"]         = ch_min
        ds_pix.attrs["ch_range"]       = ch_range
        ds_pix.attrs["channel_names"]  = np.array(channel_names,  dtype=object)
        ds_pix.attrs["context_names"]  = np.array(context_names,  dtype=object)

        dst.create_dataset("action", data=action,
                           compression="lzf",
                           chunks=(min(4096, N), 1))

        dst.create_dataset("observation.state", data=hi,
                           compression="lzf",
                           chunks=(min(4096, N), n_hi))

        dst.create_dataset("ep_len",    data=ep_len)
        dst.create_dataset("ep_offset", data=ep_offset)

        dst.attrs["lewm_ready"]        = True
        dst.attrs["state_label_names"] = np.array(state_label_names, dtype=object)
        dst.attrs["action_names"]      = np.array(action_names,      dtype=object)
        dst.attrs["n_channels"]        = 11
        dst.attrs["n_contexts"]        = 16
        dst.attrs["n_sensors_flat"]    = 11 * 16          # = 176 tokens
        dst.attrs["n_actions"]         = int(max(unique_actions)) + 1
        dst.attrs["source_file"]       = str(input_path)

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
    mode.add_argument("--input",  type=Path, help="Single input _sensors.h5")
    mode.add_argument("--all_splits", action="store_true",
                      help="Process train / test / test_hard from --data_dir")
    parser.add_argument("--output", type=Path, help="Output path for --input mode")
    parser.add_argument("--data_dir", type=Path,
                        help="Directory containing scenario4_{train,test,test_hard}_sensors.h5")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Output directory for --all_splits (default = data_dir)")
    args = parser.parse_args()

    if args.all_splits:
        if args.data_dir is None:
            parser.error("--data_dir is required with --all_splits")
        out_dir = args.out_dir or args.data_dir
        train_in  = args.data_dir / "scenario4_train_sensors.h5"
        test_in   = args.data_dir / "scenario4_test_sensors.h5"
        hard_in   = args.data_dir / "scenario4_test_hard_sensors.h5"

        ch_min, ch_range = compute_norm_stats(train_in)

        prepare_one(train_in, out_dir / "scenario4_train_lewm.h5",     ch_min, ch_range)
        prepare_one(test_in,  out_dir / "scenario4_test_lewm.h5",      ch_min, ch_range)
        prepare_one(hard_in,  out_dir / "scenario4_test_hard_lewm.h5", ch_min, ch_range)
    else:
        if args.output is None:
            parser.error("--output is required with --input")
        ch_min, ch_range = compute_norm_stats(args.input)
        prepare_one(args.input, args.output, ch_min, ch_range)


if __name__ == "__main__":
    main()
