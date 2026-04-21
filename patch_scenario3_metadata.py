#!/usr/bin/env python
"""
patch_scenario3_metadata.py
===========================
Adds missing metadata from a scenario3 ``*_sensors.h5`` source file into an
existing ``*_lewm.h5`` file, in place.

Why this exists
---------------
Earlier revisions of ``prepare_scenario3_dataset.py`` only copied the minimal
fields needed for training (``pixels``, ``action``, ``observation.state``,
``ep_len``, ``ep_offset``).  The source simulator files carry much richer
metadata (archetype labels, event flags, weather, speed regimes, ...) which
is useful for downstream analysis and archetype-stratified evaluation.

This script patches an existing lewm file to add the missing metadata
without rebuilding ``pixels`` (which is the expensive part).  It is safe
to run on a file that is already patched — existing keys are left alone.

What it adds
------------
Episode-level (``ep_meta/``):
    archetype          (E,)   int8
    archetype_onset    (E,)   int32
    eol_triggered      (E,)   bool
    success_coeff      (E,)   float32

Per-timestep (all LZF-compressed):
    event_mask         (N,)       bool
    event_types        (N, 10)    uint8
    speed_strategy     (N, 10)    uint8   (slow/normal/fast regimes per HI)
    weather/dtamb      (N,)       float32
    weather/rain       (N,)       bool

Root attributes:
    archetype_names, context_names, event_names, sensor_names,
    scenario, split, n_episodes, n_timesteps, metadata_patched

Usage
-----
  # Single file
  python patch_scenario3_metadata.py \\
      --source /path/to/scenario3_train_400_sensors.h5 \\
      --target /path/to/scenario3_train_400_lewm.h5

  # All three splits (source and target files must be co-located and follow
  # the standard naming convention)
  python patch_scenario3_metadata.py \\
      --all_splits \\
      --source_dir ~/.stable_worldmodel \\
      --target_dir ~/.stable_worldmodel
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# Datasets/groups to copy (key → whether it's per-timestep)
EP_META_KEYS = ["archetype", "archetype_onset", "eol_triggered", "success_coeff"]
PER_STEP_KEYS = ["event_mask", "event_types", "speed_strategy"]
WEATHER_KEYS  = ["weather/dtamb", "weather/rain"]

ROOT_ATTRS = [
    "archetype_names", "context_names", "event_names", "sensor_names",
    "scenario", "split", "n_episodes", "n_timesteps",
]


def _sizes_match(src: h5py.File, dst: h5py.File) -> bool:
    """Verify episode count + timestep count match before patching."""
    src_E = int(src["ep_len"].shape[0])
    dst_E = int(dst["ep_len"].shape[0])
    if src_E != dst_E:
        logging.error(f"  Episode-count mismatch: src={src_E} dst={dst_E}")
        return False
    src_N = int(src["observation/sensors"].shape[0])
    dst_N = int(dst["pixels"].shape[0])
    if src_N != dst_N:
        logging.error(f"  Timestep-count mismatch: src={src_N} dst={dst_N}")
        return False
    logging.info(f"  Sizes match: episodes={src_E}, timesteps={src_N:,}")
    return True


def _copy_if_missing(src: h5py.File, dst: h5py.File, key: str, compress: bool):
    if key in dst:
        logging.info(f"    skip  {key}  (already present)")
        return
    if key not in src:
        logging.info(f"    skip  {key}  (not in source)")
        return
    data = src[key][:]
    kwargs = {"data": data}
    if compress and data.nbytes > 1024 * 1024:   # compress only non-trivial arrays
        kwargs["compression"] = "lzf"
        # chunk on the leading axis
        chunk0 = min(4096, data.shape[0])
        kwargs["chunks"] = (chunk0,) + data.shape[1:]
    dst.create_dataset(key, **kwargs)
    logging.info(f"    add   {key}  shape={data.shape}  dtype={data.dtype}")


def _copy_attrs(src: h5py.File, dst: h5py.File):
    for a in ROOT_ATTRS:
        if a in src.attrs and a not in dst.attrs:
            v = src.attrs[a]
            # Preserve numpy string arrays as-is (h5py handles this)
            dst.attrs[a] = v
            v_str = str(v)
            if len(v_str) > 80:
                v_str = v_str[:80] + "..."
            logging.info(f"    add attr  {a}: {v_str}")


def patch_one(source: Path, target: Path) -> None:
    logging.info(f"\nPatching  {target.name}")
    logging.info(f"    from  {source.name}")
    if not source.exists():
        logging.error(f"  Source file not found: {source}")
        return
    if not target.exists():
        logging.error(f"  Target file not found: {target}")
        return

    with h5py.File(source, "r") as src, h5py.File(target, "r+") as dst:
        if not _sizes_match(src, dst):
            logging.error("  Refusing to patch — aborting.")
            return

        # ── Episode metadata ──────────────────────────────────────────────
        for key in EP_META_KEYS:
            _copy_if_missing(src, dst, f"ep_meta/{key}", compress=False)

        # ── Per-timestep arrays (large → compress) ────────────────────────
        for key in PER_STEP_KEYS:
            _copy_if_missing(src, dst, key, compress=True)

        # ── Weather group ─────────────────────────────────────────────────
        for key in WEATHER_KEYS:
            _copy_if_missing(src, dst, key, compress=True)

        # ── Root attributes ───────────────────────────────────────────────
        _copy_attrs(src, dst)

        dst.attrs["metadata_patched"] = True

    # Report final size
    size_mb = target.stat().st_size / (1024 * 1024)
    logging.info(f"  Done.  File size now: {size_mb:.1f} MB")


# ── CLI ───────────────────────────────────────────────────────────────────────

SPLIT_PAIRS = [
    ("scenario3_train_400_sensors.h5", "scenario3_train_400_lewm.h5"),
    ("scenario3_test_sensors.h5",      "scenario3_test_lewm.h5"),
    ("scenario3_test_hard_sensors.h5", "scenario3_test_hard_lewm.h5"),
]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all_splits", action="store_true")
    mode.add_argument("--source", type=Path, metavar="FILE",
                      help="Single source *_sensors.h5 file")
    parser.add_argument("--target", type=Path, metavar="FILE",
                        help="Target *_lewm.h5 (required with --source)")
    parser.add_argument("--source_dir", type=Path,
                        default=Path.home() / ".stable_worldmodel")
    parser.add_argument("--target_dir", type=Path,
                        default=Path.home() / ".stable_worldmodel")
    args = parser.parse_args()

    if args.all_splits:
        for src_name, tgt_name in SPLIT_PAIRS:
            src = args.source_dir / src_name
            tgt = args.target_dir / tgt_name
            if not src.exists():
                logging.warning(f"Skip (no source): {src}")
                continue
            if not tgt.exists():
                logging.warning(f"Skip (no target): {tgt}")
                continue
            patch_one(src, tgt)
    else:
        if args.target is None:
            parser.error("--target is required when using --source")
        patch_one(args.source, args.target)

    print("\nDone.")


if __name__ == "__main__":
    main()
