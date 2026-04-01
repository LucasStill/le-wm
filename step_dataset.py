"""
TemporalMetadataWrapper — add episode_idx and episode_pos to any le-wm dataset.

The wrapped dataset's __getitem__ already knows the (ep_idx, start) of every
window because it is stored in `clip_indices`.  We pre-compute the normalised
position for all windows at construction time so there is zero overhead at
training time.

Normalised position
-------------------
Given:
    start  — window start frame (raw integer index within the episode)
    ep_len — episode length (number of frames)
    span   — window span in frames = num_steps * frameskip

The last valid start for any window is (ep_len - span), so:

    episode_pos = start / max(ep_len - span, 1)  ∈ [0, 1]

This maps every episode to the same [0, 1] interval regardless of how long
the episode is — the canonical solution to the variable-length problem.

Example
-------
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    ...
    dataset = TemporalMetadataWrapper(dataset)

    # Each batch item now contains:
    #   item['episode_idx']  — torch.int64 scalar
    #   item['episode_pos']  — torch.float32 scalar in [0, 1]
"""

import numpy as np
import torch


class TemporalMetadataWrapper:
    """Wraps a stable_worldmodel Dataset to add temporal metadata to each batch.

    Args:
        dataset: Any stable_worldmodel Dataset instance (HDF5Dataset,
                 FolderDataset, GoalDataset, etc.).  Must expose the standard
                 `clip_indices` and `lengths` attributes.
    """

    def __init__(self, dataset) -> None:
        self.dataset = dataset

        lengths = dataset.lengths   # (N_episodes,) array of episode lengths
        span    = dataset.span      # num_steps * frameskip

        # Pre-compute normalised positions for every clip index.
        # clip_indices: list of (ep_idx, start) tuples.
        ep_indices = np.empty(len(dataset), dtype=np.int64)
        ep_positions = np.empty(len(dataset), dtype=np.float32)

        for i, (ep_idx, start) in enumerate(dataset.clip_indices):
            ep_len  = int(lengths[ep_idx])
            max_start = max(ep_len - span, 1)     # max valid start for this episode
            ep_indices[i]   = ep_idx
            ep_positions[i] = float(start) / max_start   # normalised ∈ [0, 1]

        # Store as tensors for direct use in __getitem__.
        self._ep_indices  = torch.from_numpy(ep_indices)
        self._ep_positions = torch.from_numpy(ep_positions)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        item["episode_idx"] = self._ep_indices[idx]    # int64 scalar
        item["episode_pos"] = self._ep_positions[idx]  # float32 scalar in [0, 1]
        return item

    # ------------------------------------------------------------------
    # Proxy everything else to the wrapped dataset so the rest of
    # train.py (get_col_data, get_dim, column_names, …) keeps working.
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        # Only called when the attribute isn't found on self.
        return getattr(self.dataset, name)
