"""Probe callbacks — train Transformer probes every N epochs to evaluate
frozen JEPA encoder embeddings on two downstream tasks:

  1. HI Probe: predict 10 turbofan Health Indicator (HI) states.
  2. Action-RUL Probe: predict steps-to-next-maintenance-action (regression).

Probe architecture (aligned with prior benchmark TransformerRegressor):
  - StandardScaler normalisation of encoder embeddings
  - Linear(input_dim -> d_model) projection, scaled by sqrt(d_model)
  - TransformerEncoder (batch_first, num_layers, nhead, dim_feedforward=4*d_model)
  - Output head: d_model -> d_model//2 -> ReLU -> Dropout -> n_outputs
  - Adam optimiser, early stopping on RMSE (patience configurable)

seq_len config parameter:
  - seq_len=1  : each probe input is a single JEPA embedding (current timestep
                 only).  Self-attention is trivial; functionally an MLP.
                 Random timestep subsampling is used.
  - seq_len>1  : each probe input is a window of seq_len consecutive JEPA
                 embeddings from the same episode.  The transformer can attend
                 over the trajectory window, learning temporal degradation
                 trends.  Whole episodes are gathered to preserve continuity;
                 sliding windows are built post-encoding within episode
                 boundaries.

Use seq_len=1 and seq_len=10 as two comparative configurations:
  hi_probe.probe_seq_len=1   -> single-snapshot probe (baseline)
  hi_probe.probe_seq_len=10  -> 10-step window probe (trajectory-aware)

HI metrics logged per dimension (wandb key: hi_probe/sl{N}/<component>/<metric>):
  r2, rmse, pearson_r

Action-RUL metrics (wandb key: hi_probe/sl{N}/rul/<metric>):
  rmse (steps), mae (steps), r2, pearson_r

Aggregates:
  hi_probe/sl{N}/mean_r2, hi_probe/sl{N}/mean_rmse, hi_probe/sl{N}/mean_pearson_r

CSV fallback (always written, wandb-independent):
  {trainer.log_dir}/hi_probe_metrics.csv
  columns: epoch, global_step, seq_len, task, component, r2, rmse, pearson_r
"""

import copy
import csv
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


# ── Probe model ────────────────────────────────────────────────────────────────

class _TransformerProbe(nn.Module):
    """Transformer regressor.

    Input shape : (batch, seq_len, input_dim)
    Output shape: (batch, n_outputs)

    With seq_len=1 the self-attention is trivial (single token), making this
    functionally equivalent to a 2-layer MLP — useful for comparability with
    prior work.  With seq_len>1 the transformer attends over the temporal
    window of embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        n_outputs: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, input_dim)
        x = self.input_projection(x) * (self.d_model ** 0.5)
        x = self.transformer(x)
        return self.output_layer(x[:, -1, :])   # last token → (B, n_outputs)


# ── Callback ───────────────────────────────────────────────────────────────────

class HIProbeCallback(pl.Callback):
    """Every ``eval_interval`` validation epochs:

    1. Encodes observations with the frozen JEPA encoder.
    2. Normalises embeddings with StandardScaler (fit on train split).
    3. Builds temporal windows if probe_seq_len > 1.
    4. Trains a TransformerProbe with early stopping to predict 10 HI states.
    5. Reports per-component R², RMSE, Pearson-r to wandb + CSV + console.

    Parameters
    ----------
    data_path : str
        Path to the le-wm HDF5 file (needs 'pixels' and 'observation.state').
    img_size : int
        Must match ``img_size`` used during JEPA training.
    train_split, seed : float, int
        Must match le-wm training split so the test set is held out.
    eval_interval : int
        Run probe every this many epochs (default 5).
    n_probe_epochs : int
        Max training epochs for the probe (default 50; early stopping applies).
    probe_lr : float
        Adam learning rate for the probe (default 1e-3).
    probe_patience : int
        Early-stopping patience on mean per-dim RMSE (default 10).
    d_model : int
        Transformer hidden dimension (default 64).
    nhead : int
        Number of attention heads (default 4, requires d_model % nhead == 0).
    num_layers : int
        Number of TransformerEncoder layers (default 2).
    probe_dropout : float
        Dropout inside the transformer and output head (default 0.1).
    probe_batch_size : int
        Batch size for probe training (default 256).
    probe_seq_len : int
        Number of consecutive JEPA embeddings per probe input (default 1).
        seq_len=1: single-snapshot probe (random timestep subsampling).
        seq_len>1: trajectory window probe (whole-episode gathering + sliding
                   windows within episode boundaries).
    n_subsample : int
        Max timesteps to load per split (default 30 000).
        For seq_len>1, whole episodes are gathered up to this limit.
    enc_batch_size : int
        Batch size for JEPA encoding pass (default 2048).
    rul_max_horizon : int
        Cap on steps-to-next-action for the Action-RUL probe (default 300).
        Timesteps beyond this horizon are clipped to rul_max_horizon.
        Set to 0 to disable the RUL probe entirely.
    obs_window_size : int
        Encoder input window (default 1).  When > 1, the JEPA encoder
        aggregates w consecutive observations into each embedding.  The
        probe must build w-frame windows before encoding and skip the
        first w-1 timesteps of each episode (no full window available).
    """

    def __init__(
        self,
        data_path: str,
        img_size: int = 28,
        train_split: float = 0.9,
        seed: int = 3072,
        eval_interval: int = 5,
        n_probe_epochs: int = 50,
        probe_lr: float = 1e-3,
        probe_patience: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        probe_dropout: float = 0.1,
        probe_batch_size: int = 256,
        probe_seq_len: int = 1,
        n_subsample: int = 30_000,
        enc_batch_size: int = 2048,
        rul_max_horizon: int = 300,
        obs_window_size: int = 1,
        encoder_type: str = "vit",
        # legacy alias kept for config backward compat (ignored)
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.data_path        = Path(data_path)
        self.img_size         = img_size
        self.train_split      = train_split
        self.seed             = seed
        self.eval_interval    = eval_interval
        self.n_probe_epochs   = n_probe_epochs
        self.probe_lr         = probe_lr
        self.probe_patience   = probe_patience
        self.d_model          = d_model
        self.nhead            = nhead
        self.num_layers       = num_layers
        self.probe_dropout    = probe_dropout
        self.probe_batch_size = probe_batch_size
        self.probe_seq_len    = probe_seq_len
        self.n_subsample      = n_subsample
        self.enc_batch_size   = enc_batch_size
        self.rul_max_horizon  = rul_max_horizon
        self.obs_window_size  = obs_window_size
        self.encoder_type     = encoder_type

        # Filled in setup()
        self._obs_tr:    np.ndarray | None = None
        self._obs_te:    np.ndarray | None = None
        self._hi_tr:     np.ndarray | None = None
        self._hi_te:     np.ndarray | None = None
        self._rul_tr:    np.ndarray | None = None   # (N,) steps-to-next-action
        self._rul_te:    np.ndarray | None = None
        self._ep_ids_tr: np.ndarray | None = None   # episode ID per timestep
        self._ep_ids_te: np.ndarray | None = None
        self._hi_names:  list[str] | None  = None
        self._csv_path:  Path | None       = None

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _shorten(name: str) -> str:
        """deg_CmpBst_s_mapEff_in  →  Bst/Eff"""
        n = name.replace("deg_Cmp", "").replace("deg_Trb", "Trb")
        n = n.replace("_s_mapEff_in", "/Eff").replace("_s_mapWc_in", "/Wc")
        return n

    @staticmethod
    def _build_windows(
        X: np.ndarray,          # (N, D) embeddings
        y: np.ndarray,          # (N, K) targets
        ep_ids: np.ndarray,     # (N,) episode index per timestep
        seq_len: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build sliding windows of length seq_len within episode boundaries.

        Returns
        -------
        X_win : (M, seq_len, D)
        y_win : (M, K)   — target is the label at the last timestep of each window
        """
        if seq_len == 1:
            return X[:, np.newaxis, :], y   # (N, 1, D), (N, K)

        windows, targets = [], []
        for ep_id in np.unique(ep_ids):
            mask = ep_ids == ep_id
            X_ep = X[mask]          # (T, D)
            y_ep = y[mask]          # (T, K)
            for i in range(seq_len - 1, len(X_ep)):
                windows.append(X_ep[i - seq_len + 1 : i + 1])
                targets.append(y_ep[i])

        if not windows:
            # Fallback: no episode long enough — return single-step
            logging.warning(
                "[HIProbe] No episodes long enough for seq_len=%d; "
                "falling back to seq_len=1", seq_len
            )
            return X[:, np.newaxis, :], y

        return np.array(windows, dtype=np.float32), np.array(targets, dtype=np.float32)

    @staticmethod
    def _compute_rul(
        actions: np.ndarray,    # (N_total,) float — 1.0 = maintenance event
        ep_offset: np.ndarray,  # (E,) int start index of each episode
        ep_len: np.ndarray,     # (E,) int length of each episode
        max_horizon: int,       # cap (timesteps)
    ) -> np.ndarray:
        """Compute steps-to-next-maintenance-action for every timestep.

        At each timestep t inside an episode, the target is:
          rul[t] = min(steps until next action, max_horizon)

        If no future action exists in the episode, the target is:
          rul[t] = min(steps until end of episode, max_horizon)

        Returns
        -------
        rul : (N_total,) float32
        """
        rul = np.full(len(actions), max_horizon, dtype=np.float32)
        for offset, length in zip(ep_offset, ep_len):
            ep_actions = actions[offset : offset + length]
            # Normalise to binary — handle both float and int encodings
            action_times = np.where(ep_actions > 0.5)[0]
            t_arr = np.arange(length, dtype=np.int32)
            if len(action_times) > 0:
                # searchsorted finds the first action at or after t+1 (i.e., strictly future)
                next_idx = np.searchsorted(action_times, t_arr + 1)
                has_future = next_idx < len(action_times)
                safe_idx   = np.minimum(next_idx, len(action_times) - 1)
                rul_ep = np.where(
                    has_future,
                    np.minimum(action_times[safe_idx] - t_arr, max_horizon),
                    np.minimum(length - t_arr, max_horizon),
                ).astype(np.float32)
            else:
                rul_ep = np.minimum(length - t_arr, max_horizon).astype(np.float32)
            rul[offset : offset + length] = rul_ep
        return rul

    # ── setup ──────────────────────────────────────────────────────────────

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        if self._obs_tr is not None:
            return

        logging.info(
            f"[HIProbe] Loading dataset from {self.data_path}  "
            f"(seq_len={self.probe_seq_len}, obs_window={self.obs_window_size})"
        )
        with h5py.File(self.data_path, "r") as f:
            ep_len    = f["ep_len"][:]
            ep_offset = f["ep_offset"][:]
            # Load observations.
            # The HDF5 stores sensor data under 'pixels' (not 'observation.sensors')
            # regardless of encoder_type.  We flatten to (N, n_sensors) for sensor enc.
            if "pixels" in f:
                obs_key = "pixels"
            elif "observation.sensors" in f:
                obs_key = "observation.sensors"
            else:
                raise KeyError(
                    "HDF5 file has neither 'pixels' nor 'observation.sensors' key."
                )
            pixels = f[obs_key][:]
            # For sensor encoder: flatten spatial dims → (N, n_sensors)
            if self.encoder_type == "sensor" and pixels.ndim > 2:
                pixels = pixels.reshape(len(pixels), -1)
            logging.info(f"[HIProbe] Loading observations from key '{obs_key}' "
                         f"(encoder_type={self.encoder_type}, shape={pixels.shape})")
            states    = f["observation.state"][:]
            self._hi_names = list(
                f.attrs.get("state_label_names",
                            [f"HI_{i}" for i in range(states.shape[1])])
            )
            # Action array — used to compute steps-to-next-maintenance RUL
            if "action" in f and self.rul_max_horizon > 0:
                raw_actions = f["action"][:]
                if raw_actions.ndim > 1:
                    raw_actions = raw_actions[:, 0]
                rul_full = self._compute_rul(
                    raw_actions, ep_offset, ep_len, self.rul_max_horizon
                )
                n_maint = int((raw_actions > 0.5).sum())
                logging.info(
                    f"[HIProbe] Action-RUL: {n_maint} maintenance events found "
                    f"(max_horizon={self.rul_max_horizon})"
                )
            else:
                rul_full = None
                if self.rul_max_horizon > 0:
                    logging.warning(
                        "[HIProbe] 'action' key not found in HDF5; "
                        "Action-RUL probe disabled."
                    )

        N_EP = len(ep_len)
        rng  = np.random.default_rng(self.seed)
        perm = rng.permutation(N_EP)
        n_tr = int(N_EP * self.train_split)

        tr_eps = perm[:n_tr]
        te_eps = perm[n_tr:]

        if self.probe_seq_len == 1 and self.obs_window_size <= 1:
            # Random timestep subsampling — fast, order-independent.
            # Only valid when both probe_seq_len=1 AND obs_window_size=1
            # (no temporal context needed at either stage).
            def gather_random(eps):
                idx = np.concatenate([
                    np.arange(ep_offset[e], ep_offset[e] + ep_len[e]) for e in eps
                ])
                if len(idx) > self.n_subsample:
                    idx = np.random.default_rng(self.seed + 99).choice(
                        idx, size=self.n_subsample, replace=False
                    )
                ep_ids = np.zeros(len(idx), dtype=np.int32)   # ep id unused for seq_len=1
                rul_sub = rul_full[idx] if rul_full is not None else None
                return pixels[idx], states[idx], ep_ids, rul_sub

            obs_tr, hi_tr, ep_tr, rul_tr = gather_random(tr_eps)
            obs_te, hi_te, ep_te, rul_te = gather_random(te_eps)
        else:
            # Whole-episode gathering to preserve temporal continuity for windows
            def gather_episodes(eps):
                rng2 = np.random.default_rng(self.seed + 99)
                order = rng2.permutation(len(eps))
                all_obs, all_states, all_ep_ids, all_rul = [], [], [], []
                total = 0
                for local_i, ei in enumerate(order):
                    e = eps[ei]
                    length = ep_len[e]
                    if total + length > self.n_subsample and total > 0:
                        continue   # skip to fit budget; don't truncate episodes
                    start = ep_offset[e]
                    all_obs.append(pixels[start : start + length])
                    all_states.append(states[start : start + length])
                    all_ep_ids.append(np.full(length, local_i, dtype=np.int32))
                    if rul_full is not None:
                        all_rul.append(rul_full[start : start + length])
                    total += length
                    if total >= self.n_subsample:
                        break
                rul_out = np.concatenate(all_rul) if all_rul else None
                return (
                    np.concatenate(all_obs),
                    np.concatenate(all_states),
                    np.concatenate(all_ep_ids),
                    rul_out,
                )

            obs_tr, hi_tr, ep_tr, rul_tr = gather_episodes(tr_eps)
            obs_te, hi_te, ep_te, rul_te = gather_episodes(te_eps)

        self._obs_tr, self._hi_tr, self._ep_ids_tr = obs_tr, hi_tr, ep_tr
        self._obs_te, self._hi_te, self._ep_ids_te = obs_te, hi_te, ep_te
        self._rul_tr = rul_tr
        self._rul_te = rul_te

        # CSV
        log_dir = Path(trainer.log_dir) if trainer.log_dir else Path(".")
        log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = log_dir / "hi_probe_metrics.csv"
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="") as fh:
                csv.writer(fh).writerow(
                    ["epoch", "global_step", "seq_len", "task", "component",
                     "r2", "rmse", "pearson_r", "mae"]
                )
        logging.info(f"[HIProbe] CSV → {self._csv_path}")
        logging.info(
            f"[HIProbe] {len(self._obs_tr):,} train / {len(self._obs_te):,} test "
            f"timesteps  ({N_EP} episodes, {len(self._hi_names)} HI dims)"
        )

    # ── encoding ───────────────────────────────────────────────────────────

    def _preprocess_vit(self, frame_np: np.ndarray,
                        device: torch.device,
                        mean: torch.Tensor,
                        std: torch.Tensor) -> torch.Tensor:
        """Preprocess a batch of raw frames for the ViT.

        Input:  (B, H, W) or (B, C, H, W) numpy
        Output: (B, C, H, W) torch, resized and ImageNet-normalised
        """
        t = torch.from_numpy(frame_np).float().to(device)
        if t.ndim == 3:                                   # (B, H, W) → (B, 3, H, W)
            t = t.unsqueeze(1).expand(-1, 3, -1, -1)
        t = F.interpolate(t, size=(self.img_size, self.img_size), mode="nearest")
        t = (t - mean) / std
        return t

    # Keep legacy name as alias for backward compat
    _preprocess_frame = _preprocess_vit

    @torch.no_grad()
    def _encode(
        self,
        pl_module: pl.LightningModule,
        obs: np.ndarray,
        ep_ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode raw observations -> (N_out, D) embedding array.

        Dispatches to the ViT path (image pixels) or the sensor-native path
        (flat 28-D vectors) based on ``self.encoder_type``.

        Returns
        -------
        embeddings : (N_out, D) numpy array
        valid_idx  : (N_out,) int array — indices into the original ``obs``
                     array.  When obs_window_size=1 this is np.arange(N).
                     When obs_window_size=w, the first w-1 timesteps of each
                     episode are skipped (no full window), so N_out < N.
        """
        device = next(pl_module.model.parameters()).device
        w = self.obs_window_size
        is_sensor = (self.encoder_type == "sensor")

        if not is_sensor:
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        def to_tensor_single(frames):
            """Convert single-frame batch to encoder input tensor."""
            if is_sensor:
                # frames: (B, n_sensors) — already flat after setup() reshape
                t = torch.from_numpy(frames).float().to(device)
                if t.ndim == 2:
                    t = t.unsqueeze(1)     # (B, 28) → (B, 1, 28)
                return t
            else:
                # frames: (B, H, W) → (B, 1, C, H, W)
                return self._preprocess_vit(frames, device, mean, std).unsqueeze(1)

        def to_tensor_window(win_batch):
            """Convert window batch (B_win, w, ...) to encoder input tensor."""
            if is_sensor:
                # win_batch: (B_win, w, n_sensors) — already flat after setup() reshape
                return torch.from_numpy(win_batch).float().to(device)
            else:
                # win_batch: (B_win, w, H, W) → (B_win, w, C, H, W)
                B_win, w_ = win_batch.shape[:2]
                flat = win_batch.reshape(B_win * w_, *win_batch.shape[2:])
                t_flat = self._preprocess_vit(flat, device, mean, std)
                return t_flat.view(B_win, w_, *t_flat.shape[1:])

        def encode_batch(t):
            # Use 'pixels' key — JEPA._encode_sensor() accepts pixels and reshapes
            key = "pixels"
            out = pl_module.model.encode({key: t})
            return out["emb"][:, 0, :].cpu().numpy()

        if w <= 1:
            # ── Single-frame encoding ──────────────────────────────────────
            all_emb = []
            for start in range(0, len(obs), self.enc_batch_size):
                chunk = obs[start : start + self.enc_batch_size]
                all_emb.append(encode_batch(to_tensor_single(chunk)))
            return np.concatenate(all_emb, axis=0), np.arange(len(obs))

        # ── Windowed encoding (obs_window_size > 1) ───────────────────────
        assert ep_ids is not None, "Episode IDs required for obs_window_size > 1"
        all_emb: list[np.ndarray] = []
        valid_indices: list[int] = []

        # Reduce effective batch size to keep GPU memory comparable
        eff_batch = max(self.enc_batch_size // w, 16)

        for ep_id in np.unique(ep_ids):
            mask = ep_ids == ep_id
            ep_obs = obs[mask]                            # (T_ep, ...) — image or sensor
            ep_global = np.where(mask)[0]                 # original indices

            if len(ep_obs) < w:
                continue                                  # episode too short

            n_win = len(ep_obs) - w + 1
            for b_start in range(0, n_win, eff_batch):
                b_end = min(b_start + eff_batch, n_win)
                win_batch = np.stack([
                    ep_obs[i : i + w] for i in range(b_start, b_end)
                ])
                all_emb.append(encode_batch(to_tensor_window(win_batch)))
                for i in range(b_start, b_end):
                    valid_indices.append(ep_global[i + w - 1])

        return np.concatenate(all_emb, axis=0), np.array(valid_indices)

    # ── probe training ─────────────────────────────────────────────────────

    def _train_and_eval_probe(
        self,
        X_tr: np.ndarray, y_tr: np.ndarray,   # (N_tr, D), (N_tr, K)
        X_te: np.ndarray, y_te: np.ndarray,   # (N_te, D), (N_te, K)
        ep_ids_tr: np.ndarray,
        ep_ids_te: np.ndarray,
        device: torch.device,
    ) -> dict[str, list[float]]:

        # ── 1. normalise embeddings ───────────────────────────────────────
        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_tr)
        X_te_s   = scaler.transform(X_te)

        # ── 2. build temporal windows ─────────────────────────────────────
        Xw_tr, yw_tr = self._build_windows(X_tr_s, y_tr, ep_ids_tr, self.probe_seq_len)
        Xw_te, yw_te = self._build_windows(X_te_s, y_te, ep_ids_te, self.probe_seq_len)
        # shapes: (M, seq_len, D), (M, K)

        input_dim = Xw_tr.shape[2]
        n_targets = yw_tr.shape[1]

        logging.info(
            f"[HIProbe] probe dataset: train={len(Xw_tr):,} windows, "
            f"test={len(Xw_te):,} windows  (seq_len={self.probe_seq_len})"
        )

        # ── 3. build probe ────────────────────────────────────────────────
        probe = _TransformerProbe(
            input_dim  = input_dim,
            n_outputs  = n_targets,
            d_model    = self.d_model,
            nhead      = self.nhead,
            num_layers = self.num_layers,
            dropout    = self.probe_dropout,
        ).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=self.probe_lr)

        Xtr_t = torch.from_numpy(Xw_tr).float().to(device)
        ytr_t = torch.from_numpy(yw_tr).float().to(device)
        Xte_t = torch.from_numpy(Xw_te).float().to(device)

        ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.probe_batch_size, shuffle=True
        )

        # ── 4. train with early stopping ──────────────────────────────────
        best_rmse  = float("inf")
        best_state = None
        patience_counter = 0

        with torch.enable_grad():   # Lightning sets no_grad during validation
            for epoch_p in range(self.n_probe_epochs):
                probe.train()
                for xb, yb in dl:
                    opt.zero_grad()
                    F.mse_loss(probe(xb), yb).backward()
                    opt.step()

                probe.eval()
                with torch.no_grad():
                    preds_val = probe(Xte_t).cpu().numpy()
                avg_rmse = float(np.mean(np.sqrt(
                    np.mean((yw_te - preds_val) ** 2, axis=0)
                )))

                if avg_rmse < best_rmse:
                    best_rmse  = avg_rmse
                    best_state = copy.deepcopy(probe.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= self.probe_patience:
                    logging.info(
                        f"[HIProbe] Early stop at probe epoch {epoch_p + 1} "
                        f"(best mean RMSE={best_rmse:.5f})"
                    )
                    break

        # ── 5. final eval with best weights ──────────────────────────────
        if best_state is not None:
            probe.load_state_dict(best_state)
        probe.eval()
        with torch.no_grad():
            preds = probe(Xte_t).cpu().numpy()

        r2s, rmses, pearson_rs = [], [], []
        for i in range(n_targets):
            gt, pr = yw_te[:, i], preds[:, i]
            r2s.append(float(r2_score(gt, pr)))
            rmses.append(float(np.sqrt(np.mean((gt - pr) ** 2))))
            r, _ = pearsonr(gt, pr)
            pearson_rs.append(float(r))

        return {"r2": r2s, "rmse": rmses, "pearson_r": pearson_rs}

    # ── Action-RUL probe ───────────────────────────────────────────────────

    def _train_and_eval_rul_probe(
        self,
        X_tr: np.ndarray, rul_tr: np.ndarray,   # (N_tr, D), (N_tr,)
        X_te: np.ndarray, rul_te: np.ndarray,   # (N_te, D), (N_te,)
        ep_ids_tr: np.ndarray,
        ep_ids_te: np.ndarray,
        device: torch.device,
    ) -> dict[str, float]:
        """Train a single-output TransformerProbe to predict steps-to-next-action.

        The RUL target is log1p-transformed and StandardScaler-normalised before
        training.  Predictions are de-normalised (inverse_transform + expm1) to
        report metrics in original units (timesteps).

        Returns
        -------
        dict with keys: rmse, mae, r2, pearson_r  (all in original timestep units)
        """
        # ── 1. normalise embeddings ───────────────────────────────────────
        scaler_X = StandardScaler()
        X_tr_s   = scaler_X.fit_transform(X_tr)
        X_te_s   = scaler_X.transform(X_te)

        # ── 2. log1p-transform + scale RUL targets ────────────────────────
        rul_tr_log = np.log1p(rul_tr).reshape(-1, 1).astype(np.float32)
        rul_te_log = np.log1p(rul_te).reshape(-1, 1).astype(np.float32)
        scaler_y   = StandardScaler()
        rul_tr_s   = scaler_y.fit_transform(rul_tr_log)          # (N, 1)
        rul_te_s   = scaler_y.transform(rul_te_log)

        # ── 3. build temporal windows ─────────────────────────────────────
        Xw_tr, yw_tr = self._build_windows(X_tr_s, rul_tr_s, ep_ids_tr, self.probe_seq_len)
        Xw_te, yw_te = self._build_windows(X_te_s, rul_te_s, ep_ids_te, self.probe_seq_len)

        input_dim = Xw_tr.shape[2]
        logging.info(
            f"[HIProbe] Action-RUL probe dataset: train={len(Xw_tr):,}, "
            f"test={len(Xw_te):,} windows"
        )

        # ── 4. build probe (n_outputs=1) ──────────────────────────────────
        probe = _TransformerProbe(
            input_dim  = input_dim,
            n_outputs  = 1,
            d_model    = self.d_model,
            nhead      = self.nhead,
            num_layers = self.num_layers,
            dropout    = self.probe_dropout,
        ).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=self.probe_lr)

        Xtr_t = torch.from_numpy(Xw_tr).float().to(device)
        ytr_t = torch.from_numpy(yw_tr).float().to(device)
        Xte_t = torch.from_numpy(Xw_te).float().to(device)

        ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.probe_batch_size, shuffle=True
        )

        # ── 5. train with early stopping ──────────────────────────────────
        best_rmse  = float("inf")
        best_state = None
        patience_counter = 0

        with torch.enable_grad():
            for epoch_p in range(self.n_probe_epochs):
                probe.train()
                for xb, yb in dl:
                    opt.zero_grad()
                    F.mse_loss(probe(xb), yb).backward()
                    opt.step()

                probe.eval()
                with torch.no_grad():
                    preds_s = probe(Xte_t).cpu().numpy()   # normalised space
                rmse_s = float(np.sqrt(np.mean((yw_te - preds_s) ** 2)))

                if rmse_s < best_rmse:
                    best_rmse    = rmse_s
                    best_state   = copy.deepcopy(probe.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= self.probe_patience:
                    logging.info(
                        f"[HIProbe] RUL early stop at probe epoch {epoch_p + 1}"
                    )
                    break

        # ── 6. eval — de-normalise back to timesteps ──────────────────────
        if best_state is not None:
            probe.load_state_dict(best_state)
        probe.eval()
        with torch.no_grad():
            preds_s = probe(Xte_t).cpu().numpy()   # (M, 1) normalised

        # De-normalise predictions: inverse_transform -> expm1 -> clip
        # yw_te (M, 1) is in normalised log1p space; recover ground truth the same way.
        preds_log = scaler_y.inverse_transform(preds_s)          # log1p space  (M, 1)
        preds_raw = np.expm1(preds_log).clip(min=0).ravel()      # timesteps    (M,)
        gt_log    = scaler_y.inverse_transform(yw_te)            # log1p space  (M, 1)
        gt_raw    = np.expm1(gt_log).clip(min=0).ravel()         # timesteps    (M,)

        # ── Censoring mask ────────────────────────────────────────────────
        # Timesteps where τ = max_horizon are censored: the episode ended before
        # the next action, so the true RUL is unknown (only a lower bound of
        # max_horizon).  Evaluating RMSE/R² on these samples is misleading
        # (we penalise the probe for not predicting a meaningless cap value).
        # Restrict evaluation to uncensored samples where the action actually
        # occurred within the horizon.
        uncensored = gt_raw < self.rul_max_horizon
        n_total     = len(gt_raw)
        n_uncens    = int(uncensored.sum())
        frac_uncens = n_uncens / max(n_total, 1)
        logging.info(
            f"[HIProbe] Action-RUL evaluation: {n_uncens}/{n_total} "
            f"({frac_uncens:.1%}) uncensored test samples (tau < {self.rul_max_horizon})"
        )

        if n_uncens >= 20:
            gt_eval    = gt_raw[uncensored]
            preds_eval = preds_raw[uncensored]
        else:
            logging.warning(
                "[HIProbe] Too few uncensored samples (<20); "
                "evaluating on all (censored+uncensored) — metrics may be unreliable."
            )
            gt_eval    = gt_raw
            preds_eval = preds_raw

        rmse      = float(np.sqrt(np.mean((gt_eval - preds_eval) ** 2)))
        mae       = float(np.mean(np.abs(gt_eval - preds_eval)))
        r2        = float(r2_score(gt_eval, preds_eval))
        r_val, _  = pearsonr(gt_eval, preds_eval)
        pearson_r = float(r_val)

        return {
            "rmse": rmse, "mae": mae, "r2": r2, "pearson_r": pearson_r,
            "frac_uncensored": frac_uncens,
        }

    # ── main hook ──────────────────────────────────────────────────────────

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        epoch = trainer.current_epoch
        is_interval = epoch != 0 and epoch % self.eval_interval == 0
        is_final = trainer.max_epochs is not None and epoch == trainer.max_epochs - 1
        if not (is_interval or is_final):
            return

        device = next(pl_module.model.parameters()).device

        logging.info(
            f"[HIProbe] Epoch {epoch} "
            f"(seq_len={self.probe_seq_len}, obs_window={self.obs_window_size}): "
            f"encoding ..."
        )
        pl_module.model.eval()
        X_tr, vi_tr = self._encode(pl_module, self._obs_tr, self._ep_ids_tr)
        X_te, vi_te = self._encode(pl_module, self._obs_te, self._ep_ids_te)
        pl_module.model.train()

        # ── align labels with valid encoded timesteps ─────────────────────
        hi_tr     = self._hi_tr[vi_tr]
        hi_te     = self._hi_te[vi_te]
        ep_ids_tr = self._ep_ids_tr[vi_tr]
        ep_ids_te = self._ep_ids_te[vi_te]
        rul_tr    = self._rul_tr[vi_tr] if self._rul_tr is not None else None
        rul_te    = self._rul_te[vi_te] if self._rul_te is not None else None

        if self.obs_window_size > 1:
            logging.info(
                f"[HIProbe] Windowed encoding: {len(vi_tr):,} train / "
                f"{len(vi_te):,} test valid timesteps "
                f"(skipped first {self.obs_window_size - 1} per episode)"
            )

        # ── embedding health check ────────────────────────────────────────
        emb_std = X_te.std(axis=0)
        logging.info(
            f"[HIProbe] Embedding std — min={emb_std.min():.4f}  "
            f"mean={emb_std.mean():.4f}  max={emb_std.max():.4f}"
        )
        if trainer.logger is not None:
            try:
                trainer.logger.experiment.log({
                    "hi_probe/emb_std_min":  float(emb_std.min()),
                    "hi_probe/emb_std_mean": float(emb_std.mean()),
                    "hi_probe/emb_std_max":  float(emb_std.max()),
                    "trainer/global_step":   trainer.global_step,
                })
            except Exception:
                pass

        logging.info(
            f"[HIProbe] Training TransformerProbe "
            f"(max {self.n_probe_epochs} epochs, patience={self.probe_patience}) ..."
        )
        metrics = self._train_and_eval_probe(
            X_tr, hi_tr, X_te, hi_te,
            ep_ids_tr, ep_ids_te, device,
        )
        r2s        = metrics["r2"]
        rmses      = metrics["rmse"]
        pearson_rs = metrics["pearson_r"]
        mean_r2      = float(np.mean(r2s))
        mean_rmse    = float(np.mean(rmses))
        mean_pearson = float(np.mean(pearson_rs))

        short = [self._shorten(n) for n in self._hi_names]
        sl    = self.probe_seq_len

        # ── console table ─────────────────────────────────────────────────
        sep = "─" * 66
        logging.info(f"\n[HIProbe] Epoch {epoch} | seq_len={sl} — results")
        logging.info(sep)
        logging.info(f"  {'Component':<28}  {'R²':>8}  {'RMSE':>10}  {'Pearson-r':>10}")
        logging.info(sep)
        for name, r2, rmse, pr in zip(short, r2s, rmses, pearson_rs):
            logging.info(f"  {name:<28}  {r2:>8.4f}  {rmse:>10.5f}  {pr:>10.4f}")
        logging.info(sep)
        logging.info(
            f"  {'MEAN':<28}  {mean_r2:>8.4f}  {mean_rmse:>10.5f}  {mean_pearson:>10.4f}"
        )
        logging.info(sep)

        # ── Action-RUL probe ──────────────────────────────────────────────
        rul_metrics: dict[str, float] | None = None
        if rul_tr is not None and rul_te is not None:
            logging.info(
                f"[HIProbe] Training Action-RUL probe "
                f"(max_horizon={self.rul_max_horizon}) ..."
            )
            try:
                rul_metrics = self._train_and_eval_rul_probe(
                    X_tr, rul_tr, X_te, rul_te,
                    ep_ids_tr, ep_ids_te, device,
                )
                logging.info(
                    f"[HIProbe] Action-RUL — "
                    f"RMSE={rul_metrics['rmse']:.2f} steps  "
                    f"MAE={rul_metrics['mae']:.2f} steps  "
                    f"R2={rul_metrics['r2']:.4f}  "
                    f"Pearson={rul_metrics['pearson_r']:.4f}"
                )
            except Exception as exc:
                logging.warning(f"[HIProbe] Action-RUL probe failed: {exc}")

        # ── CSV (wandb-independent, always written) ───────────────────────
        if self._csv_path is not None:
            with open(self._csv_path, "a", newline="") as fh:
                w = csv.writer(fh)
                for name, r2, rmse, pr in zip(short, r2s, rmses, pearson_rs):
                    w.writerow([epoch, trainer.global_step, sl, "hi", name,
                                f"{r2:.6f}", f"{rmse:.6f}", f"{pr:.6f}", ""])
                w.writerow([epoch, trainer.global_step, sl, "hi", "MEAN",
                            f"{mean_r2:.6f}", f"{mean_rmse:.6f}",
                            f"{mean_pearson:.6f}", ""])
                if rul_metrics is not None:
                    w.writerow([
                        epoch, trainer.global_step, sl, "rul", "steps-to-action",
                        f"{rul_metrics['r2']:.6f}",
                        f"{rul_metrics['rmse']:.6f}",
                        f"{rul_metrics['pearson_r']:.6f}",
                        f"{rul_metrics['mae']:.6f}",
                    ])
            logging.info(f"[HIProbe] Appended to {self._csv_path}")

        # ── wandb (direct .log bypasses Lightning metric aggregation) ─────
        prefix = f"hi_probe/sl{sl}"
        log_dict: dict[str, float] = {}

        # HI per-dimension metrics
        for name, r2, rmse, pr in zip(short, r2s, rmses, pearson_rs):
            log_dict[f"{prefix}/{name}/r2"]        = r2
            log_dict[f"{prefix}/{name}/rmse"]      = rmse
            log_dict[f"{prefix}/{name}/pearson_r"] = pr
        log_dict[f"{prefix}/mean_r2"]        = mean_r2
        log_dict[f"{prefix}/mean_rmse"]      = mean_rmse
        log_dict[f"{prefix}/mean_pearson_r"] = mean_pearson

        # Action-RUL metrics (evaluated on uncensored samples only)
        if rul_metrics is not None:
            log_dict[f"{prefix}/rul/rmse"]           = rul_metrics["rmse"]
            log_dict[f"{prefix}/rul/mae"]            = rul_metrics["mae"]
            log_dict[f"{prefix}/rul/r2"]             = rul_metrics["r2"]
            log_dict[f"{prefix}/rul/pearson_r"]      = rul_metrics["pearson_r"]
            log_dict[f"{prefix}/rul/frac_uncensored"] = rul_metrics["frac_uncensored"]

        if trainer.logger is not None:
            try:
                trainer.logger.experiment.log(
                    {**log_dict, "trainer/global_step": trainer.global_step}
                )
                logging.info(f"[HIProbe] Logged {len(log_dict)} metrics to wandb.")
            except Exception as e:
                logging.warning(f"[HIProbe] wandb log failed ({e}); CSV still written.")

        for k, v in log_dict.items():
            pl_module.log(k, v, on_step=False, on_epoch=True,
                          prog_bar=False, sync_dist=False)
