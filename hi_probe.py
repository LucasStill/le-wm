"""HI Probe callback — trains an MLP every N epochs to predict turbofan
Health Indicator (HI) states from frozen JEPA encoder embeddings.

Why MLP instead of Ridge?
  Ridge is a linear model that can only capture linear relationships between
  embeddings and HI states.  An MLP can capture non-linear structure in the
  latent space, giving a better signal for whether the encoder has learned a
  useful representation of engine degradation.

The callback uses the same episode-wise train/test split and seed as
le-wm training so the test set is guaranteed to be held out.

Metrics logged per HI dimension (e.g. hi_probe/HPC_eff/r2):
  - r2          : coefficient of determination
  - rmse        : root mean squared error
  - pearson_r   : Pearson correlation coefficient

Plus aggregates:
  - hi_probe/mean_r2 / mean_rmse / mean_pearson_r
"""

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


# ── Probe model ────────────────────────────────────────────────────────────────

class _MLPProbe(nn.Module):
    """Small MLP: embed_dim → hidden → ReLU → n_targets (regression)."""

    def __init__(self, embed_dim: int, hidden_dim: int, n_targets: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Callback ───────────────────────────────────────────────────────────────────

class HIProbeCallback(pl.Callback):
    """Every ``eval_interval`` validation epochs:

    1. Encodes a random subset of test observations with the frozen JEPA encoder.
    2. Trains a small MLP to predict the 10 HI degradation states.
    3. Reports per-component R², RMSE, and Pearson-r to wandb and the console.

    Parameters
    ----------
    data_path : str
        Path to the le-wm-ready HDF5 file (contains 'pixels' and
        'observation.state' keys).
    img_size : int
        Must match the ``img_size`` used during training.
    train_split, seed : float, int
        Must match le-wm training values so the test set is held out.
    eval_interval : int
        Run the probe every this many epochs (default: 5).
    n_probe_epochs : int
        Training epochs for the MLP probe (default: 100).
    hidden_dim : int
        Hidden layer width of the probe MLP (default: 256).
    n_subsample : int
        Max timesteps to encode per split for speed (default: 30 000).
    batch_size : int
        Encoding batch size (default: 2048).
    """

    def __init__(
        self,
        data_path: str,
        img_size: int = 28,
        train_split: float = 0.9,
        seed: int = 3072,
        eval_interval: int = 5,
        n_probe_epochs: int = 100,
        probe_lr: float = 1e-3,
        hidden_dim: int = 256,
        n_subsample: int = 30_000,
        batch_size: int = 2048,
    ):
        super().__init__()
        self.data_path      = Path(data_path)
        self.img_size       = img_size
        self.train_split    = train_split
        self.seed           = seed
        self.eval_interval  = eval_interval
        self.n_probe_epochs = n_probe_epochs
        self.probe_lr       = probe_lr
        self.hidden_dim     = hidden_dim
        self.n_subsample    = n_subsample
        self.batch_size     = batch_size

        # Filled in setup()
        self._obs_tr:   np.ndarray | None = None
        self._obs_te:   np.ndarray | None = None
        self._hi_tr:    np.ndarray | None = None
        self._hi_te:    np.ndarray | None = None
        self._hi_names: list[str] | None  = None

    # ── setup ──────────────────────────────────────────────────────────────

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        """Load and split dataset once at training start (rank 0 only)."""
        if self._obs_tr is not None:
            return  # Already loaded

        logging.info(f"[HIProbe] Loading dataset from {self.data_path}")

        with h5py.File(self.data_path, "r") as f:
            ep_len    = f["ep_len"][:]
            ep_offset = f["ep_offset"][:]
            obs_key   = "pixels" if "pixels" in f else "observation.sensors"
            pixels    = f[obs_key][:]              # (N, 3, 7, 4) or (N, 7, 4)
            states    = f["observation.state"][:]  # (N, 10)
            self._hi_names = list(
                f.attrs.get("state_label_names", [f"HI_{i}" for i in range(states.shape[1])])
            )

        N_EP = len(ep_len)
        rng  = np.random.default_rng(self.seed)
        perm = rng.permutation(N_EP)
        n_tr = int(N_EP * self.train_split)

        def gather(eps):
            idx = np.concatenate([
                np.arange(ep_offset[i], ep_offset[i] + ep_len[i]) for i in eps
            ])
            if len(idx) > self.n_subsample:
                sub_rng = np.random.default_rng(self.seed + 99)
                idx = sub_rng.choice(idx, size=self.n_subsample, replace=False)
            return pixels[idx], states[idx]

        self._obs_tr, self._hi_tr = gather(perm[:n_tr])
        self._obs_te, self._hi_te = gather(perm[n_tr:])

        logging.info(
            f"[HIProbe] {len(self._obs_tr):,} train / {len(self._obs_te):,} test "
            f"timesteps   ({N_EP} episodes, {len(self._hi_names)} HI components)"
        )

    # ── encoding ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode(self, pl_module: pl.LightningModule, obs: np.ndarray) -> np.ndarray:
        """Encode observations through the frozen JEPA encoder+projector."""
        device = next(pl_module.model.parameters()).device

        # Same ImageNet stats as get_img_preprocessor in train.py
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        all_emb = []
        for start in range(0, len(obs), self.batch_size):
            chunk = obs[start : start + self.batch_size]   # (B, [C,] H, W)
            t = torch.from_numpy(chunk).float().to(device)

            if t.ndim == 3:                                # (B, 7, 4) raw sensors
                t = t.unsqueeze(1).repeat(1, 3, 1, 1)     # → (B, 3, 7, 4)

            # Resize to training img_size
            t = F.interpolate(t, size=(self.img_size, self.img_size), mode="nearest")
            t = (t - mean) / std                           # ImageNet normalise
            t = t.unsqueeze(1)                             # (B, T=1, 3, H, W)

            out = pl_module.model.encode({"pixels": t})
            emb = out["emb"][:, 0, :]                      # (B, D)
            all_emb.append(emb.cpu().numpy())

        return np.concatenate(all_emb, axis=0)

    # ── probe ──────────────────────────────────────────────────────────────

    def _train_and_eval_probe(
        self,
        X_tr: np.ndarray, y_tr: np.ndarray,
        X_te: np.ndarray, y_te: np.ndarray,
        device: torch.device,
    ) -> dict[str, list[float]]:
        """Train MLP probe, return per-dimension dict of {r2, rmse, pearson_r}."""
        embed_dim = X_tr.shape[1]
        n_targets = y_tr.shape[1]

        probe = _MLPProbe(embed_dim, self.hidden_dim, n_targets).to(device)
        opt   = torch.optim.Adam(probe.parameters(), lr=self.probe_lr)

        X_tr_t = torch.from_numpy(X_tr).float().to(device)
        y_tr_t = torch.from_numpy(y_tr).float().to(device)
        ds = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
        dl = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True)

        probe.train()
        # torch.enable_grad() is required: on_validation_epoch_end is called
        # inside Lightning's no_grad context, which prevents .backward().
        with torch.enable_grad():
            for _ in range(self.n_probe_epochs):
                for xb, yb in dl:
                    opt.zero_grad()
                    F.mse_loss(probe(xb), yb).backward()
                    opt.step()

        probe.eval()
        with torch.no_grad():
            preds = probe(torch.from_numpy(X_te).float().to(device)).cpu().numpy()

        r2s, rmses, pearson_rs = [], [], []
        for i in range(n_targets):
            gt, pr = y_te[:, i], preds[:, i]
            r2s.append(float(r2_score(gt, pr)))
            rmses.append(float(np.sqrt(np.mean((gt - pr) ** 2))))
            r, _ = pearsonr(gt, pr)
            pearson_rs.append(float(r))

        return {"r2": r2s, "rmse": rmses, "pearson_r": pearson_rs}

    # ── main hook ──────────────────────────────────────────────────────────

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        epoch = trainer.current_epoch
        # Skip epoch 0 (model untrained) and only run every eval_interval epochs
        if epoch == 0 or epoch % self.eval_interval != 0:
            return

        device = next(pl_module.model.parameters()).device

        logging.info(f"[HIProbe] Epoch {epoch}: encoding observations …")
        pl_module.model.eval()
        X_tr = self._encode(pl_module, self._obs_tr)
        X_te = self._encode(pl_module, self._obs_te)
        pl_module.model.train()

        logging.info(f"[HIProbe] Training MLP probe ({self.n_probe_epochs} epochs) …")
        metrics = self._train_and_eval_probe(X_tr, self._hi_tr, X_te, self._hi_te, device)

        r2s       = metrics["r2"]
        rmses     = metrics["rmse"]
        pearson_rs= metrics["pearson_r"]

        mean_r2       = float(np.mean(r2s))
        mean_rmse     = float(np.mean(rmses))
        mean_pearson  = float(np.mean(pearson_rs))

        # ── pretty console table ──────────────────────────────────────────
        # Convert e.g. "deg_CmpBst_s_mapEff_in" → "Bst/Eff"
        def _shorten(name: str) -> str:
            n = name.replace("deg_Cmp", "").replace("deg_Trb", "Trb")
            n = n.replace("_s_mapEff_in", "/Eff").replace("_s_mapWc_in", "/Wc")
            # Handle turbine names: TrbHTrbH → TrbH, etc.
            n = n.replace("TrbTrbH", "TrbH").replace("TrbTrbL", "TrbL")
            return n

        short = [_shorten(n) for n in self._hi_names]
        sep = "─" * 64
        logging.info(f"\n[HIProbe] Epoch {epoch} results")
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

        # ── build log dict ────────────────────────────────────────────────
        # Key format: hi_probe/<component>/<metric>
        # e.g. hi_probe/HPC_eff/r2, hi_probe/HPC_eff/rmse, hi_probe/HPC_eff/pearson_r
        log_dict: dict[str, float] = {}
        for name, r2, rmse, pr in zip(short, r2s, rmses, pearson_rs):
            log_dict[f"hi_probe/{name}/r2"]        = r2
            log_dict[f"hi_probe/{name}/rmse"]      = rmse
            log_dict[f"hi_probe/{name}/pearson_r"] = pr
        log_dict["hi_probe/mean_r2"]       = mean_r2
        log_dict["hi_probe/mean_rmse"]     = mean_rmse
        log_dict["hi_probe/mean_pearson_r"] = mean_pearson

        # ── log to wandb directly (bypasses Lightning's metric aggregation) ──
        # pl_module.log_dict() called from a callback's on_validation_epoch_end
        # gets silently buffered/dropped.  Log to the wandb experiment directly.
        if trainer.logger is not None:
            try:
                # WandbLogger exposes the run via .experiment
                wandb_run = trainer.logger.experiment
                wandb_run.log({**log_dict, "trainer/global_step": trainer.global_step})
                logging.info(f"[HIProbe] Logged {len(log_dict)} metrics to wandb.")
            except Exception as e:
                logging.warning(f"[HIProbe] wandb log failed: {e}")

        # Also feed into Lightning's progress bar / CSV logger as a fallback
        for k, v in log_dict.items():
            pl_module.log(k, v, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
