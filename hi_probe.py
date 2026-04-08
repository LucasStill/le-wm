"""HI Probe callback — trains a Transformer probe every N epochs to predict
turbofan Health Indicator (HI) states from frozen JEPA encoder embeddings.

Probe architecture matches the TransformerRegressor used in prior benchmark
work for direct comparison:
  - StandardScaler normalisation of encoder embeddings
  - Linear(input_dim → d_model) projection, scaled by √d_model
  - TransformerEncoder (batch_first, num_layers, nhead, dim_feedforward=4×d_model)
  - Output head: d_model → d_model//2 → ReLU → Dropout → n_outputs
  - Adam optimiser, early stopping on mean per-dimension RMSE (patience=10)

NOTE on seq_len=1: with a single timestep the self-attention collapses to an
identity, so the transformer is functionally an MLP here.  This is intentional
for comparability.  A future extension can pass a window of consecutive
embeddings (seq_len > 1) to actually exploit temporal attention.

Metrics logged per HI dimension  (wandb key: hi_probe/<component>/<metric>):
  r2, rmse, pearson_r

Aggregates:
  hi_probe/mean_r2, hi_probe/mean_rmse, hi_probe/mean_pearson_r

CSV fallback (always written, wandb-independent):
  {trainer.log_dir}/hi_probe_metrics.csv
  columns: epoch, global_step, component, r2, rmse, pearson_r
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
    """Transformer regressor matching the benchmark TransformerRegressor.

    Input shape : (batch, seq_len=1, input_dim)
    Output shape: (batch, n_outputs)
    """

    def __init__(
        self,
        input_dim: int,
        n_outputs: int = 10,
        d_model: int = 32,
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

    1. Encodes a random subset of observations with the frozen JEPA encoder.
    2. Normalises embeddings with StandardScaler (fit on train split).
    3. Trains a TransformerProbe with early stopping to predict 10 HI states.
    4. Reports per-component R², RMSE, Pearson-r to wandb + CSV + console.

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
        Transformer hidden dimension (default 32).
    nhead : int
        Number of attention heads (default 4).
    num_layers : int
        Number of TransformerEncoder layers (default 2).
    probe_dropout : float
        Dropout inside the transformer and output head (default 0.1).
    probe_batch_size : int
        Batch size for probe training (default 256).
    n_subsample : int
        Max timesteps to encode per split (default 30 000).
    enc_batch_size : int
        Batch size for JEPA encoding pass (default 2048).
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
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        probe_dropout: float = 0.1,
        probe_batch_size: int = 256,
        n_subsample: int = 30_000,
        enc_batch_size: int = 2048,
        # legacy alias kept for config backward compat (ignored)
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.data_path       = Path(data_path)
        self.img_size        = img_size
        self.train_split     = train_split
        self.seed            = seed
        self.eval_interval   = eval_interval
        self.n_probe_epochs  = n_probe_epochs
        self.probe_lr        = probe_lr
        self.probe_patience  = probe_patience
        self.d_model         = d_model
        self.nhead           = nhead
        self.num_layers      = num_layers
        self.probe_dropout   = probe_dropout
        self.probe_batch_size = probe_batch_size
        self.n_subsample     = n_subsample
        self.enc_batch_size  = enc_batch_size

        # Filled in setup()
        self._obs_tr:   np.ndarray | None = None
        self._obs_te:   np.ndarray | None = None
        self._hi_tr:    np.ndarray | None = None
        self._hi_te:    np.ndarray | None = None
        self._hi_names: list[str] | None  = None
        self._csv_path: Path | None       = None

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _shorten(name: str) -> str:
        """deg_CmpBst_s_mapEff_in  →  Bst/Eff"""
        n = name.replace("deg_Cmp", "").replace("deg_Trb", "Trb")
        n = n.replace("_s_mapEff_in", "/Eff").replace("_s_mapWc_in", "/Wc")
        return n

    # ── setup ──────────────────────────────────────────────────────────────

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        if self._obs_tr is not None:
            return

        logging.info(f"[HIProbe] Loading dataset from {self.data_path}")
        with h5py.File(self.data_path, "r") as f:
            ep_len    = f["ep_len"][:]
            ep_offset = f["ep_offset"][:]
            obs_key   = "pixels" if "pixels" in f else "observation.sensors"
            pixels    = f[obs_key][:]
            states    = f["observation.state"][:]
            self._hi_names = list(
                f.attrs.get("state_label_names",
                            [f"HI_{i}" for i in range(states.shape[1])])
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
                idx = np.random.default_rng(self.seed + 99).choice(
                    idx, size=self.n_subsample, replace=False
                )
            return pixels[idx], states[idx]

        self._obs_tr, self._hi_tr = gather(perm[:n_tr])
        self._obs_te, self._hi_te = gather(perm[n_tr:])

        # CSV file lives next to Lightning's log dir
        log_dir = Path(trainer.log_dir) if trainer.log_dir else Path(".")
        log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = log_dir / "hi_probe_metrics.csv"
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["epoch", "global_step", "component", "r2", "rmse", "pearson_r"]
                )
        logging.info(f"[HIProbe] CSV → {self._csv_path}")

        logging.info(
            f"[HIProbe] {len(self._obs_tr):,} train / {len(self._obs_te):,} test "
            f"timesteps  ({N_EP} episodes, {len(self._hi_names)} HI dims)"
        )

    # ── encoding ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode(self, pl_module: pl.LightningModule, obs: np.ndarray) -> np.ndarray:
        device = next(pl_module.model.parameters()).device
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        all_emb = []
        for start in range(0, len(obs), self.enc_batch_size):
            chunk = obs[start : start + self.enc_batch_size]
            t = torch.from_numpy(chunk).float().to(device)
            if t.ndim == 3:
                t = t.unsqueeze(1).repeat(1, 3, 1, 1)
            t = F.interpolate(t, size=(self.img_size, self.img_size), mode="nearest")
            t = (t - mean) / std
            t = t.unsqueeze(1)                           # (B, T=1, C, H, W)
            out = pl_module.model.encode({"pixels": t})
            all_emb.append(out["emb"][:, 0, :].cpu().numpy())

        return np.concatenate(all_emb, axis=0)

    # ── probe training ─────────────────────────────────────────────────────

    def _train_and_eval_probe(
        self,
        X_tr: np.ndarray, y_tr: np.ndarray,
        X_te: np.ndarray, y_te: np.ndarray,
        device: torch.device,
    ) -> dict[str, list[float]]:
        """Normalise → train TransformerProbe w/ early stopping → eval."""

        # ── 1. normalise embeddings (fit on train) ────────────────────────
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        input_dim = X_tr_s.shape[1]
        n_targets = y_tr.shape[1]

        # ── 2. build probe ────────────────────────────────────────────────
        probe = _TransformerProbe(
            input_dim  = input_dim,
            n_outputs  = n_targets,
            d_model    = self.d_model,
            nhead      = self.nhead,
            num_layers = self.num_layers,
            dropout    = self.probe_dropout,
        ).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=self.probe_lr)

        # seq_len=1 wrapper: (B, D) → (B, 1, D)
        Xtr_t = torch.from_numpy(X_tr_s).float().to(device).unsqueeze(1)
        ytr_t = torch.from_numpy(y_tr).float().to(device)
        Xte_t = torch.from_numpy(X_te_s).float().to(device).unsqueeze(1)

        ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.probe_batch_size, shuffle=True
        )

        # ── 3. train with early stopping ──────────────────────────────────
        best_rmse  = float("inf")
        best_state = None
        patience_counter = 0

        # enable_grad: Lightning sets no_grad during validation
        with torch.enable_grad():
            for epoch_p in range(self.n_probe_epochs):
                probe.train()
                for xb, yb in dl:
                    opt.zero_grad()
                    F.mse_loss(probe(xb), yb).backward()
                    opt.step()

                # quick eval for early stopping
                probe.eval()
                with torch.no_grad():
                    preds_val = probe(Xte_t).cpu().numpy()
                per_dim_rmse = np.sqrt(
                    np.mean((y_te - preds_val) ** 2, axis=0)
                )
                avg_rmse = float(np.mean(per_dim_rmse))

                if avg_rmse < best_rmse:
                    best_rmse  = avg_rmse
                    best_state = copy.deepcopy(probe.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= self.probe_patience:
                    logging.info(
                        f"[HIProbe] Early stop at probe epoch {epoch_p+1} "
                        f"(patience={self.probe_patience}, best mean RMSE={best_rmse:.5f})"
                    )
                    break

        # ── 4. final eval with best weights ──────────────────────────────
        if best_state is not None:
            probe.load_state_dict(best_state)
        probe.eval()
        with torch.no_grad():
            preds = probe(Xte_t).cpu().numpy()

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
        if epoch == 0 or epoch % self.eval_interval != 0:
            return

        device = next(pl_module.model.parameters()).device

        logging.info(f"[HIProbe] Epoch {epoch}: encoding …")
        pl_module.model.eval()
        X_tr = self._encode(pl_module, self._obs_tr)
        X_te = self._encode(pl_module, self._obs_te)
        pl_module.model.train()

        logging.info(
            f"[HIProbe] Training TransformerProbe "
            f"(max {self.n_probe_epochs} epochs, patience={self.probe_patience}) …"
        )
        metrics    = self._train_and_eval_probe(X_tr, self._hi_tr, X_te, self._hi_te, device)
        r2s        = metrics["r2"]
        rmses      = metrics["rmse"]
        pearson_rs = metrics["pearson_r"]
        mean_r2      = float(np.mean(r2s))
        mean_rmse    = float(np.mean(rmses))
        mean_pearson = float(np.mean(pearson_rs))

        short = [self._shorten(n) for n in self._hi_names]

        # ── console table ─────────────────────────────────────────────────
        sep = "─" * 66
        logging.info(f"\n[HIProbe] Epoch {epoch} — TransformerProbe results")
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

        # ── CSV logging (always, wandb-independent) ───────────────────────
        if self._csv_path is not None:
            with open(self._csv_path, "a", newline="") as f:
                w = csv.writer(f)
                for name, r2, rmse, pr in zip(short, r2s, rmses, pearson_rs):
                    w.writerow([epoch, trainer.global_step, name,
                                f"{r2:.6f}", f"{rmse:.6f}", f"{pr:.6f}"])
                # summary row
                w.writerow([epoch, trainer.global_step, "MEAN",
                            f"{mean_r2:.6f}", f"{mean_rmse:.6f}", f"{mean_pearson:.6f}"])
            logging.info(f"[HIProbe] Appended to {self._csv_path}")

        # ── wandb (direct .log to bypass Lightning aggregation) ───────────
        log_dict: dict[str, float] = {}
        for name, r2, rmse, pr in zip(short, r2s, rmses, pearson_rs):
            log_dict[f"hi_probe/{name}/r2"]       = r2
            log_dict[f"hi_probe/{name}/rmse"]     = rmse
            log_dict[f"hi_probe/{name}/pearson_r"] = pr
        log_dict["hi_probe/mean_r2"]        = mean_r2
        log_dict["hi_probe/mean_rmse"]      = mean_rmse
        log_dict["hi_probe/mean_pearson_r"] = mean_pearson

        if trainer.logger is not None:
            try:
                trainer.logger.experiment.log(
                    {**log_dict, "trainer/global_step": trainer.global_step}
                )
                logging.info(f"[HIProbe] Logged {len(log_dict)} metrics to wandb.")
            except Exception as e:
                logging.warning(f"[HIProbe] wandb log failed ({e}); CSV still written.")

        # Lightning fallback (progress bar / CSV logger)
        for k, v in log_dict.items():
            pl_module.log(k, v, on_step=False, on_epoch=True,
                          prog_bar=False, sync_dist=False)
