"""Sensor Autoencoder Baseline Training Script
================================================

Frame-level reconstruction baseline: the same SensorEncoder used by JEPA is
paired with an MLP decoder and trained to reconstruct sensor inputs (MSE loss).
No temporal modelling — each timestep is encoded/decoded independently.

This isolates the contribution of *prediction pressure* (JEPA, AR-LSTM) vs
*reconstruction pressure* (this baseline) on the quality of the learned
representations, measured via the HIProbeCallback.

Usage (local):
    cd le-wm/
    python baselines/ae/train_ae.py \\
        --config-path baselines/ae/config \\
        --config-name train_ae_scenario3 \\
        data.dataset.cache_dir=/path/to/data

Usage (cluster):  see baselines/ae/train_ae_scenario3.slurm
"""

import json
import logging
import os
import sys
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

# ── Add le-wm root to path so shared modules are importable ──────────────────
_LE_WM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LE_WM_ROOT))

import stable_pretraining as spt       # noqa: E402
import stable_worldmodel as swm        # noqa: E402
from hi_probe import HIProbeCallback   # noqa: E402
from module import SIGReg              # noqa: E402
from utils import (                    # noqa: E402
    ModelObjectCallBack,
    get_column_normalizer,
)
from jepa import SensorEncoder, TemporalAggregator  # noqa: E402
from module import MLP                              # noqa: E402
from baselines.ae.model import SensorDecoder, SensorAutoencoder  # noqa: E402


# ── Forward pass ─────────────────────────────────────────────────────────────

def ae_forward(self, batch, stage, cfg):
    """Encode every frame independently, decode back to sensor space, MSE loss."""
    batch["action"] = torch.nan_to_num(batch.get("action", torch.zeros(1)), 0.0)

    output       = self.model.encode(batch)
    emb_flat     = output["_emb_flat"]      # (B*T, d_model)
    sensors_flat = output["_sensors_flat"]  # (B*T, n_sensors)

    recon = self.model.decode(emb_flat)     # (B*T, n_sensors)

    output["recon_loss"] = F.mse_loss(recon, sensors_flat)
    output["loss"]       = output["recon_loss"]

    self.log_dict(
        {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k},
        on_step=True, sync_dist=True,
    )
    return output


# ── Offline-wandb patch (same as train.py / train_ar_lstm.py) ────────────────

def _patch_wandb_offline(manager: spt.Manager) -> None:
    def _safe_init_and_sync_wandb(self):
        import lightning
        if not isinstance(
            self._trainer.logger, lightning.pytorch.loggers.wandb.WandbLogger
        ):
            return
        logging.info("📈 Using Wandb")
        exp = self._trainer.logger.experiment
        if exp.offline:
            previous_run = self._wandb_previous_dir()
            if previous_run is None:
                logging.info("[wandb] Offline mode: first run, no config to reuse.")
                return
            logging.info(f"[wandb] Reusing config from previous run: {previous_run}")
            with open(previous_run / "files/wandb-config.json", "r") as f:
                last_config = json.load(f)
            exp.config.update(last_config)
            logging.info("[wandb] Config reloaded.")

    type(manager).init_and_sync_wandb = _safe_init_and_sync_wandb


# ── Main ─────────────────────────────────────────────────────────────────────

@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "config"),
    config_name="train_ae_scenario3",
)
def run(cfg):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    logging.info("=" * 60)
    logging.info("Sensor Autoencoder Baseline")
    logging.info("=" * 60)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)

    transforms = [get_column_normalizer(source="action", target="action", dataset=dataset)]
    dataset.transform = spt.data.transforms.Compose(*transforms)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    n_sensors       = cfg.get("n_sensors", 84)
    embed_dim       = cfg.wm.embed_dim
    obs_window_size = cfg.get("obs_window_size", 1)

    encoder = SensorEncoder(
        n_sensors  = n_sensors,
        d_model    = cfg.sensor_encoder.d_model,
        nhead      = cfg.sensor_encoder.nhead,
        num_layers = cfg.sensor_encoder.num_layers,
        dropout    = cfg.sensor_encoder.dropout,
    )
    decoder = SensorDecoder(
        d_model    = embed_dim,
        n_sensors  = n_sensors,
        hidden_dim = cfg.decoder.hidden_dim,
    )

    # Build TemporalAggregator when obs_window_size > 1 (same as JEPA)
    projector    = None
    temporal_agg = None
    if obs_window_size > 1:
        projector = MLP(
            input_dim  = cfg.sensor_encoder.d_model,
            output_dim = embed_dim,
        )
        temporal_agg = TemporalAggregator(
            embed_dim  = embed_dim,
            max_window = obs_window_size,
            nhead      = cfg.sensor_encoder.nhead,
            num_layers = 1,
            dropout    = cfg.sensor_encoder.dropout,
        )

    world_model = SensorAutoencoder(
        encoder         = encoder,
        decoder         = decoder,
        obs_window_size = obs_window_size,
        temporal_agg    = temporal_agg,
        projector       = projector,
    )

    param_count = sum(p.numel() for p in world_model.parameters() if p.requires_grad)
    logging.info(f"Total trainable parameters: {param_count:,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizers = {
        "model_opt": {
            "modules":   "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {
                "type":         "LinearWarmupCosineAnnealingLR",
                "warmup_steps": max(1, len(train_loader) // 2),
                "max_steps":    cfg.trainer.max_epochs * len(train_loader),
            },
            "interval": "step",
        },
    }

    # ── Callbacks ─────────────────────────────────────────────────────────────
    run_id  = cfg.get("subdir") or "ae"
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelObjectCallBack(
            dirpath=run_dir,
            filename=cfg.output_model_name,
            epoch_interval=1,
        ),
    ]

    hi_probe_cfg = cfg.get("hi_probe", None)
    if hi_probe_cfg is not None and hi_probe_cfg.get("enabled", False):
        cache_dir = cfg.data.dataset.get("cache_dir")
        name      = cfg.data.dataset.get("name", "dataset")
        if not cache_dir or cache_dir == "???":
            logging.error("[HIProbe] data.dataset.cache_dir not set — pass via CLI")
        else:
            data_path = str(Path(cache_dir) / f"{name}.h5")
            callbacks.append(HIProbeCallback(
                data_path        = data_path,
                img_size         = cfg.get("img_size", 28),
                train_split      = cfg.train_split,
                seed             = cfg.seed,
                eval_interval    = hi_probe_cfg.get("eval_interval", 5),
                n_probe_epochs   = hi_probe_cfg.get("n_probe_epochs", 150),
                probe_lr         = hi_probe_cfg.get("probe_lr", 1e-3),
                probe_patience   = hi_probe_cfg.get("probe_patience", 20),
                d_model          = hi_probe_cfg.get("d_model", 64),
                nhead            = hi_probe_cfg.get("nhead", 4),
                num_layers       = hi_probe_cfg.get("num_layers", 2),
                probe_dropout    = hi_probe_cfg.get("probe_dropout", 0.1),
                probe_batch_size = hi_probe_cfg.get("probe_batch_size", 256),
                probe_seq_len    = hi_probe_cfg.get("probe_seq_len", 1),
                n_subsample      = hi_probe_cfg.get("n_subsample", 30_000),
                enc_batch_size   = hi_probe_cfg.get("enc_batch_size", 2048),
                rul_max_horizon  = hi_probe_cfg.get("rul_max_horizon", 300),
                obs_window_size  = cfg.get("obs_window_size", 1),
                encoder_type     = "sensor",
            ))
            logging.info(
                f"[HIProbe] Registered — evaluating every "
                f"{hi_probe_cfg.get('eval_interval', 5)} epochs"
            )

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = None
    if cfg.wandb.enabled:
        wandb_kwargs = dict(cfg.wandb.config)
        if os.environ.get("WANDB_MODE") == "offline":
            wandb_kwargs["offline"] = True
        logger = WandbLogger(**wandb_kwargs)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # ── Trainer ───────────────────────────────────────────────────────────────
    # spt.Module requires a sigreg argument — pass one but ae_forward never calls it.
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    module = spt.Module(
        model   = world_model,
        sigreg  = SIGReg(**cfg.loss.sigreg.kwargs),
        forward = partial(ae_forward, cfg=cfg),
        optim   = optimizers,
    )

    manager = spt.Manager(
        trainer = trainer,
        module  = module,
        data    = spt.data.DataModule(train=train_loader, val=val_loader),
        ckpt_path = None,
    )

    if os.environ.get("WANDB_MODE") == "offline":
        _patch_wandb_offline(manager)

    manager()


if __name__ == "__main__":
    run()
