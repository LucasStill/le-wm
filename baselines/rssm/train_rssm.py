"""
DreamerV3 RSSM Baseline Training Script
=========================================

Mirrors the AR-LSTM training script (baselines/ar_lstm/train_ar_lstm.py) so
the only visible differences are (a) the model swap to RSSMWorldModel and
(b) the loss function, which is reconstruction MSE + balanced KL instead of
JEPA latent prediction.

Usage (local):
    cd le-wm/
    python baselines/rssm/train_rssm.py \
        --config-path baselines/rssm/config \
        --config-name train_rssm_scenario4 \
        data.dataset.cache_dir=/path/to/data \
        hi_probe.enabled=true

Usage (cluster):
    ./train_rssm_scenario4_orailix.sh
"""
from __future__ import annotations

import logging
import os
import sys
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

# ── Add le-wm root to path so shared modules are importable ──────────────────
_LE_WM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LE_WM_ROOT))

import stable_pretraining as spt           # noqa: E402
import stable_worldmodel as swm            # noqa: E402
from hi_probe import HIProbeCallback       # noqa: E402
from utils import (                        # noqa: E402
    ModelObjectCallBack,
    get_column_normalizer,
)
from baselines.rssm.model import build_rssm  # noqa: E402


# ── Forward / loss step ──────────────────────────────────────────────────────

def rssm_forward(self, batch, stage, cfg):
    """Encoder + dynamics + decoder, with KL-balanced ELBO loss.

    Loss = recon_loss + kl_loss
       recon_loss = mean over (B, T, n_sensors) of (recon - target)^2
       kl_loss    = balanced KL between posterior and prior, with free bits

    Note: AR-LSTM/JEPA's `cfg.wm.history_size`, `h_step`, `num_preds` are
    *not* used here. Dreamer trains on contiguous chunks of length
    `cfg.data.dataset.num_steps` (set explicitly in the YAML).
    """
    output = self.model.world_model_step(batch)

    recon  = output["recon"]   # (B, T, n_sensors)
    target = output["target"]  # (B, T, n_sensors), already symlog'd if symlog_inputs

    recon_loss = (recon - target).pow(2).mean()

    kl_total, dyn_loss, rep_loss = self.model.dynamics.kl_loss(
        output["post"],
        output["prior"],
        free      = cfg.loss.kl_free,
        dyn_scale = cfg.loss.dyn_scale,
        rep_scale = cfg.loss.rep_scale,
    )
    kl_loss = kl_total.mean()

    loss = recon_loss + kl_loss

    output["recon_loss"] = recon_loss.detach()
    output["kl_loss"]    = kl_loss.detach()
    output["dyn_loss"]   = dyn_loss.mean().detach()
    output["rep_loss"]   = rep_loss.mean().detach()
    output["loss"]       = loss

    losses_dict = {
        f"{stage}/{k}": v.detach() if torch.is_tensor(v) else v
        for k, v in output.items() if k.endswith("loss")
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


# ── Main training entry point ─────────────────────────────────────────────────

@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "config"),
    config_name="train_rssm_scenario4",
)
def run(cfg):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    logging.info("=" * 60)
    logging.info("RSSM Baseline (DreamerV3) — TurboSens")
    logging.info(f"Encoder MLP   : {cfg.rssm.encoder_layers}L x {cfg.rssm.encoder_hidden}")
    logging.info(f"RSSM          : stoch={cfg.rssm.stoch}x{cfg.rssm.discrete}, deter={cfg.rssm.deter}")
    logging.info(f"Decoder MLP   : {cfg.rssm.decoder_layers}L x {cfg.rssm.decoder_hidden}")
    logging.info(f"num_steps     : {cfg.data.dataset.num_steps}")
    logging.info(f"num_actions   : {cfg.num_actions}")
    logging.info("=" * 60)

    # ── Data (mirrors train.py / train_ar_lstm.py) ───────────────────────────
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset)

    transforms = []
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            if not col.startswith("observation."):
                setattr(cfg, f"{col}_dim_inferred", dataset.get_dim(col))

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
    world_model = build_rssm(cfg)

    param_count = sum(p.numel() for p in world_model.parameters() if p.requires_grad)
    logging.info(f"Total trainable parameters: {param_count:,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizers = {
        "model_opt": {
            "modules":   "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": max(1, len(train_loader) // 2),
                "max_steps":    cfg.trainer.max_epochs * len(train_loader),
            },
            "interval":  "step",
        },
    }

    # ── Callbacks ─────────────────────────────────────────────────────────────
    run_id  = cfg.get("subdir") or "rssm"
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
            logging.error(
                "[HIProbe] data.dataset.cache_dir not set — pass via CLI."
            )
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
                d_model          = hi_probe_cfg.get("d_model", 128),
                nhead            = hi_probe_cfg.get("nhead", 4),
                num_layers       = hi_probe_cfg.get("num_layers", 2),
                probe_dropout    = hi_probe_cfg.get("probe_dropout", 0.1),
                probe_batch_size = hi_probe_cfg.get("probe_batch_size", 256),
                probe_seq_len    = hi_probe_cfg.get("probe_seq_len", 1),
                n_subsample      = hi_probe_cfg.get("n_subsample", 200_000),
                enc_batch_size   = hi_probe_cfg.get("enc_batch_size", 2048),
                rul_max_horizon  = hi_probe_cfg.get("rul_max_horizon", 300),
                obs_window_size  = 1,
                encoder_type     = "rssm",
            ))
            logging.info(
                f"[HIProbe] Registered — evaluating every "
                f"{hi_probe_cfg.get('eval_interval', 5)} epochs"
            )

    # ── Lightning module ──────────────────────────────────────────────────────
    # Note: no SIGReg term — Dreamer's KL plays the role of the latent
    # regulariser. Pass a no-op identity for the spt.Module slot if needed.
    module = spt.Module(
        model   = world_model,
        forward = partial(rssm_forward, cfg=cfg),
        optim   = optimizers,
    )

    # ── Logger ─────────────────────────────────────────────────────────────────
    logger = None
    if cfg.wandb.enabled:
        wandb_kwargs = dict(cfg.wandb.config)
        if os.environ.get("WANDB_MODE") == "offline":
            wandb_kwargs["offline"] = True
        logger = WandbLogger(**wandb_kwargs)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    pl.seed_everything(cfg.seed)
    trainer.fit(module, datamodule=spt.data.DataModule(train=train_loader, val=val_loader))


if __name__ == "__main__":
    run()
