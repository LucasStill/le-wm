import json
import logging
import os
import types
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA, SensorEncoder, TemporalAggregator
from hi_probe import HIProbeCallback
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def _patch_wandb_offline(manager: spt.Manager) -> None:
    """Patch stable_pretraining's init_and_sync_wandb to handle the first
    offline run gracefully.

    Bug: when WANDB_MODE=offline and no previous run exists,
    _wandb_previous_dir() returns None, but the original code unconditionally
    does `None / "files/wandb-config.json"`, crashing with TypeError.
    """

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

    manager.init_and_sync_wandb = types.MethodType(_safe_init_and_sync_wandb, manager)


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    # ── encoder_type dispatch ─────────────────────────────────────────────────
    # "vit"    : ViT-on-fake-image (legacy, keeps backward compat)
    # "sensor" : SensorEncoder — 28 scalar tokens, no image preprocessing
    encoder_type = cfg.get("encoder_type", "vit")

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)

    # Build transforms. ViT needs the img_preprocessor; sensor encoder just
    # uses a StandardScaler normaliser (applied via get_column_normalizer).
    transforms = []
    if encoder_type == "vit":
        transforms.append(
            get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)
        )

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            # 'pixels' handled by img_preprocessor above (vit only)
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            # Set column dims on cfg.wm for downstream use.
            # Skip observation.* keys — their shape is fixed by the encoder.
            if not col.startswith("observation."):
                setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    ##############################
    ##       model / optim      ##
    ##############################

    if encoder_type == "sensor":
        # ── Sensor-native encoder ─────────────────────────────────────────
        sen_cfg = cfg.get("sensor_encoder", {})
        embed_dim = cfg.wm.get("embed_dim", 64)
        encoder = SensorEncoder(
            n_sensors  = cfg.get("n_sensors", 28),
            d_model    = sen_cfg.get("d_model", embed_dim),
            nhead      = sen_cfg.get("nhead", 4),
            num_layers = sen_cfg.get("num_layers", 2),
            dropout    = sen_cfg.get("dropout", 0.1),
        )
        hidden_dim = encoder.hidden_size
        logging.info(
            f"[Encoder] SensorEncoder — n_sensors={cfg.get('n_sensors', 28)}, "
            f"d_model={hidden_dim}"
        )
    else:
        # ── ViT encoder (original) ────────────────────────────────────────
        encoder = spt.backbone.utils.vit_hf(
            cfg.encoder_scale,
            patch_size=cfg.patch_size,
            image_size=cfg.img_size,
            pretrained=False,
            use_mask_token=False,
        )
        hidden_dim = encoder.config.hidden_size
        embed_dim  = cfg.wm.get("embed_dim", hidden_dim)
        logging.info(
            f"[Encoder] ViT-{cfg.encoder_scale} — hidden_dim={hidden_dim}, "
            f"embed_dim={embed_dim}"
        )

    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

    # Scale projector hidden dim proportionally to input/output dims.
    # Avoids a 64→2048→64 bottleneck (32× over-expansion) for sensor encoder
    # while keeping a reasonable 192→768→64 expansion for ViT.
    proj_hidden = max(hidden_dim * 4, embed_dim * 4)
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=proj_hidden,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=proj_hidden,
        norm_fn=torch.nn.BatchNorm1d,
    )

    obs_window_size = cfg.get("obs_window_size", 1)
    temporal_agg = None
    if obs_window_size > 1:
        temporal_agg = TemporalAggregator(
            embed_dim=embed_dim,
            max_window=obs_window_size + 4,     # small margin above window size
            nhead=4,
            num_layers=1,
            dropout=0.1,
        )
        logging.info(
            f"[ObsWindow] obs_window_size={obs_window_size} — "
            f"TemporalAggregator created (embed_dim={embed_dim})"
        )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
        temporal_agg=temporal_agg,
        obs_window_size=obs_window_size,
        encoder_type=encoder_type,
    )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    ##########################
    ##       callbacks      ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelObjectCallBack(dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1),
    ]

    # Optional: HI probe — evaluates downstream degradation prediction every N epochs.
    # Config lives at top-level cfg.hi_probe (NOT cfg.data.hi_probe) because
    # the data config is loaded under cfg.data.* by Hydra.
    hi_probe_cfg = cfg.get("hi_probe", None)
    logging.info(f"[HIProbe] Config found: {hi_probe_cfg is not None}, "
                 f"enabled: {hi_probe_cfg.get('enabled', False) if hi_probe_cfg else False}")

    if hi_probe_cfg is not None and hi_probe_cfg.get("enabled", False):
        cache_dir = cfg.data.dataset.get("cache_dir")
        name      = cfg.data.dataset.get("name", "dataset")
        if not cache_dir or cache_dir == "???":
            logging.error(
                "[HIProbe] data.dataset.cache_dir is not set — "
                "pass it via CLI: data.dataset.cache_dir=/path/to/data"
            )
        else:
            data_path = str(Path(cache_dir) / f"{name}.h5")
            logging.info(f"[HIProbe] data_path = {data_path}")
            callbacks.append(HIProbeCallback(
                data_path        = data_path,
                img_size         = cfg.img_size,
                train_split      = cfg.train_split,
                seed             = cfg.seed,
                eval_interval    = hi_probe_cfg.get("eval_interval", 5),
                n_probe_epochs   = hi_probe_cfg.get("n_probe_epochs", 50),
                probe_lr         = hi_probe_cfg.get("probe_lr", 1e-3),
                probe_patience   = hi_probe_cfg.get("probe_patience", 10),
                d_model          = hi_probe_cfg.get("d_model", 64),
                nhead            = hi_probe_cfg.get("nhead", 4),
                num_layers       = hi_probe_cfg.get("num_layers", 2),
                probe_dropout    = hi_probe_cfg.get("probe_dropout", 0.1),
                probe_batch_size = hi_probe_cfg.get("probe_batch_size", 256),
                probe_seq_len    = hi_probe_cfg.get("probe_seq_len", 1),
                n_subsample      = hi_probe_cfg.get("n_subsample", 30_000),
                enc_batch_size   = hi_probe_cfg.get("enc_batch_size", 2048),
                rul_max_horizon  = hi_probe_cfg.get("rul_max_horizon", 300),
                obs_window_size  = obs_window_size,
                encoder_type     = encoder_type,
            ))
            logging.info(
                f"[HIProbe] ✓ Registered — evaluating every "
                f"{hi_probe_cfg.get('eval_interval', 5)} epochs"
            )

    ##########################
    ##       training       ##
    ##########################

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
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

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    # Fix stable_pretraining bug: offline mode crashes on first run
    if os.environ.get("WANDB_MODE") == "offline":
        _patch_wandb_offline(manager)

    manager()
    return


if __name__ == "__main__":
    run()
