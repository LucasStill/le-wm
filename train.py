import logging
import os
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


def _encode_strided_actions(raw_action, ctx_len, s, action_encoder):
    """Max-pool raw actions over each stride interval, then encode.

    For binary maintenance actions, max-pool is OR: the interval is marked
    as a maintenance interval if *any* step in it had a maintenance event.
    Works for any action_dim (linear encoder, arbitrary shape).
    """
    intervals = [raw_action[:, k * s : (k + 1) * s].amax(dim=1) for k in range(ctx_len)]
    pooled = torch.stack(intervals, dim=1)   # (B, ctx_len, action_dim)
    return action_encoder(pooled)            # (B, ctx_len, D)


def _zero_pad_prefix(ctx_emb, ctx_act, zero_pad_prob, training):
    """Randomly zero-pad a prefix of context embeddings/actions during training.

    Teaches the model that early-episode positions have no prior context
    (pads with zeros, which = action 0 = do_nothing for actions).
    """
    if not training or zero_pad_prob <= 0.0:
        return ctx_emb, ctx_act
    ctx_len = ctx_emb.size(1)
    if ctx_len < 2 or torch.rand(1).item() >= zero_pad_prob:
        return ctx_emb, ctx_act
    n_pad = torch.randint(1, ctx_len, (1,)).item()
    ctx_emb = ctx_emb.clone()
    ctx_act = ctx_act.clone()
    ctx_emb[:, :n_pad] = 0.0
    ctx_act[:, :n_pad] = 0.0
    return ctx_emb, ctx_act


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len        = cfg.wm.history_size
    s              = cfg.wm.get("h_step", 1)   # temporal stride (1 = original dense)
    lambd          = cfg.loss.sigreg.weight
    zero_pad_prob  = cfg.wm.get("zero_pad_prob", 0.0)

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]      # (B, T, D),  T = H*s + obs_window_size

    # Strided context: z_0, z_s, z_2s, ..., z_{(H-1)*s}  →  (B, H, D)
    ctx_emb = emb[:, :ctx_len * s : s]
    # Max-pool raw actions over each stride interval then encode.
    # For s=1 this is equivalent to encoding the single step (no change).
    ctx_act = _encode_strided_actions(
        batch["action"], ctx_len, s, self.model.action_encoder
    )
    # Strided target: z_s, z_2s, ..., z_{H*s}  →  predict one stride ahead
    tgt_emb = emb[:, s : ctx_len * s + 1 : s]

    # Zero-pad prefix augmentation: randomly mask early positions with zeros
    # to simulate inference at early episode timesteps (no prior context).
    ctx_emb, ctx_act = _zero_pad_prefix(ctx_emb, ctx_act, zero_pad_prob, self.training)

    pred_emb = self.model.predict(ctx_emb, ctx_act)

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output



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

    # Optional: TORCH_COMPILE=1 env var enables torch.compile for ~20-40% speedup.
    # Adds ~2 min warmup on first run (graph compilation). H100 benefits most.
    if os.environ.get("TORCH_COMPILE", "0") == "1":
        world_model = torch.compile(world_model)

    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    logger = None
    if cfg.wandb.enabled:
        wandb_kwargs = dict(cfg.wandb.config)
        # When running on a no-internet node (WANDB_MODE=offline), pass offline=True
        # directly to WandbLogger so it never attempts a network connection.
        # Without this, wandb.init() hangs trying to create a new project on the server.
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
    trainer.fit(world_model, datamodule=data_module)
    return


if __name__ == "__main__":
    run()
