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

from jepa import JEPA
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

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

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

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
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

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
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

    # Optional: HI probe — evaluates downstream degradation prediction every N epochs
    hi_probe_cfg = cfg.get("hi_probe", None)
    if hi_probe_cfg is not None and hi_probe_cfg.get("enabled", False):
        data_path = str(
            Path(cfg.data.dataset.cache_dir) / f"{cfg.data.dataset.name}.h5"
        )
        callbacks.append(HIProbeCallback(
            data_path      = data_path,
            img_size       = cfg.img_size,
            train_split    = cfg.train_split,
            seed           = cfg.seed,
            eval_interval  = hi_probe_cfg.get("eval_interval", 5),
            n_probe_epochs = hi_probe_cfg.get("n_probe_epochs", 100),
            hidden_dim     = hi_probe_cfg.get("hidden_dim", 256),
            n_subsample    = hi_probe_cfg.get("n_subsample", 30_000),
        ))
        logging.info(
            f"[HIProbe] Enabled — evaluating every "
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
