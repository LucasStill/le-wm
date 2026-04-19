"""
AR-LSTM Baseline Training Script
==================================

This script trains the AR-LSTM world model on the TurboSens dataset.
It is intentionally written to be as close as possible to train.py so
that the *only* visible difference is the predictor module — demonstrating
that swapping world model architectures on TurboSens takes ~10 lines.

Usage (local):
    cd le-wm/
    python baselines/ar_lstm/train_ar_lstm.py \
        --config-path baselines/ar_lstm/config \
        --config-name train_ar_lstm \
        data.dataset.cache_dir=/path/to/data \
        hi_probe.enabled=true

Usage (cluster):
    sbatch baselines/ar_lstm/train_ar_lstm.slurm
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
from baselines.ar_lstm.model import build_ar_lstm  # noqa: E402


def _patch_wandb_offline(manager: spt.Manager) -> None:
    """Patch spt.Manager.init_and_sync_wandb for offline mode.

    Bug in stable_pretraining: when WANDB_MODE=offline and a previous offline
    run exists, _wandb_previous_dir() returns None (offline runs have no server
    path), but the original code does `None / "files/wandb-config.json"` →
    TypeError.  Replace with a version that guards against None.
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

    type(manager).init_and_sync_wandb = _safe_init_and_sync_wandb


# ── Training forward pass (identical to train.py: lejepa_forward) ─────────────
# The LSTM predictor exposes the same predict() interface as the transformer,
# so the forward pass needs zero modification.

def ar_lstm_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, compute losses.

    Identical to lejepa_forward in train.py — the LSTM and transformer
    predictors share the same encode()/predict() interface.
    """
    ctx_len = cfg.wm.history_size  # H: number of context embeddings
    s       = cfg.wm.get("h_step", 1)   # temporal stride (1 = original dense)
    lambd   = cfg.loss.sigreg.weight

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    # Step 1: encode all T_raw frames → Z = (B, T, embed_dim)
    output  = self.model.encode(batch)
    emb     = output["emb"]       # (B, T, D),  T = H*s + obs_window_size
    act_emb = output["act_emb"]   # (B, T, D)

    # Step 2: strided context and target
    # Context: z_0, z_s, z_2s, ..., z_{(H-1)*s}  →  (B, H, D)
    ctx_emb = emb[:, :ctx_len * s : s]
    # Last action in each stride interval (captures any maintenance event)
    ctx_act = act_emb[:, s - 1 : ctx_len * s : s]
    # Target: z_s, z_2s, ..., z_{H*s}  →  predict one stride ahead
    tgt_emb = emb[:, s : ctx_len * s + 1 : s]

    # Step 3: LSTM predicts ẑ_{t+s} for each context position
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # (B, H, D)

    # Step 4: losses (identical to JEPA)
    output["pred_loss"]   = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"]        = output["pred_loss"] + lambd * output["sigreg_loss"]

    losses_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items() if "loss" in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


# ── Main training entry point ─────────────────────────────────────────────────

@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "config"),  # absolute → no CLI doubling
    config_name="train_ar_lstm",
)
def run(cfg):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    logging.info("=" * 60)
    logging.info("AR-LSTM Baseline — TurboSens")
    logging.info(f"Predictor  : LSTM  (hidden={cfg.lstm.hidden_dim}, "
                 f"layers={cfg.lstm.num_layers})")
    logging.info(f"Encoder    : SensorEncoder  (d={cfg.wm.embed_dim}, w={cfg.obs_window_size})")
    logging.info(f"History H  : {cfg.wm.history_size}")
    logging.info("=" * 60)

    # ── Data (identical to train.py) ──────────────────────────────────────────
    # TurboSens makes this easy: point at the HDF5 file and go.
    with open_dict(cfg):
        cfg.wm.action_dim = 1  # binary maintenance action

    # num_steps is already computed in the config via ${eval:'...'} interpolation
    # (history_size + num_preds + obs_window_size - 1), so no need to pass it again.
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset)

    transforms = []
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            if not col.startswith("observation."):
                setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

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
    # This is the only section that differs from train.py.
    # build_ar_lstm() returns a JEPA container with an LSTM predictor;
    # everything else — data, probes, checkpointing — is unchanged.
    world_model = build_ar_lstm(cfg)

    param_count = sum(p.numel() for p in world_model.parameters() if p.requires_grad)
    logging.info(f"Total trainable parameters: {param_count:,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizers = {
        "model_opt": {
            "modules":   "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                # spt.Manager normally injects these from dataset size; we compute explicitly.
                "warmup_steps": max(1, len(train_loader) // 2),          # ~0.5 epoch warmup
                "max_steps":    cfg.trainer.max_epochs * len(train_loader),
            },
            "interval":  "step",
        },
    }

    # ── Callbacks ─────────────────────────────────────────────────────────────
    run_id  = cfg.get("subdir") or "ar_lstm"
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelObjectCallBack(
            dirpath=run_dir,
            filename=cfg.output_model_name,
            epoch_interval=1,
        ),
    ]

    # HI probe: reused without modification from the JEPA training pipeline.
    # The probe only calls model.encode(), which the LSTM exposes identically.
    hi_probe_cfg = cfg.get("hi_probe", None)
    if hi_probe_cfg is not None and hi_probe_cfg.get("enabled", False):
        cache_dir = cfg.data.dataset.get("cache_dir")
        name      = cfg.data.dataset.get("name", "dataset")
        if not cache_dir or cache_dir == "???":
            logging.error(
                "[HIProbe] data.dataset.cache_dir not set — "
                "pass via CLI: data.dataset.cache_dir=/path/to/data"
            )
        else:
            data_path = str(Path(cache_dir) / f"{name}.h5")
            callbacks.append(HIProbeCallback(
                data_path        = data_path,
                img_size         = cfg.img_size,
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

    # ── Lightning module ───────────────────────────────────────────────────────
    module = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(ar_lstm_forward, cfg=cfg),
        optim=optimizers,
    )

    # ── Logger ─────────────────────────────────────────────────────────────────
    logger = None
    if cfg.wandb.enabled:
        wandb_kwargs = dict(cfg.wandb.config)
        if os.environ.get("WANDB_MODE") == "offline":
            wandb_kwargs["offline"] = True
        logger = WandbLogger(**wandb_kwargs)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    # Save config snapshot alongside the checkpoint
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=train_loader, val=val_loader),
        ckpt_path=None,
    )

    if os.environ.get("WANDB_MODE") == "offline":
        _patch_wandb_offline(manager)

    manager()


if __name__ == "__main__":
    run()
