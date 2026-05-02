"""
DreamerV3 RSSM Baseline for TurboSens
======================================

A Recurrent State-Space Model (RSSM) baseline that mirrors the DreamerV3
world model component (no actor, no critic, no reward head). Reconstructs
the raw sensor observation, in contrast to the JEPA / AR-LSTM baselines
which never reconstruct.

Design choices (locked in with the user):
  * Encoder: 2-layer MLP on the n_sensors-dim flat observation.
  * Dynamics: RSSM with a categorical stochastic latent (32 cats x 32 dims)
    on top of a deterministic GRU state (deter=512). Matches V3 default.
  * Decoder: 2-layer MLP reconstructing the n_sensors-dim observation.
  * KL: balanced (dyn / rep), with free bits (kl_free=1.0).
  * No reward head, no continuation head, no policy.

The model exposes the same `encode(info)` interface as JEPA so it can plug
into HIProbeCallback unchanged. `encode()` runs the encoder MLP through
`dynamics.observe`, returning the per-step model state
``feat_t = [h_t, z_t]`` of dimension `deter + stoch * discrete` under
``info["emb"]``.

Training uses a separate `world_model_step()` method that returns the
posterior, prior, decoded observation and embed needed by `dreamer_forward`
in train_rssm.py.

Reference: NM512/dreamerv3-torch/networks.py (RSSM class).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import distributions as torchd

# ── Add le-wm root so we can import shared modules ───────────────────────────
_LE_WM_ROOT = Path(__file__).resolve().parents[2]
if str(_LE_WM_ROOT) not in sys.path:
    sys.path.insert(0, str(_LE_WM_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Categorical distribution helper (one-hot with straight-through gradient).
# Lifted from dreamerv3-torch/tools.py::OneHotDist, simplified.
# ─────────────────────────────────────────────────────────────────────────────

class OneHotDist(torchd.OneHotCategoricalStraightThrough):
    """Straight-through one-hot categorical with optional uniform mixing.

    DreamerV3 mixes the categorical with a small uniform component
    (`unimix_ratio`) so the prior never assigns zero probability to a
    category, which keeps the KL well-defined when the posterior chooses
    something the prior would have rejected.
    """

    def __init__(self, logits=None, probs=None, unimix_ratio: float = 0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        # Argmax of probs, with straight-through: forward = hard, backward = probs.
        idx = super().probs.argmax(dim=-1)
        hard = F.one_hot(idx, num_classes=super().probs.shape[-1]).to(super().probs)
        return hard + (super().probs - super().probs.detach())

    def sample(self, sample_shape=torch.Size(), seed=None):
        if seed is not None:
            raise NotImplementedError("seeded sampling not supported")
        sample = super().sample(sample_shape)
        # Straight-through: gradient flows through probs.
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs.unsqueeze(0)
        sample += probs - probs.detach()
        return sample


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks: GRU cell with optional LayerNorm, MLP encoder/decoder.
# ─────────────────────────────────────────────────────────────────────────────

class LayerNormGRUCell(nn.Module):
    """GRU cell with LayerNorm on the gate pre-activations.

    DreamerV3 uses this in place of the vanilla `nn.GRUCell` for stability
    when the deter state is large.
    """

    def __init__(self, input_size: int, hidden_size: int, norm: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size + hidden_size, 3 * hidden_size, bias=False)
        self.norm = nn.LayerNorm(3 * hidden_size, eps=1e-3) if norm else nn.Identity()

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([x, h], dim=-1)
        gates = self.norm(self.linear(cat))
        reset, cand, update = gates.chunk(3, dim=-1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1.0)  # bias toward keeping previous state
        return update * cand + (1.0 - update) * h


class MLPEncoder(nn.Module):
    """Plain MLP encoder for proprioceptive / sensor input.

    DreamerV3's MultiEncoder collapses to this when the observation has
    no spatial dimensions (sensor data only).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        out_dim: int = 512,
        num_layers: int = 2,
        symlog_inputs: bool = True,
    ):
        super().__init__()
        self.symlog_inputs = symlog_inputs
        layers: list[nn.Module] = []
        d = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_dim, bias=False),
                       nn.LayerNorm(hidden_dim, eps=1e-3),
                       nn.SiLU()]
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.symlog_inputs:
            x = torch.sign(x) * torch.log1p(torch.abs(x))
        return self.net(x)


class MLPDecoder(nn.Module):
    """MLP decoder reconstructing the n_sensors-dim observation.

    Output is interpreted as the mean of an isotropic Gaussian with unit
    variance, so the negative log-likelihood reduces to MSE/2 plus a
    constant — we just use plain MSE in the trainer.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_dim, bias=False),
                       nn.LayerNorm(hidden_dim, eps=1e-3),
                       nn.SiLU()]
            d = hidden_dim
        layers += [nn.Linear(d, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


# ─────────────────────────────────────────────────────────────────────────────
# RSSM (categorical V3 variant).
# ─────────────────────────────────────────────────────────────────────────────

class RSSM(nn.Module):
    """Recurrent State-Space Model with categorical stochastic latent.

    Each step:
      prior  p(z_t | h_t, a_{t-1})  via  img_step
      post   q(z_t | h_t, x_t)      via  obs_step  (uses the obs embed)

    All gradients flow through both prior and post; the KL is balanced
    via two stop-gradient terms (rep_loss, dyn_loss).

    Parameters
    ----------
    embed_size  : dimensionality of the encoder MLP output.
    num_actions : number of discrete actions (one-hot encoded).
    stoch       : number of stochastic latents (z dims). Default 32.
    discrete    : number of categories per stoch dim. Default 32.
                  Set to 0 to fall back to a Gaussian latent (V1/V2 style).
    deter       : deterministic GRU state size. Default 512.
    hidden      : intermediate MLP hidden size inside the cell. Default 512.
    """

    def __init__(
        self,
        embed_size: int,
        num_actions: int,
        stoch: int = 32,
        discrete: int = 32,
        deter: int = 512,
        hidden: int = 512,
        unimix_ratio: float = 0.01,
        norm: bool = True,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.num_actions = num_actions
        self.stoch = stoch
        self.discrete = discrete
        self.deter = deter
        self.hidden = hidden
        self.unimix_ratio = unimix_ratio
        assert discrete > 0, "Only categorical RSSM is supported in this baseline."

        stoch_flat = stoch * discrete

        # Pre-cell: project (z, a) -> hidden
        self.img_in = nn.Sequential(
            nn.Linear(stoch_flat + num_actions, hidden, bias=False),
            nn.LayerNorm(hidden, eps=1e-3) if norm else nn.Identity(),
            nn.SiLU(),
        )
        self.cell = LayerNormGRUCell(hidden, deter, norm=norm)

        # Post-cell: project deter -> hidden -> stochastic logits (prior path)
        self.img_out = nn.Sequential(
            nn.Linear(deter, hidden, bias=False),
            nn.LayerNorm(hidden, eps=1e-3) if norm else nn.Identity(),
            nn.SiLU(),
        )
        # Posterior path: project (deter, embed) -> hidden -> logits
        self.obs_out = nn.Sequential(
            nn.Linear(deter + embed_size, hidden, bias=False),
            nn.LayerNorm(hidden, eps=1e-3) if norm else nn.Identity(),
            nn.SiLU(),
        )

        self.prior_logits = nn.Linear(hidden, stoch_flat)
        self.post_logits = nn.Linear(hidden, stoch_flat)

        # Learnable initial deter state.
        self.W = nn.Parameter(torch.zeros(1, deter))

    @property
    def feat_size(self) -> int:
        return self.deter + self.stoch * self.discrete

    # ── state initialisation ────────────────────────────────────────────────

    def initial(self, batch_size: int, device) -> dict:
        deter = torch.tanh(self.W).expand(batch_size, -1).contiguous()
        x = self.img_out(deter)
        logits = self.prior_logits(x).view(batch_size, self.stoch, self.discrete)
        stoch = OneHotDist(logits=logits, unimix_ratio=self.unimix_ratio).mode()
        return {"deter": deter, "stoch": stoch, "logit": logits}

    # ── single-step transitions ────────────────────────────────────────────

    def img_step(self, prev_state: dict, prev_action: torch.Tensor) -> dict:
        """Prior step: advance deter via GRU, sample z from p(z | h)."""
        prev_stoch = prev_state["stoch"]
        prev_stoch = prev_stoch.reshape(prev_stoch.shape[0], -1)  # flatten cats
        x = self.img_in(torch.cat([prev_stoch, prev_action], dim=-1))
        deter = self.cell(x, prev_state["deter"])
        x = self.img_out(deter)
        logits = self.prior_logits(x).view(-1, self.stoch, self.discrete)
        stoch = OneHotDist(logits=logits, unimix_ratio=self.unimix_ratio).sample()
        return {"deter": deter, "stoch": stoch, "logit": logits}

    def obs_step(
        self,
        prev_state: dict,
        prev_action: torch.Tensor,
        embed: torch.Tensor,
        is_first: torch.Tensor,
    ) -> tuple[dict, dict]:
        """Posterior step: same as img_step but z is also conditioned on embed."""
        # Reset prev_state where is_first.
        if is_first.any():
            mask = is_first.float().unsqueeze(-1)              # (B, 1)
            init = self.initial(prev_state["deter"].shape[0], embed.device)
            for k in prev_state:
                expand_mask = mask.view(mask.shape[0], *([1] * (prev_state[k].ndim - 1)))
                prev_state[k] = prev_state[k] * (1.0 - expand_mask) + init[k] * expand_mask
            prev_action = prev_action * (1.0 - mask)

        prior = self.img_step(prev_state, prev_action)
        x = self.obs_out(torch.cat([prior["deter"], embed], dim=-1))
        logits = self.post_logits(x).view(-1, self.stoch, self.discrete)
        stoch = OneHotDist(logits=logits, unimix_ratio=self.unimix_ratio).sample()
        post = {"deter": prior["deter"], "stoch": stoch, "logit": logits}
        return post, prior

    # ── full-sequence rollouts ─────────────────────────────────────────────

    def observe(
        self,
        embed: torch.Tensor,        # (B, T, embed_size)
        action: torch.Tensor,       # (B, T, num_actions)  one-hot
        is_first: torch.Tensor,     # (B, T)
    ) -> tuple[dict, dict]:
        """Roll posterior through a sequence. Returns post and prior dicts
        whose entries are (B, T, ...) tensors."""
        B, T, _ = embed.shape
        state = self.initial(B, embed.device)
        prev_action = torch.zeros(B, self.num_actions, device=embed.device)

        post_list, prior_list = [], []
        for t in range(T):
            post_t, prior_t = self.obs_step(state, prev_action, embed[:, t], is_first[:, t])
            post_list.append(post_t)
            prior_list.append(prior_t)
            state = post_t
            prev_action = action[:, t]

        def stack(seq: list[dict]) -> dict:
            return {k: torch.stack([s[k] for s in seq], dim=1) for k in seq[0]}

        return stack(post_list), stack(prior_list)

    # ── feature extraction & loss ──────────────────────────────────────────

    def get_feat(self, state: dict) -> torch.Tensor:
        """Return feat = concat(stoch_flat, deter), the standard Dreamer feature
        used by all downstream heads."""
        stoch = state["stoch"]
        flat_shape = list(stoch.shape[:-2]) + [self.stoch * self.discrete]
        stoch = stoch.reshape(*flat_shape)
        return torch.cat([stoch, state["deter"]], dim=-1)

    def kl_loss(
        self,
        post: dict,
        prior: dict,
        free: float = 1.0,
        dyn_scale: float = 0.5,
        rep_scale: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Balanced KL with free-bits clamp.

        Returns (total_kl, dyn_loss, rep_loss). Each is a (B, T) scalar.
        """
        def dist(logits):
            return torchd.Independent(
                OneHotDist(logits=logits, unimix_ratio=self.unimix_ratio), 1
            )

        post_d = dist(post["logit"])
        prior_d = dist(prior["logit"])
        post_sg = dist(post["logit"].detach())
        prior_sg = dist(prior["logit"].detach())

        rep_loss = torchd.kl.kl_divergence(post_d, prior_sg)   # train post toward (sg) prior
        dyn_loss = torchd.kl.kl_divergence(post_sg, prior_d)   # train prior toward (sg) post
        rep_loss = torch.clamp(rep_loss, min=free)
        dyn_loss = torch.clamp(dyn_loss, min=free)
        total = dyn_scale * dyn_loss + rep_scale * rep_loss
        return total, dyn_loss, rep_loss


# ─────────────────────────────────────────────────────────────────────────────
# RSSMWorldModel: glues encoder + RSSM + decoder, exposes JEPA-compatible
# encode() so the HI probe can plug in unchanged.
# ─────────────────────────────────────────────────────────────────────────────

class RSSMWorldModel(nn.Module):
    """World model that bundles MLPEncoder + RSSM + MLPDecoder.

    Two interfaces:
      - `encode(info)`   : JEPA-compatible call used by HIProbeCallback.
                           Returns ``info["emb"] = feat`` of shape
                           (B, T, deter + stoch*discrete).
      - `world_model_step(info)` : full forward pass returning post, prior,
                           recon and embed for the loss in train_rssm.py.
    """

    def __init__(
        self,
        n_sensors: int,
        num_actions: int,
        embed_size: int = 512,
        encoder_hidden: int = 512,
        decoder_hidden: int = 512,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        symlog_inputs: bool = True,
        rssm_kwargs: dict | None = None,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        self.num_actions = num_actions
        self.embed_size = embed_size
        self.symlog_inputs = symlog_inputs

        self.encoder = MLPEncoder(
            input_dim=n_sensors,
            hidden_dim=encoder_hidden,
            out_dim=embed_size,
            num_layers=encoder_layers,
            symlog_inputs=symlog_inputs,
        )

        rssm_kwargs = rssm_kwargs or {}
        self.dynamics = RSSM(
            embed_size=embed_size,
            num_actions=num_actions,
            **rssm_kwargs,
        )

        self.decoder = MLPDecoder(
            input_dim=self.dynamics.feat_size,
            output_dim=n_sensors,
            hidden_dim=decoder_hidden,
            num_layers=decoder_layers,
        )

        # Compatibility shims so this module looks like a JEPA from outside.
        self.encoder_type = "rssm"
        self.obs_window_size = 1

    # ── helpers ─────────────────────────────────────────────────────────────

    def _flatten_obs(self, info: dict) -> torch.Tensor:
        """Pull observations from info['observation.sensors'] or info['pixels']
        and flatten any spatial dims to (B, T, n_sensors)."""
        if "observation.sensors" in info:
            x = info["observation.sensors"].float()
        else:
            x = info["pixels"].float()
        if x.ndim > 3:
            B, T = x.shape[:2]
            x = x.reshape(B, T, -1)
        return x

    def _one_hot_actions(self, action: torch.Tensor) -> torch.Tensor:
        """Convert scalar action ids (B, T) -> one-hot (B, T, num_actions).

        Robust to extra trailing dims (e.g. (B, T, 1)) by squeezing.
        """
        if action.ndim == 3 and action.shape[-1] == 1:
            action = action.squeeze(-1)
        action = torch.nan_to_num(action, 0.0).long().clamp(0, self.num_actions - 1)
        return F.one_hot(action, num_classes=self.num_actions).float()

    # ── JEPA-compatible encode() for HIProbeCallback ───────────────────────

    def encode(self, info: dict) -> dict:
        """Run encoder + dynamics over the input sequence; return per-step feat.

        The HI probe passes single-frame (B, 1, n_sensors) batches when
        ``probe_seq_len == 1``, in which case the dynamics is rolled for one
        step starting from the learned init state. For larger seq_len it
        rolls through the whole window.

        The probe expects ``info["emb"]`` to have shape (B, T, D); we honour
        that contract by returning feat = [stoch_flat, deter] per timestep.
        """
        obs = self._flatten_obs(info)                               # (B, T, n_sensors)
        B, T, _ = obs.shape

        if "action" in info:
            action = self._one_hot_actions(info["action"])          # (B, T, A)
        else:
            action = torch.zeros(B, T, self.num_actions, device=obs.device)

        is_first = torch.zeros(B, T, dtype=torch.bool, device=obs.device)
        is_first[:, 0] = True

        embed = self.encoder(rearrange(obs, "b t d -> (b t) d"))    # (B*T, embed_size)
        embed = rearrange(embed, "(b t) d -> b t d", b=B)

        post, _prior = self.dynamics.observe(embed, action, is_first)
        feat = self.dynamics.get_feat(post)                         # (B, T, deter+stoch*disc)

        info["emb"] = feat
        return info

    # ── full forward step used by the trainer ──────────────────────────────

    def world_model_step(self, info: dict) -> dict:
        """Encoder -> RSSM -> decoder. Returns dict with everything the
        loss function needs."""
        obs = self._flatten_obs(info)                               # (B, T, n_sensors)
        B, T, _ = obs.shape

        action = self._one_hot_actions(info["action"])              # (B, T, A)
        is_first = torch.zeros(B, T, dtype=torch.bool, device=obs.device)
        is_first[:, 0] = True

        embed = self.encoder(rearrange(obs, "b t d -> (b t) d"))    # (B*T, embed_size)
        embed = rearrange(embed, "(b t) d -> b t d", b=B)

        post, prior = self.dynamics.observe(embed, action, is_first)
        feat = self.dynamics.get_feat(post)
        recon = self.decoder(rearrange(feat, "b t d -> (b t) d"))
        recon = rearrange(recon, "(b t) d -> b t d", b=B)

        # Symlog target if encoder used symlog inputs (V3 convention: train
        # decoder to match the symlog'd target so the encoder/decoder share
        # the same nonlinearity).
        target = obs
        if self.symlog_inputs:
            target = torch.sign(obs) * torch.log1p(torch.abs(obs))

        return {
            "post":   post,
            "prior":  prior,
            "feat":   feat,
            "embed":  embed,
            "recon":  recon,
            "target": target,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory: build_rssm(cfg)
# ─────────────────────────────────────────────────────────────────────────────

def build_rssm(cfg) -> RSSMWorldModel:
    """Build an RSSMWorldModel from a Hydra config."""
    rssm_cfg = cfg.get("rssm", {})

    rssm_kwargs = dict(
        stoch        = rssm_cfg.get("stoch", 32),
        discrete     = rssm_cfg.get("discrete", 32),
        deter        = rssm_cfg.get("deter", 512),
        hidden       = rssm_cfg.get("hidden", 512),
        unimix_ratio = rssm_cfg.get("unimix_ratio", 0.01),
        norm         = rssm_cfg.get("norm", True),
    )

    model = RSSMWorldModel(
        n_sensors      = cfg.get("n_sensors", 176),
        num_actions    = cfg.get("num_actions", 7),
        embed_size     = rssm_cfg.get("embed_size", 512),
        encoder_hidden = rssm_cfg.get("encoder_hidden", 512),
        decoder_hidden = rssm_cfg.get("decoder_hidden", 512),
        encoder_layers = rssm_cfg.get("encoder_layers", 2),
        decoder_layers = rssm_cfg.get("decoder_layers", 2),
        symlog_inputs  = rssm_cfg.get("symlog_inputs", True),
        rssm_kwargs    = rssm_kwargs,
    )

    nparams = sum(p.numel() for p in model.parameters())
    logging.info(
        f"[RSSM] n_sensors={cfg.get('n_sensors', 176)}, num_actions={cfg.get('num_actions', 7)}, "
        f"feat={model.dynamics.feat_size}, params={nparams:,}"
    )
    return model
