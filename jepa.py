"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


# ── Temporal aggregator for obs_window_size > 1 ──────────────────────────────

class TemporalAggregator(nn.Module):
    """Aggregate a window of w per-frame embeddings into a single embedding.

    Architecture:
      - Learnable positional embedding for positions 0..max_window-1
      - Single TransformerEncoderLayer (batch_first)
      - Output = last token (most recent frame), which attends over the full window

    Input:  (B, w, D)   — w per-frame embeddings from the ViT+projector
    Output: (B, D)       — single aggregated embedding
    """

    def __init__(self, embed_dim: int, max_window: int = 32,
                 nhead: int = 4, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, max_window, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, w, D)
        w = x.size(1)
        x = x + self.pos_emb[:, :w]
        x = self.transformer(x)
        return x[:, -1]                    # last token → (B, D)


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        temporal_agg=None,
        obs_window_size: int = 1,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.temporal_agg = temporal_agg
        self.obs_window_size = obs_window_size

    def encode(self, info):
        """Encode observations and actions into embeddings.

        info: dict with 'pixels' (B, T_raw, C, H, W) and optionally 'action'.

        When obs_window_size (w) > 1 the method:
          1. Encodes ALL T_raw frames with the shared ViT (each frame once).
          2. Builds sliding windows of w consecutive frame embeddings.
          3. Aggregates each window via the TemporalAggregator.
          4. Returns T_out = T_raw - w + 1 embeddings per batch item.

        With w=1 the behaviour is unchanged from the original code.
        """
        pixels = info['pixels'].float()             # (B, T_raw, C, H, W)
        b = pixels.size(0)
        w = self.obs_window_size

        # ── Step 1: encode every frame with the ViT (shared weights) ──────
        pixels_flat = rearrange(pixels, "b t ... -> (b t) ...")   # (B*T_raw, C, H, W)
        output = self.encoder(pixels_flat, interpolate_pos_encoding=True)
        frame_emb = output.last_hidden_state[:, 0]               # CLS → (B*T_raw, hidden)

        if w <= 1 or self.temporal_agg is None:
            # ── No temporal aggregation ───────────────────────────────────
            emb = self.projector(frame_emb)                       # (B*T_raw, embed_dim)
            info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
        else:
            # ── Step 2: project per-frame, then build sliding windows ─────
            frame_proj = self.projector(frame_emb)                # (B*T_raw, embed_dim)
            frame_proj = rearrange(frame_proj, "(b t) d -> b t d", b=b)
            T_raw = frame_proj.size(1)
            T_out = T_raw - w + 1

            # Sliding windows: (B, T_out, w, embed_dim)
            windows = torch.stack(
                [frame_proj[:, t : t + w] for t in range(T_out)], dim=1
            )
            # ── Step 3: aggregate each window ─────────────────────────────
            windows_flat = rearrange(windows, "b t w d -> (b t) w d")
            agg = self.temporal_agg(windows_flat)                 # (B*T_out, embed_dim)
            info["emb"] = rearrange(agg, "(b t) d -> b t d", b=b)

        # ── Action embeddings ─────────────────────────────────────────────
        if "action" in info:
            if w > 1:
                # Align actions with the output timesteps (last frame of each
                # window).  Feed the full action sequence to the Conv1d-based
                # Embedder so it can see temporal context, then slice.
                act_emb_full = self.action_encoder(info["action"])  # (B, T_raw, D)
                info["act_emb"] = act_emb_full[:, w - 1 :]         # (B, T_out, D)
            else:
                info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        
        return cost
