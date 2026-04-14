"""
AR-LSTM Predictor for TurboSens
================================

Drop-in replacement for the transformer ARPredictor used in JEPA.
Exposes the same forward(x, c) → (B, T, D) interface, so it can be
plugged into the JEPA container without touching any other code.

The LSTM is causal by construction: at each position t, the hidden
state has seen only inputs 0..t, so output[t] is a valid prediction
of the embedding at t+1 — identical semantics to the causal transformer.

Usage
-----
    from baselines.ar_lstm.model import LSTMPredictor, build_ar_lstm
    model = build_ar_lstm(cfg)          # returns a JEPA instance
    model.predict(emb, act_emb)         # identical to JEPA.predict()
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# ── Add le-wm root so we can import JEPA, SensorEncoder, etc. ────────────────
_LE_WM_ROOT = Path(__file__).resolve().parents[2]
if str(_LE_WM_ROOT) not in sys.path:
    sys.path.insert(0, str(_LE_WM_ROOT))

from jepa import JEPA, SensorEncoder, TemporalAggregator  # noqa: E402
from module import Embedder, MLP                           # noqa: E402


# ── LSTM predictor ────────────────────────────────────────────────────────────

class LSTMPredictor(nn.Module):
    """Autoregressive LSTM predictor — same interface as ARPredictor.

    Takes a context of T (embedding, action) pairs and returns T
    one-step-ahead predictions in the same latent space.

    The LSTM is causal: output at position t has seen inputs 0..t,
    predicting the latent state at t+1.  This mirrors the causal masking
    used in the transformer ARPredictor.

    Parameters
    ----------
    embed_dim    : dimensionality of latent embeddings (input and output)
    act_emb_dim  : dimensionality of action embeddings (concatenated with emb)
    hidden_dim   : LSTM hidden state size
    num_layers   : number of stacked LSTM layers
    dropout      : dropout between LSTM layers (only active when num_layers > 1)
    """

    def __init__(
        self,
        embed_dim: int   = 64,
        act_emb_dim: int = 64,
        hidden_dim: int  = 256,
        num_layers: int  = 2,
        dropout: float   = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.drop = nn.Dropout(dropout) if (num_layers > 1 and dropout > 0) else nn.Identity()

        # LSTMCell loop instead of nn.LSTM — avoids flatten_parameters() / cuDNN init,
        # which fails on V100 (SM 7.0) when PyTorch is compiled with cuDNN >= 9.x.
        # Semantics are identical: same weights, same computation, same gradients.
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = (embed_dim + act_emb_dim) if i == 0 else hidden_dim
            self.cells.append(nn.LSTMCell(in_dim, hidden_dim))

        # Map hidden state back to embedding space
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, embed_dim)    — context latent embeddings
        c : (B, T, act_emb_dim)  — context action embeddings

        Returns
        -------
        (B, T, embed_dim)  — predicted next-step embeddings (causal)
        """
        inp = torch.cat([x, c], dim=-1)          # (B, T, embed_dim + act_emb_dim)
        B, T, _ = inp.shape

        # Initialise hidden + cell states to zero
        h = [torch.zeros(B, self.hidden_dim, device=inp.device, dtype=inp.dtype)
             for _ in self.cells]
        c_state = [torch.zeros_like(h[i]) for i in range(self.num_layers)]

        outs = []
        for t in range(T):
            x_t = inp[:, t]                      # (B, input_dim)
            for i, cell in enumerate(self.cells):
                h[i], c_state[i] = cell(x_t, (h[i], c_state[i]))
                x_t = self.drop(h[i])            # inter-layer dropout; Identity if disabled
            outs.append(h[-1])                   # top-layer hidden state

        out = torch.stack(outs, dim=1)            # (B, T, hidden_dim)
        return self.output_head(out)              # (B, T, embed_dim)


# ── Factory: build a JEPA container with an LSTM predictor ───────────────────

def build_ar_lstm(cfg) -> JEPA:
    """Build a JEPA world model with an LSTM predictor.

    This is the only function that differs from train.py.  Everything else
    — data loading, HI probing, OOD evaluation — is identical.

    Parameters
    ----------
    cfg : OmegaConf DictConfig with fields:
        n_sensors          (int, default 28)
        sensor_encoder.*   (d_model, nhead, num_layers, dropout)
        wm.embed_dim       (int, default 64)
        wm.action_dim      (int)
        data.dataset.frameskip (int, default 1)
        lstm.*             (hidden_dim, num_layers, dropout)
        obs_window_size    (int, default 1)

    Returns
    -------
    JEPA instance with LSTMPredictor — same interface as the JEPA baseline.
    """
    sen_cfg   = cfg.get("sensor_encoder", {})
    embed_dim = cfg.wm.get("embed_dim", 64)

    # ── Encoder (identical to JEPA sensor baseline) ───────────────────────────
    encoder = SensorEncoder(
        n_sensors  = cfg.get("n_sensors", 28),
        d_model    = sen_cfg.get("d_model", embed_dim),
        nhead      = sen_cfg.get("nhead", 4),
        num_layers = sen_cfg.get("num_layers", 2),
        dropout    = sen_cfg.get("dropout", 0.1),
    )
    hidden_dim = encoder.hidden_size  # == embed_dim for sensor encoder

    # ── LSTM predictor (new) ──────────────────────────────────────────────────
    lstm_cfg = cfg.get("lstm", {})
    predictor = LSTMPredictor(
        embed_dim   = embed_dim,
        act_emb_dim = embed_dim,          # action is projected to same dim
        hidden_dim  = lstm_cfg.get("hidden_dim", 256),
        num_layers  = lstm_cfg.get("num_layers", 2),
        dropout     = lstm_cfg.get("dropout", 0.1),
    )

    # ── Action encoder (identical to JEPA) ───────────────────────────────────
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

    # ── Projectors (identical to JEPA) ────────────────────────────────────────
    proj_hidden = max(hidden_dim * 4, embed_dim * 4)
    projector = MLP(
        input_dim=hidden_dim, output_dim=embed_dim,
        hidden_dim=proj_hidden, norm_fn=nn.BatchNorm1d,
    )
    pred_proj = MLP(
        input_dim=embed_dim, output_dim=embed_dim,
        hidden_dim=proj_hidden, norm_fn=nn.BatchNorm1d,
    )

    # ── Optional TemporalAggregator (obs_window_size > 1) ─────────────────────
    obs_window_size = cfg.get("obs_window_size", 1)
    temporal_agg = None
    if obs_window_size > 1:
        temporal_agg = TemporalAggregator(
            embed_dim=embed_dim,
            max_window=obs_window_size + 4,
            nhead=4, num_layers=1, dropout=0.1,
        )

    return JEPA(
        encoder         = encoder,
        predictor       = predictor,   # ← only line that differs from JEPA baseline
        action_encoder  = action_encoder,
        projector       = projector,
        pred_proj       = pred_proj,
        temporal_agg    = temporal_agg,
        obs_window_size = obs_window_size,
        encoder_type    = "sensor",
    )
