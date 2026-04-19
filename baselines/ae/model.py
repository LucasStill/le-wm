"""Sensor Autoencoder baseline.

Encoder : identical SensorEncoder used by JEPA (transformer over n_sensors tokens
          → mean pool → d_model embedding).  Weights are NOT shared with JEPA at
          eval time — each baseline is trained independently.

Decoder : 2-layer MLP (d_model → hidden_dim → hidden_dim → n_sensors).

obs_window_size > 1 : a TemporalAggregator (same as JEPA) fuses w consecutive
          frame embeddings into one.  The decoder reconstructs the last frame
          of each window, matching JEPA's last-token alignment convention.

The encode() interface matches JEPA so the HIProbeCallback works unchanged.
"""

import torch.nn as nn
from einops import rearrange


class SensorDecoder(nn.Module):
    def __init__(self, d_model: int = 64, n_sensors: int = 84, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_sensors),
        )

    def forward(self, z):
        return self.net(z)  # (N, n_sensors)


class SensorAutoencoder(nn.Module):
    """Sensor autoencoder with optional temporal aggregation.

    obs_window_size=1 : each frame encoded/decoded independently (snapshot AE).
    obs_window_size=w : sliding windows of w frames fused by TemporalAggregator;
                        decoder reconstructs the last frame of each window.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        obs_window_size: int = 1,
        temporal_agg: nn.Module | None = None,
        projector: nn.Module | None = None,
    ):
        super().__init__()
        self.encoder         = encoder
        self.decoder         = decoder
        self.obs_window_size = obs_window_size
        self.temporal_agg    = temporal_agg
        self.projector       = projector or nn.Identity()

    # ── JEPA-compatible interface ─────────────────────────────────────────────

    def encode(self, info: dict) -> dict:
        """Encode a batch of sensor windows.

        info['pixels'] : (B, T_raw, n_sensors)
        Returns info with added keys:
          info['emb']           : (B, T_out, d_model)  T_out = T_raw - w + 1
          info['_emb_flat']     : (B*T_out, d_model)
          info['_sensors_flat'] : (B*T_out, n_sensors) — last frame of each window
        """
        if 'observation.sensors' in info:
            sensors = info['observation.sensors'].float()
        else:
            pixels = info['pixels'].float()
            b, t = pixels.shape[:2]
            sensors = pixels.reshape(b, t, -1)

        b, t_raw, _ = sensors.shape
        w = self.obs_window_size

        # Encode every raw frame with the shared SensorEncoder
        sensors_flat = rearrange(sensors, 'b t d -> (b t) d')
        frame_emb    = self.encoder(sensors_flat)                   # (B*T_raw, d_model)

        # Optional temporal aggregation (mirrors JEPA._apply_temporal_agg)
        if w > 1 and self.temporal_agg is not None:
            frame_proj = self.projector(frame_emb)                  # (B*T_raw, d_model)
            frame_proj = rearrange(frame_proj, '(b t) d -> b t d', b=b)
            t_out = t_raw - w + 1
            windows = rearrange(
                [frame_proj[:, i:i + w] for i in range(t_out)],
                'tout b w d -> (b tout) w d',
            )
            emb_flat = self.temporal_agg(windows)                   # (B*T_out, d_model)
            # Reconstruction target: last frame of each window
            targets = sensors[:, w - 1:, :]                         # (B, T_out, n_sensors)
        else:
            emb_flat = self.projector(frame_emb)                    # (B*T_raw, d_model)
            targets  = sensors                                       # (B, T_raw, n_sensors)
            t_out    = t_raw

        targets_flat = rearrange(targets, 'b t d -> (b t) d')

        info['emb']           = rearrange(emb_flat, '(b t) d -> b t d', b=b, t=t_out)
        info['_emb_flat']     = emb_flat
        info['_sensors_flat'] = targets_flat
        return info

    def decode(self, emb_flat):
        return self.decoder(emb_flat)  # (B*T_out, n_sensors)
