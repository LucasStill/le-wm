"""Sensor Autoencoder baseline.

Encoder : identical SensorEncoder used by JEPA (transformer over n_sensors tokens
          → mean pool → d_model embedding).  Weights are NOT shared with JEPA at
          eval time — each baseline is trained independently.

Decoder : 2-layer MLP (d_model → hidden_dim → hidden_dim → n_sensors).

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
    """Frame-level sensor autoencoder.

    Each timestep is encoded and decoded independently — no temporal modelling.
    This is the reconstruction-pressure baseline against JEPA (latent prediction)
    and AR-LSTM (recurrent latent prediction).
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    # ── JEPA-compatible interface ─────────────────────────────────────────────

    def encode(self, info: dict) -> dict:
        """Encode a batch of sensor windows.

        Accepts the same info dict format as JEPA.encode():
          info['pixels'] : (B, T, n_sensors)  — sensor data stored under 'pixels' key
        Returns info with added keys:
          info['emb']          : (B, T, d_model) — frame embeddings
          info['_emb_flat']    : (B*T, d_model)  — flat view (for decode in forward)
          info['_sensors_flat']: (B*T, n_sensors) — flat sensors (for recon loss)
        """
        if 'observation.sensors' in info:
            sensors = info['observation.sensors'].float()
        else:
            pixels = info['pixels'].float()
            b, t = pixels.shape[:2]
            sensors = pixels.reshape(b, t, -1)

        b, t, _ = sensors.shape
        sensors_flat = rearrange(sensors, 'b t d -> (b t) d')
        emb_flat = self.encoder(sensors_flat)                         # (B*T, d_model)

        info['emb']           = rearrange(emb_flat, '(b t) d -> b t d', b=b)
        info['_emb_flat']     = emb_flat
        info['_sensors_flat'] = sensors_flat
        return info

    def decode(self, emb_flat):
        return self.decoder(emb_flat)  # (B*T, n_sensors)
