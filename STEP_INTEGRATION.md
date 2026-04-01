# STEP × LeWorldModel Integration

## High-level summary

Both systems learn a latent embedding of sequential data and share the intuition that
**temporal structure should be reflected in the latent space**. The key difference is the
signal source and the scale of the temporal window:

| | STEP | le-wm |
|---|---|---|
| Domain | Turbofan time series (CMAPSS) | Robotic video + actions |
| Encoder | Transformer AE | ViT (CLS token) |
| Latent shaping | Triplet cosine + prototypes | SIGReg (Gaussian regularizer) |
| Temporal scale | Full episode (100s of steps) | Short window (T=4 frames, frameskip 5) |
| Trajectory metadata | Explicit (unit_id, RUL) | Implicit (episode sequential chunks) |
| Downstream | RUL regression | Planning / action inference |

---

## Architecture walkthrough

### le-wm forward pass

```
batch  ─► encode()  ─► emb (B, T, D)   ← this is the latent we want to shape
                     ─► act_emb (B, T, A)
              ↓
           predict()  ─►  pred_emb (B, T-1, D)   ← predictor output
              ↓
         pred_loss = MSE(pred_emb, tgt_emb)
         sigreg_loss = SIGReg(emb)          ← keeps distribution isotropic
         loss = pred_loss + λ · sigreg_loss
```

`emb` is produced by:
1. ViT CLS token per frame  →  `pixels_emb` (B·T, hidden_size)
2. MLP projector            →  `emb`        (B, T, embed_dim=64)

This is the exact analogue of STEP's `z` — the per-timestep latent vector.

### STEP forward pass

```
anchor/positive/negative windows ─► Transformer AE ─► z (B, latent_dim)
                                                      ↓
recon_loss  = MSE(reconstructed, input)
triplet_loss = ReLU( sim(z_anc, z_neg) − sim(z_anc, z_pos) + margin )
proto_loss  = MSE(z_endpoint, prototype)
loss = recon_loss + α · triplet_loss + β · proto_loss
```

---

## Where the losses live

### SIGReg (already in le-wm)

SIGReg pushes the **marginal distribution** of `emb` toward an isotropic Gaussian using
random projection statistics. It does not care about temporal ordering — it only ensures
the embedding space is well-spread and does not collapse.

### What STEP's triplet loss adds

The triplet cosine loss adds **relational structure**: it explicitly says
"frame at time t should lie closer to frame at time t+1 (same trajectory)
than to a frame from a different trajectory." This is orthogonal to SIGReg:
SIGReg shapes the marginal, the triplet shapes pairwise geometry.

---

## Integration plan

### Step 1 — Intra-window temporal triplet (implemented below)

**No dataset changes required.**

Each batch element is already a window of T=4 consecutive frames from the same episode.
We can form triplets directly from `emb` (B, T, D):

```
anchor   = emb[b, t]         same sequence, current frame
positive = emb[b, t+1]       same sequence, next frame (temporally adjacent)
negative = emb[b', t]        different sequence (different episode / position)
```

The loss:
```
L_temp = E[ ReLU( cos_sim(z_a, z_neg) − cos_sim(z_a, z_pos) + margin ) ]
```

This encourages consecutive frames in the same trajectory window to be more similar
than frames drawn from random other trajectory windows.

**Signal strength**: modest, because T is small (4 frames). Even with frameskip=5 that's
only ~20 env steps. But it still provides consistent directional pressure on the
encoder, especially since B=128 trajectories give many inter-trajectory negatives.

### Step 2 — Episode-level temporal triplet (future work)

For a richer STEP-like signal (ordering across the full episode, not just 4 frames),
we would need episode metadata in the batch:

- `episode_idx` — which episode each sequence comes from
- `episode_pos`  — normalized position t/T_episode ∈ [0, 1]

Then the triplet definition becomes:
```
anchor   = emb[b, t]  (episode i, position p)
positive = emb[b', t] where episode_idx[b'] == i and |pos[b'] − pos[b]| < δ_pos
negative = emb[b'', t] where |pos[b''] − pos[b]| > δ_neg
           (can be same or different episode but distant in time)
```

This exactly mirrors the STEP design (RUL → normalized episode position). This would
require either:
- Checking whether `stable_worldmodel.HDF5Dataset` already exposes index/position,
  and if so adding it to `keys_to_load`; or
- Wrapping the dataset with a custom sampler that tracks these indices.

### Step 3 — Prototype anchors (future)

Once temporal ordering is working, the prototypes from STEP can be reintroduced:
- e1 ≈ mean embedding of first k frames of each episode (initial state prototype)
- e2 ≈ mean embedding of last k frames (goal state prototype)

These would define an angular basis and give the user an interpretable "task progress"
angle in latent space — analogous to STEP's degradation angle.

---

## Files changed

```
le-wm/
├── step_loss.py          ← NEW:  temporal triplet loss
├── train.py              ← MOD:  add temporal loss to lejepa_forward
└── config/train/
    └── lewm.yaml         ← MOD:  add temporal_triplet loss config
```

---

## Design decisions

**Why apply the loss to `emb`, not to a separate head?**
`emb` is exactly what the predictor operates on and what gets used for planning. Shaping
it directly is the cleanest approach. A projection head (à la SimCLR) is possible if the
triplet loss interferes with the predictor, but is premature at this stage.

**Why cosine similarity (not L2)?**
Cosine similarity is scale-invariant and focuses purely on direction — this is what
makes STEP's angular prototypes interpretable. SIGReg already handles the scale/spread,
so directional structure is the complementary signal.

**Interaction with SIGReg:**
SIGReg and the triplet loss are complementary. SIGReg prevents collapse and keeps the
distribution globally isotropic; the triplet loss adds local relational structure. In
practice you may need to tune the weights (suggested starting point: `weight: 0.05–0.1`).

**Why `margin = 0.5`?**
Cosine similarities live in [−1, 1]. A margin of 0.5 means we want the positive to be
at least 0.5 cosine units closer than the negative — a moderate but meaningful gap.
Start here and tune based on whether the loss is saturating (→ raise margin) or too
noisy (→ lower margin).
