"""
STEP-style temporal losses for le-wm.

References: "Progressive Latent Representations of Time Series" (STEP paper).

Two losses, increasing in expressiveness:

  temporal_triplet_loss   — Step 1.  Intra-window, no dataset metadata needed.
                            Uses the T=4 consecutive frames in each window.

  episode_triplet_loss    — Step 2.  Episode-aware, requires episode_idx and
                            episode_pos in the batch (added by TemporalMetadataWrapper).
                            Implements the STEP-style dead-zone margin design:

                              |Δpos| < pos_radius          → positive pair
                              |Δpos| > pos_radius + neg_gap → negative pair
                              (dead zone in between, ignored)

                            episode_pos is normalised ∈ [0, 1] so the margins
                            are interpretable as fractions of the episode length,
                            fully independent of how long individual episodes are.

Both losses use cosine similarity (vectors are L2-normalised before comparison).
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Step 1 — Intra-window temporal triplet (no metadata required)
# ---------------------------------------------------------------------------

def temporal_triplet_loss(emb: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """Triplet cosine loss using the temporal ordering within each window.

    For every adjacent pair (t, t+1) in the same trajectory window, enforce:

        cos(z[b,t], z[b,t+1]) > cos(z[b,t], z[b',t]) + margin

    where b' ≠ b is a randomly drawn different trajectory.

    No dataset metadata is required — triplets are formed purely from the
    (B, T, D) tensor that le-wm already produces.  Works even when every
    batch element is from a completely different episode.

    Args:
        emb:    (B, T, D)  per-frame encoder embeddings.
        margin: Triplet margin in cosine space ∈ (0, 2].  Default 0.5.

    Returns:
        Scalar loss.
    """
    B, T, D = emb.shape
    if T < 2:
        return emb.new_tensor(0.0)

    emb_n = F.normalize(emb, dim=-1)          # (B, T, D)

    anchors   = emb_n[:, :-1]                 # (B, T-1, D)  frame t
    positives = emb_n[:, 1:]                  # (B, T-1, D)  frame t+1 (same trajectory)

    # Negatives: random different trajectory, same time offset.
    neg_idx   = torch.randperm(B, device=emb.device)
    negatives = emb_n[neg_idx, :-1]           # (B, T-1, D)

    pos_sim = (anchors * positives).sum(dim=-1)   # (B, T-1)
    neg_sim = (anchors * negatives).sum(dim=-1)   # (B, T-1)

    return F.relu(neg_sim - pos_sim + margin).mean()


# ---------------------------------------------------------------------------
# Step 2 — Episode-level temporal triplet (STEP-style dead-zone margins)
# ---------------------------------------------------------------------------

def episode_triplet_loss(
    emb: torch.Tensor,
    episode_idx: torch.Tensor,
    episode_pos: torch.Tensor,
    margin: float = 0.5,
    pos_radius: float = 0.10,
    neg_gap: float = 0.10,
) -> torch.Tensor:
    """Episode-aware temporal triplet loss with STEP-style dead-zone margins.

    Triplet selection (mirroring the STEP health_margin / margin design):

        The criterion is purely about temporal distance — episode membership
        is NOT the deciding factor, just as in STEP where the negative does
        not have to come from a different engine unit, only from a different
        health state (sufficiently different RUL).

          |Δpos| < pos_radius            → POSITIVE   (temporally close,
                                           same or different episode)
          |Δpos| > pos_radius + neg_gap  → NEGATIVE   (temporally far,
                                           same or different episode)
          pos_radius ≤ |Δpos| ≤ ...+neg_gap → ignored (dead zone)

        Note: in practice, positive pairs with small |Δpos| will mostly come
        from the same episode (sequences from different episodes at the same
        normalised position may genuinely have different robot states), but
        the loss does not enforce this — it only cares about temporal distance,
        just like STEP.

    episode_pos is normalised to [0, 1] by TemporalMetadataWrapper, so the
    margins are episode-length agnostic (pos_radius=0.1 → "within the first /
    last 10 % of the episode").

    Embeddings: we use the first frame of each window as the sequence
    representative.  This keeps the loss focused on the encoder rather than
    being confounded by within-window variation.

    Args:
        emb:          (B, T, D)  per-frame encoder embeddings.
        episode_idx:  (B,)  int64 episode index for each sequence.
        episode_pos:  (B,)  float32 normalised intra-episode position ∈ [0, 1].
        margin:       Triplet margin in cosine space.  Default 0.5.
        pos_radius:   Max |Δpos| for a pair to count as positive.  Default 0.10.
        neg_gap:      Dead-zone width added beyond pos_radius.
                      Negative threshold = pos_radius + neg_gap.  Default 0.10.

    Returns:
        Scalar loss, or 0.0 if no valid triplets were found.
    """
    B, T, D = emb.shape
    device = emb.device

    # Use the first frame of each window as the per-sequence representative.
    emb_n = F.normalize(emb[:, 0], dim=-1)    # (B, D)

    # --- Build pair masks  (B, B) -------------------------------------------
    # The criterion is purely temporal distance in normalised position space,
    # mirroring STEP where "different health state" (= temporally far) is what
    # makes a negative, not "different engine unit".
    pos_delta = (episode_pos.unsqueeze(0) - episode_pos.unsqueeze(1)).abs()  # (B, B)

    neg_threshold = pos_radius + neg_gap

    # Positive: temporally close (within pos_radius).
    # Restricted to same-episode pairs to avoid treating cross-episode windows
    # at similar positions as positives — their robot states may differ due to
    # random initial conditions.
    same_ep = (episode_idx.unsqueeze(0) == episode_idx.unsqueeze(1))  # (B, B)
    is_pos  = same_ep & (pos_delta < pos_radius)

    # Negative: temporally far, same or different episode.
    # Pairs inside the dead zone [pos_radius, neg_threshold] are ignored.
    # Cross-episode pairs within pos_radius are also in the dead zone (neither
    # positive nor negative) — they are left ambiguous rather than forced.
    is_neg  = pos_delta > neg_threshold

    # Remove diagonal (self-pairs).
    eye = torch.eye(B, dtype=torch.bool, device=device)
    is_pos = is_pos & ~eye
    is_neg = is_neg & ~eye

    # --- Vectorised triplet sampling -----------------------------------------
    # For each anchor i, draw one random positive j and one random negative k.
    # Collect all valid (i, j, k) tuples and average the hinge loss over them.

    losses = []
    for i in range(B):
        pos_cands = is_pos[i].nonzero(as_tuple=False).view(-1)
        neg_cands = is_neg[i].nonzero(as_tuple=False).view(-1)
        if pos_cands.numel() == 0 or neg_cands.numel() == 0:
            continue

        # Sample one positive and one negative uniformly.
        j = pos_cands[torch.randint(pos_cands.numel(), (1,), device=device).item()]
        k = neg_cands[torch.randint(neg_cands.numel(), (1,), device=device).item()]

        pos_sim = (emb_n[i] * emb_n[j]).sum()
        neg_sim = (emb_n[i] * emb_n[k]).sum()
        losses.append(F.relu(neg_sim - pos_sim + margin))

    if not losses:
        return emb.new_tensor(0.0)
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Step 3 — Prototype loss  (future, stub kept for reference)
# ---------------------------------------------------------------------------

def prototype_loss(
    emb: torch.Tensor,
    episode_pos: torch.Tensor,
    initial_proto: torch.Tensor,
    final_proto: torch.Tensor,
    endpoint_radius: float = 0.05,
) -> torch.Tensor:
    """Pull start/end-of-episode embeddings toward canonical prototype vectors.

    Mirrors STEP's prototype loss.  Embeddings from the *start* of an episode
    (episode_pos < endpoint_radius) are pulled toward `initial_proto` and
    embeddings from the *end* (episode_pos > 1 - endpoint_radius) toward
    `final_proto`.

    Args:
        emb:             (B, D)  one embedding per sequence (e.g. first frame).
        episode_pos:     (B,)   normalised position ∈ [0, 1].
        initial_proto:   (D,)   target embedding for episode start.
        final_proto:     (D,)   target embedding for episode end.
        endpoint_radius: Fraction of the episode that counts as an endpoint.

    Returns:
        Scalar loss.
    """
    is_initial = episode_pos < endpoint_radius
    is_final   = episode_pos > (1.0 - endpoint_radius)

    total = emb.new_tensor(0.0)
    count = 0

    if is_initial.any():
        init_emb = emb[is_initial]
        total += F.mse_loss(init_emb, initial_proto.expand_as(init_emb))
        count += 1

    if is_final.any():
        fin_emb = emb[is_final]
        total += F.mse_loss(fin_emb, final_proto.expand_as(fin_emb))
        count += 1

    return total / max(count, 1)
