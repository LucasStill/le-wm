# OrailixTower AR-LSTM le-wm results — auto-updated

Scenario-4 TurboSens, AR-LSTM baseline, 10 epochs each, bf16-mixed,
AdamW lr=7e-5 cosine, hi_probe at epoch 5 + final epoch.

Predictor: `LSTMPredictor(hidden_dim=256, num_layers=2, dropout=0.1)`.
Everything else identical to dragon's JEPA config — see `identity.md`.

## Run table

| ID | H | S | P | bs (eff. 512) | Wall | fit/loss | fit/pred | fit/ar | fit/sigreg | params |
|----|---|---|---|---------------|------|----------|----------|--------|-----------|--------|
| L1 | 16 | 1 | 4 | 512 / acc=1   | ~7 h | **0.379** | **0.129** | **0.078** | 1.664 | 1,138,068 |
| L2 | 32 | 1 | 4 | 256 / acc=2   | —    | —        | —        | —      | —          | 1,138,068 |
| L3 | 16 | 5 | 4 | 128 / acc=4   | —    | —        | —        | —      | —          | 1,138,068 |
| L4 | 32 | 5 | 4 | 64  / acc=8   | —    | —        | —        | —      | —          | 1,138,068 |

(Empty rows fill in as runs complete. Param count is the same across
all four because H, S, P do not affect any module's `nn.Parameter`s —
they only change the AR-rollout depth and dataloader windowing.)

**Param-count parity note** (per dragon's reply 2026-04-25 09:34Z):
my AR-LSTM at `hidden=256, layers=2` is **1.138 M trainable params**
vs dragon's JEPA encoder-only **0.828 M** quoted in their campaign
report — a **+37 %** delta. Dragon and I agreed to *not* calibrate to
exact parity (cost: ~1 h GPU for marginal value). The paper text
should call this out explicitly so reviewers don't conflate predictor
choice with model size. Concrete mitigation: report final per-config
total-param counts alongside metrics in the paper table.

## Hi_probe HI-regression (mean over 10 components)

| ID | epoch | R² | RMSE | Pearson-r |
|----|-------|----|------|-----------|
| L1 | 5  | -0.87 | 0.00291 | 0.224 |
| L1 | 9  | -1.24 | 0.00304 | 0.242 |
| L2 | 5  | — | — | — |
| L2 | 9  | — | — | — |
| L3 | 5  | — | — | — |
| L3 | 9  | — | — | — |
| L4 | 5  | — | — | — |
| L4 | 9  | — | — | — |

## Action-RUL probe

| ID | epoch | RMSE (steps) | MAE | R² | Pearson |
|----|-------|--------------|-----|----|---------|
| L1 | 5 | 27.97 | 22.10 | -0.18 | 0.034 |
| L1 | 9 | 27.23 | 22.10 | -0.12 | 0.024 |
| L2 | 5 | — | — | — | — |
| L2 | 9 | — | — | — | — |
| L3 | 5 | — | — | — | — |
| L3 | 9 | — | — | — | — |
| L4 | 5 | — | — | — | — |
| L4 | 9 | — | — | — | — |

## Wandb runs (on https://wandb.ai/thil-ecole-polytechnique/turbofan_S4/runs/)

- L1: `12uiyu7c` (offline; sync_wandb daemon will push)
- L2: pending
- L3: pending
- L4: pending

## Key findings so far (after L1 only)

1. **AR-LSTM training loss is lower than JEPA's E1** at every component
   (fit/pred 0.129 vs 0.152; fit/ar 0.078 vs 0.092). The LSTM fits the
   world-modeling objective slightly better at H=16 S=1 P=4. Note the
   +37 % param delta — not a free win, possibly capacity-driven.
2. **Probe quality is comparable**: HI mean Pearson-r at epoch 9 is
   0.242 (L1) vs 0.25 (E1). Within noise.
3. **Non-monotonic-in-epoch pattern flipped**: dragon's JEPA had its best
   probe at epoch 5 (Pearson 0.30 → 0.25 by epoch 9). My LSTM has
   epoch-9 better than epoch-5 (0.242 > 0.224). Could be sampling, could
   be that the LSTM's representation stabilises later — needs more runs
   to call.
4. **Action-RUL is noise for both architectures** (R² ≈ -0.1, Pearson
   ≈ 0.02 — both at L1 and at dragon E1). Confirms dragon's earlier
   finding: 10 epochs is not enough for RUL, regardless of predictor.
