# OrailixTower AR-LSTM le-wm results — auto-updated

Scenario-4 TurboSens, AR-LSTM baseline, 10 epochs each, bf16-mixed,
AdamW lr=7e-5 cosine, hi_probe at epoch 5 + final epoch.

Predictor: `LSTMPredictor(hidden_dim=256, num_layers=2, dropout=0.1)`.
Everything else identical to dragon's JEPA config — see `identity.md`.

## Run table

| ID | H | S | P | bs (eff. 512) | Wall | fit/loss | fit/pred | fit/ar | fit/sigreg | params |
|----|---|---|---|---------------|------|----------|----------|--------|-----------|--------|
| L1 | 16 | 1 | 4 | 512 / acc=1   | ~7 h | **0.379** | **0.129** | **0.078** | 1.664 | 1,138,068 |
| L2 | 32 | 1 | 4 | 256 / acc=2   | ~13 h | **0.234** | **0.078** | **0.050** | 1.039 | 1,138,068 |
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
| L2 | 5  | -1.38 | 0.00321 | 0.188 |
| L2 | 9  | -1.41 | 0.00320 | 0.169 |
| L3 | 5  | — | — | — |
| L3 | 9  | — | — | — |
| L4 | 5  | — | — | — |
| L4 | 9  | — | — | — |

## Action-RUL probe

| ID | epoch | RMSE (steps) | MAE | R² | Pearson |
|----|-------|--------------|-----|----|---------|
| L1 | 5 | 27.97 | 22.10 | -0.18 | 0.034 |
| L1 | 9 | 27.23 | 22.10 | -0.12 | 0.024 |
| L2 | 5 | 27.13 | 22.02 | -0.114 | 0.004 |
| L2 | 9 | 27.40 | 22.16 | -0.136 | 0.013 |
| L3 | 5 | — | — | — | — |
| L3 | 9 | — | — | — | — |
| L4 | 5 | — | — | — | — |
| L4 | 9 | — | — | — | — |

## Wandb runs (on https://wandb.ai/thil-ecole-polytechnique/turbofan_S4/runs/)

- L1: `12uiyu7c` (offline; sync_wandb daemon will push)
- L2: `5iu3jbtt` (offline; sync_wandb daemon will push)
- L3: pending
- L4: pending

## Bigger-probe diagnostic on L1

Per dragon's request: load L1's frozen encoder at probe-epoch 5 and 9
(ckpts `epoch_6_object.ckpt` and `epoch_10_object.ckpt`, same weights as
the default-probe was run against), train a much bigger TransformerProbe
(`d_model=512, num_layers=6`, vs the default `128, 3`), keep optimizer +
patience identical (Adam lr=1e-3, patience=20, max 150 epochs, same
data split, same probe_seq_len=16). Standalone script
(`baselines/ar_lstm/bigger_probe_diag.py`) — no re-training.

| ckpt | probe size            | HI mean Pearson | HI mean R² | HI mean RMSE | early-stop |
|------|----------------------|-----------------|------------|--------------|------------|
| ep5  | default (128 / 3)    | **+0.224**      | -0.87      | 0.00291      | ep ~31     |
| ep5  | bigger  (512 / 6)    | **-0.071**      | -2.26      | 0.00356      | ep 29      |
| ep9  | default (128 / 3)    | **+0.242**      | -1.24      | 0.00304      | ep ~31     |
| ep9  | bigger  (512 / 6)    | **+0.019**      | -2.02      | 0.00351      | ep 33      |

### Interpretation — neither dragon hypothesis A nor B; a third path

Dragon's matrix:
- **A**: bigger probe recovers Pearson > 0.4 → encoder fine, default probe is bottleneck.
- **B**: bigger probe gives ~same as default → encoder lacks HI signal.

Observed: **bigger probe gives WORSE Pearson than default** at both
checkpoints (-0.07 / +0.02 vs +0.22 / +0.24). RMSE also worse. Both
bigger probes early-stopped at best test-RMSE — not undertraining.

Conclusion: **the default probe (128 / 3) is approximately the right size
for the available probe-train budget**. The 8× bigger head has 8× more
parameters to fit on the same 28K-window training set and overfits — it
finds a low-RMSE solution that has poor *linear* correlation with the
targets (likely shrinks predictions toward the mean and predicts noise
patterns).

Implications:
1. **Encoder DOES carry HI signal.** Default probe extracts +0.24
   Pearson from L1's frozen encoder; that's real, not noise.
2. **Bottleneck is NOT probe capacity.** Pushing capacity destroys the
   signal rather than recovering it. The default probe is well-sized.
3. **The 28K probe-train budget is the real constraint.** A larger head
   *might* help if we also raised `n_subsample` (e.g., to 200K) — that's
   a follow-up test, not done here.
4. **JEPA E1's default-probe Pearson 0.30 ≈ ceiling for this probe setup.**
   Getting past 0.3-0.4 will require either (a) more probe-train data,
   or (b) different encoder objective (not a different probe).

Per-component bigger-probe results show a few HI dims with meaningful
magnitude (e.g. ep9: HI_0 +0.34, HI_3 -0.60, HI_4 +0.32) but they mostly
cancel in the mean — the bigger probe latches onto idiosyncratic noise
*per component* rather than a coherent signal across all of them.

CSV with per-component rows: `logs/bigger_probe_results.csv`.

## Key findings so far (after L1 + L2)

1. **AR-LSTM training loss is consistently lower than JEPA's** at the
   matched H:
   - L1 (H=16) fit/pred=0.129 vs E1 fit/pred=0.152 (LSTM 15 % lower)
   - L2 (H=32) fit/pred=0.078 vs E2 fit/pred=0.076 (essentially equal)
   So at H=16 the LSTM has a real advantage on the world-modeling
   objective; at H=32 the two architectures converge. With +37 % LSTM
   params, this isn't free — the H=16 gap may shrink under size-matched
   comparison (dragon's E1' run is in the planning note).

2. **Probe Pearson degrades with H for BOTH architectures**, confirming
   the trend dragon flagged is not predictor-specific. Pearson @9:
   - LSTM: L1 0.242 → L2 0.169 (-30 %)
   - JEPA: E1 0.25  → E2 0.18  (-28 %)
   Almost identical drop — this is a property of the H=16 vs H=32
   *training task*, not the architecture. **Major paper-worthy finding.**

3. **Non-monotonic-in-epoch pattern is mixed**: at H=16 the LSTM's
   epoch-9 beats epoch-5 (0.242 > 0.224); at H=32 it flips to match
   the JEPA pattern (epoch-5 0.188 > epoch-9 0.169). So both
   architectures show degradation from epoch-5 to epoch-9 at H=32.
   Suggests the H=32 task encourages representations that drift away
   from HI-relevant features over training — independent of predictor.

4. **Action-RUL is noise for all four runs** (LSTM L1, L2 and JEPA E1,
   E2; R² ≈ -0.12 to -0.15, Pearson ≈ 0 to 0.03). Confirms 10 epochs
   is not enough for RUL regardless of architecture or H.
