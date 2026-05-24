# Scenario-4 TurboSens — cross-architecture results

_Last regenerated: 2026-05-24 20:48 UTC_

Auto-aggregated from `coordination/{dragon,orailixtower}/` and
`hi_probe_metrics.csv`. **Do not edit by hand** — run `./aggregate_results.sh`.

---

## What's running right now

### Dragon (JEPA, RTX 5090)


- **Tmux `campaign`** — full E3 training (H=16 S=5 P=4, bs=128 accum=4, 10 epochs).
  E3 probe finished cleanly at 09:29Z. Full run launched at 09:29Z, process 730874,
  GPU 97%, 17.9 GB VRAM. ETA ~22:00 UTC.
- **Tmux `wandb_sync`** — daemon syncing offline wandb runs every 5 min.
- **Tmux `peer_sync`** — daemon rsyncing coordination/ with OrailixTower every 10 min.


### OrailixTower (AR-LSTM, RTX A6000)


- **Tmux `s4_arlstm_Lbig`** — overnight run: W=4, H=32, S=1, P=4
  (TemporalAggregator engaged for first time in sweep). bs=256 accum=2,
  bf16-mixed, compile_encoder=true, 10 epochs. Started 19:41 UTC. Healthy:
  GPU 99 %, 13.99 GB VRAM, 72 trainable params, first backward ✓.
  ETA ~09:30 UTC tomorrow. Log: `logs/s4_arlstm_Lbig_20260426_2141.log`.
- **Tmux `wandb_sync`** — offline→cloud sync daemon, still ticking.
- Other tmux sessions are zombie post-run shells (bigger_probe,
  eval_sweep, sanity, s4_arlstm_L2) — keep-alive only.


---

## Per-architecture results

### JEPA (dragon)


Scenario-4 TurboSens, JEPA encoder (828K params), bf16-mixed, AdamW lr=7e-5
cosine, hi_probe at epoch 5 + final epoch.

## ⚡ Headline findings (updated 2026-04-27 with multi-seed)

**1. In-distribution and OOD trade off across (H, S, W).**

Multi-seed mean ± std (5-6 valid seeds per cell):

| run | H  | S  | regular Pearson    | test_hard Pearson |
|-----|----|----|--------------------|-------------------|
| E1  | 16 | 1  | 0.446 ± 0.039     | 0.234 ± 0.180     |
| **E2**  | 32 | 1  | **0.597 ± 0.025** | 0.327 ± 0.135     |
| E3  | 16 | 5  | 0.450 ± 0.054     | 0.284 ± 0.104     |
| E5  | 8  | 1  | 0.400 ± 0.218     | 0.178 ± 0.196     |
| **E6probe** (1ep) | 16 | 10 | 0.406 ± 0.095 | **0.345 ± 0.044** |

**E2 best in-dist** (0.597); **E6probe best OOD** (0.345) — and with the
TIGHTEST std (0.044) of any cell. Larger S consistently helps OOD;
larger H consistently helps in-dist.

**2. Aggregate OOD hides held-out-archetype effect.** OT's per-archetype
breakdown of L1 × test_hard reveals the dataset has a SYMMETRIC arch-OOD
split: test holds out B_fan_booster, test_hard holds out A_compressor.
The "L1 OOD 0.358" headline is almost entirely B_fan_booster signal.
**Reporting must be per-archetype.**

**3. Single-seed numbers are unreliable** (±0.2 std common). All
pre-2026-04-27 single-seed claims must be re-quoted with multi-seed
mean ± std. Probe-init seed=1 NaNs degenerately and is filtered.

## ⚡ Methodology finding (2026-04-26)

In-training hi_probe was data-starved at `n_subsample=30000`. eval_sweep at
769k probe-train samples gives Pearson **0.563** for E1 vs the in-training
**0.250** — a 2.25× understating of encoder quality.

OT confirmed via bigger-probe diagnostic: 8× larger probe head on the same
28k budget _overfits_ (Pearson drops to ~0.02). Bottleneck is data, not
capacity. **Action: bumped `n_subsample` 30k → 200k for all future runs.**

## Run table

| ID | H  | S | P | bs / accum | wall | fit/loss | fit/pred | fit/ar | fit/sigreg |
|----|----|---|---|------------|------|----------|----------|--------|-----------|
| E1 | 16 | 1 | 4 | 512 / 1    | 4.4h | 0.406    | 0.152    | 0.092  | 1.695     |
| E2 | 32 | 1 | 4 | 256 / 2    | 8.2h | 0.254    | 0.076    | 0.048  | 1.188     |
| E3 | 16 | 5 | 4 | 128 / 4    | 20h  | 0.259    | 0.078    | 0.038  | 1.203     |
| E4 | 32 | 5 | 4 | 64 / 8     | **deferred** | — | — | — | — |
| E5 | 8  | 1 | 4 | 512 / 1    | running | — | — | — | — |

E4 deferred per the adaptive plan: would have cost ~40h to confirm the same
direction E2/E3 already established.

## HI Pearson by evaluation pipeline

| run | in-training (28k probe-train) | eval_sweep regular (769k) | eval_sweep test_hard |
|-----|------------------------------|---------------------------|----------------------|
| E1  | 0.250                        | **0.563**                 | 0.152               |
| E2  | 0.180                        | 0.196                     | **0.447**           |
| E3  | 0.252                        | -0.010                    | 0.370               |

**Trust the eval_sweep columns.** In-training column is the under-budget probe
that misled us early — kept here for documentation.

## Per-HI on E1 (illustrative)

| HI    | regular | test_hard |
|-------|---------|-----------|
| HI_0  | 0.43    | 0.02 |
| HI_1  | 0.57    | 0.07 |
| HI_2  | 0.51    | 0.12 |
| HI_3  | 0.49    | **0.42** ← robust |
| HI_4  | 0.51    | 0.17 |
| HI_5  | 0.34    | 0.19 |
| HI_6  | 0.73    | 0.23 |
| HI_7  | 0.63    | 0.20 |
| HI_8  | 0.71    | **-0.27** ← sign flip! |
| HI_9  | 0.72    | 0.36 |

## Action-RUL probe

Universally near zero (Pearson < 0.03, R² ≤ 0). Not pursued; class imbalance
on rare maintenance events likely the cause. Deferred.

## Wandb runs (https://wandb.ai/thil-ecole-polytechnique/turbofan_S4/runs/)

- E1: `zbpmvuo4`
- E2: `oz81yajr`
- E3: `5qv43zup` (E3 actually has its own; verify with WandB UI)
- E5: pending (auto-synced when finished)

### AR-LSTM (orailixtower)


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

## Per-archetype HI Pearson on L1 (T3)

`baselines/ar_lstm/per_archetype_diag.py`. L1's frozen encoder, eval_sweep
HI-probe pipeline (~770K probe-train windows), split test predictions by
the 4 scenario-4 archetypes (A_compressor, B_fan_booster, C_turbine,
D_balanced).

### TEST set — single-seed (subject to same noise caveat as T2)

| archetype     | n windows (sl=1) | sl=1 Pearson | sl=10 Pearson |
|---------------|-----------------:|-------------:|--------------:|
| A_compressor  | 31,224 | **0.533** | 0.531 |
| B_fan_booster | **0**  | — (none in test set) | — |
| C_turbine     | 23,432 | 0.236 | 0.267 |
| D_balanced    | 29,791 | 0.360 | 0.417 |
| **L1 aggregate (ref)** | 84,447 | 0.564 | 0.510 |

### Findings

1. **Regular test contains zero B_fan_booster windows.** All B episodes
   are routed to test_hard. This is a real **dataset-design** finding
   relevant for the dataset paper: the test_hard split is partly defined
   by B as the held-out OOD archetype, not just by perturbed weather /
   degradation profiles. (Test_hard run pending — will populate the full
   4×Pearson matrix once GPU is free.)

2. **>2× spread between archetypes in distribution.** A_compressor (0.53)
   is much easier to predict HI for than C_turbine (0.24); D_balanced sits
   in the middle (0.36). The aggregate L1 Pearson 0.564 is mostly carried
   by A_compressor (~37 % of test windows, easiest archetype).

3. **sl=10 windowing helps D_balanced and C_turbine but NOT A_compressor.**
   A is at saturation already — extra sequence context can't help. C and
   D both gain ~0.03–0.06 Pearson from sl=1 → sl=10. So "sequence context
   helps" is archetype-specific, not universal.

### Implications for the paper

The dataset's **per-archetype heterogeneity** is a feature worth reporting:
benchmark numbers should be quoted both aggregate AND per-archetype to
avoid a single archetype dominating the headline. C_turbine is the
hardest of the three in-distribution archetypes — likely the right
"benchmark difficulty" reference point.

CSV: `logs/per_archetype_results.csv`. Test_hard pending.

## Sanity / lower-bound baselines (T2) — single-seed, **NOISY**

⚠️ **All numbers below are single-seed point estimates and are highly
unstable.** `eval_sweep.train_probe` does NOT seed the probe-head init or
the DataLoader shuffle. Two back-to-back runs of the *same* config give
swings of up to ~0.5 in mean Pearson (raw_sensors @ sl=1 test: smoke 0.593
vs full-run nan, same code path, different probe-init seed). All single-
seed Pearson numbers in this document — including the calibrated L1/L2
rows below — should be treated as ±0.2 minimum until we re-run with
multiple seeds.

`baselines/ar_lstm/sanity_baselines.py` — runs the eval_sweep task-1
pipeline (~770K probe-train windows, default TransformerProbe head) on
two encoder-free / random baselines:

- **raw_sensors**: skip the encoder. Probe input is the raw 176-dim
  sensor vector.
- **random_encoder**: build JEPA + LSTMPredictor via `build_ar_lstm`,
  keep at random init (no checkpoint), encode normally → 64-dim
  embeddings → probe.

### Single-seed numbers (caveat: noisy)

|                 | sl=1   | sl=10 | sl=50 |
|-----------------|-------:|------:|------:|
| **TEST**        |        |       |       |
| raw_sensors     | nan¹   | nan¹  | 0.373 |
| random_encoder  | 0.157  | **0.629** | 0.247 |
| L1 trained (ref)| 0.564  | 0.510 | 0.549 |
| L2 trained (ref)| 0.495  | 0.533 | 0.504 |
| **TEST_HARD**   |        |       |       |
| raw_sensors     | 0.474  | 0.468 | **0.808** |
| random_encoder  | 0.002  | 0.360 | nan¹  |
| L1 trained (ref)| 0.325  | 0.378 | 0.241 |
| L2 trained (ref)| -0.014 | 0.330 | 0.376 |

¹ `nan` when probe predictions have zero variance on a column → degenerate
local minimum. Different probe-init seed avoids it.

### What stands out even through the noise

1. **random_encoder @ sl=10 test = 0.629** beats every trained encoder at
   the same sl (L1 0.510, L2 0.533). A random-init JEPA + 10-frame probe
   windows extracts more HI signal than the trained encoders. Strong
   suggestion that **trained-encoder benefit is small or zero on this
   benchmark task**.
2. **raw_sensors @ sl=50 test_hard = 0.808** — by far the best OOD number
   we've seen. Direct readout from raw sensors with sequence context
   generalises better than any encoder pipeline.
3. **raw_sensors @ sl=1 test smoke = 0.593** (single sample, but matches
   in spirit) is *higher* than L1's 0.564 and E1's 0.563. Encoders may
   be losing HI-relevant signal in service of the JEPA / AR-rollout
   objective.

### Implication for the dataset paper

If these patterns hold under multi-seed evaluation, the dataset-paper
headline becomes: **"This benchmark task is hard enough that trained
encoders do not outperform raw-feature linear-on-transformer probes."**
That's a positive finding for a dataset paper — it positions the dataset
as a real challenge rather than something where pretraining trivially
wins.

### NEXT STEP STRONGLY RECOMMENDED — multi-seed re-run

Run each baseline (raw_sensors, random_encoder, L1, L2) × {test, test_hard}
× {sl=1, 10, 50} with at least **3 seeds** and report mean ± std. Cost
estimate: ~1.5 h GPU. Without this, none of the T2 / calibrated numbers
are paper-grade.

Files: `logs/sanity_baselines_results.csv` (per-component), 
`logs/sanity_baselines_20260426_1652.log` (full run log).

## Calibrated eval_sweep task-1 — L1 + L2 (the "true Pearson" rows)

Per dragon's request: re-evaluate L1 and L2 frozen encoders (epoch_10 ckpts)
with `eval_sweep.py --tasks 1`, the ~770K-probe-train-window pipeline that
gave dragon's E1 a Pearson of 0.563. These are the numbers that go into
the paper table — directly comparable to dragon's JEPA rows.

### Test set (in-distribution)

| run | sl=1 Pearson | sl=10 | sl=50 | mean R² (sl=1) | RMSE (sl=1) |
|-----|-------------:|------:|------:|---------------:|------------:|
| L1 (H=16, P=4) | **0.564** | 0.510 | 0.549 | 0.251 | 0.00235 |
| L2 (H=32, P=4) | **0.495** | 0.533 | 0.504 | 0.237 | 0.00256 |
| dragon E1 (ref) | 0.563 | — | — | — | — |

L1's sl=1 = 0.564 ≈ E1 = 0.563. Same encoder behaviour at H=16 between
AR-LSTM and JEPA on the in-distribution probe. L2 (H=32) is *below* L1
at sl=1 (0.495) but pulls ahead at sl=10 (0.533) — H=32 representations
need sequence context to be useful.

### Test_hard set (OOD)

| run | sl=1 Pearson | sl=10 | sl=50 | mean R² (sl=1) | RMSE (sl=1) |
|-----|-------------:|------:|------:|---------------:|------------:|
| L1 (H=16, P=4) | **0.325** | 0.378 | 0.241 | -0.068 | 0.00416 |
| L2 (H=32, P=4) | **-0.014** | 0.330 | 0.376 | -0.123 | 0.00448 |

OOD shift is brutal for both:
- L1 drops 0.564 → 0.325 (-42 %)
- L2 drops 0.495 → -0.014 (-103 %, sign flip at sl=1)

L2 partially recovers at sl=10/50, but at sl=1 the H=32 encoder is
essentially anti-correlated under shift — strong evidence that H=32
specialises to in-distribution structure that breaks OOD without
sequence context. **At sl=10 OOD, L1 (0.378) > L2 (0.330)** — H=16
generalises better OOD.

### In-training vs calibrated Pearson — confirms data-bound probe

| run | in-training (28K) | calibrated test (770K) | ratio |
|-----|------------------:|-----------------------:|------:|
| L1 epoch 9 | 0.242 | 0.564 | 2.33× |
| L2 epoch 9 | 0.169 | 0.495 | 2.93× |

Together with the bigger-probe diagnostic (28K + 8× capacity → 0.02,
overfit), the picture is unambiguous: **the in-training probe was
data-starved, not capacity-bound**. The 30K → 200K bump in
`n_subsample` for future runs is well-motivated and will give
in-training metrics that better track the calibrated number.

### Files

- Per-component results JSON: `eval_results/all_ckpts_test/summary.json`,
  `eval_results/all_ckpts_test_hard/summary.json`
- Note: `eval_sweep.write_flat_csv` raises `AttributeError` when
  `--tasks 1` is the only request (`task2_delta_hi` is `None`, not `{}`).
  Printed metrics are fine; only the flat CSV is missing. Trivial fix.

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


---

## Dragon's per-epoch hi_probe trajectory (raw, from `hi_probe_metrics.csv`)

**HI mean across 10 components, per epoch:**

| epoch | global_step | seq_len | R²    | RMSE     | Pearson-r |
|-------|-------------|---------|-------|----------|-----------|
| 0 | 0 | 1 | -0.0102 | 0.003734 | 0.0000 |
| 0 | 0 | 16 | -0.0412 | 0.003529 | 0.0000 |
| 0 | 0 | 16 | -0.3201 | 0.002349 | 0.2431 |
| 0 | 0 | 16 | -0.3280 | 0.002368 | 0.2461 |
| 0 | 0 | 16 | -0.3971 | 0.002551 | 0.1171 |
| 0 | 0 | 32 | -0.2332 | 0.002280 | 0.2527 |
| 0 | 11989 | 16 | 0.3748 | 0.002696 | 0.6543 |
| 0 | 12156 | 16 | 0.1727 | 0.003115 | 0.4556 |
| 0 | 12240 | 16 | -0.6998 | 0.002728 | 0.2757 |
| 0 | 12293 | 32 | -0.6615 | 0.002760 | 0.2056 |
| 5 | 73440 | 16 | -1.3470 | 0.003159 | 0.1223 |
| 5 | 73608 | 1 | -0.0046 | 0.003721 | 0.0000 |
| 5 | 73758 | 32 | -1.3238 | 0.003168 | 0.1360 |
| 5 | 73842 | 16 | -0.7902 | 0.002746 | 0.2974 |
| 5 | 73884 | 8 | 0.2436 | 0.002993 | 0.5737 |
| 9 | 122400 | 16 | -1.3839 | 0.003136 | 0.2513 |
| 9 | 122680 | 1 | -0.0014 | 0.003714 | 0.0000 |
| 9 | 122930 | 32 | -0.9075 | 0.002908 | 0.1769 |
| 9 | 123070 | 16 | -0.8999 | 0.002946 | 0.2526 |
| 9 | 123140 | 8 | 0.1566 | 0.003165 | 0.5126 |

**Action-RUL probe per epoch:**

| epoch | global_step | seq_len | R²    | RMSE   | Pearson-r |
|-------|-------------|---------|-------|--------|-----------|
| 0 | 0 | 1 | -0.1631 | 35.2572 | 0.0000 |
| 0 | 0 | 16 | -0.1502 | 33.4724 | 0.0963 |
| 0 | 0 | 16 | -0.1547 | 27.6228 | 0.0166 |
| 0 | 0 | 16 | -0.1561 | 27.6384 | 0.0000 |
| 0 | 0 | 16 | -0.1635 | 27.7273 | 0.0130 |
| 0 | 0 | 16 | -0.1750 | 33.8323 | 0.0669 |
| 0 | 0 | 32 | -0.1524 | 27.5914 | -0.0044 |
| 0 | 11989 | 16 | -0.1858 | 33.9875 | 0.0510 |
| 0 | 12156 | 16 | -0.1871 | 34.0060 | 0.0000 |
| 0 | 12240 | 16 | -0.1613 | 27.7005 | 0.0327 |
| 0 | 12293 | 32 | -0.1330 | 27.3584 | 0.0233 |
| 5 | 73440 | 16 | -0.1936 | 28.0839 | -0.0016 |
| 5 | 73608 | 1 | -0.1686 | 35.3393 | 0.0335 |
| 5 | 73758 | 32 | -0.2197 | 28.3859 | 0.0038 |
| 5 | 73842 | 16 | -0.0831 | 26.7522 | 0.0257 |
| 5 | 73884 | 8 | -0.1920 | 34.0804 | 0.0325 |
| 9 | 122400 | 16 | -0.1529 | 27.6004 | 0.0028 |
| 9 | 122680 | 1 | -0.1719 | 35.3900 | 0.0400 |
| 9 | 122930 | 32 | -0.1524 | 27.5912 | -0.0148 |
| 9 | 123070 | 16 | -0.1418 | 27.4670 | 0.0244 |
| 9 | 123140 | 8 | -0.1964 | 34.1426 | 0.0505 |


---

## Coordination logs

### Dragon log
```

  PRIORITY: this is LOWER priority than the T7 counterfactual chain
  you have running. Fire T9 after T7+T6 land if there's still GPU
  time today. No rush; dataset paper headline numbers are already
  strong.

  STATUS DRAGON SIDE — all phase 2 + counterfactual artifacts on
  GitHub tracking branch (commit cf93126). figures/, eval_results/
  SUMMARY.md, all logs/findings_2026-04-27.md. Master agent can
  read everything from there. Awaiting T7 and any follow-ups.
```

### OrailixTower log
```

2026-05-03T21:01Z  AR-LSTM counterfactual chain (L1, L2, L_big) launched — protocol matches dragon's JEPA E1/E2/E5/E6probe/E7probe: --n_episodes 30 --horizon 100 --actions 0..6 --branch_fracs 0.25 0.5 0.75 --history H_train (16 for L1, 32 for L2/L_big — dir name says w4_H32 so L_big uses 32, not 16 as the task hint suggested).
2026-05-03T20:58Z  L1 done: 630 rollouts, 0 NaN, mean Pearson=+0.112, mean ‖model‖/‖sim‖=2.486. results_L1.json saved.
2026-05-03T21:34Z  L2 done: 630 rollouts, 0 NaN, mean Pearson=+0.131, mean ‖model‖/‖sim‖=2.980. results_L2.json saved.
2026-05-03T20:54Z  L_big first attempt FAILED — all 630 rollouts errored "Sizes of tensors must match except in dimension 2. Expected size 29 but got size 32" because counterfactual_fidelity.py doesn't trim actions to match emb length when obs_window_size>1 (W=4 for L_big → encoder produces H-3 emb rows from H input frames). Discovered that the OLD T7-B run on Apr 28 had this fix LOCALLY but it never got committed (found it in git stash@{0}).
2026-05-03T21:01Z  Applied W>1 fix locally: trim `actions = info["action"][:, -T_emb:]` so action length matches embedding length. Also use T_emb in the HS computation. Diff is 6 lines; preserves parity with dragon's JEPA runs (all W=1 so no behaviour change there).
2026-05-03T23:01Z  L_big rerun done: 630 rollouts, 0 NaN, mean Pearson=+0.106, mean ‖model‖/‖sim‖=1.658. results_Lbig.json saved.
2026-05-03T23:02Z  Ran action_conditioned_cf_probe.py: produced metrics_{L1,L2,Lbig}.json. Wrote render_action_cf_probe_unified.py to combine OT's metrics with dragon's existing metrics_E*/RSSM.json into a unified ACTION_CF_PROBE.md (didn't touch action_conditioned_cf_probe.py — kept harness parity).
2026-05-03T23:02Z  Headline cross-arch ranking (mean Pearson across 6 actions): E5=0.204 > E6probe=0.197 > E1=0.165 > E2=0.141 > L2=0.131 > L1=0.112 > Lbig=0.106 > E7probe=0.077. RSSM action-blind (Pearson NaN, ‖model‖/‖sim‖≈0). Sign-agreement leader: L_big=0.744. n_pairs=90 for every (action × ckpt).
2026-05-03T23:02Z  Open issue for dragon: counterfactual_fidelity.py W>1 fix is a real bug — the script only worked on Apr 28 because of the un-committed local patch. Recommend committing the fix on dragon's side too. Diff is small (6 lines around line 76-86 in current file).
```

---

_To refresh: `./aggregate_results.sh`. Hooked into the `peer_sync` daemon
so it regenerates after every pull tick (~every 10 min)._
