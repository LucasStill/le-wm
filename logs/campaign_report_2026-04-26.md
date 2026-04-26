# Scenario-4 campaign report — 2026-04-26 (UPDATED, supersedes 2026-04-25)

## Executive summary (read this first)

We have run a sweep of (H, S) for the JEPA world model on dragon, and a parallel
AR-LSTM sweep on OrailixTower at matched (H, S, P=4). Two findings dominate.

### Finding 1: the in-training hi_probe was systematically understating encoder quality

The default config capped the probe at `n_subsample=30000` timesteps. We discovered
this by accident:

- **E1 (H=16 S=1) in-training hi_probe (28k probe-train windows): Pearson = 0.25**
- **E1 evaluated with `eval_sweep` (769k probe-train windows, default head): Pearson = 0.563**

OrailixTower's bigger-probe diagnostic confirmed the bottleneck is **probe-train
data size, not probe head capacity**: an 8× bigger probe head on the same 28k
windows actually overfits and gives _worse_ Pearson (-0.07 / +0.02) than the
default head (+0.22 / +0.24).

**Action taken:** bumped `hi_probe.n_subsample` 30000 → 200000 in
`config/train/lewm.yaml`. All future runs see ~7× more probe-train data,
adding ~2-3 min per probe evaluation. Past runs (E1-E3, L1-L2) are unaffected
— they're being re-evaluated via `eval_sweep` for accurate paper-table numbers.

### Finding 2: in-distribution Pearson and OOD generalization are anti-correlated across the (H, S) grid

Same encoders, two test splits:

| run         | regular `test_lewm` Pearson | OOD `test_hard_lewm` Pearson | gap |
|-------------|----------------------------|------------------------------|-----|
| E1 H=16 S=1 | **0.563** (best in-dist)   | 0.152 (worst OOD)            | -73% |
| E2 H=32 S=1 | 0.196                      | **0.447**                    | +128% |
| E3 H=16 S=5 | -0.010                     | **0.370**                    | +∞ |

E1 has the **hardest** pretext task (least context, finest stride) and produces
features that fit the training distribution sharply but transfer poorly. E2 and E3
both ease the pretext task — more context (E2) or larger gaps (E3) — and produce
features that fit the training distribution loosely but transfer **much better**
to OOD.

This is a **specificity ↔ generality tradeoff**:

> Hardening the JEPA pretext task forces the encoder to encode rich per-frame
> details, which fit the training distribution well. Easing the pretext task lets
> the encoder rely on coarser temporal structure, producing more abstract
> features that survive distribution shift.

Implications:
- **For deployment** (predicting HI on new engines we haven't trained on), E2 or
  E3 are likely the better encoders: 0.45 / 0.37 OOD Pearson is meaningful.
- **For ablation experiments** (where train/test distributions match), E1 looks
  best but is misleadingly overfit.
- The earlier "E2/E3 hurt the representation" reading was wrong; it conflated
  "fits training data less" with "is a worse representation". They're not the
  same.

### Hypothesis test in flight: E5 (H=8 S=1)

Launched 2026-04-26 12:17 UTC. Smaller history → even *harder* pretext task than
E1. The specificity-generality story predicts:

- E5 regular test Pearson > 0.563 (more overfit)
- E5 test_hard Pearson < 0.152 (worse OOD transfer)

If observed, the (H, S) → (specificity, generality) axis is confirmed. Wall time
~3-4h, finishes ~16:30 UTC today.

---

## Run inventory

### Dragon (JEPA, RTX 5090, 828K params)

| ID | H  | S | P | bs / accum   | wall  | fit/loss | regular Pearson | test_hard Pearson |
|----|----|---|---|--------------|-------|----------|----------------|--------------------|
| E1 | 16 | 1 | 4 | 512 / 1      | 4.4h  | 0.406    | **0.563**      | 0.152              |
| E2 | 32 | 1 | 4 | 256 / 2      | 8.2h  | 0.254    | 0.196          | **0.447**          |
| E3 | 16 | 5 | 4 | 128 / 4      | 20h   | 0.259    | -0.010         | 0.370              |
| E4 | 32 | 5 | 4 | 64 / 8       | deferred | —     | —              | —                  |
| E5 | 8  | 1 | 4 | 512 / 1      | running | —      | —              | —                  |

### OrailixTower (AR-LSTM, RTX A6000, 1.14M params)

| ID | H  | S | P | bs / accum   | wall  | fit/loss | regular Pearson | test_hard Pearson |
|----|----|---|---|--------------|-------|----------|----------------|--------------------|
| L1 | 16 | 1 | 4 | 512 / 1      | ~7h   | 0.379    | _eval pending_ | _eval pending_     |
| L2 | 32 | 1 | 4 | 256 / 2      | ~13h  | 0.234    | _eval pending_ | _eval pending_     |
| L3 | 16 | 5 | 4 | 128 / 4      | not run | —      | —              | —                  |
| L4 | 32 | 5 | 4 | 64 / 8       | not run | —      | —              | —                  |

L1/L2 in-training Pearson @ ep.9: 0.242 / 0.169 — likely also under-stated by
the 28k cap; eval_sweep run requested.

---

## Per-HI breakdown for E1 (illustrative)

Showing how OOD transfer is component-specific:

| HI    | E1 regular | E1 test_hard | Δ (relative) |
|-------|-----------|--------------|--------------|
| HI_0  | 0.43      | 0.02         | -94% |
| HI_1  | 0.57      | 0.07         | -87% |
| HI_2  | 0.51      | 0.12         | -76% |
| HI_3  | 0.49      | **0.42**     | -14% (robust) |
| HI_4  | 0.51      | 0.17         | -67% |
| HI_5  | 0.34      | 0.19         | -44% |
| HI_6  | 0.73      | 0.23         | -68% |
| HI_7  | 0.63      | 0.20         | -68% |
| HI_8  | 0.71      | **-0.27**    | -138% (sign flip) |
| HI_9  | 0.72      | 0.36         | -50% |

HI_3 is robustly recovered across distributions; HI_8 spectacularly flips sign
on OOD (the encoder learned a relationship that inverts under distribution
shift — likely a spurious correlation specific to train_lewm).

---

## What we're NOT doing

- **More raw data**: 500 engines × ~14k frames each is comparable to or larger
  than C-MAPSS (the standard benchmark). Lucas's preferred framing for the
  paper is "use the trajectories we have more effectively" via the world
  model's action-conditional rollouts, rather than collecting more.
- **Longer than 10 epochs**: training loss plateaus before then; longer runs
  consistently show probe quality drift, not improvement.
- **Action-RUL deep dive**: probe Pearson ≈ 0 universally. Class imbalance
  (rare maintenance events) likely the cause. Not central to paper, deferred.
- **E4** (H=32 S=5): would have taken ~40h to confirm a tautology given that
  E2 and E3 already individually showed the same direction.
- **L3, L4** (AR-LSTM at S=5): same reasoning; deferred unless we want to
  confirm cross-architecturally.

---

## Open questions / next moves

1. **E5 result** (in ~3-4h): does H=8 S=1 confirm the specificity-generality axis?
2. **L1/L2 eval_sweep** (queued on OT): does AR-LSTM show the same axis as JEPA?
3. **E1' parity-with-L1**: scale JEPA predictor to ~1.14M params; only meaningful
   if we want a same-size cross-architecture comparison. Lucas mentioned this.
4. **Bigger probe + bigger probe-train data** (combined): would push Pearson
   higher than 0.56. Not urgent if we've established the cross-config story.
5. **Online-interaction agent** (third agent) is building an API to roll out
   the trained world model. Once available, we evaluate counterfactual prediction
   quality — that's the headline experimental contribution for the paper.

---

## Files and where to find things

- This report: `logs/campaign_report_2026-04-26.md`
- Live aggregate: `RESULTS.md` (auto-refreshed every 10 min)
- GitHub mirror: https://github.com/LucasStill/le-wm/tree/tracking
- Per-run probes: `eval_results/<dataset>/<run>.json`
- WandB project: https://wandb.ai/thil-ecole-polytechnique/turbofan_S4
- Coordination: `coordination/{dragon,orailixtower}/` (shared mailbox)
- Earlier report (now outdated): `logs/campaign_report_2026-04-25.md`
