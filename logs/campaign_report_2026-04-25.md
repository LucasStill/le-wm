# Scenario-4 campaign report — 2026-04-25 ~07:30 UTC

## TL;DR

- **2 full 10-epoch trainings completed**: E1 (H=16 S=1) and E2 (H=32 S=1).
- **E2's training loss is half of E1's** — the world model fits the data better with more history.
- **E2's representation is *worse* for downstream HI/RUL probes** — the gain is not generic.
- **E3 (H=16 S=5) and E4 (H=32 S=5) initially OOM'd** due to a VRAM-scaling miscalculation
  on my part; **fixed and re-launched** at corrected batch sizes. E3 probe currently running
  at ~18 GB VRAM, healthy.

## Models trained (full 10-epoch runs only)

| ID | Config              | bs / accum | num_steps | Epoch time | Wall time | Wandb run |
|----|---------------------|------------|-----------|------------|-----------|-----------|
| E1 | H=16 S=1 P=4        | 512 / 1    | 20        | 26:18      | ~4.4 h    | `zbpmvuo4` |
| E2 | H=32 S=1 P=4        | 256 / 2    | 36        | ~49 min    | ~8.2 h    | `oz81yajr` |

Both used: bf16-mixed, nw=4, AdamW lr=7e-5 cosine schedule, JEPA encoder (828K params),
hi_probe at epochs 5 & 9 (probe_seq_len = HISTORY_LEN), `wandb.config.project=turbofan_S4`,
data = `/home/lthil/.stable_worldmodel/scenario4_train_lewm.h5`.

## Final metrics

### Training losses (lower = better)

| Run | fit/loss | fit/pred_loss | fit/ar_loss | fit/sigreg_loss |
|-----|----------|---------------|-------------|-----------------|
| E1  | 0.406    | 0.152         | 0.092       | 1.695           |
| E2  | **0.254**| **0.076**     | **0.048**   | **1.188**       |

E2 cuts every training loss roughly in half. **More history → better world-model fit.**

### Hi_probe HI-regression mean across 10 components

| Run | Epoch | R²    | RMSE     | Pearson-r |
|-----|-------|-------|----------|-----------|
| E1  | 5     | -0.79 | 0.00275  | **0.30**  |
| E1  | 9     | -0.90 | 0.00295  | 0.25      |
| E2  | 5     | -1.32 | 0.00317  | 0.14      |
| E2  | 9     | -0.91 | 0.00291  | 0.18      |

**Pearson-r is the cleaner read** (R² is noisy when both predictor and target are near-zero).
Best representation by this metric: **E1 at epoch 5 (Pearson 0.30)**. E2 underperforms E1 across
the board on HI-predictability.

### Action-RUL probe (downstream task: predict steps-until-next-maintenance)

| Run | Epoch | RMSE (steps) | MAE (steps) | R²    | Pearson |
|-----|-------|--------------|-------------|-------|---------|
| E1  | 5     | 26.75        | 21.84       | -0.08 | 0.026   |
| E1  | 9     | 27.47        | 22.21       | -0.14 | 0.024   |
| E2  | 5     | 28.39        | 22.73       | -0.22 | 0.004   |
| E2  | 9     | 27.59        | 22.27       | -0.15 | -0.015  |

All near-zero Pearson means **neither encoder learned a usable RUL representation in 10
epochs**. The maintenance-event prediction signal is essentially noise. May need
substantially more training, or RUL is fundamentally harder than the linear probe can
recover from these reps.

## What this tells us so far

1. **More history (H=32 vs H=16) helps the world-model objective but hurts probe quality.**
   The encoder is specializing toward the JEPA pred/AR loss at the expense of generic
   degradation features. Classic specificity-vs-generality tradeoff.
2. **Probe quality is non-monotonic in training epochs.** E1's best probe (epoch 5,
   Pearson 0.30) is *better* than its final-epoch result (0.25). This suggests we may
   want **early stopping on probe metrics**, or that 10 epochs is past the sweet spot.
3. **No model has learned RUL.** The Action-RUL R² is consistently negative. Either the
   task needs very different representations, or 10 epochs isn't enough.
4. **The JEPA loss alone is not a perfect proxy for downstream usefulness.** Future runs
   should track probe metrics live and consider stopping/checkpointing on Pearson-r rather
   than fit/loss.

## What's running now (07:30 UTC)

**E3 probe** — H=16 S=5 P=4, bs=128 accum=4 (effective bs=512), 1 epoch.
- Process 718746, 100% GPU, 17.9 GB VRAM (well under 28 GB safety limit).
- Started ~07:25. Estimated probe finish ~08:40 (longer because of larger num_steps).
- If probe succeeds: full 10-epoch E3 launches automatically.
- After E3: E4 at H=32 S=5, bs=64 accum=8.

## Failed runs (archived to `logs/campaign/failed_attempt_1/`)

| Run | Why it failed |
|-----|---------------|
| E2_probe (v1) | Hydra struct-mode rejected `trainer.accumulate_grad_batches` — needed `+` prefix. Fixed. |
| All v1 cascading failures | Same Hydra error; orchestrator marked everything as can't-fit and exited. |
| E3_probe / E3_probe_fallback (v1.5) | Real VRAM OOM. I underestimated activation cost: stride S=5 quintuples num_steps, and the encoder processes the full num_steps frames before striding. bs=256 accum=2 → 30.9 GB allocated → OOM. |
| E4 (v1.5) | Same root cause. |

## Lesson learned for VRAM modelling

Empirical fit from E1+E2 measurements:

```
VRAM_GB ≈ 0.00146 * batch_size * num_steps
```

where `num_steps = (H + P - 1) * S + W` (the full window the dataloader returns).

Reality: even though the world model only operates on H+P=20 strided tokens, the
**encoder still processes all `num_steps` raw frames**, then the code does
`emb[:, ::s]` to subsample. So the encoder's activations dominate VRAM and they
scale with `num_steps`, not just `H` or `H+P`.

Targeting 28 GB as safety threshold:

| Config         | num_steps | Max bs (28 GB) | Plan         |
|----------------|-----------|----------------|--------------|
| E1 H16 S1      | 20        | 958            | bs=512 ✓     |
| E2 H32 S1      | 36        | 532            | bs=256/acc=2 |
| E3 H16 S5      | 96        | 199            | bs=128/acc=4 |
| E4 H32 S5      | 176       | 109            | bs=64/acc=8  |

All runs preserve effective batch = 512 via gradient accumulation, keeping
optimizer dynamics comparable.

## Files

- Plan: `logs/experiment_plan.md`
- Per-run logs: `logs/s4_real_20260424_1709.log` (E1), `logs/campaign/E2.log`
- Per-run status: `logs/campaign/*.status`
- Orchestrators: `run_experiments.sh` (v1, used for E2), `run_experiments_v2.sh` (current)
- Hydra outputs (configs, hi_probe CSVs): `outputs/2026-04-2{4,5}/<HH-MM-SS>/`
- WandB synced runs: `https://wandb.ai/thil-ecole-polytechnique/turbofan_S4/runs/zbpmvuo4` (E1),
  `oz81yajr` (E2). Older offline-only runs in `wandb/` directory.

## Next steps once E3 + E4 finish

- Add E3, E4 rows to the metrics tables above.
- Decide: do we want a **longer training** of E1 (best probe so far) to confirm whether
  representation degrades further past epoch 9, or stabilizes?
- Consider: probe-on-fit-loss-checkpoint vs probe-on-all-checkpoints (already do per-epoch).
- Consider: tracking probe Pearson-r as model selection criterion instead of pred_loss.
