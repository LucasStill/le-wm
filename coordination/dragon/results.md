# Dragon JEPA le-wm results — auto-updated

Scenario-4 TurboSens, JEPA encoder (828K params), bf16-mixed, AdamW lr=7e-5
cosine, hi_probe at epoch 5 + final epoch.

## ⚡ Headline finding (2026-04-26)

**In-distribution Pearson and OOD generalization are anti-correlated** across
the (H, S) sweep:

| run | H | S | regular `test` Pearson | OOD `test_hard` Pearson |
|-----|---|---|------------------------|--------------------------|
| E1  | 16 | 1 | **0.563** (best in-dist) | 0.152 (worst OOD)       |
| E2  | 32 | 1 | 0.196                    | **0.447**                |
| E3  | 16 | 5 | -0.010                   | **0.370**                |

E1 (hardest pretext task: shortest history, finest stride) overfits to training
distribution. E2 and E3 (easier pretext tasks) produce more abstract features
that survive distribution shift. **For deployment, E2 or E3 are likely the
better encoder.**

E5 (H=8, S=1) launched 12:17 UTC to test the prediction: smaller H ⇒ even
harder pretext task ⇒ should be even more overfit (regular > 0.56, OOD < 0.15).
ETA ~16:30 UTC.

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
