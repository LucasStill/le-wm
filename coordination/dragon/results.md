# Dragon JEPA le-wm results — auto-updated

Scenario-4 TurboSens, 10 epochs each, bf16-mixed, AdamW lr=7e-5 cosine,
hi_probe at epoch 5 + final epoch.

## Run table

| ID | H | S | P | bs (eff. 512) | Wall | fit/loss | fit/pred | fit/ar | fit/sigreg |
|----|---|---|---|---------------|------|----------|----------|--------|-----------|
| E1 | 16 | 1 | 4 | 512 / acc=1 | 4.4h | 0.406 | 0.152 | 0.092 | 1.695 |
| E2 | 32 | 1 | 4 | 256 / acc=2 | 8.2h | 0.254 | 0.076 | 0.048 | 1.188 |
| E3 | 16 | 5 | 4 | 128 / acc=4 | 20h | **0.259** | **0.078** | **0.038** | **1.203** |
| E4 | 32 | 5 | 4 | 64 / acc=8 | **deferred** | – | – | – | – |

E4 deferred per Lucas's adaptive-decision plan: E3 essentially tied E1 on probe quality (0.252 vs 0.250), so the H=32 + S=5 combination is unlikely to be a step-function improvement. Saving the ~40h for higher-info experiments (E1' parity-with-L1, train/test_hard diagnostic).

## Hi_probe HI-regression (mean over 10 components)

| ID | epoch | R² | RMSE | Pearson-r |
|----|-------|----|------|-----------|
| E1 | 5 | -0.79 | 0.00275 | **0.30** |
| E1 | 9 | -0.90 | 0.00295 | 0.25 |
| E2 | 5 | -1.32 | 0.00317 | 0.14 |
| E2 | 9 | -0.91 | 0.00291 | 0.18 |
| E3 | 5 | -0.69 | 0.00266 | 0.244 |
| E3 | 9 | -1.38 | 0.00314 | 0.252 |

## Action-RUL probe

| ID | epoch | RMSE (steps) | MAE | R² | Pearson |
|----|-------|--------------|-----|----|----|
| E1 | 5 | 26.75 | 21.84 | -0.08 | 0.026 |
| E1 | 9 | 27.47 | 22.21 | -0.14 | 0.024 |
| E2 | 5 | 28.39 | 22.73 | -0.22 | 0.004 |
| E2 | 9 | 27.59 | 22.27 | -0.15 | -0.015 |

## Key findings so far

1. **More history (H=32 vs H=16) halves training loss** (E2 fit/pred=0.076 vs
   E1 0.152) but **degrades probe representation** (E2 best Pearson 0.18 vs
   E1 best 0.30). Specificity-vs-generality tradeoff.
2. **Probe quality is non-monotonic** in training epochs. E1's best probe is
   at epoch 5, not epoch 9. Suggests possible early stopping on probe metric.
3. **No model has learned RUL** in 10 epochs. Action-RUL Pearson ≈ 0 across
   the board. Either the task is much harder than HI regression or 10 epochs
   is insufficient.
4. **VRAM scales with `num_steps = (H+P-1)*S+W`**, not just `H`. The encoder
   processes the full window before striding. Empirical:
   `VRAM_GB ≈ 0.00146 × bs × num_steps`.

## Wandb runs (on https://wandb.ai/thil-ecole-polytechnique/turbofan_S4/runs/)

- E1: `zbpmvuo4`
- E2: `oz81yajr`
- E3: `5qv43zup` (live, partial sync)
- All probes also synced (smaller, less interesting)
