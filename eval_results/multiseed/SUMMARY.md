# Multi-seed eval_sweep task-1 — HI mean Pearson-r

_Aggregated 30 runs across 10 (ckpt, split) cells._

| ckpt   | regular test (mean ± std)        | test_hard (mean ± std)           | n_valid |
|--------|----------------------------------|----------------------------------|---------|
| E1     | +0.447 ± 0.033 (1 NaN)           | +0.053 ± 0.138 (1 NaN)           | 4 |
| E2     | +0.588 ± 0.028 (1 NaN)           | +0.262 ± 0.073                   | 5 |
| E3     | +0.431 ± 0.032 (1 NaN)           | +0.340 ± 0.041 (1 NaN)           | 4 |
| E5     | +0.514 ± 0.007 (1 NaN)           | -0.028 ± 0.092 (1 NaN)           | 4 |
| E6probe | +0.469 ± 0.153 (1 NaN)           | +0.369 ± 0.052                   | 5 |

## Per-seed breakdown

| ckpt   | split      | seed | Pearson |
|--------|------------|------|---------|
| E1     | test       | 0    | +0.424 |
| E1     | test       | 1    | +nan ← NaN |
| E1     | test       | 2    | +0.470 |
| E1     | test_hard  | 0    | -0.044 |
| E1     | test_hard  | 1    | +nan ← NaN |
| E1     | test_hard  | 2    | +0.150 |
| E2     | test       | 0    | +0.608 |
| E2     | test       | 1    | +nan ← NaN |
| E2     | test       | 2    | +0.569 |
| E2     | test_hard  | 0    | +0.215 |
| E2     | test_hard  | 1    | +0.346 |
| E2     | test_hard  | 2    | +0.225 |
| E3     | test       | 0    | +0.409 |
| E3     | test       | 1    | +nan ← NaN |
| E3     | test       | 2    | +0.454 |
| E3     | test_hard  | 0    | +0.311 |
| E3     | test_hard  | 1    | +nan ← NaN |
| E3     | test_hard  | 2    | +0.369 |
| E5     | test       | 0    | +0.509 |
| E5     | test       | 1    | +nan ← NaN |
| E5     | test       | 2    | +0.519 |
| E5     | test_hard  | 0    | -0.093 |
| E5     | test_hard  | 1    | +nan ← NaN |
| E5     | test_hard  | 2    | +0.037 |
| E6probe | test       | 0    | +0.361 |
| E6probe | test       | 1    | +nan ← NaN |
| E6probe | test       | 2    | +0.577 |
| E6probe | test_hard  | 0    | +0.340 |
| E6probe | test_hard  | 1    | +0.338 |
| E6probe | test_hard  | 2    | +0.429 |
