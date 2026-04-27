# Multi-seed eval_sweep task-1 — HI mean Pearson-r

_Aggregated 78 runs across 16 (ckpt, split) cells._

| ckpt   | regular test (mean ± std)        | test_hard (mean ± std)           | n_valid |
|--------|----------------------------------|----------------------------------|---------|
| E1     | +0.446 ± 0.039 (1 NaN)           | +0.234 ± 0.180 (1 NaN)           | 10 |
| E2     | +0.597 ± 0.025 (1 NaN)           | +0.327 ± 0.135                   | 11 |
| E2ep10 | +0.598 ± 0.026                   | +0.286 ± 0.114                   | 6 |
| E2ep5  | +0.357 ± 0.129                   | +0.307 ± 0.120                   | 6 |
| E3     | +0.450 ± 0.054 (1 NaN)           | +0.284 ± 0.104 (1 NaN)           | 10 |
| E5     | +0.400 ± 0.218 (1 NaN)           | +0.178 ± 0.196 (1 NaN)           | 10 |
| E6probe | +0.406 ± 0.095 (1 NaN)           | +0.345 ± 0.044                   | 11 |
| E7probe | +0.519 ± 0.058                   | +0.394 ± 0.025                   | 6 |

## Per-seed breakdown

| ckpt   | split      | seed | Pearson |
|--------|------------|------|---------|
| E1     | test       | 0    | +0.424 |
| E1     | test       | 1    | +nan ← NaN |
| E1     | test       | 2    | +0.470 |
| E1     | test       | 3    | +0.403 |
| E1     | test       | 4    | +0.431 |
| E1     | test       | 5    | +0.501 |
| E1     | test_hard  | 0    | -0.044 |
| E1     | test_hard  | 1    | +nan ← NaN |
| E1     | test_hard  | 2    | +0.150 |
| E1     | test_hard  | 3    | +0.374 |
| E1     | test_hard  | 4    | +0.334 |
| E1     | test_hard  | 5    | +0.358 |
| E2     | test       | 0    | +0.608 |
| E2     | test       | 1    | +nan ← NaN |
| E2     | test       | 2    | +0.569 |
| E2     | test       | 3    | +0.619 |
| E2     | test       | 4    | +0.571 |
| E2     | test       | 5    | +0.616 |
| E2     | test_hard  | 0    | +0.215 |
| E2     | test_hard  | 1    | +0.346 |
| E2     | test_hard  | 2    | +0.225 |
| E2     | test_hard  | 3    | +0.417 |
| E2     | test_hard  | 4    | +0.212 |
| E2     | test_hard  | 5    | +0.544 |
| E2ep10 | test       | 0    | +0.608 |
| E2ep10 | test       | 2    | +0.569 |
| E2ep10 | test       | 3    | +0.619 |
| E2ep10 | test_hard  | 0    | +0.215 |
| E2ep10 | test_hard  | 2    | +0.225 |
| E2ep10 | test_hard  | 3    | +0.417 |
| E2ep5  | test       | 0    | +0.209 |
| E2ep5  | test       | 2    | +0.413 |
| E2ep5  | test       | 3    | +0.448 |
| E2ep5  | test_hard  | 0    | +0.367 |
| E2ep5  | test_hard  | 2    | +0.386 |
| E2ep5  | test_hard  | 3    | +0.169 |
| E3     | test       | 0    | +0.409 |
| E3     | test       | 1    | +nan ← NaN |
| E3     | test       | 2    | +0.454 |
| E3     | test       | 3    | +0.517 |
| E3     | test       | 4    | +0.385 |
| E3     | test       | 5    | +0.484 |
| E3     | test_hard  | 0    | +0.311 |
| E3     | test_hard  | 1    | +nan ← NaN |
| E3     | test_hard  | 2    | +0.369 |
| E3     | test_hard  | 3    | +0.330 |
| E3     | test_hard  | 4    | +0.104 |
| E3     | test_hard  | 5    | +0.308 |
| E5     | test       | 0    | +0.509 |
| E5     | test       | 1    | +nan ← NaN |
| E5     | test       | 2    | +0.519 |
| E5     | test       | 3    | +0.475 |
| E5     | test       | 4    | +0.011 |
| E5     | test       | 5    | +0.484 |
| E5     | test_hard  | 0    | -0.093 |
| E5     | test_hard  | 1    | +nan ← NaN |
| E5     | test_hard  | 2    | +0.037 |
| E5     | test_hard  | 3    | +0.260 |
| E5     | test_hard  | 4    | +0.338 |
| E5     | test_hard  | 5    | +0.346 |
| E6probe | test       | 0    | +0.361 |
| E6probe | test       | 1    | +nan ← NaN |
| E6probe | test       | 2    | +0.577 |
| E6probe | test       | 3    | +0.367 |
| E6probe | test       | 4    | +0.366 |
| E6probe | test       | 5    | +0.360 |
| E6probe | test_hard  | 0    | +0.340 |
| E6probe | test_hard  | 1    | +0.338 |
| E6probe | test_hard  | 2    | +0.429 |
| E6probe | test_hard  | 3    | +0.329 |
| E6probe | test_hard  | 4    | +0.338 |
| E6probe | test_hard  | 5    | +0.297 |
| E7probe | test       | 0    | +0.475 |
| E7probe | test       | 2    | +0.496 |
| E7probe | test       | 3    | +0.585 |
| E7probe | test_hard  | 0    | +0.376 |
| E7probe | test_hard  | 2    | +0.423 |
| E7probe | test_hard  | 3    | +0.382 |
