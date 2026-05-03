# Action-conditioned counterfactual probe — quantitative match

For every (episode, branch_t) we compute Δ_sim(a) = sim_HI(a) − sim_HI(do_nothing) and Δ_model(a) = model_HI(a) − model_HI(do_nothing) across the full horizon. The four metrics below answer _does the world model react to action a like the simulator does?_:

- **Pearson** between the Δ vectors flattened over (ep, branch, τ, HI dim).
- **R²** of regressing Δ_sim on Δ_model (intercept-free; 1.0 = perfect, 0.0 = trivial mean, negative = worse than mean).
- **Sign agreement**: fraction of non-zero (ep, branch, τ, HI) tuples where sign(Δ_sim) = sign(Δ_model). 0.5 = chance.
- **Magnitude ratio**: ‖Δ_model‖_RMS / ‖Δ_sim‖_RMS. 1.0 = matched amplitude; near 0 = model is action-blind; >>1 = model overreacts.

## Headline — full-horizon metrics per (architecture, action)

| ckpt | action | n pairs | Pearson | R² | sign agree | ‖model‖/‖sim‖ |
|------|--------|---------|---------|-----|-----------|---------------|
| E1 | fan_overhaul | 90 | +0.006 | -2.013 | 0.102 | 1.077 |
| E1 | hpc_overhaul | 90 | +0.146 | -0.414 | 0.786 | 0.774 |
| E1 | turbine_overhaul | 90 | +0.442 | -0.282 | 0.760 | 0.885 |
| E1 | full_overhaul | 90 | +0.165 | -2.103 | 0.643 | 0.429 |
| E1 | patch | 90 | +0.197 | -1.745 | 0.885 | 1.601 |
| E1 | wash | 90 | +0.034 | -87.246 | 0.437 | 9.337 |
|      |        |         |         |     |           |               |
| E2 | fan_overhaul | 90 | +0.324 | -1.720 | 0.713 | 1.455 |
| E2 | hpc_overhaul | 90 | -0.177 | -3.351 | 0.288 | 1.205 |
| E2 | turbine_overhaul | 90 | +0.236 | -1.805 | 0.810 | 1.311 |
| E2 | full_overhaul | 90 | +0.211 | -2.725 | 0.716 | 0.637 |
| E2 | patch | 90 | +0.226 | -3.198 | 0.799 | 2.040 |
| E2 | wash | 90 | +0.026 | -128.626 | 0.674 | 11.351 |
|      |        |         |         |     |           |               |
| E5 | fan_overhaul | 90 | +0.267 | -1.205 | 0.106 | 0.914 |
| E5 | hpc_overhaul | 90 | +0.111 | -1.009 | 0.360 | 0.564 |
| E5 | turbine_overhaul | 90 | +0.537 | +0.178 | 0.987 | 0.909 |
| E5 | full_overhaul | 90 | +0.153 | -1.665 | 0.993 | 0.426 |
| E5 | patch | 90 | +0.131 | -1.332 | 0.923 | 1.370 |
| E5 | wash | 90 | +0.025 | -63.521 | 0.533 | 7.964 |
|      |        |         |         |     |           |               |
| E6probe | fan_overhaul | 90 | +0.339 | -1.079 | 0.880 | 0.975 |
| E6probe | hpc_overhaul | 90 | +0.143 | -1.727 | 0.265 | 0.779 |
| E6probe | turbine_overhaul | 90 | +0.471 | -0.104 | 0.988 | 1.023 |
| E6probe | full_overhaul | 90 | +0.099 | -1.925 | 0.796 | 0.490 |
| E6probe | patch | 90 | +0.112 | -2.028 | 0.817 | 1.596 |
| E6probe | wash | 90 | +0.022 | -87.258 | 0.599 | 9.335 |
|      |        |         |         |     |           |               |
| E7probe | fan_overhaul | 90 | -0.193 | -1.200 | 0.495 | 0.948 |
| E7probe | hpc_overhaul | 90 | -0.031 | -0.860 | 0.597 | 0.817 |
| E7probe | turbine_overhaul | 90 | +0.487 | -0.265 | 0.987 | 1.206 |
| E7probe | full_overhaul | 90 | +0.089 | -1.753 | 0.895 | 0.566 |
| E7probe | patch | 90 | +0.086 | -2.839 | 0.864 | 1.839 |
| E7probe | wash | 90 | +0.022 | -115.747 | 0.547 | 10.749 |
|      |        |         |         |     |           |               |
| RSSM | fan_overhaul | 90 | +nan | -0.156 | nan | 0.000 |
| RSSM | hpc_overhaul | 90 | +nan | -0.458 | nan | 0.000 |
| RSSM | turbine_overhaul | 90 | +nan | -0.404 | nan | 0.000 |
| RSSM | full_overhaul | 90 | +nan | -2.840 | nan | 0.000 |
| RSSM | patch | 90 | +nan | -0.079 | nan | 0.000 |
| RSSM | wash | 90 | +nan | -0.001 | nan | 0.000 |
|      |        |         |         |     |           |               |

## Cross-architecture summary (mean across the 6 non-zero actions)

| ckpt | mean Pearson | mean R² | mean sign agree | mean ‖model‖/‖sim‖ |
|------|--------------|---------|------------------|---------------------|
| E1 | +0.165 | -15.634 | 0.602 | 2.351 |
| E2 | +0.141 | -23.571 | 0.667 | 3.000 |
| E5 | +0.204 | -11.426 | 0.650 | 2.025 |
| E6probe | +0.197 | -15.687 | 0.724 | 2.366 |
| E7probe | +0.077 | -20.444 | 0.731 | 2.687 |
| RSSM | — | -0.656 | — | 0.000 |

## Horizon decay (mean across actions, per τ)

| ckpt | τ=10 Pearson | τ=50 Pearson | τ=99 Pearson | τ=10 sign | τ=50 sign | τ=99 sign |
|------|--------------|--------------|--------------|-----------|-----------|-----------|
| E1 | +0.128 | +0.179 | +0.185 | 0.632 | 0.597 | 0.582 |
| E2 | +0.118 | +0.176 | +0.125 | 0.613 | 0.678 | 0.659 |
| E5 | +0.192 | +0.216 | +0.209 | 0.700 | 0.622 | 0.614 |
| E6probe | +0.228 | +0.207 | +0.190 | 0.720 | 0.718 | 0.701 |
| E7probe | +0.178 | +0.076 | +0.065 | 0.741 | 0.720 | 0.722 |
| RSSM | — | — | — | — | — | — |
