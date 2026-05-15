# RSSM/DreamerV3 baseline — overnight evaluation plan

Authored by Claude during the bs=512 training run on 2026-05-02.
This document describes the evals that will run automatically once
training completes, and proposes a follow-up suite for the morning.

The training run itself: tmux `s4_rssm`, log `logs/s4_rssm_bs512_2201.log`,
config bs=512 / accum=1 / T=64 / max_epochs=10, ETA ~4-5 h.

---

## What ships automatically tonight

When the persistent monitor sees `Training complete` in the tee'd log, a
post-training pipeline kicks off:

1. **Locate latest checkpoint** at
   `/home/lthil/.stable_worldmodel/rssm_scenario4_T64_S32x32_D512/`,
   matching `rssm_s4_T64_S32x32_D512_epoch_*_object.ckpt`. Final epoch
   is the one used for eval.

2. **Task 1 — HI state probing on `turbosens2_test.h5`.**
   Freeze the encoder, run `model.encode()` over every frame, fit a Ridge
   probe → predict the 10-dim HI state. Mirrors the AR-LSTM Task 1 (it
   uses the same `model.encode()` contract). Reports per-component R²,
   RMSE, Pearson-r, and the mean across HI dims. Output:
   `eval_results/rssm_s4/task1_test.json`.

3. **Task 1 — HI state probing on `turbosens2_test_hard.h5`** (OOD).
   Same probe as above re-applied on the harder split. Output:
   `eval_results/rssm_s4/task1_test_hard.json`.
   The training-side HIProbeCallback fires only on the training-split
   train/val data so this OOD measurement is the first time we see how
   the encoder generalises.

4. **Task 3 — Latent forecasting (RSSM imagination).**
   For 500 random rollout start points in the test set, observe an
   H-step context with `dynamics.observe()`, then roll the **prior**
   forward with `dynamics.img_step()` for τ ∈ [1, 50] using the
   ground-truth future actions. Decode each predicted feat to HI via
   the Task-1 probe. Reports:
   - `rmse_clean[τ]` — trajectories with no maintenance
   - `rmse_event[τ]` — trajectories that include any non-zero action
   - `action_divergence_gap[τ] = rmse_event - rmse_clean`
   This is the same metric AR-LSTM/JEPA report. Output:
   `eval_results/rssm_s4/task3_test.json`.

5. **Comparison report.**
   `RESULTS_RSSM.md` rendered with side-by-side tables vs the AR-LSTM /
   JEPA numbers in `eval_results/all_ckpts_test/summary.json` and
   `eval_results/counterfactual/SUMMARY.md`. Saved at the repo root.

Wallclock budget for the eval pipeline: 15-30 min on the dragon RTX 5090
once training is done. This won't push us anywhere near the 9-10 h
training budget.

### What's deliberately skipped tonight

- **Counterfactual fidelity vs the simulator.** `counterfactual_fidelity.py`
  expects `model.predictor` (JEPA-style); RSSM has `dynamics`. Adapter
  needed (~50 lines). Listed in the morning plan.
- **Multi-seed runs.** A single seed (3072, the train default) is enough
  for a first baseline number; multi-seed needs 3-5× compute.
- **Per-archetype slicing.** The dataset attrs include
  `archetype_names`/`region_names`; reusing the AR-LSTM
  `per_archetype_diag.py` would be ideal but it's AR-LSTM-specific.

---

## Proposed morning experiments

In rough priority order — pick the ones that fit your remaining budget.

### A. Counterfactual fidelity ✅ DONE
Implemented in `counterfactual_fidelity_rssm.py`; ran in ~8 min on the
final ckpt. Results in `eval_results/counterfactual_rssm/{results_RSSM.json,
summary.json}` and surfaced in `RESULTS_RSSM.md`.

### A2. Re-train RSSM with stronger KL pressure (high priority, ~5 h)
The default V3 hyperparams (kl_free=1.0, dyn_scale=0.5, rep_scale=0.1)
let the latent collapse on this dataset. Worth one re-run with
kl_free=0.0 (no free bits — forces the latent to encode information)
and/or larger rep_scale=0.5. Same wallclock as the first run. If HI
Pearson moves into JEPA territory (0.4+), the conclusion changes from
"RSSM underfits" to "RSSM is competitive with hyperparam tuning".

### A3. Smaller-feat RSSM so sl=10 fits (low priority, ~1 day)
Training a stoch=16 / discrete=16 / deter=256 RSSM cuts feat to
~512 dim and would let `seq_len=10` probes fit on the 32 GB card.
Useful for parity with JEPA's sl=10 reporting in the appendix.

### B. Per-checkpoint HI probe trace (low priority, ~5 min)
Use every saved checkpoint (epochs 1-10) to plot mean HI R² vs epoch.
This shows the speed at which the latent acquires HI structure.
- Just loop the Task-1 probe over the 10 ckpts; reuse the training
  encoded train set so probe fitting is cheap.
- Output: `eval_results/rssm_s4/hi_r2_vs_epoch.png`.

### C. Per-archetype HI breakdown (medium priority, ~10 min)
Slice the test set by `archetype_names` (4 archetypes: A_compressor,
B_fan_booster, C_turbine, D_balanced) and recompute Task 1. Reveals
whether RSSM degrades on a specific failure mode.
- Adapt `baselines/ar_lstm/per_archetype_diag.py`.

### D. OOD generalization curve (medium priority, ~15 min)
Read `event_mask` and `valid_flight_mask` from `test_hard.h5`, slice
the HI-probe test set by event severity (no-event / 1 event / 2+
events per episode), report Task 1 R² per slice. Quantifies how the
event distribution shift hurts the encoder.

### E. Capacity / hidden-size sweep (low priority, multi-hour)
Repeat training with `deter=256` and `deter=1024` to characterize
parameter efficiency. Worth doing if we want a **scaling figure** for
the paper. Each new size: ~5 h.

### F. Action-conditioning ablation (medium priority, ~5 h)
Re-train RSSM with all actions zeroed out (`hist_actions = 0`) so the
model can't use the maintenance signal. Compare counterfactual fidelity
gap at τ=99 — should jump significantly if the model is using actions.
This is the cleanest "did we learn action causality" test.

### G. Linear probe vs Transformer probe (sanity, ~5 min)
Right now Task 1 uses a Ridge probe. Re-run with the same
`HIProbeCallback`-style transformer probe used at training to confirm
the encoder is well-conditioned regardless of probe capacity.

---

## Gotchas observed during the first run

- **OOM at `seq_len=10` for the TransformerProbe.** RSSM feat is 1536-dim
  (stoch 32×32 + deter 512); JEPA's was 64-dim. The windowed train tensor at
  sl=10 is `(745k windows, 10 timesteps, 1536 feat) ≈ 46 GB` and doesn't fit
  on a 32 GB card. `multiseed_eval_rssm.py` and `per_archetype_rssm.py` both
  hit this. Fixed by defaulting both to `--seq-lens 1` only and patching
  `multiseed_eval_rssm.py` to write CSV per-sl (so partial sl=1 results
  survive even if sl=10 OOMs). To restore sl=10 you'd need either a probe
  that streams windows or a smaller feat — keep on the followup list.
- **Encoder collapse on HI prediction (real result, not a bug).** At epoch 10
  the model has very low recon loss (0.002) and KL pinned at the kl_free=1.0
  boundary — meaning the posterior collapses to the prior and the latent
  carries little task-relevant info. Ridge Task 1 Pearson is ≈ 0.05 (vs
  JEPA E2 0.60), Pearson NaN per archetype, gap@τ=50 ≈ 0 in latent
  forecasting. Counterfactual differentials are non-trivial though
  (fan_overhaul 0.0023, full_overhaul 0.0052) — same ballpark as JEPA E1 —
  so the dynamics did learn action effects even if the encoder didn't preserve
  HI. Worth a paragraph in the writeup; possible follow-ups in §A below.

## Notes / caveats

- The current RSSM uses `loader.batch_size=512` to match AR-LSTM
  exactly. If reviewers want the canonical Dreamer setup, the original
  Hafner-V3 config has bs=16 with much longer training; we're trading
  optimization budget for wallclock. Worth a paragraph in the writeup.
- HIProbe training-time results in wandb are computed every 5 epochs
  on the training-split eval window only; treat those as a coarse
  curve, not the headline number — the offline `eval_rssm.py` results
  on `test_lewm.h5` are the comparable metric.
- All AR-LSTM/JEPA numbers in `eval_results/` were generated with
  history_size=16 (`H=16`) on the eval side. RSSM Task 3 will use
  H=16 too for fair comparison even though training used T=64
  windows.
