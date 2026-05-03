# Dragon handoff report

Remote Claude: append a new section under **Timeline** for every meaningful event
(run launched, throughput measured, config changed, bug found, decision made).
Keep entries short (1–3 bullets). Use UTC timestamps: `date -u +%H:%M`.

Update **Current state** in-place whenever the active run changes.
Keep **Open questions** current so the user can answer them on reconnect.

---

## Current state

- **RSSM/DreamerV3 baseline @ bs=512 (active)** — tmux `s4_rssm`, started 2026-05-02
  22:01 UTC. Launcher: `train_rssm_scenario4_orailix.sh` with env-var overrides
  `BATCH_SIZE=512 NUM_WORKERS=8 ACCUM_GRAD=1 MAX_EPOCHS=10 WANDB_MODE=offline`.
  Tee log: `logs/s4_rssm_bs512_2201.log`. Wandb run: `wandb/offline-run-20260502_220125-9inzwnrj`.
  Config: T=64, stoch=32x32, deter=512, hi_probe=on (eval_interval=5 + final epoch),
  bf16-mixed, optimizer.lr=1e-4. Effective bs=512 matches AR-LSTM scenario4 exactly.
- **Live status (22:11 UTC, ~10 min in):** healthy. global_step=4149 in 481 s →
  **8.63 it/s @ bs=512 = 4,418 samples/s** (23.7× faster than bs=16). Loss 0.602
  (recon 0.002, kl 0.60, dyn/rep 1.00). GPU 64-70 %, VRAM 7.93 GB.
- **Measured ETA:** 23.7 min/epoch training + ~1 min/epoch val/hi_probe →
  **~4–5 h for 10 epochs**. Well inside the 9–10 h budget.
- **Older AE/JEPA campaign (Apr 24 → Apr 27):** finished. tmux `s4` and `campaign`
  are gone; corresponding entries in the Decisions log + Timeline below describe
  what was done. Not active any more.

## Open questions for the user

- _(none yet — bs=512 launch is the agreed plan; user said "let it run")_

## Decisions log

- Added MAX_EPOCHS env-var to launcher script (feature branch only, not main).
- Baseline probe (bs=256, nw=4, hi_probe=off, wandb=disabled) ran 2026-04-24 ~15:52–15:56 UTC; completed 24645 steps / 1 epoch in 3m07s at **130.77 it/s**.
- GPU VRAM at checkpoint: 0.03 GB allocated / 0.99 GB reserved — lots of headroom; safe to test bs=512 and 1024.
- Launching bs=512 probe (nw=4, hi_probe=on, wandb=offline) to measure impact; will then try bs=1024.
- bs=512 is better than bs=256: 39,941 samples/s vs 33,477 (+19%). Chose to try bs=1024 next.
- bs=1024 is slightly worse than bs=512: 38,984 samples/s vs 39,941 (-2%). GPU already saturated at bs=512; larger batches don't help.
- VRAM at bs=512: ~2 GB; at bs=1024: ~3.5 GB (both well under 28 GB limit). VRAM is NOT the bottleneck.
- Chose bs=512 as optimal batch size. Checking nw=8 vs nw=4 next.
- nw=8 result: 77.96 it/s — identical to nw=4 (78.01 it/s). Cached pixels saturate GPU regardless of worker count. Keep nw=4 (less OS overhead).
- **CRITICAL VRAM FINDING**: H=16 P=4 at bs=512 uses 14.9 GB VRAM (vs 2 GB at H=1). bs=1024 would extrapolate to ~30 GB → OOM. bs=512 is BOTH throughput-optimal AND memory-safe for real training config.

## Timeline

### 15:52 UTC — baseline throughput run (bs=256)
- Launched by previous session: nw=4, bs=256, bf16, wandb=disabled, hi_probe=off, max_epochs=1.
- 24645 steps, epoch 0 in 3m07s, **130.77 it/s**.
- GPU VRAM reserved ~1 GB at checkpoint — large headroom for bigger batches.

### 16:03 UTC — handoff received; baseline verified; bs=512 probe launched
- Session s4 was finished (no tmux), run had completed cleanly.
- Updated launcher script to expose MAX_EPOCHS env var.
- Launched bs=512 probe: nw=4, bf16, hi_probe=on, wandb=offline, max_epochs=1.

### 16:06 UTC — bs=512 probe complete; bs=1024 probe launched
- bs=512 result: 78.01 it/s, 12322 steps, epoch in 2:37. VRAM ~2 GB (out of 32 GB).
- Samples/s: 39,941 (+19% over bs=256 at 33,477).
- Launched bs=1024 probe immediately (same other params).

### 16:10 UTC — bs=1024 probe complete; decided bs=512 is optimal; nw=8 probe launched
- bs=1024 result: 38.07 it/s, 6161 steps, epoch in 2:41. VRAM ~3.5 GB.
- Samples/s: 38,984 (-2% vs bs=512). Larger batch doesn't help — GPU already saturated.
- Batch size sweep done: optimal is **bs=512**.
- Launched nw=8 probe (bs=512, max_epochs=1) to check worker count impact.

### 16:13 UTC — nw=8 probe complete; real-config 1-epoch probe launched
- nw=8: 77.96 it/s — tied with nw=4 (78.01 it/s). No benefit to more workers with cached data.
- Keeping nw=4. Tuning done: **bs=512, nw=4**.
- Launched H=16 P=4 1-epoch probe to get epoch time and check VRAM before full run.
- CRITICAL: H=16 P=4 at bs=512 → 14.9 GB VRAM (98% GPU). bs=1024 → ~30 GB → OOM. bs=512 confirmed.
- Dataset: 7,010,872 timesteps, 500 episodes, ~14,022 frames/episode avg.

### 16:41 UTC — real-config 1-epoch probe complete
- **7.77 it/s**, 12,307 steps/epoch, **epoch time 26:18**. VRAM 14.9 GB. GPU 98-99% throughout.
- Validation + hi_probe + checkpoint after epoch: ~1 min. Hi_probe is fast (small dataset).
- 100-epoch estimate: ~44 hours (~1.85 days).

### 17:04 UTC — 10-epoch training launched (first attempt, killed)
- Config: bs=512, nw=4, H=16, P=4, MAX_EPOCHS=10, WANDB_MODE=offline, hi_probe=on.
- Discovered hi_probe skips epoch 0 and only fires on `epoch % eval_interval == 0`; with
  max_epochs=10 and eval_interval=5, final epoch (9) never got evaluated.
- Killed the run ~2 min in to patch hi_probe.py.

### 17:09 UTC — hi_probe.py patched; 10-epoch training relaunched
- Patch: `on_validation_epoch_end` now also fires on the final epoch (epoch == max_epochs-1).
- Effect: hi_probe runs at epoch 5 (interval) AND epoch 9 (final). Backwards-compatible with
  longer runs (e.g., max_epochs=100 → fires at 5,10,...,95 + 99).
- Relaunched training with the patched callback. Same hyperparams.

### 21:31 UTC — experimental campaign queued
- Wrote `logs/experiment_plan.md` covering E2/E3/E4 (H=32, H=16 S=2, H=32 S=2).
- Extended launcher with `ACCUM_GRAD` env-var for gradient accumulation (preserves
  effective bs=512 when native bs must drop for VRAM).
- Wrote `run_experiments.sh` orchestrator: waits for `s4` to end, then runs each
  experiment with probe-first + OOM fallback. All runs keep P=4 for comparability.
- Launched orchestrator in tmux `campaign` — will auto-kick-off E2 when E1 finishes.
- Monitor `bm27v1cyh` on E1 still active.

### 2026-05-03 ~04:30 UTC — RSSM (DreamerV3) run found stuck; killed and relaunched
- Inherited a `train_rssm.py` run from previous session (PID 2332842, started 2026-05-02 21:18
  local). Diagnosis:
  - Process at 100% CPU and 1.4 GB GPU (40% util) but **no log writes for 7+ hours**.
  - `outputs/.../train_rssm.log` last touched 21:18:34 (after HIProbe init).
  - `s4_rssm_20260502_2118.log` (tee'd) last touched 21:18:45 (after env_info callback).
  - `wandb/.../run-*.wandb` stopped at 21:27, `wandb/.../files/output.log` at 21:23.
  - `wandb/debug-cli.lthil.log` ballooned to **1.27 GB** by 21:28 — strongly suggests a
    runaway wandb-core write loop / deadlock during training startup.
- Killed: `tmux kill-session -t s4_rssm`, `pkill -9 -f train_rssm`, `pkill -9 wandb-core`.
  GPU returned to 0%/2 MiB.
- Relaunched via `train_rssm_scenario4_orailix.sh` in tmux `s4_rssm`, with
  `STABLEWM_HOME=/home/lthil/.stable_worldmodel` and default `WANDB_MODE=offline`.
  Same hyperparams: bs=16, T=64, stoch=32x32, deter=512, hi_probe=on, max_epochs=10.
- Monitoring for first iteration / sanity-check output to confirm it's actually stepping.

### 2026-05-02 21:45 UTC — bs=16 run was healthy but 4-day ETA detected
- The relaunch (PID 2335665) was confirmed live: at 21:45 it was at global_step=10899
  in 935.86s of runtime → **11.65 it/s, loss=0.60 (recon=0.003, kl=0.6, dyn=1.0, rep=1.0)**.
- Computed steps/epoch: HDF5Dataset had 6,979,372 stride-1 windows → train split (0.9)
  = 6,281,434 → **392,589 steps/epoch at bs=16 → ~9.4 h/epoch → ~3.9 days for 10 epochs.**
- User flagged this as unacceptable (initial mental budget was ~2 hours; settled on
  9–10 h budget on reconnect).

### 2026-05-02 21:55 UTC — killed bs=16, probed bs=128, then launched bs=512
- Killed `s4_rssm` tmux + `train_rssm` + `wandb-core` PIDs. GPU returned to 0 %/2 MiB.
  No checkpoint had been saved (only 0.3 % into epoch 0), nothing lost.
- bs=128 probe (`tmux probe128`, WANDB_MODE=disabled): VRAM 4.85 GB, GPU 49 %.
  Only +8 pp utilization for 8x batch → **GRU sequential is the bottleneck**, not memory.
  Killed the probe (couldn't read it/s with wandb disabled).
- Launched real run **bs=512, accum=1, nw=8, MAX_EPOCHS=10, WANDB_MODE=offline**
  in tmux `s4_rssm` at 22:01:25 UTC. Tee log: `logs/s4_rssm_bs512_2201.log`,
  wandb dir: `wandb/offline-run-20260502_220125-9inzwnrj`. Effective bs=512 matches
  AR-LSTM scenario4 exactly (fair comparison preserved, no dataset subsampling).
- At 22:07 UTC: GPU 70 % / 7.93 GB VRAM, wandb .wandb file growing (96 KB).
  No tqdm/wandb-summary visible yet (Lightning quirk + sparse logging cadence at
  large bs). Run is healthy. Theoretical ETA: 9–12 h for 10 epochs.

### 2026-05-02 22:11 UTC — first wandb-summary flushed; speed confirmed
- `_runtime=481`, `global_step=4149`, `epoch=0`. Throughput **8.63 it/s @ bs=512
  = 4,418 samples/s** — 23.7× faster than bs=16 (186 samples/s). The GRU
  bottleneck saturates well at bs=512 even though GPU is "only" 64-70 % util.
- Loss curve mirrors bs=16 run: recon=0.002, kl=0.60, dyn/rep=1.00, total=0.602.
  KL/dyn/rep are pinned to the kl_free=1.0 boundary as expected for early training.
- Refined ETA: 12,268 steps / 8.63 it/s = 23.7 min/epoch + val/hi_probe overhead
  → **~4–5 h for 10 epochs**. Comfortably under the 9–10 h budget.
- Persistent monitor armed (`grep` over the tee log) for OOMs / errors / epoch
  boundaries / "Training complete". Wakeup scheduled at 22:37 UTC for a re-check.

### 2026-05-02 22:37 UTC — wakeup re-check; epoch 0+1 done; pace holding
- `_runtime=2046`, `global_step=16,999`, `epoch=1` (just kicked off epoch 2).
  Throughput **8.30 it/s** (vs 8.63 at first flush — slight slowdown after the
  first hi_probe pass at end of epoch 0, expected).
- Validation metrics from epoch 0/1: `validate/loss=0.602`, `recon=0.0022`,
  `kl=0.60`, `dyn/rep=1.00` — train and val tracking each other tightly.
- VRAM jumped from 7.9 → 13.7 GB (hi_probe state retained between epochs);
  still safe under the 28 GB limit. GPU 70 %.
- One checkpoint saved: `rssm_s4_T64_S32x32_D512_epoch_1_object.ckpt` (22 MB).
- Refined ETA: 24.6 min/epoch (training) + ~1 min/epoch val/hi_probe →
  **finish ~02:20 UTC**, ~4.3 h total. Within the 9–10 h budget.
- Eval pipeline scaffolding shipped tonight: `multiseed_eval_rssm.py`,
  `eval_rssm.py`, `per_archetype_rssm.py`, `counterfactual_fidelity_rssm.py`,
  `render_results.py`, `BASELINE_TEMPLATE.md`. tmux `eval_watch` polling for
  `Training complete.` to fire `run_post_training_eval.sh`. Smoke test of
  multiseed eval against `epoch_1` ckpt running concurrently as a sanity check.

### 2026-05-03 00:38 UTC — wakeup mid-training audit; 6 epochs done
- 6 ckpts on disk (`epoch_{1..6}_object.ckpt`). Currently in epoch 6 (~5 min in,
  expecting ~25 min). No `Training complete.` yet, no post-training eval log yet.
- All 4 tmux sessions alive: `s4_rssm`, `eval_watch`, `peer_sync`, `wandb_sync`.
- Throughput holding at ~25 min/epoch (slightly slower epochs that include
  the hi_probe pass — epoch 5 took 28 min because of that).
- 4 epochs to go (incl. epoch 9 with hi_probe + eval) → **finish ~02:18 UTC**.
- Smoke test of `multiseed_eval_rssm.py` against the epoch_1 ckpt completed
  at 22:48 UTC: pipeline runs cleanly end-to-end, but Pearson r is NaN per HI
  dim because the encoder at epoch 1 is still close to random (KL pinned at
  the kl_free=1.0 boundary). Expected to resolve as KL pressure forces the
  latent to encode useful info; if Pearson is still NaN at epoch 10, that's a
  real result (RSSM default V3 hyperparams underfit) not a pipeline bug.
- Recon-loss validation curve so far (small but monotone-ish):
  epoch 0 = 0.00223, epoch 1 = 0.00204, epoch 2 = 0.00208, epoch 3 = 0.00200,
  epoch 4 = 0.00202, epoch 5 = 0.00204. Loss ~0.6020, KL/dyn/rep at the
  free-bits floor.
- HIProbe at epoch 5 boundary trained the TransformerProbe to mean RMSE=0.00372
  (in the JEPA/AR-LSTM ballpark — encouraging that the encoder is now learning
  signal even if the Pearson story is messy).

### 2026-05-03 02:14 UTC — Training complete (10 ckpts saved); eval pipeline fired
- 10 epochs, ~24-28 min/epoch (slower epochs are the ones with hi_probe).
- Total wallclock: 22:01 → 02:14 = **4 h 13 min** for training proper, well under
  the 9-10 h budget.
- Final-epoch hi_probe at 02:13 hit best mean RMSE 0.00371 (= epoch 5's 0.00372,
  basically flat — the encoder representation didn't materially change between
  epoch 5 and epoch 10).
- `eval_watch` autonomously fired `run_post_training_eval.sh` ~30 s after
  `Training complete.` printed.

### 2026-05-03 02:34 UTC — eval pipeline finished with two failures
- Pipeline ran 5 steps in ~20 min. **task3** (test + test_hard) and
  **counterfactual_fidelity_rssm.py** finished cleanly. **multiseed** and
  **per_archetype** OOM'd at `seq_len=10`: RSSM feat = 1536 (32×32 stoch +
  512 deter, vs JEPA's 64) makes the windowed train tensor (745k × 10 × 1536)
  ≈ 46 GB, doesn't fit on 32 GB. Both crashed mid-script and lost partial
  data because writes happen after both seq_lens complete.
- Patched `multiseed_eval_rssm.py` to write CSV per-sl and default `--seq-lens 1`
  only. Updated launcher to pass `--seq-lens 1` for both multiseed +
  per_archetype. Hard-card decision: keep sl=10 disabled until we either run a
  smaller-feat RSSM or write a streaming probe.
- Initial render produced `RESULTS_RSSM.md` with **headline metric missing**
  (multiseed CSV empty) but Ridge-probe + counterfactual + Task 3 + 3-of-4
  archetype rows present.

### 2026-05-03 02:35 UTC — eval_redo tmux launched to fill gaps
- Re-runs multiseed (test + test_hard, seeds 0-5, sl=1 only) → per_archetype
  (test + test_hard, sl=1 only) → re-renders RESULTS_RSSM.md. Each multiseed
  step ~12 min (6 seeds × ~2 min/probe), per_archetype ~1 min. Total ~25-30 min.
- Wakeup queued at 03:18 UTC to verify completion + write final summary.

### Key findings so far (will be finalized after eval_redo)
- **Encoder collapse on HI:** Ridge Task 1 mean Pearson **+0.048** on test,
  **+0.058** on test_hard — vs JEPA E2's +0.597 on test. Per-archetype Pearson
  is NaN (probe predictions are constant within each archetype slice). KL is
  pinned at the kl_free=1.0 floor, indicating the posterior collapsed to the
  prior.
- **Action effects DID survive the collapse:** counterfactual differentials
  fan_overhaul 0.0023 / full_overhaul 0.0052 / hpc_overhaul 0.0029 — same
  ballpark as JEPA E1 (0.0037 / 0.0047 / 0.0028). The RSSM dynamics learned
  what each action does even though the encoder didn't preserve HI structure.
- **Latent forecasting gap ≈ 0:** RMSE_event − RMSE_clean is essentially zero
  across all τ ∈ [1, 50] on both splits, so latent forecasting can't
  distinguish maintenance from clean trajectories. Consistent with the
  collapsed encoder story.
- **Recon loss is fine** (0.002) — the decoder fits the input perfectly. The
  collapse is in the latent space's task-relevance, not in pixel space.
- **Recommendation for paper:** report this as the headline RSSM number with
  a paragraph on "default V3 hyperparams collapse on TurboSens 2"; the obvious
  follow-up is to retrain with `kl_free=0.0` to remove the free-bits relief.
  Queued as item A2 in `EVAL_PLAN_RSSM.md`.

### 2026-05-03 03:16 UTC — eval_redo finished; RESULTS_RSSM.md finalized
- Multiseed completed (6 seeds × 2 splits, sl=1 only): all 12 probe runs
  produced **NaN Pearson** with R² ≈ -0.04 (test) / -0.11 (test_hard) and
  RMSE ≈ 0.0032 / 0.0045. Confirms encoder collapse is universal across probe
  seeds, not a one-off.
- Per-archetype completed: 4 archetypes × 2 splits, all with NaN Pearson too.
- `RESULTS_RSSM.md` re-rendered with: (a) the RSSM headline row labelled
  `NaN (6/6 probes)` instead of being silently dropped, (b) a backup table
  reporting R²/RMSE means since those are non-degenerate, (c) a TL;DR
  paragraph at the top with the headline + recommendation.
- Final artifact list:
  - `RESULTS_RSSM.md` (headline report)
  - `EVAL_PLAN_RSSM.md` (what ran + 7 follow-up experiments)
  - `BASELINE_TEMPLATE.md` (how to add the next baseline)
  - `eval_results/rssm_s4/{multiseed_results.csv, per_archetype.csv,
    task_results_test.json, task_results_test_hard.json}`
  - `eval_results/counterfactual_rssm/{results_RSSM.json, summary.json}`
  - 10 RSSM checkpoints under `~/.stable_worldmodel/rssm_scenario4_T64_S32x32_D512/`
- Total wallclock for the night: training 22:01 → 02:14 (4h13m) +
  eval pipeline 02:14 → 02:34 (20 min) + eval re-do 02:35 → 03:16 (41 min) =
  **5h15m end-to-end**, well inside the 9-10 h budget.

### Open follow-ups
1. **A2 in `EVAL_PLAN_RSSM.md`**: re-train RSSM with `kl_free=0.0` to test
   whether the collapse is hyperparameter-driven. If the new run hits JEPA
   territory (Pearson 0.4+) that's the actual paper number; if not, the
   "RSSM doesn't fit TurboSens 2" conclusion stands.
2. **A3**: smaller-feat RSSM (stoch=16, deter=256) to enable sl=10 probing
   without OOM.
3. **B-G**: per-checkpoint trace, capacity sweep, action-conditioning
   ablation, etc. — see `EVAL_PLAN_RSSM.md`.
