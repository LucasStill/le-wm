# Dragon handoff report

Remote Claude: append a new section under **Timeline** for every meaningful event
(run launched, throughput measured, config changed, bug found, decision made).
Keep entries short (1–3 bullets). Use UTC timestamps: `date -u +%H:%M`.

Update **Current state** in-place whenever the active run changes.
Keep **Open questions** current so the user can answer them on reconnect.

---

## Current state

- **E1 (active):** 10-epoch H=16 training — tmux `s4`, log `logs/s4_real_20260424_1709.log`.
  Config: bs=512 nw=4 bf16 H=16 S=1 P=4. Epoch 8 done at 21:14. 1 epoch to go. Finish ~21:41 UTC.
- **Campaign queued:** tmux `campaign` is waiting for `s4` to end, then auto-runs E2→E3→E4.
  See `logs/experiment_plan.md` for the design and `run_experiments.sh` for the orchestrator.
  Per-run logs land in `logs/campaign/*.log`, status files in `logs/campaign/*.status`.
- **E2 next:** H=32 S=1 P=4, bs=256 + accum=2 (effective bs=512 preserved). Probe first, then 10 epochs.
- **E3 then E4:** H=16 S=5 (~4.5h) and H=32 S=5 (~8-9h). P=4 fixed across all runs for comparability.
  Stride bumped from 2 → 5 at user request to probe longer temporal spans (episodes ~14k frames).
- Hi_probe fires at epoch 5 (interval) and epoch 9 (final) for every run — `hi_probe.py` patch applies.
- See `logs/session_2026-04-24_changes.md` for a full explainer of earlier changes.

## Open questions for the user

- _(none yet)_

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
