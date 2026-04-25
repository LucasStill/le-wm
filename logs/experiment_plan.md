# Scenario-4 experimental plan (NeurIPS 2026 — dragon campaign)

## Goal

Sweep temporal context in the **le-wm JEPA world model** on Scenario 4, varying
history depth `H` (how many past tokens the model sees) and stride `S` (temporal
spread per token), holding all other hyperparameters fixed so the final table is
a clean apples-to-apples comparison. `P=4` prediction horizon is kept constant
across runs — per user instruction.

Later campaigns (AR-LSTM, AE baselines) will reuse the same `H/S/P` to enable
head-to-head comparison.

## Design principles

1. **P=4 is fixed.** Every experiment predicts 4 future steps.
2. **Effective batch size = 512** everywhere. When native `bs=512` exceeds VRAM
   (H=32 case), we use `bs=256 + accumulate_grad_batches=2` to preserve the
   effective batch. Same gradient statistics, similar LR dynamics.
3. **Same encoder / optimizer / LR schedule / sigreg / loss weights.** Only
   `H` and `S` change. The architecture stays identical.
4. **10 epochs per run** for this round. That's the user's current budget per
   configuration.
5. **hi_probe enabled** with `probe_seq_len=H`, fires at epoch 5 and final
   epoch (epoch 9 — via the `hi_probe.py` patch added earlier today).
6. **Always do a 1-epoch probe first** for any config we haven't measured, to
   catch OOMs and get an honest epoch-time estimate.

## Temporal-span arithmetic

The effective number of raw timesteps each sample touches is:
`num_steps = (H + P - 1) * S + W`  (W=1 throughout).

| Config      | H  | S | P | num_steps | Timesteps of context the sample sees |
|-------------|----|---|---|-----------|---------------------------------------|
| E1 (done)   | 16 | 1 | 4 | 20        | 20 (baseline)                         |
| E2          | 32 | 1 | 4 | 36        | 36  (2× tokens, 1.8× span)            |
| E3          | 16 | 5 | 4 | 96        | 96  (same tokens as E1, ~5× span)     |
| E4          | 32 | 5 | 4 | 176       | 176 (2× tokens AND 5× span)           |

**Why S=5 not S=2** (user decision, 2026-04-24): episodes average ~14,022 frames, so
even S=5 spans only ~0.7% of an episode. S=2 would be a weak test (0.3%); S=5 is bold
enough to see a real effect. Stride applies to BOTH history AND predictions, so S=5
means predicting 4 timesteps spaced 5 apart (a coarse 20-step-ahead horizon) — this
is a different prediction task than E1's dense horizon, but appropriate given that
gas-turbine degradation is slow.

This decouples the two axes the user cares about:
- **E2 vs E1**: does more history *tokens* help, holding stride fixed?
- **E3 vs E1**: does longer *temporal span* help, holding compute fixed?
- **E4 vs E2/E3**: does combining them give multiplicative gains, or does the
  model saturate on either alone?

## Expected resource profile

Extrapolated from measured runs. Actual numbers will be filled in by the
probe step.

| Config | bs  | accum | Est. VRAM | Est. epoch time | 10-epoch wall |
|--------|-----|-------|-----------|-----------------|---------------|
| E1     | 512 | 1     | 14.9 GB   | 26:18 (measured)| ~4.4 h (done) |
| E2     | 256 | 2     | ~17–22 GB | ~45–55 min      | ~8–9 h        |
| E3     | 512 | 1     | ~15 GB    | ~27 min         | ~4.5 h        |
| E4     | 256 | 2     | ~17–22 GB | ~45–55 min      | ~8–9 h        |

Why E2 probably OOMs at bs=512: seq_len goes from 20 → 36 (1.8×), and attention
memory is O(T²) (3.2× for this factor) while FFN activations are O(T) (1.8×).
Extrapolating the 14.9 GB @ H=16 gives ~30–40 GB at bs=512 — at/past the 28 GB
safety threshold. Dropping to bs=256 halves activation memory; +accum=2
preserves effective batch. E4 has the same resource profile as E2 because H=32
dominates the seq-length blowup (S changes num_steps by a constant factor but
not the per-step compute — same 32 history tokens processed either way).

## Execution plan

An orchestrator script `run_experiments.sh` runs in tmux session `campaign` and
does the following in order (each step blocks on the previous):

1. **Wait** for the current `s4` session (E1) to finish.
2. **E2 probe** (1 epoch, H=32 S=1, bs=256 accum=2). Check for OOM / NaN; capture
   actual VRAM, it/s, epoch time.
3. **E2 training** (10 epochs, same config as E2 probe).
4. **E3 training** (10 epochs, H=16 S=5, bs=512). Probe first — num_steps=96 is a
   big jump from E1's 20 frames, so worth a sanity-check. Model compute is the same
   as E1 (H=16 is what drives compute, not S), so VRAM should match E1's 14.9 GB.
5. **E4 training** (10 epochs, H=32 S=5, bs=256 accum=2). Reuses whichever
   (bs, accum) worked for E2. Same resource profile as E2.

Each step writes a status file to `logs/campaign/` so we can see progress.

## Failure handling

- **OOM** → halve `bs` and double `ACCUM_GRAD`, retry once. If still OOM, skip
  that experiment and note it. Never exceed 28 GB target.
- **NaN loss** → stop the offending run, log the fact, continue to the next
  experiment.
- **>3× slower than estimate** → continue but flag; might indicate bad cache
  behavior.
- **Everything is captured in logs/campaign/*.log** — one file per experiment.

## What comparisons to report at the end

For each experiment, the table row contains:
- Final `fit/pred_loss`, `fit/ar_loss` (training metrics)
- Final `validate/pred_loss` (generalization)
- `hi_probe/mean_r2` and `hi_probe/mean_pearson_r` at epochs 5 and 9 (the
  representation-quality signal — the headline number)
- `hi_probe/rul/r2` and `hi_probe/rul/rmse` at final epoch (downstream task)
- Wall-clock epoch time, VRAM usage

Future comparison runs (AR-LSTM, AE) will use the same `H/S/P` grid so the
rows in the paper's table line up.

## What I did NOT change

- No modifications to model architecture, JEPA, SIGReg, loss weights, or
  optimizer/LR schedule.
- No change to `main` branch.
- No change to dataset.
- No change to `hi_probe` methodology (same callback, `eval_interval=5`, same
  probe architecture — uses the end-of-training patch from earlier today).

---

## Status

- **E1 (current):** H=16 S=1 P=4, bs=512 — **running**, ~21:41 UTC expected finish.
- **E2:** queued in `run_experiments.sh` — kicks off automatically once E1 ends.
- **E3, E4:** queued after E2.
