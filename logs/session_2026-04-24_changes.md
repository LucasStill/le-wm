# Session changes — 2026-04-24 (Scenario-4 training handoff)

## What was on the stove when I arrived

A baseline throughput test had just completed (130.77 it/s @ bs=256, H=1 P=1). Dataloader
bottleneck was already fixed (pixels cached in RAM, not LZF chunk-decompressed per sample).
Mission: tune batch size and workers, then launch a real training run.

## Sweep results (all at w=1 H=1 S=1 P=1, bf16, pixels cached in RAM)

| bs   | nw | it/s   | samples/s | epoch | VRAM    |
|------|----|--------|-----------|-------|---------|
| 256  | 4  | 130.77 | 33,477    | 3:07  | ~2 GB   |
| 512  | 4  | 78.01  | 39,941    | 2:37  | ~2 GB   |
| 1024 | 4  | 38.07  | 38,984    | 2:41  | ~3.5 GB |
| 512  | 8  | 77.96  | 39,941    | 2:37  | ~2 GB   |

**Takeaways**
- bs=512 is the throughput winner (+19% samples/s vs bs=256).
- bs=1024 is marginally worse — GPU was already compute-saturated at bs=512.
- nw=8 ≈ nw=4. Cached pixels mean the dataloader is never the bottleneck, so extra workers
  just add OS overhead.

## Real-config probe (w=1 H=16 S=1 P=4, bs=512, nw=4, bf16, 1 epoch)

- **7.77 it/s**, 12,307 steps, **epoch 26:18**
- **VRAM 14.9 GB** (vs 2 GB at H=1 — ~7.5x scaling with history)
- GPU 98-99% throughout training
- Post-epoch (val + hi_probe eval + checkpoint): ~50s

**Why the jump in VRAM?** With H=16, backprop-through-time keeps activations for 20 frames
(16 history + 4 predictions) per sample. The model params are tiny (828K) — nearly all of
the 14.9 GB is activation memory.

**bs=1024 would hit ~30 GB → OOM.** bs=512 is the correct choice for *both* throughput and
VRAM safety.

## Why the 10-epoch estimate is ~4.4 hours (not much longer)

Breakdown:
- Training: 10 × 26:18 ≈ **263 min** (this is 95% of the runtime)
- Validation + checkpoint after each epoch: 10 × ~20s ≈ 3 min
- hi_probe evaluations: 2 × ~30s ≈ 1 min
- **Total: ~267 min ≈ 4.45 h**

It seems fast because:
1. The model is **tiny (828K params, ~3 MB)** — almost all VRAM is activations, not weights.
   Most wall-clock goes into moving activations around, not heavy matmul.
2. The dataset is only **~12k training batches/epoch** at bs=512 (vs 24k at bs=256).
3. **No data-loading bottleneck** — pixels cached in RAM eliminates the LZF chunk thrashing
   that plagued earlier runs.
4. **bf16-mixed** keeps tensor cores fed; the RTX 5090 sits at 98–99% utilization the entire
   time — we are compute-saturated, not I/O- or memory-bound.

For context: 100 epochs at the same rate would be **~44 h (~1.85 days)**. A full 10-epoch run
is just 10% of that. The per-epoch cost is what it is; scaling is linear.

## Code changes I made this session

### 1. `train_lewm_scenario4_dragon.sh` — added `MAX_EPOCHS` env-var
The launcher already exposed BATCH_SIZE/NUM_WORKERS/etc., but not max_epochs. I added it so
that probe runs (max_epochs=1) and partial runs (max_epochs=10) can use the same launcher
path without hand-rolling a python command. Default stays at 100.

### 2. `hi_probe.py` — run hi_probe at final epoch, not just intervals
Before: `if epoch == 0 or epoch % eval_interval != 0: return` — this skipped epoch 0 AND only
fired on multiples of `eval_interval`.

With `max_epochs=10` and `eval_interval=5`, that meant hi_probe only ran at epoch 5 — the
final epoch (index 9) never got evaluated. The user wanted probe metrics at the end of
training, so I changed the guard:

```python
is_interval = epoch != 0 and epoch % self.eval_interval == 0
is_final = trainer.max_epochs is not None and epoch == trainer.max_epochs - 1
if not (is_interval or is_final):
    return
```

Now with `max_epochs=10, eval_interval=5`, hi_probe fires at **epoch 5** (interval) and
**epoch 9** (final). For longer runs (max_epochs=100), it fires at every 5 plus epoch 99.
The change is backwards-compatible: short runs with `max_epochs < eval_interval` now get at
least one probe evaluation at the end, which wasn't the case before.

## What's running now

- tmux session `s4`, log `logs/s4_real_20260424_1709.log`
- 10 epochs, bs=512, nw=4, H=16 P=4, bf16, wandb offline, hi_probe at epochs 5 & 9
- Expected wall time: ~4.4 h → finishing around **21:30 UTC** (2026-04-24)
- Persistent monitor armed to alert on every epoch, OOM, or error

## What I did NOT do

- Did not touch `main` branch (feature branch only).
- Did not delete or modify anything in `.stable_worldmodel/`.
- Did not force-push, amend, or rewrite history.
- Did not change LR, optimizer, scheduler, model architecture, or the SIGReg / JEPA config.
