# Scenario-4 TurboSens — cross-architecture results

_Last regenerated: 2026-04-26 10:43 UTC_

Auto-aggregated from `coordination/{dragon,orailixtower}/` and
`hi_probe_metrics.csv`. **Do not edit by hand** — run `./aggregate_results.sh`.

---

## What's running right now

### Dragon (JEPA, RTX 5090)


- **Tmux `campaign`** — full E3 training (H=16 S=5 P=4, bs=128 accum=4, 10 epochs).
  E3 probe finished cleanly at 09:29Z. Full run launched at 09:29Z, process 730874,
  GPU 97%, 17.9 GB VRAM. ETA ~22:00 UTC.
- **Tmux `wandb_sync`** — daemon syncing offline wandb runs every 5 min.
- **Tmux `peer_sync`** — daemon rsyncing coordination/ with OrailixTower every 10 min.


### OrailixTower (AR-LSTM, RTX A6000)


- **Nothing training.** L2 full completed 08:50 UTC. GPU idle. Per
  Lucas's standing "don't auto-launch" instruction, L3/L4 are paused.
- **Tmux `s4_arlstm_L2`** — still alive (post-training shell). Attach
  to see the final printout.
- **Tmux `wandb_sync`** — offline→cloud sync daemon, still ticking.


---

## Per-architecture results

### JEPA (dragon)


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

### AR-LSTM (orailixtower)


Scenario-4 TurboSens, AR-LSTM baseline, 10 epochs each, bf16-mixed,
AdamW lr=7e-5 cosine, hi_probe at epoch 5 + final epoch.

Predictor: `LSTMPredictor(hidden_dim=256, num_layers=2, dropout=0.1)`.
Everything else identical to dragon's JEPA config — see `identity.md`.

## Run table

| ID | H | S | P | bs (eff. 512) | Wall | fit/loss | fit/pred | fit/ar | fit/sigreg | params |
|----|---|---|---|---------------|------|----------|----------|--------|-----------|--------|
| L1 | 16 | 1 | 4 | 512 / acc=1   | ~7 h | **0.379** | **0.129** | **0.078** | 1.664 | 1,138,068 |
| L2 | 32 | 1 | 4 | 256 / acc=2   | ~13 h | **0.234** | **0.078** | **0.050** | 1.039 | 1,138,068 |
| L3 | 16 | 5 | 4 | 128 / acc=4   | —    | —        | —        | —      | —          | 1,138,068 |
| L4 | 32 | 5 | 4 | 64  / acc=8   | —    | —        | —        | —      | —          | 1,138,068 |

(Empty rows fill in as runs complete. Param count is the same across
all four because H, S, P do not affect any module's `nn.Parameter`s —
they only change the AR-rollout depth and dataloader windowing.)

**Param-count parity note** (per dragon's reply 2026-04-25 09:34Z):
my AR-LSTM at `hidden=256, layers=2` is **1.138 M trainable params**
vs dragon's JEPA encoder-only **0.828 M** quoted in their campaign
report — a **+37 %** delta. Dragon and I agreed to *not* calibrate to
exact parity (cost: ~1 h GPU for marginal value). The paper text
should call this out explicitly so reviewers don't conflate predictor
choice with model size. Concrete mitigation: report final per-config
total-param counts alongside metrics in the paper table.

## Hi_probe HI-regression (mean over 10 components)

| ID | epoch | R² | RMSE | Pearson-r |
|----|-------|----|------|-----------|
| L1 | 5  | -0.87 | 0.00291 | 0.224 |
| L1 | 9  | -1.24 | 0.00304 | 0.242 |
| L2 | 5  | -1.38 | 0.00321 | 0.188 |
| L2 | 9  | -1.41 | 0.00320 | 0.169 |
| L3 | 5  | — | — | — |
| L3 | 9  | — | — | — |
| L4 | 5  | — | — | — |
| L4 | 9  | — | — | — |

## Action-RUL probe

| ID | epoch | RMSE (steps) | MAE | R² | Pearson |
|----|-------|--------------|-----|----|---------|
| L1 | 5 | 27.97 | 22.10 | -0.18 | 0.034 |
| L1 | 9 | 27.23 | 22.10 | -0.12 | 0.024 |
| L2 | 5 | 27.13 | 22.02 | -0.114 | 0.004 |
| L2 | 9 | 27.40 | 22.16 | -0.136 | 0.013 |
| L3 | 5 | — | — | — | — |
| L3 | 9 | — | — | — | — |
| L4 | 5 | — | — | — | — |
| L4 | 9 | — | — | — | — |

## Wandb runs (on https://wandb.ai/thil-ecole-polytechnique/turbofan_S4/runs/)

- L1: `12uiyu7c` (offline; sync_wandb daemon will push)
- L2: `5iu3jbtt` (offline; sync_wandb daemon will push)
- L3: pending
- L4: pending

## Key findings so far (after L1 + L2)

1. **AR-LSTM training loss is consistently lower than JEPA's** at the
   matched H:
   - L1 (H=16) fit/pred=0.129 vs E1 fit/pred=0.152 (LSTM 15 % lower)
   - L2 (H=32) fit/pred=0.078 vs E2 fit/pred=0.076 (essentially equal)
   So at H=16 the LSTM has a real advantage on the world-modeling
   objective; at H=32 the two architectures converge. With +37 % LSTM
   params, this isn't free — the H=16 gap may shrink under size-matched
   comparison (dragon's E1' run is in the planning note).

2. **Probe Pearson degrades with H for BOTH architectures**, confirming
   the trend dragon flagged is not predictor-specific. Pearson @9:
   - LSTM: L1 0.242 → L2 0.169 (-30 %)
   - JEPA: E1 0.25  → E2 0.18  (-28 %)
   Almost identical drop — this is a property of the H=16 vs H=32
   *training task*, not the architecture. **Major paper-worthy finding.**

3. **Non-monotonic-in-epoch pattern is mixed**: at H=16 the LSTM's
   epoch-9 beats epoch-5 (0.242 > 0.224); at H=32 it flips to match
   the JEPA pattern (epoch-5 0.188 > epoch-9 0.169). So both
   architectures show degradation from epoch-5 to epoch-9 at H=32.
   Suggests the H=32 task encourages representations that drift away
   from HI-relevant features over training — independent of predictor.

4. **Action-RUL is noise for all four runs** (LSTM L1, L2 and JEPA E1,
   E2; R² ≈ -0.12 to -0.15, Pearson ≈ 0 to 0.03). Confirms 10 epochs
   is not enough for RUL regardless of architecture or H.


---

## Dragon's per-epoch hi_probe trajectory (raw, from `hi_probe_metrics.csv`)

**HI mean across 10 components, per epoch:**

| epoch | global_step | seq_len | R²    | RMSE     | Pearson-r |
|-------|-------------|---------|-------|----------|-----------|
| 0 | 0 | 16 | -0.3201 | 0.002349 | 0.2431 |
| 0 | 0 | 16 | -0.3280 | 0.002368 | 0.2461 |
| 0 | 0 | 16 | -0.3971 | 0.002551 | 0.1171 |
| 0 | 0 | 32 | -0.2332 | 0.002280 | 0.2527 |
| 0 | 12240 | 16 | -0.6998 | 0.002728 | 0.2757 |
| 0 | 12293 | 32 | -0.6615 | 0.002760 | 0.2056 |
| 5 | 73440 | 16 | -1.3470 | 0.003159 | 0.1223 |
| 5 | 73758 | 32 | -1.3238 | 0.003168 | 0.1360 |
| 5 | 73842 | 16 | -0.7902 | 0.002746 | 0.2974 |
| 9 | 122400 | 16 | -1.3839 | 0.003136 | 0.2513 |
| 9 | 122930 | 32 | -0.9075 | 0.002908 | 0.1769 |
| 9 | 123070 | 16 | -0.8999 | 0.002946 | 0.2526 |

**Action-RUL probe per epoch:**

| epoch | global_step | seq_len | R²    | RMSE   | Pearson-r |
|-------|-------------|---------|-------|--------|-----------|
| 0 | 0 | 16 | -0.1547 | 27.6228 | 0.0166 |
| 0 | 0 | 16 | -0.1561 | 27.6384 | 0.0000 |
| 0 | 0 | 16 | -0.1635 | 27.7273 | 0.0130 |
| 0 | 0 | 32 | -0.1524 | 27.5914 | -0.0044 |
| 0 | 12240 | 16 | -0.1613 | 27.7005 | 0.0327 |
| 0 | 12293 | 32 | -0.1330 | 27.3584 | 0.0233 |
| 5 | 73440 | 16 | -0.1936 | 28.0839 | -0.0016 |
| 5 | 73758 | 32 | -0.2197 | 28.3859 | 0.0038 |
| 5 | 73842 | 16 | -0.0831 | 26.7522 | 0.0257 |
| 9 | 122400 | 16 | -0.1529 | 27.6004 | 0.0028 |
| 9 | 122930 | 32 | -0.1524 | 27.5912 | -0.0148 |
| 9 | 123070 | 16 | -0.1418 | 27.4670 | 0.0244 |


---

## Coordination logs

### Dragon log
```
   - bigger probe recovers Pearson significantly higher → encoder is fine,
     default probe was too weak (paper-worthy finding!)
   - bigger probe gives ~same as default → encoder genuinely lacks the HI
     signal, need objective changes
  Either result is publishable. Append to your results.md under a new
  "Bigger-probe diagnostic on L1" section.

  PLANNED NEXT (after both diagnostics land): E1' with scaled-up JEPA
  predictor for L1 parity (~1.14M). Lucas also wants to explore even larger
  S in a follow-up. We'll discuss specifics once you finish the diagnostic.
```

### OrailixTower log
```
2026-04-25T11:10Z  Decision: predictor LSTMCell python loop is too slow vs cuDNN. Switched LSTMPredictor default to nn.LSTM (use_cudnn flag, fallback kept). Added optional torch.compile on the encoder.
2026-04-25T11:13Z  Smoke-tested both. nn.LSTM alone: 3.25 → 3.61 it/s (+11%). +torch.compile(encoder): 3.61 → 5.19 it/s (+44% over baseline). No drift in losses or grads.
2026-04-25T11:17Z  L1 full 10-epoch run launched in tmux s4_arlstm_L1 (now wrapped in `exec bash` so the session stays alive after training). compile_encoder=true. Wandb offline run id: 12uiyu7c. ETA ~18:00 UTC.
2026-04-25T18:19Z  L1 full completed (10 epochs in ~7 h, 40:16 per epoch at 5.06 it/s). Final fit/pred=0.129, fit/ar=0.078; HI mean Pearson@9=0.242, RUL R²=-0.12. See results.md for the full row.
2026-04-25T18:19Z  Lucas asked me NOT to auto-launch L2 — paused queue, GPU idle, awaiting next instruction.
2026-04-25T19:30Z  Read dragon's log update. Two requests noted: per-HI rows in results.md (cheap), bigger-probe diagnostic on L1 frozen encoder at ep.5 and ep.9 (~30-90 min GPU). Will queue after L2 lands.
2026-04-25T19:47Z  Lucas said resume baselines. L2 full launched in tmux s4_arlstm_L2 (H=32 S=1 P=4 bs=256 accum=2, compile_encoder=true). ETA ~09:15 UTC tomorrow.
2026-04-25T19:48Z  L2 first launch crashed: torch.compile mode=reduce-overhead uses CUDA graphs which break under accumulate_grad_batches>1. Switched to mode=default. Re-launched at 19:49Z, training healthy.
2026-04-26T08:50Z  L2 full completed: 24586 steps × 10 epochs in ~13 h, 75 min/epoch at 5.44 it/s. Final fit/loss=0.234, fit/pred=0.078, fit/ar=0.050. HI Pearson@9=0.169, RUL R²=-0.14. See results.md for full row + comparison vs E2.
2026-04-26T11:55Z  GPU idle. Per Lucas's standing instruction, NOT auto-launching L3. Awaiting next direction.
```

---

_To refresh: `./aggregate_results.sh`. Hooked into the `peer_sync` daemon
so it regenerates after every pull tick (~every 10 min)._
