# OrailixTower current state — auto-updated

**As of:** 2026-04-25 ~18:20 UTC

## Running now

- **Nothing training.** L1 full completed cleanly at 18:19 UTC. GPU is
  idle (~0 %, 384 MiB). Lucas asked me not to auto-launch L2 — pending
  next instruction.
- **Tmux `s4_arlstm_L1`** — still alive (post-training shell). Attach
  with `tmux attach -t s4_arlstm_L1` to see the final printout.
- **Tmux `wandb_sync`** — offline→cloud sync daemon, still ticking.

## L1 full — 10 epochs, COMPLETED

- Wall ~7 h (started 11:17 UTC, ended 18:19 UTC). 40:16 per epoch
  at 5.06 it/s steady-state. Matched the compile-on estimate of 6.7 h.
- Final fit/loss=0.379, fit/pred=0.129, fit/ar=0.078, fit/sigreg=1.66.
- HI probe (mean over 10 components, seq_len=16): Pearson 0.242 @9
  (vs 0.224 @5), RMSE 0.00304 @9, R² -1.24 @9.
- Action-RUL: RMSE 27.23 @9, R² -0.12, Pearson 0.024 — essentially
  noise (same shape as dragon's E1).
- See `results.md` for the row + comparison vs dragon's E1.

## L1 probe (1 epoch) — completed 10:58 UTC

- 12,307 train steps in 62 min  →  3.25 it/s, peak 15.9 GB VRAM.
- Final val/loss=34.1, val/pred=~0.07, val/ar=~0.04, val/sigreg≈207.
- HI probe metrics are noise (1-epoch encoder) — Pearson nan,
  R² negative. Expected; this is just a sanity baseline.
- Probe confirmed (a) no OOM, (b) all 1.14 M params receive gradients,
  (c) data path is correct, (d) hi_probe + RUL probe wiring is intact.

## Speed-fix landed before L1 full

Two changes to `baselines/ar_lstm/`:

1. `model.py`: `LSTMPredictor` now defaults to `nn.LSTM` (cuDNN-fused)
   instead of the LSTMCell python loop. `lstm.use_cudnn=false` falls
   back to the old path (V100 escape hatch).
2. `train_ar_lstm.py`: optional `compile_encoder` flag — wraps
   `world_model.encoder` with `torch.compile(mode='reduce-overhead')`.
   On A6000 + cu13.0 + torch 2.11, the SensorEncoder forward+backward
   speeds up enough to push 100-step throughput from 3.61 → **5.19 it/s
   steady-state (+44 %)**. Falls back gracefully on compile errors.

Mathematically equivalent to the slow path — same gates, same weights,
same gradients (modulo ~1e-6 fp rounding). Comparison vs JEPA is
*not* biased by these changes.

## Setup recap (done)

- `prepare_scenario4_dataset.py --all_splits` finished — produced
  `scenario4_{train,test,test_hard}_lewm.h5` in
  `/home/lucas/.stable_worldmodel/` (4.9 GB / 595 MB / 612 MB,
  pixels (N, 11, 16), 7 actions 0–6).
- `baselines/ar_lstm/config/train_ar_lstm_scenario4.yaml` created.
- `baselines/ar_lstm/model.py` patched: `build_ar_lstm` now passes
  `max_sensors` through to `SensorEncoder` (was hardcoded to default
  128, would have crashed on scenario4's n_sensors=176).
- `train_ar_lstm_scenario4_orailix.sh` launcher written (mirror of
  `train_lewm_scenario4_dragon.sh`, env-var overrides for the same
  knobs plus three LSTM-specific knobs).
- 1-epoch smoke test on the headline config (H=16 S=1 P=4, bs=64,
  bf16-mixed) ran to completion: 1,138,068 trainable params,
  `pred_loss≈0.57`, `ar_loss≈0.30`, `sigreg≈18` after a couple of
  steps. No CUDA / cuDNN issues on the A6000 + cu13.0 + torch 2.11.

## Queued

Following your H/S grid (`P=4` constant, 10 epochs each). I will run a
1-epoch probe before every 10-epoch run, same convention as you.

| Order | ID | H  | S | bs (eff. 512)      | num_steps | est. 10-ep wall |
|-------|----|----|---|--------------------|-----------|-----------------|
| 1     | L1 | 16 | 1 | 512 / accum=1      | 20        | ~6.7 h (running) |
| 2     | L2 | 32 | 1 | 256 / accum=2      | 36        | ~13.5 h          |
| 3     | L3 | 16 | 5 | 128 / accum=4      | 96        | ~8.5 h           |
| 4     | L4 | 32 | 5 | 64  / accum=8      | 176       | ~16 h            |

**Total sweep ≈ 45 h** (with cuDNN LSTM + torch.compile). The 1.8× gap
vs your ~25 h on the 5090 is almost entirely raw GPU throughput
(5090 ≈ 1.5–2× A6000 per FLOP), not the LSTM kernel.

I'm holding off on launching L1 until Lucas confirms — I want him to
verify the SSH/sync and the grid before I burn ~25 h of GPU time.

## GPU / disk / memory

- VRAM idle: ~0 MiB (no run yet)
- Available: 48 GB total — 50% larger headroom than dragon's 5090,
  so the same `(bs, accum)` tuples should work
- Disk: `/home/lucas` has 857 GB free; dataset + lewm h5s ≈ 6 GB
- Wandb mode: `offline` — will sync via `sync_wandb.sh` in a separate
  tmux daemon once runs start

## Next planned action by me

1. Wait for Lucas to confirm SSH key exchange + IP for dragon
   (so the sync script actually works).
2. Launch L1 probe (1 epoch) → if healthy, full 10-epoch L1.
3. Then L2 → L3 → L4 sequentially. Each starts with a probe.
4. After every run, append to `results.md` and `log.md` and trigger
   `coordination/sync_coordination.sh`.

## Open questions for Lucas

(All resolved 2026-04-25 09:40 UTC — sync working, dragon agent answered
my questions in `coordination/dragon/log.md`.)

- ~~Dragon IP / hostname for SSH~~ → `lthil@192.168.112.108`, working.
- ~~wandb project name~~ → keep `turbofan_S4`.
- ~~LSTM size~~ → keep `hidden_dim=256, num_layers=2`; +37 % param
  delta vs JEPA encoder is acceptable, will be called out in the paper.

**Only remaining ask:** explicit go/no-go from Lucas to launch L1
(probe first, then 10-epoch full). Dragon already greenlit on its
side, but burning ~25 h cumulative GPU on this host is your call.

## How to reach me asynchronously

Drop notes in `coordination/dragon/log.md` on your side (I rsync from
there). For urgent stuff, tell Lucas — he's the live operator.
