# Dragon current state — auto-updated

**As of:** 2026-04-25 ~09:40 UTC

## Running now

- **Tmux `campaign`** — full E3 training (H=16 S=5 P=4, bs=128 accum=4, 10 epochs).
  E3 probe finished cleanly at 09:29Z. Full run launched at 09:29Z, process 730874,
  GPU 97%, 17.9 GB VRAM. ETA ~22:00 UTC.
- **Tmux `wandb_sync`** — daemon syncing offline wandb runs every 5 min.
- **Tmux `peer_sync`** — daemon rsyncing coordination/ with OrailixTower every 10 min.

## Queued (auto-launched after E3 finishes)

- **E4 probe** (1 epoch, H=32 S=5 P=4, bs=64 accum=8) — ETA ~1.5 h
- **E4 full** (10 epochs, same as probe) — ETA ~15 h

## Done

- E1: H=16 S=1 P=4, 10 epochs (4.4 h)
- E2: H=32 S=1 P=4, 10 epochs (8.2 h)
- See `results.md` for metrics.

## GPU / disk / memory

- VRAM: 17.9 / 32 GB while training; idle 2 MiB
- Disk: ~5 GB dataset cached in RAM during training
- Local wandb dir: 45 MB
- All offline wandb runs synced to `wandb.ai/thil-ecole-polytechnique/turbofan_S4`

## Next planned action by me

When E3+E4 finish (around 2026-04-26 evening UTC), write up a final results
table including all 4 JEPA configs. Then wait for further direction.

## How to reach me asynchronously

Drop notes in `coordination/orailixtower/log.md` (your write side, I'll read
it on my next sync). For urgent stuff, tell Lucas — he's running this session.
