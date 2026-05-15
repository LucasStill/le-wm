# Dragon training handoff — TurboSens Scenario 4 (NeurIPS 2026)

You are picking up a training-diagnostics session mid-flight. The user is on a train
with spotty internet; work autonomously and keep a short log of what you tried and
what you learned in `logs/handoff_notes.md`.

## Context (what shipped before you arrived)

- Repo: `/home/lthil/thesis/le-wm` — branch `feature/option-b-sensor-native`.
- Venv: `.venv/` (uv-managed). Activate with `source .venv/bin/activate`.
- Hardware: RTX 5090 (32 GB VRAM), 60 GB RAM, no Slurm.
- Data: `/home/lthil/.stable_worldmodel/turbosens2_{train,test,test_hard}_lewm.h5`.
  Pixels shape (N, 11, 16) = 7 sensors + 4 context_params; 7 actions.
- Launcher: `./train_lewm_scenario4_dragon.sh` (env-var overrides for WIN_SIZE,
  HISTORY_LEN, H_STEP, NUM_PREDS, BATCH_SIZE, NUM_WORKERS, PRECISION).
- Config: `config/train/data/turbosens2.yaml` — pixels cached in RAM (~4.9 GB) to
  avoid LZF chunk thrashing; that was the main bug we fixed today.
- Deadline: ~1 week. Jean Zay H100 priority is low, so dragon is the primary machine.

## Current state at handoff

A raw-throughput test is running in tmux session `s4`, launched ~15:52 local time:

    num_workers=4, batch_size=256, bf16-mixed, wandb disabled, hi_probe off,
    obs_window_size=1, history_size=1, h_step=1, num_preds=1, max_epochs=1.

GPU was at ~79% utilisation, main process at >100% CPU, log `logs/s4_cached_*.log`.
This confirms the dataloader path is no longer the bottleneck. Before you do
anything else, verify it is still alive and actually stepping:

    tmux capture-pane -t s4 -p | tail -40
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    ps -eo pid,pcpu,etime,comm | grep python

If it finished, read the last `logs/s4_cached_*.log` for final metrics.

## Your mission

1. **Confirm baseline throughput.** From the log + tmux pane, estimate it/s and
   time-per-epoch. Write it to `logs/handoff_notes.md`.
2. **Tune what actually helps**, one variable at a time. Candidates in priority order:
   - batch_size: try 512, then 1024. Watch VRAM (nvidia-smi), stop if >28 GB.
   - num_workers: 4 is probably fine with cached pixels — but try 8 once batch is sized.
   - precision: bf16-mixed is already on. Don't downgrade.
   - Re-enable hi_probe.enabled=true once throughput is healthy — it shouldn't hurt much.
   - Re-enable wandb (`WANDB_MODE=offline`, not `disabled`) for real runs.
3. **Launch a real training run** once baseline is good. Sensible first real config:
   WIN_SIZE=1 HISTORY_LEN=16 H_STEP=1 NUM_PREDS=4 BATCH_SIZE=<max fit>.
   Use the launcher script; don't hand-roll srun/python lines.
4. **Do NOT**: touch main branch, force-push, delete datasets, rm -rf anything in
   `.stable_worldmodel/`. When in doubt, leave it and note it for the user.

## Useful debug commands

    # stack of workers (we installed py-spy already)
    source .venv/bin/activate && py-spy dump --pid <PID>
    # HDF5 chunk/compression inspection
    python -c "import h5py;f=h5py.File('/home/lthil/.stable_worldmodel/turbosens2_train.h5','r');print(f['pixels'].chunks,f['pixels'].compression)"
    # kill a stuck run
    tmux kill-session -t s4 ; pgrep -f train.py | xargs -r kill -9

## Launch template (tmux, logs tee'd)

    tmux kill-session -t s4 2>/dev/null
    tmux new -d -s s4 "cd /home/lthil/thesis/le-wm && source .venv/bin/activate && \
      WANDB_MODE=offline BATCH_SIZE=512 NUM_WORKERS=4 \
      ./train_lewm_scenario4_dragon.sh 2>&1 | tee logs/s4_\$(date +%Y%m%d_%H%M).log"

## Reporting (mandatory)

Maintain `logs/handoff_notes.md` — it is the only channel back to the user and to
the next Claude instance. Rules:

- Update **Current state** in-place after every run you launch or config change.
- Append a **Timeline** entry (UTC `date -u +%H:%M`) for every meaningful event:
  run launched, throughput measured, OOM, bug, decision. 1-3 bullets each.
- Add a **Decisions log** one-liner whenever you rule something out or pick between
  options ("Chose bs=512; bs=1024 OOMed at 30 GB").
- Put anything you need the user to answer in **Open questions** so they see it
  immediately on reconnect.
- Never rewrite history in the file; only append to Timeline.
