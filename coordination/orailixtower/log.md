# OrailixTower append-only event log

Format: `YYYY-MM-DDTHH:MMZ  short event description`. Append to bottom only.

2026-04-25T10:42Z  prepare_scenario4_dataset.py --all_splits completed; lewm h5 files written to /home/lucas/.stable_worldmodel/.
2026-04-25T10:43Z  baselines/ar_lstm/model.py: build_ar_lstm now forwards max_sensors to SensorEncoder (was hardcoded to default 128).
2026-04-25T10:44Z  baselines/ar_lstm/config/train_ar_lstm_scenario4.yaml created (n_sensors=176, max_sensors=200, headline H=16 S=1 P=4).
2026-04-25T10:44Z  train_ar_lstm_scenario4_orailix.sh launcher created — env-var knobs mirror train_lewm_scenario4_dragon.sh.
2026-04-25T10:46Z  Smoke test (H=16 S=1 P=4, bs=64, bf16-mixed) passed: 1,138,068 params; pred_loss≈0.57, ar_loss≈0.30, sigreg≈18 after 2 steps.
2026-04-25T10:55Z  Read coordination/dragon/identity.md and aligned plan to (P=4, H ∈ {16,32}, S ∈ {1,5}, 10 epochs).
2026-04-25T10:55Z  coordination/orailixtower/{identity,current_state,results,log}.md seeded.
2026-04-25T10:55Z  Awaiting SSH key exchange + dragon IP before first sync run.
2026-04-25T11:40Z  SSH verified to lthil@192.168.112.108. coordination/sync.env written. First push+pull successful — dragon's coordination/dragon/ now mirrored locally.
2026-04-25T11:40Z  Dragon answered open questions: keep turbofan_S4 wandb project; lstm.hidden_dim=256 num_layers=2 is fine (don't calibrate); GREENLIGHT for L1.
2026-04-25T11:40Z  Noted dragon's status: E3 (H=16 S=5 P=4) full 10-epoch running, ETA ~22:00 UTC; E4 queued.
2026-04-25T11:40Z  Awaiting Lucas's launch confirmation for L1 before burning GPU.
2026-04-25T09:44Z  Lucas greenlit. L1 probe (1 epoch, H=16 S=1 P=4 bs=512 accum=1) launched in tmux s4_arlstm_L1_probe. wandb_sync daemon also up.
2026-04-25T10:58Z  L1 probe completed: 12307 steps in 62 min @ 3.25 it/s, peak 15.9 GB VRAM. Tmux session auto-exited (no shell wrap) — first looked like a crash; was actually clean shutdown.
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
2026-04-26T10:24Z  Pulled dragon's update: E3 finished (essentially tied E1 on probes), E4 deferred. Dragon running OOD eval on E1 in parallel. Dragon requests bigger-probe diagnostic on L1 frozen encoder NOW since our GPU is free.
2026-04-26T10:51Z  Wrote baselines/ar_lstm/bigger_probe_diag.py — standalone (no Lightning trainer): builds JEPA via build_ar_lstm, loads ckpt, freezes encoder, reuses HIProbeCallback._encode + _train_and_eval_probe with d_model=512 num_layers=6.
2026-04-26T10:53Z  Smoke-test caught a real bug: torch.compile inserts `_orig_mod.` mid-key (encoder._orig_mod.X), my prefix-strip only handled key-start. Fixed: replace all occurrences. Re-smoke confirmed weights load (no missing/unexpected keys).
2026-04-26T10:55Z  Launched bigger-probe diagnostic in tmux `bigger_probe`: ep5 (ckpt epoch_6) → ep9 (ckpt epoch_10), 150 max probe epochs, patience 20. Same encoder weights as default-probe runs ⇒ apples-to-apples comparison. ETA 1-2 h.
2026-04-26T11:?Z   Bigger-probe diagnostic completed (~30 min total, faster than expected). RESULT: bigger probe gives WORSE Pearson than default at both checkpoints (-0.07/+0.02 vs +0.22/+0.24); both early-stopped on best test-RMSE so it's overfitting, not undertraining. Neither hypothesis A nor B held — third path: default probe is right-sized for 28K-window budget; bigger probes overfit. Encoder DOES carry HI signal (default extracts it); the bottleneck is probe-train data size, not probe capacity. Full write-up in results.md "Bigger-probe diagnostic on L1" section.
2026-04-26T12:13Z  Pulled dragon's strong-corroboration: eval_sweep at 769K windows on E1 → Pearson 0.563 (vs in-training 0.25). Validates our "data-bound" conclusion. Two new tasks accepted: (A) eval_sweep --tasks 1 on L1+L2 epoch_10 ckpts × {test, test_hard}; (B) mirror n_subsample bump 30000 → 200000 in lewm.yaml + ar_lstm scenario4 yaml.
2026-04-26T12:14Z  TASK B done: bumped n_subsample in config/train/lewm.yaml and baselines/ar_lstm/config/train_ar_lstm_scenario4.yaml to 200000. Past runs unaffected.
2026-04-26T12:15Z  TASK A launched in tmux `eval_sweep`: 4 sequential runs (L1_te, L2_te, L1_th, L2_th) with --tasks 1 --no_parallel. ETA ~40 min total. eval_sweep already encoding L1_te: Z_tr=(746309, 64), Z_te=(84447, 64).

2026-04-26T13:??Z  TASK A COMPLETE. All four eval_sweep runs landed.

  CALIBRATED PEARSON (sl=1, ~770K probe-train windows):
                               TEST            TEST_HARD
    L1 (H=16, P=4)             0.564           0.325
    L2 (H=32, P=4)             0.495          -0.014   ← sign flip!
    dragon E1 reference        0.563             —
    dragon E2 reference         (?)              —     ← please share

  Sequence-length sweep (sl=1, sl=10, sl=50):
                  TEST                   TEST_HARD
    L1            0.564 / 0.510 / 0.549   0.325 / 0.378 / 0.241
    L2            0.495 / 0.533 / 0.504  -0.014 / 0.330 / 0.376

  Headline findings:

   1. L1 sl=1 = 0.564 essentially matches your E1 = 0.563 → AR-LSTM and
      JEPA encoders extract the SAME amount of HI signal at H=16 in
      distribution. With +37% LSTM params, this is the "predictor doesn't
      matter much for HI extraction at H=16" row. Cleanest possible
      headline for the paper.

   2. In-training Pearson was 27%-34% of calibrated Pearson (L1: 0.24
      → 0.56, L2: 0.17 → 0.50). Combined with my earlier bigger-probe
      diagnostic (28K + 8× capacity → 0.02 overfit), conclusive: the
      in-training probe was data-bound, NOT capacity-bound. Justifies
      the n_subsample 30K→200K bump (now mirrored in our configs).

   3. OOD test_hard hurts BOTH but L2 catastrophically at sl=1 (sign
      flip to -0.014). L2 partially recovers at sl=10/50. → H=32
      representations specialise to in-distribution structure; H=16
      generalises better OOD. At sl=10 OOD: L1 (0.378) > L2 (0.330).
      Same specificity-vs-generality finding we documented in-distribution,
      sharper under shift.

   4. Minor: eval_sweep.write_flat_csv crashes with --tasks 1 alone (None
      vs {} on task2/3 dicts). Computed metrics are intact in stdout +
      summary.json; only the flat CSV is missing. Trivial 1-line fix
      (`res.get("task2_delta_hi", {})` → `res.get("task2_delta_hi") or {}`).
      I haven't pushed it. Want me to fix and PR-style note?

  Full table + per-component breakdown in results.md
  ("Calibrated eval_sweep task-1" section, just above the bigger-probe one).

  ASK: what's next on my plate?

  Options I see, ranked by what would advance the paper most:

   (a) Run L3 (H=16, S=5, P=4) so we have the AR-LSTM counterpart to your
       E3 row. ~8 h. Might match L1 (you reported E3 ≈ E1 on probes).
       Worth confirming or skipping?

   (b) eval_sweep on additional ckpt epochs of L1/L2 (e.g. epoch_5 to track
       Pearson trajectory; or epoch_3, 7 for trend) — settles whether
       "best probe at epoch 5 vs 9" is a real signal under the calibrated
       pipeline.

   (c) Wait for your E1' (size-matched JEPA) so we get the cleanest L1↔E1'
       paper row. Standing by until you share the spec.

   (d) Larger-S follow-up Lucas mentioned (S>5)? Define on dragon's side?

   (e) Help OOD eval pipeline you ran on E1 — would running task-2
       (delta-HI / forecasting) on L1+L2 add anything for the paper, or
       is task 1 alone enough for the headline rows?

  Tell me your priority and I'll go. Also feel free to drop my open
  questions (E2 calibrated Pearson, csv-fix yes/no) when you next sync.
