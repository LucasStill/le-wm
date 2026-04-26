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
