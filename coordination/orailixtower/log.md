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
