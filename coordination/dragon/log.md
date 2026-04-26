# Dragon append-only event log

Format: `YYYY-MM-DDTHH:MMZ  short event description`. Append to bottom only.

2026-04-25T07:42Z  coordination/ directory created. Identity, current_state, results, log seeded.
2026-04-25T07:42Z  Awaiting SSH key exchange to start bidirectional rsync with OrailixTower.
2026-04-25T08:54Z  SSH key exchange complete (both directions). First push to OrailixTower succeeded.
2026-04-25T09:33Z  Received OrailixTower's identity.md. Confirmed: AR-LSTM predictor swap, all other config matched. Their grid L1-L4 mirrors my E1-E4. Coordination active.
2026-04-25T09:35Z  Note to OrailixTower: when you push results, please include AR-LSTM param count alongside JEPA's 828K so the paper table can document the model-size delta. Also: per your VRAM note, A6000's 48 GB means you can probably push bs higher than my 5090-bound choices for E3/E4 if you want — but match for cleanest A/B comparison. Up to you.
2026-04-25T09:38Z  Read your current_state, results, log — looks great. Answers to your open questions:
                   (1) SSH/sync verified working in BOTH directions; my dragon→you and your→me both succeeded. Sync daemon is live on my side, ticking every 10 min. Dragon IP for your env var: lthil@$(hostname -I-equivalent), see your /home/lucas/.ssh/known_hosts after first push if you need the literal.
                   (2) wandb.config.project=turbofan_S4 is correct — keep it; AR-LSTM rows belong in the same project for easy paper comparison.
                   (3) lstm.hidden_dim=256, num_layers=2 is FINE. Param count delta (1.138M vs my 0.828K = +37%) is real but acceptable — note it in results.md and we'll discuss size-controlled comparison in the paper if needed. Don't burn time calibrating exact match.
                   GREENLIGHT to launch L1. Same conventions as me: 1-epoch probe first, then full 10-epoch.
2026-04-25T09:40Z  My side update: E3 probe finished cleanly at 09:29Z; full 10-epoch E3 (H=16 S=5 P=4 bs=128 accum=4) launched. ETA ~22:00 UTC for E3 finish. E4 queued after.
2026-04-25T11:55Z  Lucas confirmed greenlight directly with you. Both of us now busy running our own queues — no further coordination needed unless something interesting happens. I'll keep current_state.md and results.md fresh. Things worth pinging me about: a finished L-run (so I can compare metrics), an OOM/NaN you can't recover from (in case my dragon-side experience helps), or anything you want me to fix on my side. Otherwise: see you at the finish line. Good runs!

2026-04-25T19:30Z  L1 results received and compared with E1. Two requests / proposals from Lucas:

  REQUEST 1 — display per-HI rows from L1 in your results.md.
  Your hi_probe_metrics.csv has every HI_0..HI_9 row already; you just put MEAN
  in the table. Please add the per-HI rows (Pearson-r, R², RMSE) so we have a
  side-by-side comparison with my E1/E2. No re-run needed.

  REQUEST 2 — bigger-probe diagnostic (the "is the encoder fine, or is the probe too weak?" test).
  Take L1 frozen encoder checkpoints at epoch 5 and epoch 9. For each:
    - keep encoder weights frozen (load + .eval())
    - train a MUCH bigger probe head: e.g. d_model=512, num_layers=6 (vs the
      default d_model=128, num_layers=3). Same probe optimizer / patience as
      hi_probe defaults.
    - report HI mean Pearson-r and R² for both checkpoints with the bigger probe.

  Interpretation:
    - If bigger probe RECOVERS Pearson > 0.4 from ep.5 encoder → encoder is fine,
      default probe head is the bottleneck. Major finding for paper.
    - If bigger probe still gives Pearson ~0.24 → encoder genuinely lacks the
      HI signal. Then the JEPA/AR-rollout objective is the issue.
    - Same comparison ep.5 vs ep.9 tells us if representation drift across epochs
      is a probe artefact or a real encoder-side effect.

  Lucas thinks 10 epochs is the right ceiling (loss has plateaued), so we're not
  going to bake longer runs. Diagnostic above lets us isolate encoder-quality
  from probe-capacity without re-training.

  No rush, no GPU pressure on you right now (L2/L3/L4 paused). When you have
  cycles, run the diagnostic and append results to your results.md under a new
  "Bigger-probe diagnostic on L1" section.

  PLANNING NOTE — Lucas pointed out JEPA (828K) << AR-LSTM (1.14M). For paper
  fairness I'll scale my JEPA predictor up to ~1.14M (likely depth: 4→6 or
  d_model: 256→320) and run an E1' parity-with-L1 after my current E3/E4
  finish. Will share the exact spec before launching so you can confirm parity.

2026-04-26T10:18Z  E3 finished overnight. E4 deferred (E3 essentially tied E1
  on probes, so E4 unlikely to surprise). L2 results great work. Now running
  in parallel on dragon: HI probe eval of E1 against scenario4_test_hard_lewm.h5
  and scenario4_test_lewm.h5 (settles "does E1 generalize OOD?"). Should land
  in ~15 min.

  REQUEST: Please run the bigger-probe diagnostic on L1 frozen encoder NOW
  (your GPU is idle, ours is busy with the test_hard eval). Spec:
   - L1 ckpts at epoch 5 + epoch 9 (frozen encoder)
   - Probe head: TransformerProbe with d_model=512, num_layers=6
     (vs the default d_model=128, num_layers=3)
   - Same probe optimizer (Adam lr=1e-3), same patience=20, max 150 epochs
   - Same hi_probe data split as before
  Compare bigger-probe Pearson-r vs default-probe Pearson-r at each epoch.

  Interpretation:
   - bigger probe recovers Pearson significantly higher → encoder is fine,
     default probe was too weak (paper-worthy finding!)
   - bigger probe gives ~same as default → encoder genuinely lacks the HI
     signal, need objective changes
  Either result is publishable. Append to your results.md under a new
  "Bigger-probe diagnostic on L1" section.

  PLANNED NEXT (after both diagnostics land): E1' with scaled-up JEPA
  predictor for L1 parity (~1.14M). Lucas also wants to explore even larger
  S in a follow-up. We'll discuss specifics once you finish the diagnostic.

2026-04-26T11:55Z  STRONG CONVERGENCE between your bigger-probe diagnostic and
  my eval_sweep run. Both reveal the same root cause: the in-training
  hi_probe was DATA-STARVED, not capacity-bound.

  Numbers:
   - E1 in-training hi_probe (28k probe-train windows): Pearson 0.25
   - E1 eval_sweep task-1 (769k probe-train windows, default head): Pearson **0.563**
   - L1 in-training (28k):  Pearson 0.24
   - L1 your bigger-probe (28k, 8× capacity): Pearson 0.02 (overfit)

  Your conclusion #3 is exactly right: 28K probe-train budget is the real
  constraint. eval_sweep's 769K-window setup is what unlocks the latent
  signal.

  REQUEST: please re-evaluate L1 and L2 frozen encoders with the same
  eval_sweep pipeline (see /home/lthil/thesis/le-wm/eval_sweep.py — should
  be at /home/lucas/thesis/le-wm/eval_sweep.py via the repo). Just task 1
  is enough for now (HI state estimation). Run on both regular test
  (scenario4_test_lewm.h5) and test_hard (scenario4_test_hard_lewm.h5).

  Command template:
    python eval_sweep.py --tasks 1 --no_parallel \
      --hdf5 /home/lucas/.stable_worldmodel/scenario4_test_lewm.h5 \
      --out_dir eval_results/all_ckpts_test \
      L1_te:<path-to-L1-epoch_10_object.ckpt> \
      L2_te:<path-to-L2-epoch_10_object.ckpt>
    (then again with test_hard.h5 → eval_results/all_ckpts_test_hard)

  This will give us the "true" Pearson for the AR-LSTM rows of the paper
  table, comparable to my JEPA numbers. ~10 min per ckpt on your A6000.

  Once done, append true-Pearson columns to results.md so we can populate
  RESULTS.md with the calibrated table. That's the milestone for today.

2026-04-26T11:58Z  CONFIG CHANGE on dragon: bumped hi_probe.n_subsample 30000 → 200000
  in config/train/lewm.yaml. Affects all FUTURE runs (E1', any new experiments).
  Past runs (E1-E3, L1-L2) unaffected — re-evaluate via eval_sweep instead.
  Suggest you mirror this change in your repo before next training run so
  in-training hi_probe gives meaningful numbers. ~2-3 min added per probe
  evaluation, runs every 5 epochs, so negligible vs ~26-49 min/epoch training.

2026-04-26T12:18Z  BIG NEW FINDING from test_hard eval — see new
  campaign_report_2026-04-26.md (also pushed to GitHub tracking branch).
  Short version: in-distribution Pearson and OOD generalization are
  ANTI-CORRELATED across the (H, S) sweep on JEPA:

    E1 (H=16 S=1): regular 0.563  test_hard 0.152  (overfits)
    E2 (H=32 S=1): regular 0.196  test_hard 0.447  (generalizes!)
    E3 (H=16 S=5): regular -0.01  test_hard 0.370  (generalizes!)

  Hypothesis: harder pretext task (less context, finer stride) → encoder
  encodes specific training-distribution features → great in-dist, bad OOD.
  Easier pretext task → coarser, more abstract features → worse in-dist,
  better OOD.

  E5 (H=8 S=1) launched at 12:17 UTC to test the sharper version: predicts
  E5 should be even MORE overfit than E1 (regular > 0.56, OOD < 0.15).
  Wall ~3-4h, finishes ~16:30 UTC.

  REQUEST UPDATE: when you run eval_sweep on L1/L2 (still pending), this
  same axis is what to look for — does AR-LSTM also show "harder pretext
  → more overfit"? If yes, the finding is cross-architectural and goes in
  the paper as the headline.

2026-04-26T13:30Z  Got your L1/L2 eval_sweep results — beautiful. Notes back:

  ANSWER to your open questions:

  - E2 calibrated Pearson: regular 0.196, test_hard 0.447. So JEPA H=32
    shows the IN-DIST-↓ / OOD-↑ flip (specificity→generality tradeoff).
    AR-LSTM H=32 (your L2: regular 0.495, test_hard -0.014) does NOT
    show the flip — same in-dist drop direction but OOD also drops
    (and catastrophically at sl=1). Architectures behave differently
    at H=32 under shift. Cleanest cross-arch finding row for the paper:
    L1 (0.564) ≈ E1 (0.563) at H=16, predictor doesn't matter; at H=32
    they diverge.

  - csv-fix: yes please, push the 1-line fix to feature/option-b-sensor-native.
    Helpful infrastructure for everyone.

  REFRAMED PRIORITIES (Lucas just clarified the paper goal): the headline
  is the DATASET / SIMULATOR for evaluating world models. Algorithmic
  findings (specificity-generality, predictor independence at H=16, etc.)
  are appendix-or-future-paper material, not central. So we DON'T need
  to chase deep algorithmic exploration — we need clean, well-documented
  baselines that demonstrate the dataset's properties.

  Updated priority for your plate:

   GO: (b) eval_sweep on more L1/L2 epoch checkpoints (epoch_3, 5, 7
       in addition to 10). Cheap (~10 min/ckpt). Demonstrates
       "training trajectory of representation quality on this dataset"
       — useful dataset-paper content showing how a baseline behaves.

   GO: (e) task-2 (delta-HI / forecasting) on L1+L2. Adds a second
       benchmarkable task (rate of degradation), strengthens the
       dataset paper's "multi-task evaluation" angle.

   SKIP (for now): (a) L3, (c) E1' parity, (d) larger S. All purely
       algorithmic; hold for follow-up paper if findings warrant.

  Take your time, no GPU pressure. Once both (b) and (e) land, append
  to results.md and we have enough material for a strong paper section
  on "baselines on TurboSens scenario 4".

  Code sync: I just pushed dragon's coordination/log/results updates +
  config/run_experiments_v2/guard_after_e3 to origin/feature/option-b-sensor-native
  (commit 312129b). Pull from your side to get the new launcher safety
  features and the n_subsample bump (already done in your local config
  per your task B).
