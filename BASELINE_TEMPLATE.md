# How to add a new baseline world-model architecture (TurboSens scenario 4)

Built from the RSSM/Dreamer baseline added on 2026-05-02 (this directory)
and the AR-LSTM baseline that landed on 2026-04-24. Both share the same
training/eval scaffolding; the per-architecture work is small and lives
under `baselines/<name>/`.

This doc is meant as the working contract: if you follow each step, the
new baseline drops into the existing eval comparison tables (Table~1,
counterfactual fidelity table, per-archetype appendix, multiseed Pearson
summary) without bespoke glue.

---

## TL;DR — what you have to provide

A new baseline must expose these contracts:

1. A `train_<name>.py` Hydra entrypoint that reads the same scenario-4
   HDF5 dataset and saves checkpoints under
   `~/.stable_worldmodel/<run_subdir>/<name>_*_epoch_<N>_object.ckpt`
   via `utils.ModelObjectCallBack`.
2. A model class with **`encode(info)`** returning `info["emb"]` of shape
   `(B, T, D)` — the same contract `HIProbeCallback` uses.
3. **For action-conditioned rollouts** (Task 3 + counterfactual fidelity):
   either expose a `model.predict(emb_ctx, act_emb)` that mirrors JEPA's
   API, or write a one-page `<name>_rollout()` adapter — see
   `counterfactual_fidelity_rssm.py::rssm_rollout_HI` for the RSSM example.
4. A launcher script `train_<name>_scenario4_<host>.sh` that mirrors
   `train_rssm_scenario4_orailix.sh` (env-var overrides for BATCH_SIZE,
   NUM_WORKERS, MAX_EPOCHS, ACCUM_GRAD, LR; effective batch must default
   to 512 to match AR-LSTM/JEPA).

That's it. Everything below is the eval & reporting machinery.

---

## File-by-file walkthrough

The RSSM baseline ships these files under `baselines/rssm/`:

| file                                     | role                                                |
|------------------------------------------|-----------------------------------------------------|
| `model.py`                               | `RSSMWorldModel` with `encode()` + `world_model_step()` + `dynamics.{observe,img_step}` |
| `config/train_rssm_scenario4.yaml`       | Hydra config: data, encoder/dyn/decoder dims, KL weights, optimizer |
| `train_rssm.py`                          | Hydra entrypoint; uses `swm.data.HDF5Dataset`, Lightning, hi_probe callback |
| `smoke_test.py`                          | Synthetic forward/backward sanity test, < 5 s     |
| `eval_rssm.py`                           | Bespoke Task 1 (Ridge) + Task 3 (latent forecast) for RSSM |
| `multiseed_eval_rssm.py`                 | TransformerProbe Task 1 across 6 probe seeds × {sl=1, sl=10} |
| `per_archetype_rssm.py`                  | Tiny wrapper that imports `RSSMWorldModel` and delegates to AR-LSTM's `per_archetype_diag.main()` |
| `render_results.py`                      | Joins all eval artifacts → `RESULTS_RSSM.md`        |

And these one-offs at the repo root:

| file                                | role                                              |
|-------------------------------------|---------------------------------------------------|
| `train_rssm_scenario4_orailix.sh`   | tmux-friendly launcher with env-var overrides     |
| `counterfactual_fidelity_rssm.py`   | RSSM-specific rollout in the counterfactual harness |
| `run_post_training_eval.sh`         | Orchestrator: kicks off all evals + render        |
| `EVAL_PLAN_RSSM.md`                 | What ran tonight + proposed follow-ups            |

For a new baseline, copy `baselines/rssm/` → `baselines/<name>/` and adapt.

---

## Step-by-step: bringing up a new baseline

### 1. Model class

Put it under `baselines/<name>/model.py`. The two contracts that matter:

```python
def encode(self, info: dict) -> dict:
    # info["pixels"]: (B, T, n_sensors) where n_sensors=176 (11×16)
    # info["action"]: (B, T, 1) int64 (0..6); may be all-zeros at probe time
    info["emb"] = ...   # (B, T, D)
    return info

def world_model_step(self, info: dict) -> dict:
    # Returns whatever your loss function consumes. Must include "recon",
    # "target" if you want HIProbeCallback to log a recon_loss curve.
```

Anything else (latent transitions, reconstruction head) is yours.

### 2. Hydra config

Mirror `baselines/rssm/config/train_rssm_scenario4.yaml`. Match the
**data** block exactly so all baselines train on the same windows:

```yaml
data:
  dataset:
    num_steps: 64                       # contiguous chunk length per sample
    frameskip: 1
    name: scenario4_train_lewm
    cache_dir: /home/lthil/.stable_worldmodel
    keys_to_load: [pixels, action]
    keys_to_cache: [action, pixels]     # cache pixels in RAM (~4.9 GB)
```

Match the **trainer** block too: `precision: bf16-mixed`, `gradient_clip_val: 0.3`,
`max_epochs: 10`. **Match the effective batch** to AR-LSTM/JEPA via:

```yaml
loader:
  batch_size: 512                       # if your model fits at 512, just use it
  # else: pick the largest physical bs that fits, then set
  # +trainer.accumulate_grad_batches=K so bs * K = 512.
```

If your model is GRU/RNN-bound (RSSM was), expect GPU utilisation in the
60-80 % range at saturation rather than 95 %+. That's fine — total
samples-per-second is what matters for wallclock.

### 3. Launcher script

Copy `train_rssm_scenario4_orailix.sh`. Keep the same env-var override
contract (BATCH_SIZE, NUM_WORKERS, MAX_EPOCHS, ACCUM_GRAD, LR,
PRECISION, WANDB_MODE) so the dragon handoff playbook works unchanged.
Default `WANDB_MODE=offline` and tee into `logs/` with a UTC-stamped
filename.

### 4. Smoke test

Synthetic forward+backward at small scale:

```bash
cd ~/thesis/le-wm
source .venv/bin/activate
python baselines/<name>/smoke_test.py
```

Should pass in seconds and produces no NaNs. This catches >90 % of
"model wiring is wrong" problems before you waste GPU time.

### 5. Train

Inside a tmux. Cap monitoring with the persistent watcher pattern from
`/home/lthil/thesis/le-wm/.eval_watch.sh` (see RSSM bring-up): a
background tmux that polls the live log for `Training complete.` and
fires the eval pipeline.

```bash
tmux new -d -s s4_<name> "cd ~/thesis/le-wm && source .venv/bin/activate && \
  WANDB_MODE=offline BATCH_SIZE=512 NUM_WORKERS=8 MAX_EPOCHS=10 \
  ./train_<name>_scenario4_<host>.sh 2>&1 | tee logs/s4_<name>_$(date -u +%Y%m%d_%H%M).log"
```

Useful reality-checks during training (RSSM bs=512 on 5090 numbers,
2026-05-02):

| signal                           | RSSM value          | what to flag                  |
|----------------------------------|---------------------|-------------------------------|
| GPU utilisation                  | 60-70 %             | <30 % → dataloader-bound      |
| VRAM                             | ~8 GB / 32 GB       | >28 GB → close to OOM         |
| samples/sec at bs=512            | ~4,400              | <500 → check stride / model   |
| epoch time at bs=512             | ~24 min             | >2 h → bs too small or stride=1 issue |
| recon loss at end of epoch 0     | 0.002-0.003         | >0.05 → model not fitting     |
| total loss at end of epoch 0     | 0.5-0.7             | >2.0 → KL not converging      |

If wandb-summary.json is sparse (large bs ⇒ infrequent log ticks), watch
the `.wandb` file growing as the live signal.

### 6. Eval suite

Once `Training complete.` lands, the eval pipeline runs five steps in
priority order:

1. **`baselines/<name>/multiseed_eval_<name>.py`** — TransformerProbe
   Task 1 across 6 probe seeds × {sl=1, sl=10} on **test** and
   **test_hard**. This produces the `multiseed_results.csv` row that
   slots into the headline table next to E_1/E_2/L_*.
2. **`baselines/<name>/eval_<name>.py`** — Custom Task 1 (Ridge) and
   Task 3 (latent forecasting) on test + test_hard. Writes
   `task_results_test{,_hard}.json`.
3. **`baselines/<name>/per_archetype_<name>.py`** — Task 1 sliced by
   archetype (A_compressor / B_fan_booster / C_turbine / D_balanced).
   Writes `per_archetype.csv`.
4. **`counterfactual_fidelity_<name>.py`** — RMSE vs the deterministic
   `EpisodeReplayer` across `actions × episodes × branch_fracs ×
   horizon`. Writes `eval_results/counterfactual_<name>/{results_*.json,summary.json}`.
5. **`baselines/<name>/render_results.py`** — Joins all of the above
   and emits `RESULTS_<NAME>.md` at the repo root.

Total wallclock for the eval pipeline on the dragon RTX 5090: 15-30 min
(the counterfactual pass is the longest, ~10-15 min for 30 ep × 7
actions × 3 branch positions × 100 horizon).

### 7. Multi-training-seed (optional, expensive)

The RSSM baseline I'm shipping tonight is a single training seed (3072)
with multi-probe-seed error bars. If reviewer feedback demands
training-seed multi-seed (it likely will for the camera-ready), add:

```bash
for seed in 0 1 2 3 4 5; do
    SEED=$seed BATCH_SIZE=512 ./train_<name>_scenario4_<host>.sh
done
```

Each seed is one full training run (~4 h for RSSM at bs=512). Plan
~24 h of GPU time.

---

## What goes in the report

`RESULTS_<NAME>.md` should contain, in order:

1. **Headline table** — multi-seed mean Pearson on test + test_hard,
   side by side with E_*/L_* from `eval_results/multiseed/SUMMARY.md`.
   This is the figure the reviewer reads first.
2. **sl=10 supplement** — same numbers with windowed probe input.
3. **Per-component HI breakdown** — per-HI R²/RMSE/Pearson on both
   splits. Flags which HI dimensions the encoder is and isn't picking
   up.
4. **Per-archetype breakdown** — Pearson by failure mode. Flags
   diversity issues.
5. **Task 3 latent forecasting** — RMSE_clean / RMSE_event /
   action_divergence_gap at τ ∈ {1, 5, 10, 25, 50}.
6. **Counterfactual fidelity** — RMSE@τ ∈ {10, 50, 99} per action;
   differential |Δ_model − Δ_sim| in the same shape as
   `eval_results/counterfactual/SUMMARY.md`.

Every table should reference a CSV/JSON artifact that lives under
`eval_results/<name>_*/`, so the paper's compile pipeline can reach
them. Don't put numbers in the report that aren't backed by a file.

---

## Common gotchas (RSSM bring-up taught us these)

- **Default `BATCH_SIZE=16` was too small** at the launcher level — the
  effective bs needs to match AR-LSTM/JEPA (512). Bumping bs alone
  helped only a little for sequence-bound models; the real win was
  ensuring effective bs matched.
- **`tee` + tqdm**: Lightning's tqdm bar uses `\r` and is invisible in
  tee'd logs. Watch the wandb-summary.json (or the `.wandb` file size)
  for live progress — don't expect step counts in the tee log.
- **Wandb-summary.json flushes infrequently.** At bs=512 the first flush
  was at ~480 s of runtime. Don't conclude the run is stuck just
  because the file isn't there yet. Cross-check `lsof -p <PID>` and
  GPU util.
- **`torch.load` needs the model class importable.** If you don't add
  `from baselines.<name>.model import <Class>` before `torch.load`, the
  unpickle will crash with an unfindable class. Every eval wrapper in
  this repo does this `noqa: F401` import explicitly.
- **stride-1 sliding windows** mean `len(dataset) ≈ n_timesteps` even
  with `num_steps=64`. At ~7 M total timesteps you have ~7 M training
  windows. With bs=16 that's ~393 k steps/epoch (~9 h on a 5090);
  with bs=512 it's ~12 k steps/epoch (~24 min). Choose bs based on
  steps-per-epoch arithmetic, not just memory.
- **Counterfactual rollouts need a probe.** The HI decoder isn't
  shared across baselines; each architecture's eval trains a fresh
  probe on its own embeddings. The rollout function returns a feat
  sequence; the probe maps feat → HI. Use `eval_sweep.train_probe(seq_len=1)`
  for symmetry with the existing JEPA pipeline.

---

## Concrete checklist before you call it done

- [ ] `python baselines/<name>/smoke_test.py` exits 0 with no NaN.
- [ ] Training run completes 10 epochs and writes
      `<ckpt_dir>/<name>_*_epoch_10_object.ckpt`.
- [ ] `python baselines/<name>/multiseed_eval_<name>.py` produces a CSV
      whose `MEAN` rows have non-NaN Pearson at sl=1.
- [ ] `python baselines/<name>/per_archetype_<name>.py` produces 4 rows
      per (split, sl) — one per archetype.
- [ ] `python counterfactual_fidelity_<name>.py` writes a `summary.json`
      whose `differentials` dict has 6 actions × n_episodes entries
      (action 0 is the baseline so it doesn't appear).
- [ ] `RESULTS_<NAME>.md` renders without missing-file warnings.
- [ ] Headline Pearson on test_hard is non-trivially positive
      (raw_sensors floor is ~0.50; a self-supervised baseline should
      be in the 0.10-0.40 range — the gap is what the paper is about).
- [ ] `EVAL_PLAN_<NAME>.md` lists the open follow-ups so the next person
      doesn't re-derive them.
