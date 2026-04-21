
# LeWorldModel
### Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

[Lucas Maes*](https://x.com/lucasmaes_), [Quentin Le Lidec*](https://quentinll.github.io/), [Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr), [Yann LeCun](https://yann.lecun.com/) and [Randall Balestriero](https://randallbalestriero.github.io/)

**Abstract:** Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pretrained encoders, or auxiliary supervision to avoid representation collapse. In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With ~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks. Beyond control, we show that LeWM's latent space encodes meaningful physical structure through probing of physical quantities. Surprise evaluation confirms that the model reliably detects physically implausible events.

<p align="center">
   <b>[ <a href="https://arxiv.org/pdf/2603.19312v1">Paper</a> | <a href="https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e?usp=sharing">Checkpoints</a> | <a href="https://huggingface.co/collections/quentinll/lewm">Data</a> | <a href="https://le-wm.github.io/">Website</a> ]</b>
</p>

<br>

<p align="center">
  <img src="assets/lewm.gif" width="80%">
</p>

If you find this code useful, please reference it in your paper:
```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## Using the code
This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environment management, planning, and evaluation, and [stable-pretraining](https://github.com/galilai-group/stable-pretraining) for training. Together they reduce this repository to its core contribution: the model architecture and training objective.

**Installation:**
```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

## Data

Datasets use the HDF5 format for fast loading. Download the data from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME` (defaults to `~/.stable-wm/`). You can override this path:
```bash
export STABLEWM_HOME=/path/to/your/storage
```

Dataset names are specified without the `.h5` extension. For example, `config/train/data/pusht.yaml` references `pusht_expert_train`, which resolves to `$STABLEWM_HOME/pusht_expert_train.h5`.

## Training

`jepa.py` contains the PyTorch implementation of LeWM. Training is configured via [Hydra](https://hydra.cc/) config files under `config/train/`.

Before training, set your WandB `entity` and `project` in `config/train/lewm.yaml`:
```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

To launch training:
```bash
python train.py data=pusht
```

Checkpoints are saved to `$STABLEWM_HOME` upon completion.

For baseline scripts, see the stable-worldmodel [scripts](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) folder.

## Planning

Evaluation configs live under `config/eval/`. Set the `policy` field to the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix:

```bash
# ✓ correct
python eval.py --config-name=pusht.yaml policy=pusht/lewm

# ✗ incorrect
python eval.py --config-name=pusht.yaml policy=pusht/lewm_object.ckpt
```

## Pretrained Checkpoints

Pre-trained checkpoints are available on [Google Drive](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e). Download the checkpoint archive and place the extracted files under `$STABLEWM_HOME/`.

<div align="center">

| Method | two-room | pusht | cube | reacher |
|:---:|:---:|:---:|:---:|:---:|
| pldm | ✓ | ✓ | ✓ | ✓ |
| lejepa | ✓ | ✓ | ✓ | ✓ |
| ivl | ✓ | ✓ | ✓ | — |
| iql | ✓ | ✓ | ✓ | — |
| gcbc | ✓ | ✓ | ✓ | — |
| dinowm | ✓ | ✓ | — | — |
| dinowm_noprop | ✓ | ✓ | ✓ | ✓ |

</div>

## Loading a checkpoint

Each tar archive contains two files per checkpoint:
- `<name>_object.ckpt` — a serialized Python object for convenient loading; this is what `eval.py` and the `stable_worldmodel` API use
- `<name>_weight.ckpt` — a weights-only checkpoint (`state_dict`) for cases where you want to load weights into your own model instance

To load the object checkpoint via the `stable_worldmodel` API:

```python
import stable_worldmodel as swm

# Load the cost model (for MPC)
cost = swm.policy.AutoCostModel('pusht/lewm')
```

This function accepts:
- `run_name` — checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix
- `cache_dir` — optional override for the checkpoint root (defaults to `$STABLEWM_HOME`)

The returned module is in `eval` mode with its PyTorch weights accessible via `.state_dict()`.

## TurboSens Benchmark

TurboSens evaluates world models on the OpenDeckSMR turbofan simulator.
The benchmark is split into two scripts:

- **`eval_sweep.py`** — four downstream probing tasks that measure what the
  encoder/predictor has learned about the underlying system (HI state,
  degradation dynamics, latent forecasting, health classification).
- **`eval_ood.py`** — anomaly/OOD detection on five live simulator scenarios.

Both scripts accept any mix of JEPA and AR-LSTM checkpoints in a single run.

### Checkpoint format

All eval scripts use a consistent `name:path` format:
```bash
"display_name:/absolute/path/to/checkpoint_epoch_N_object.ckpt"
```
The `display_name` is the model label in all output files and figures.

---

### Downstream benchmark sweep (`eval_sweep.py`)

**Tasks:**

| # | Task | Target | Probe | Metrics |
|---|---|---|---|---|
| 1 | HI state estimation | HI (10-dim) | TransformerProbe, `seq_len ∈ {1, 10, 50}` | R², RMSE, Pearson |
| 2 | Degradation velocity ΔHI + maintenance alarm | ΔHI(t), `tnm = time-to-next-maint` | TransformerProbe | R², AUROC@K |
| 3 | Latent forecasting | HI at τ=1…50 via AR rollout | Task-1 probe as decoder | RMSE(τ), clean vs event gap |
| 4 | Health-state classification | binary: `healthy` vs `pre-maintenance` | Logistic regression on mean-pooled z | AUROC, balanced acc, F1 |

**Task 4 labels are derived purely from action timestamps** (no HI values used):
- `healthy` — within 200 steps after a repair (or episode start)
- `degraded` — within 300 steps before the next maintenance event
- everything else is excluded (ambiguous mid-episode drift)

This directly tests whether the latent encodes *relative degradation level*
without leaking the HI target through the labels.

**Run on Jean-Zay (recommended):**
```bash
sbatch eval_sweep.slurm   # edit CKPTS array first
```

**Run locally / interactively:**
```bash
CKPTS=(
    "ar_lstm_h1:/path/to/ar_lstm_h1_epoch_100_object.ckpt"
    "jepa_w1_h50:/path/to/lewm_w1_H50_epoch_100_object.ckpt"
)

# All tasks (default):
python eval_sweep.py "${CKPTS[@]}" --out_dir results/sweep/ --no_parallel

# Subsets:
python eval_sweep.py "${CKPTS[@]}" --tasks 1        # HI probe only (fastest)
python eval_sweep.py "${CKPTS[@]}" --tasks 1 2 4    # skip slow forecasting
python eval_sweep.py "${CKPTS[@]}" --tasks 4        # health classification only
```

**Useful flags:**
```bash
--tasks N [N ...]            # select a subset of {1, 2, 3, 4}
--task1_seq_lens  N [N ...]  # override {1, 10, 50} for Task 1
--delta_hi_seq_lens N ...    # override seq_lens for Task 2
--health_seq_lens   N ...    # override seq_lens for Task 4
--forecast_horizon N         # max τ for Task 3 (default 50)
--step_size N                # Task-3 starting-frame subsampling (default 10)
--no_parallel                # force sequential even on multi-GPU hosts
--hdf5 /path/to/data.h5      # override the HDF5 path baked into the script
```

**Output files** written to `--out_dir`:
```
results/sweep/
├── {name}.json          ← per-checkpoint results (all tasks)
├── summary.json         ← combined list
├── metrics_flat.csv     ← flat table (Tasks 1, 2, 2b, 4)
└── task3_curves.csv     ← RMSE(τ) curves for plotting
```

**Generate paper figures** from the saved CSVs (no GPU needed, rerun freely):
```bash
python plot_results.py \
    --csv    results/sweep/metrics_flat.csv \
    --curves results/sweep/task3_curves.csv \
    --out_dir figures/sweep/
```

---

### AR-LSTM baseline

A simple autoregressive LSTM baseline is provided under `baselines/ar_lstm/`.
It reuses the same SensorEncoder as JEPA and exposes an identical
`encode()` / `predict()` interface, so all evaluation scripts work without
modification.

**Train on Jean-Zay:**
```bash
# Edit H, HIDDEN, LAYERS in the slurm header as desired
sbatch baselines/ar_lstm/train_ar_lstm.slurm
```

**Key hyperparameters:**
```bash
H=3        # predictor context window (try 1, 10, 50)
HIDDEN=256 # LSTM hidden state size
LAYERS=2   # stacked LSTM layers
```

The checkpoint is saved as `ar_lstm_h${H}_hd${HIDDEN}_l${LAYERS}_epoch_N_object.ckpt`
under `$STABLEWM_HOME/<subdir>/` and can be passed directly to `eval_sweep.py`
and `eval_ood.py`.

---

### OOD Detection (`eval_ood.py`)

Evaluates four anomaly detectors (surprise, Mahalanobis, k-NN, reconstruction)
on five OOD scenarios generated by the live OpenDeckSMR simulator. Independent
from the `eval_sweep.py` tasks — reports its own AUC-ROC / score-shift metrics.

**Run on Jean-Zay (self-contained — starts simulator worker on the same node):**
```bash
# Edit CKPTS array in eval_ood.slurm first
sbatch eval_ood.slurm
# Results written to $STABLEWM_HOME/eval_ood/<SLURM_JOB_ID>/
```

**Run without simulator** (uses dataset-extreme proxy instead):
```bash
python eval_ood.py "${CKPTS[@]}" --out_dir results/ood/ --no_simulator
```

**Generate all figures from a completed run:**
```bash
python plot_ood.py \
    --summary_json $STABLEWM_HOME/eval_ood/<JOB_ID>/ood_summary.json \
    --out_dir      figures/ood
```

Produces 9 figures in both `.pdf` and `.png`:

| File | Content |
|---|---|
| `auc_heatmap` | AUC-ROC per detector × scenario |
| `score_shift` | Normalised score shift (OOD − ID) / σ_ID |
| `trajectories` | Example degradation state trajectories per scenario |
| `episode_lengths` | Episode length distributions |
| `paper_summary` | Compact two-panel summary for the paper |
| `correlated_pattern` | HPC+HPT selective degradation pattern |
| `episode_auc` | Per-timestep vs per-episode AUC |
| `recon_profile` | Reconstruction error over episode lifetime |
| `spike_fault` | Spike fault pattern visualisation |

**Run connectivity test (if simulator communication fails):**
```bash
python test_zmq.py                     # reads $WORK/.simulator_addr
python test_zmq.py tcp://r1i3n21:5555  # explicit address
python test_zmq.py --batch 100         # include throughput benchmark
```

## Contact & Contributions
Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues)! For questions or collaborations, please contact `lucas.maes@mila.quebec`
