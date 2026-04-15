
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

TurboSens evaluates world models on the OpenDeckSMR turbofan simulator across
four tasks: HI state estimation, degradation velocity, latent forecasting, and
OOD detection.  All evaluation scripts accept any mix of JEPA and AR-LSTM
checkpoints in the same run.

### Checkpoint format

All eval scripts use a consistent `name:path` format:
```bash
"display_name:/absolute/path/to/checkpoint_epoch_N_object.ckpt"
```
The `display_name` is used as the model label in all output files and figures.

---

### Tasks 1–3 — Downstream benchmark sweep

**What it runs:**

| Task | Description | Approx. time / ckpt |
|---|---|---|
| 1 | HI state estimation — R², RMSE, Pearson per component, for `seq_len ∈ {1, 10, 50}` | ~10 min |
| 2 | Degradation velocity (ΔHI) + maintenance alarm AUC | ~10 min |
| 3 | Latent forecasting + action-divergence gap φ(τ) | ~40 min |

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

# Task 1 only (fastest — just HI probing):
python eval_sweep.py "${CKPTS[@]}" --out_dir results/sweep/ --tasks 1

# Tasks 1 and 2 (skip slow forecasting):
python eval_sweep.py "${CKPTS[@]}" --out_dir results/sweep/ --tasks 1 2
```

**`--no_parallel` explained:** On a single-GPU machine this flag has no effect
on results — it only skips the multiprocessing setup overhead.  On multi-GPU
machines (without the flag) each checkpoint would be dispatched to a separate
GPU in parallel.  Always use `--no_parallel` on Jean-Zay single-GPU jobs for
cleaner logs.

**Output files** written to `--out_dir`:
```
results/sweep/
├── {name}.json          ← per-checkpoint results (all tasks)
├── summary.json         ← combined list of all results
├── metrics_flat.csv     ← flat table for plotting (Tasks 1 & 2)
└── task3_curves.csv     ← RMSE(τ) curves for plotting (Task 3)
```

**Generate paper figures** from the saved CSVs (no GPU needed, rerun freely):
```bash
python plot_results.py \
    --csv    results/sweep/metrics_flat.csv \
    --curves results/sweep/task3_curves.csv \
    --out_dir figures/sweep/
```

Produces (PDF + PNG at 300 dpi):

| File | Content |
|---|---|
| `fig_task1_r2.pdf` | Per-component R² bar chart, all models side-by-side |
| `fig_task3_curves.pdf` | RMSE(τ) for clean / event / gap, 3-panel |
| `fig_task3_comparison.pdf` | Overlay RMSE curves for all models |

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
# In the slurm:
H=3        # predictor context window (try 1, 10, 50)
HIDDEN=256 # LSTM hidden state size
LAYERS=2   # stacked LSTM layers
```

The checkpoint is saved as `ar_lstm_h${H}_hd${HIDDEN}_l${LAYERS}_epoch_N_object.ckpt`
under `$STABLEWM_HOME/<subdir>/` and can be passed directly to `eval_sweep.py`
and `eval_ood.py`.

---

### Task 4 — OOD Detection

Evaluates four anomaly detectors (surprise, Mahalanobis, k-NN, reconstruction)
on five OOD scenarios generated by the live OpenDeckSMR simulator.

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

**Generate all figures from a completed run** (no GPU needed, rerun freely):
```bash
python plot_ood.py \
    --summary_json $STABLEWM_HOME/eval_ood/<JOB_ID>/ood_summary.json \
    --out_dir      figures/ood
```

This produces 9 figures in both `.pdf` and `.png`:

| File | Content |
|---|---|
| `auc_heatmap` | AUC-ROC per detector × scenario (both models) |
| `score_shift` | Normalised score shift (OOD − ID) / σ_ID |
| `trajectories` | Example degradation state trajectories per scenario |
| `episode_lengths` | Episode length distributions |
| `paper_summary` | Compact two-panel summary for the paper |
| `correlated_pattern` | HPC+HPT selective degradation pattern |
| `episode_auc` | Per-timestep vs per-episode AUC comparison |
| `recon_profile` | Reconstruction error over episode lifetime |
| `spike_fault` | Spike fault pattern visualisation |

**Run connectivity test (if simulator communication fails):**
```bash
# From le-wm venv on any node:
python test_zmq.py                     # reads $WORK/.simulator_addr
python test_zmq.py tcp://r1i3n21:5555  # explicit address
python test_zmq.py --batch 100         # include throughput benchmark
```

## Contact & Contributions
Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues)! For questions or collaborations, please contact `lucas.maes@mila.quebec`
