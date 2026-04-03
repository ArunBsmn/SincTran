# SincTran

**An end-to-end transformer pipeline for imagined speech electroencephalogram classification.**

SincTran combines learnable sinc bandpass filter banks (one per physiological EEG band) with a depthwise-separable CNN spatial/temporal encoder and a Transformer with a CLS token. Band contributions are weighted by a learned attention mechanism before spatial mixing, and the Transformer output at the CLS position is passed to an MLP classification head.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Reproducibility](#reproducibility)

---

## Repository Structure

```bash
SINCTRAN/
├── Code/
│   ├── core_model.py        # SincTran architecture + SincBandFilter + TransparentEncoderLayer
│   ├── core_dataset.py      # EEG preprocessing, ZCA whitening, stratified k-fold data loaders
│   ├── core_loaders.py      # Dataset loaders for ASU and BCI Competition datasets
│   ├── core_loss.py         # Cross-entropy loss wrapper
│   ├── core_train.py        # Training loop, evaluation, metric logging, confusion matrices
│   ├── core_utils.py        # Reproducibility, timing, FLOPs, metadata serialisation
│   │
│   ├── main_asu.py          # SincTran training pipeline — ASU dataset
│   ├── main_bci.py          # SincTran training pipeline — BCI Competition Track 3
│   ├── main_driver.py       # Entry point for main experiment runs
│   │
│   ├── sota_asu.py          # SOTA baseline training pipeline — ASU dataset
│   ├── sota_bci.py          # SOTA baseline training pipeline — BCI Competition Track 3
│   ├── sota_driver.py       # Entry point for SOTA benchmark runs
│   │
│   ├── int_module.py        # Interpretation utilities: band attention, attention rollout, GradCAM, UMAP
│   ├── int_asu.py           # Interpretation pipeline — ASU dataset
│   ├── int_bci.py           # Interpretation pipeline — BCI Competition Track 3
│   └── int_driver.py        # Entry point for interpretation runs
│
├── CITATION.cff
├── requirements.txt
└── README.md
```

All scripts live under `Code/` and are intentionally flat within it — all imports resolve within the same directory with no relative-path dependencies. Run all driver scripts from inside `Code/`.

---

## Datasets

Two imagined-speech EEG benchmarks are supported.

**ASU Speech Imagery Dataset**
Four tasks (n1–n4) covering long words, short words, vowels, and mixed word pairs. Raw `.mat` files are expected under a single root directory; `core_loaders.py` resolves subfolder layout automatically. [ASU Dataset](https://doi.org/10.1088/1741-2552/aa8235)

**BCI Competition 2020 Track 3**
Five-class imagined-speech dataset (15 subjects). Expects the standard competition folder layout with a `Track3_clean.csv` label file for the held-out test partition. [BCI Dataset](https://doi.org/10.17605/OSF.IO/PQ7VB)

---

## Installation

Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
```

All imports use only the standard library plus the packages listed in `requirements.txt`. No package installation beyond this step is required.

---

## Usage

Each experiment group has a dedicated driver script. Configure the run by editing the `RUN_CFG` dict at the bottom of the relevant driver, then execute it from inside the `Code/` directory.

### Main experiments (SincTran)

```bash
cd Code
python main_driver.py
```

Set `DATA_PATH` and `RESULTS_ROOT` at the top of `main_asu.py` or `main_bci.py` before running.

### SOTA baselines

```bash
python sota_driver.py
```

Supported baselines: `EEGNet`, `MSVTNet`, `ShallowFBCSPNet`, `CTNet`, `EEGConformer`, `ATCNet`, `EEGNeX` (all via [braindecode](https://braindecode.org)).

### Interpretation

```bash
python int_driver.py
```

Requires trained weights saved by a main-experiment run. Set `WEIGHTS_ROOT` and `RESULTS_ROOT` in `int_asu.py` / `int_bci.py`.

Interpretation outputs per subject/fold:

| Output file | Description |
| --- | --- |
| `band_attn_norm_*.svg` | Band-attention weights (L1-normalised), per class |
| `band_attn_lin_*.svg` | Raw linear band-attention scores, per class |
| `cls_umap_*.svg` | UMAP of CLS-token embeddings |
| `mlp_umap_*.svg` | UMAP of MLP first-layer activations |
| `attn_rollout_*.svg` | CLS-token attention rollout heatmap (Abnar & Zuidema 2020) |
| `gradcam_*.svg` | Spatial Grad-CAM heatmap |

Fold-average versions of rollout and GradCAM figures are written automatically after the last fold.

---

## Configuration

Each pipeline script (`main_asu.py`, `main_bci.py`, `sota_asu.py`, etc.) contains three self-contained config dicts — `DATA_CFG`, `MODEL_CFG`, and `TRAIN_CFG` — alongside a `DATA_PATH` / `RESULTS_ROOT` header block. No external config files are required; all hyperparameters are co-located with the pipeline that uses them.

Driver scripts expose a `RUN_CFG` dict for selecting dataset, task, subjects, fold count, and epoch budget without touching the pipeline files.

---

## Reproducibility

`core_utils.set_all_seeds` fixes seeds for Python, NumPy, and PyTorch (including `cudnn.deterministic`). All pipelines call this before data loading. The default seed across all experiments is `37`.

Normalisation and ZCA whitening statistics are always fitted on the training split only and applied to validation and test splits, with statistics recomputed per fold.
