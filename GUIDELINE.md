# Command-Line Guide

A practical reference for every command-line tool in this project: what it does,
its arguments, and copy-paste examples. For architecture and file-by-file
details, see [README.md](README.md).

## Contents

- [Setup](#setup)
- [Shell note (PowerShell vs bash)](#shell-note-powershell-vs-bash)
- [Quick reference](#quick-reference)
- [1. Prepare data](#1-prepare-data)
- [2. Train](#2-train)
- [3. Evaluate](#3-evaluate)
- [4. Predict / inference](#4-predict--inference)
- [5. Use the pretrained OpenEarthMap-SAR weight](#5-use-the-pretrained-openearthmap-sar-weight)
- [6. Fine-tune (warm start)](#6-fine-tune-warm-start)
- [7. Diagnostic tools](#7-diagnostic-tools)
- [8. Tests](#8-tests)
- [End-to-end recipes](#end-to-end-recipes)
- [Troubleshooting](#troubleshooting)

---

## Setup

From the repository root, create and activate a virtual environment and install
dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script, allow it for the current user with
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use `cmd`'s
`.venv\Scripts\activate.bat`.

All commands below are run from the repository root with the environment active.

## Shell note (PowerShell vs bash)

Every example here is written as a **single line** so it works everywhere.

If you want to split a long command across lines:

- **PowerShell** uses a backtick `` ` `` at the end of each line (no trailing
  space after it).
- **bash / Git Bash** uses a backslash `\`.

The README uses `\` (bash style). On PowerShell, either keep commands on one line
or replace `\` with `` ` ``.

## Quick reference

| Command | Purpose |
| --- | --- |
| `python tools/oem/create_labeled_splits.py` | Build train/val/test split files |
| `python train.py` | Train (or fine-tune with `--init-weights`) |
| `python evaluate.py` | Evaluate a checkpoint on a split (writes metrics JSON) |
| `python test.py` | Evaluate on the `test` split (shortcut) |
| `python predict.py` | Run inference, save colorized masks / panels |
| `python tools/oem/check_dataset.py` | Visually sanity-check image/mask pairs |
| `python tools/oem/check_missing_files.py` | Find/filter samples with missing files |
| `python tools/oem/count_pixels.py` | Count raw mask values and percentages |
| `python tools/oem/test_dataset_loader.py` | Load random samples through the dataset |
| `python tests/smoke_test.py` | Run the synthetic-data smoke tests |

Configs available:

| Config | Use |
| --- | --- |
| `configs/unet/unet_effb4_oem.yml` | Train from an ImageNet-pretrained encoder |
| `configs/unet/unet_effb4_oem_pretrained.yml` | Use the external OpenEarthMap-SAR weights (inference/eval) |
| `configs/unet/unet_effb4_oem_finetune.yml` | Fine-tune (warm start) from a pretrained checkpoint |

---

## 1. Prepare data

### `tools/oem/create_labeled_splits.py`

Builds 60/10/30 `train`/`val`/`test` split text files from the labeled data.
Run this once before training.

| Argument | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--config` | yes | - | Path to the YAML config |
| `--seed` | no | `42` | Random seed for the split |

```powershell
python tools/oem/create_labeled_splits.py --config configs/unet/unet_effb4_oem.yml
```

---

## 2. Train

### `train.py`

Builds datasets/model/loss/optimizer, runs the epoch loop, logs per-epoch
metrics to JSON, and saves `last`/`best` checkpoints under the config's
`experiment.output_dir`.

| Argument | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--config` | yes | - | Path to the YAML config |
| `--init-weights` | no | none | Warm-start: load these weights before training (a project checkpoint or a bare `state_dict`) to fine-tune from |

Train from the ImageNet-pretrained encoder:

```powershell
python train.py --config configs/unet/unet_effb4_oem.yml
```

Outputs:

- `experiments/<name>/checkpoints/last_checkpoint.pth`
- `experiments/<name>/checkpoints/best_checkpoint.pth` (best val mIoU)
- `experiments/<name>/logs/training_logs.json`

Warm-starting is covered in [section 6](#6-fine-tune-warm-start).

---

## 3. Evaluate

### `evaluate.py`

Loads a checkpoint and reports loss / OA / mIoU / mF1 on a chosen split, writing
`<split>_metrics.json`.

| Argument | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--config` | yes | - | Path to the YAML config |
| `--checkpoint` | yes | - | Checkpoint to load |
| `--split` | no | `test` | `train`, `val`, or `test` |
| `--output` | no | `<output_dir>/logs/<split>_metrics.json` | Where to write the metrics JSON |

```powershell
python evaluate.py --config configs/unet/unet_effb4_oem.yml --checkpoint experiments/exp_01_unet_effb4_oem/checkpoints/best_checkpoint.pth --split test
```

### `test.py`

Shortcut for `evaluate.py` with `--split test`. Same arguments as `evaluate.py`
(you can omit `--split`).

```powershell
python test.py --config configs/unet/unet_effb4_oem.yml --checkpoint experiments/exp_01_unet_effb4_oem/checkpoints/best_checkpoint.pth
```

---

## 4. Predict / inference

### `predict.py`

Runs inference on a single image or a directory and saves colorized prediction
masks (and optional side-by-side panels).

| Argument | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--config` | yes | - | Path to the YAML config |
| `--checkpoint` | yes | - | Checkpoint to load |
| `--input` | yes | - | An image file **or** a directory of images |
| `--output` | no | `<output_dir>/predictions` | Output directory |
| `--panel` | no | off | Also save an input+prediction side-by-side panel per image |
| `--tile_size` | no | none | Enable sliding-window inference with this tile size (pixels) |
| `--overlap` | no | `128` | Tile overlap in pixels (only with `--tile_size`) |

Outputs per input image `<name>`:

- `<name>_pred.png` - the colorized land-cover mask
- `<name>_panel.png` - input + prediction side by side (only with `--panel`)

Full-image inference on a folder, with panels:

```powershell
python predict.py --config configs/unet/unet_effb4_oem.yml --checkpoint experiments/exp_01_unet_effb4_oem/checkpoints/best_checkpoint.pth --input path/to/images --output outputs/preds --panel
```

Sliding-window inference for imagery too large to fit in memory (tiles are run
independently and their softmax probabilities are averaged over overlaps):

```powershell
python predict.py --config configs/unet/unet_effb4_oem.yml --checkpoint experiments/exp_01_unet_effb4_oem/checkpoints/best_checkpoint.pth --input path/to/images --tile_size 512 --overlap 128
```

Full-image inference reflect-pads inputs to a multiple of 32, so any image size
works; use `--tile_size` only when memory is the limit. (1024x1024 imagery fits
fine without tiling.)

---

## 5. Use the pretrained OpenEarthMap-SAR weight

The external `pretrain_weight/RGB_Real_5_u-efficientnet-b4.pth` is a fully
trained model. It differs from a checkpoint trained here (9-class head with a
leading background class, SCSE decoder attention, `[0,1]` input scaling, bare
`state_dict`); `configs/unet/unet_effb4_oem_pretrained.yml` and the loader
handle all of that. **Use this config with these weights** - it sets
`normalization: zero_one`, which they require.

Predict on optical imagery (e.g. the CEI data):

```powershell
python predict.py --config configs/unet/unet_effb4_oem_pretrained.yml --checkpoint pretrain_weight/RGB_Real_5_u-efficientnet-b4.pth --input data/CEI_data --output outputs/CEI_data_pred --panel
```

Evaluate on a labeled split:

```powershell
python evaluate.py --config configs/unet/unet_effb4_oem_pretrained.yml --checkpoint pretrain_weight/RGB_Real_5_u-efficientnet-b4.pth --split test
```

---

## 6. Fine-tune (warm start)

Instead of training from scratch, load a pretrained checkpoint as the starting
point (`--init-weights`) and continue training at a low learning rate. The
`unet_effb4_oem_finetune.yml` config matches the pretrained architecture so the
warm start loads cleanly, and uses a low LR (`2e-5`).

**Stage 1 - fine-tune on OpenEarthMap** (stronger base from the full labeled
split; warm-started from the external weights):

```powershell
python train.py --config configs/unet/unet_effb4_oem_finetune.yml --init-weights pretrain_weight/RGB_Real_5_u-efficientnet-b4.pth
```

**Stage 2 - fine-tune on your own domain (e.g. CEI)**: warm-start from the
Stage-1 `best_checkpoint.pth` using a config that points at your labeled domain
data. This requires labels for that domain (see below).

```powershell
python train.py --config configs/unet/<your_cei_config>.yml --init-weights experiments/exp_03_oem_finetune/checkpoints/best_checkpoint.pth
```

Label format for a new domain: single-channel PNG masks, same filename as the
image, using the **raw OpenEarthMap encoding** so the dataset remaps them to
training indices `0`-`7`:

| Pixel value | Class |
| --- | --- |
| `0` | background / unknown (ignored in the loss) |
| `1` | Bareland |
| `2` | Rangeland |
| `3` | Developed space |
| `4` | Road |
| `5` | Tree |
| `6` | Water |
| `7` | Agriculture land |
| `8` | Building |

Pixels left as `0` are ignored during training, so partial ("sparse") labeling
is allowed - label only the regions you are confident about.

---

## 7. Diagnostic tools

All under `tools/oem/`. Handy for verifying data before a long training run.

### `check_dataset.py`

Structural/visual sanity check of image/mask pairs.

| Argument | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--config` | yes | - | Path to the YAML config |
| `--split` | no | `train` | `train`, `val`, or `test` |
| `--num_samples` | no | `5` | Number of samples to inspect |

```powershell
python tools/oem/check_dataset.py --config configs/unet/unet_effb4_oem.yml --split train --num_samples 5
```

### `check_missing_files.py`

Reports and filters out samples whose image or mask file is missing, writing a
filtered split file.

| Argument | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--config` | yes | - | Path to the YAML config |
| `--split` | no | `train` | Which split to check |
| `--input` | no | `<split>.txt` | Input split filename |
| `--output` | no | `<split>_filtered.txt` | Filtered output filename |

```powershell
python tools/oem/check_missing_files.py --config configs/unet/unet_effb4_oem.yml --split train
```

### `count_pixels.py`

Counts raw mask values and their dataset percentages (class balance).

| Argument | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--config` | yes | - | Path to the YAML config |
| `--split` | no | `train` | Which split to count |
| `--max_files` | no | all | Cap the number of files scanned |

```powershell
python tools/oem/count_pixels.py --config configs/unet/unet_effb4_oem.yml --split train --max_files 100
```

### `test_dataset_loader.py`

Loads random samples through `OpenEarthMapDataset` and prints tensor
shapes/dtypes/unique values. Takes **no arguments** (it uses
`configs/unet/unet_effb4_oem.yml` and the `train` split).

```powershell
python tools/oem/test_dataset_loader.py
```

---

## 8. Tests

### `tests/smoke_test.py`

Exercises the runtime paths with synthetic data (no dataset or trained weights
needed): metrics, visualization, loss forward/backward, model construction, and
both full-image and tiled inference. Checks that need
`segmentation_models_pytorch` are skipped (not failed) when it is not installed.
Takes no arguments.

```powershell
python tests/smoke_test.py
```

---

## End-to-end recipes

### A. First run on OpenEarthMap (train your own)

```powershell
python tools/oem/create_labeled_splits.py --config configs/unet/unet_effb4_oem.yml
python train.py --config configs/unet/unet_effb4_oem.yml
python test.py --config configs/unet/unet_effb4_oem.yml --checkpoint experiments/exp_01_unet_effb4_oem/checkpoints/best_checkpoint.pth
```

### B. Predict with the pretrained weight (no training)

```powershell
python predict.py --config configs/unet/unet_effb4_oem_pretrained.yml --checkpoint pretrain_weight/RGB_Real_5_u-efficientnet-b4.pth --input data/CEI_data --output outputs/CEI_data_pred --panel
```

### C. Two-stage fine-tune (OEM then your domain)

```powershell
python train.py --config configs/unet/unet_effb4_oem_finetune.yml --init-weights pretrain_weight/RGB_Real_5_u-efficientnet-b4.pth
python train.py --config configs/unet/<your_cei_config>.yml --init-weights experiments/exp_03_oem_finetune/checkpoints/best_checkpoint.pth
```

---

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `Missing expression after unary operator '--'` | You used bash `\` line continuation in PowerShell. Use one line, or `` ` `` at line ends. |
| Predictions are noisy / nonsensical with the pretrained weight | Wrong normalization. These weights need `dataset.normalization: zero_one` (already set in the pretrained/finetune configs); `imagenet` breaks them. |
| `dataset.num_classes and model.num_classes must match` | The config's `dataset.num_classes` must equal `model.num_classes` (both `8` here). `pretrained_classes` is separate and does not need to match. |
| `CUDA out of memory` during training | Lower `training.batch_size`, or reduce `dataset.crop_size`, in the config. |
| Mixed precision seems ignored | AMP only runs on CUDA; it is skipped on CPU/MPS even when `mix_precision: true`. |
| Checkpoint fails to load | Architecture must match the weights. For the OpenEarthMap-SAR weights use `unet_effb4_oem_pretrained.yml` (SCSE + `pretrained_classes: 9`). |
