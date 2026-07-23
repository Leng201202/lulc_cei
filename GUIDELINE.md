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
- [9. Correct labels in GIMP (CEI workflow)](#9-correct-labels-in-gimp-cei-workflow)
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
| `python tools/cei/predict_for_gimp.py` | Predict CEI images into a GIMP-editable workspace |
| `python tools/cei/import_from_gimp.py` | Turn GIMP-corrected masks into dataset labels |
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
| `--format` | no | `png` | Mask file format: `png` or `tiff` |
| `--tta` | no | off | Average predictions over 4-way flips (slower, cleaner mask) |

Outputs per input image `<name>`:

- `<name>_pred.png` - the colorized land-cover mask (`<name>_pred.tif` with `--format tiff`)
- `<name>_panel.png` - input + prediction side by side (only with `--panel`)

Use `--format tiff` when the mask is going to be hand-corrected in an image
editor: TIFF is lossless, so the exact class colors survive the round trip. For a
full correction workflow, use the CEI tools in [section 9](#9-correct-labels-in-gimp-cei-workflow)
instead of calling `predict.py` directly.

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

## 9. Correct labels in GIMP (CEI workflow)

The CEI images have no ground-truth labels. Rather than draw them from scratch,
we let the trained model produce a *draft* label for each image, hand-correct the
draft in GIMP, and keep the result as ground truth. Two scripts bracket the
manual editing step.

```text
best checkpoint                                        corrected masks
      |                                                       |
      v                                                       v
predict_for_gimp.py  -->  edit *_mask.tif in GIMP  -->  import_from_gimp.py
      |                                                       |
   review/ workspace                                  data/CEI_data/labels/
   (mask.tif + image.png                              (*.png, raw codes 1-8,
    + palette + instructions)                          ready for training)
```

### Step 1 - `tools/cei/predict_for_gimp.py`

Predicts every CEI image and lays out a GIMP workspace.

| Argument | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--config` | yes | - | Path to the YAML config |
| `--checkpoint` | yes | - | Trained weights (use your best run) |
| `--input` | yes | - | A CEI image file **or** a directory of them |
| `--output` | yes | - | Directory to create the workspace in |
| `--tta` | no | off | 4-way flip TTA: a cleaner draft means less hand-correcting |
| `--start-index` | no | none | Only tiles whose trailing number is >= this |
| `--end-index` | no | none | Only tiles whose trailing number is <= this |
| `--overwrite` | no | off | Re-predict tiles that already have a mask (destroys corrections) |
| `--tile_size` / `--overlap` | no | none / `128` | Sliding-window inference (CEI tiles are 1024x1024 and do not need it) |

Input images may be `.tif`, `.tiff`, `.png`, `.jpg`, or `.jpeg`.

When you add a new batch of tiles to an existing workspace, select them by their
trailing number so the earlier ones are left alone -- e.g. after adding
`maesuai_85..112` to a folder that already had `1..84`:

```powershell
python tools/cei/predict_for_gimp.py --config configs/unet/oem/unet_effb4_oem_v2.yml --checkpoint experiments/exp_02_unet_effb4_oem_paper/checkpoints/best_checkpoint.pth --input data/CEI_data --output outputs/cei_maesuai/review --start-index 85 --tta
```

Tiles that already have a `_mask.tif` in the output folder are skipped anyway, so
re-running the command is safe and never overwrites work you have corrected by
hand. Pass `--overwrite` only when you deliberately want a fresh draft.

```powershell
python tools/cei/predict_for_gimp.py --config configs/unet/oem/unet_effb4_oem_v2.yml --checkpoint experiments/exp_02_unet_effb4_oem_paper/checkpoints/best_checkpoint.pth --input data/CEI_data --output outputs/cei_exp02/review --tta
```

This writes, per image `<name>`:

- `<name>_mask.tif` - the draft mask, in the official class colors. **Edit this one.**
- `<name>_image.png` - the aerial image, to open as a reference layer.
- `_original/<name>.tif` - a pristine copy, used later to report what you changed.

plus `oem_palette.gpl` (import into GIMP so the class colors are exact) and
`HOW_TO_EDIT.md` (the click-by-click GIMP instructions).

### Step 2 - edit in GIMP

Follow `HOW_TO_EDIT.md` in the workspace. The short version:

1. Import `oem_palette.gpl` once (**Windows > Dockable Dialogs > Palettes**,
   right-click > **Import Palette...**).
2. Open `<name>_mask.tif`, then **File > Open as Layers** the `<name>_image.png`
   and drag it *below* the mask; lower the mask's opacity to ~60% to see through it.
3. Paint corrections with the **Pencil** (`N`), not the Paintbrush - the pencil
   has hard edges, the paintbrush anti-aliases and invents in-between colors.
4. Restore opacity to 100% and **File > Export As...** back over the same `.tif`.

Paint a region black ("Unlabeled") when you genuinely cannot tell what it is;
those pixels are excluded from the loss rather than teaching the model a guess.

### Step 3 - `tools/cei/import_from_gimp.py`

Turns the corrected masks back into dataset labels.

| Argument | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--review` | yes | - | The workspace folder from step 1 |
| `--output` | yes | - | Directory to write the label PNGs into |
| `--preview` | no | off | Also save a colorized PNG of each imported label |

```powershell
python tools/cei/import_from_gimp.py --review outputs/cei_exp02/review --output data/CEI_data/labels --preview
```

Each pixel is snapped to the nearest class color (so an accidental anti-aliased
edge still resolves to a real class) and written as a single-channel PNG in the
**raw OpenEarthMap encoding**: classes `1-8`, unlabeled `0`. That is exactly what
`OpenEarthMapDataset` reads - it remaps `1-8` down to training indices `0-7` and
turns everything else into the ignore index.

The script prints, per mask, how many pixels you corrected and the resulting
class distribution. Two warnings are worth acting on:

- **"only N% of pixels were exactly on-palette"** - the mask went through a lossy
  format (JPEG) or was painted with a soft brush. Re-export as TIFF from GIMP.
- **"size changed"** - the mask must stay the same size as its image. The mask is
  skipped; undo the crop/scale in GIMP.

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
