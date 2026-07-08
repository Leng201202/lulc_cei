# LULC CEI

Semantic-segmentation project for land-use/land-cover (LULC) mapping. The
current experiment configuration targets the [OpenEarthMap](https://open-earth-map.org/)
dataset and a U-Net with an EfficientNet-B4 encoder.

The pipeline now runs end to end: dataset loading, model/loss/optimizer
construction, a training loop with mixed precision, validation with
segmentation metrics, checkpointing, evaluation, and inference on new imagery
are all implemented. The dataset factory is wired for OpenEarthMap; IRSA_Map
and LoveDA are reserved but not yet implemented.

## Repository Structure

```text
lulc_cei/
├── configs/
│   └── unet/
│       └── unet_effb4_oem.yml
├── data/                     # local datasets (git-ignored)
│   ├── OpenEarthMap/
│   ├── IRSA_Map/
│   └── LoveDA/
├── experiments/              # checkpoints, logs, predictions (generated)
├── src/
│   ├── datasets/
│   │   ├── dataset_factory.py
│   │   ├── openearthmap_dataset.py
│   │   ├── irsa_dataset.py       # empty (reserved)
│   │   └── loveda_dataset.py     # empty (reserved)
│   ├── engine/
│   │   ├── trainer.py
│   │   └── validator.py
│   ├── losses/
│   │   └── loss_factory.py
│   ├── metrics/
│   │   └── segmentation_metrics.py
│   ├── models/
│   │   └── model_factory.py
│   └── utils/
│       ├── config.py
│       └── visualization.py
├── tools/
│   └── oem/
│       ├── check_dataset.py
│       ├── check_missing_files.py
│       ├── count_pixels.py
│       ├── create_labeled_splits.py
│       └── test_dataset_loader.py
├── tests/
│   └── smoke_test.py
├── train.py
├── evaluate.py
├── predict.py
├── test.py
├── requirements.txt
└── .gitignore
```

Generated folders such as `.venv/`, `.git/`, and `__pycache__/` are local
environment or version-control metadata, not project modules.

## Pipeline Overview

```text
YAML configuration
        |
        v
OpenEarthMap split + region files
        |
        v
Dataset loading, mask remapping (1-8 -> 0-7), augmentation
        |
        v
U-Net (EfficientNet-B4) -> [B, 8, H, W] logits
        |
        +--> CE + Dice loss -> AdamW update (optional AMP)
        |
        +--> argmax -> confusion matrix -> OA / mIoU / mF1 / per-class
        |
        v
Checkpoints (last + best-by-mIoU), JSON logs, colorized predictions
```

## Folder and File Reference

### Root entry points

| Path | Responsibility |
| --- | --- |
| `train.py` | Trains a model: builds datasets/model/loss/optimizer, runs the epoch loop, logs per-epoch metrics to JSON, and saves `last`/`best` checkpoints. |
| `evaluate.py` | Loads a checkpoint and reports loss/OA/mIoU/mF1 on a chosen split, writing `<split>_metrics.json`. |
| `predict.py` | Runs inference on a single image or a directory and saves colorized prediction masks (and optional side-by-side panels). |
| `test.py` | Thin wrapper around `evaluate.py` that defaults `--split` to `test`. |
| `requirements.txt` | Python dependencies. Versions are not pinned. |
| `.gitignore` | Excludes data, environments, caches, checkpoints, predictions, and logs. |

### `configs/`

`configs/unet/unet_effb4_oem.yml` defines the current OpenEarthMap experiment.

| Section | Purpose | Consumed by |
| --- | --- | --- |
| `experiment` | Experiment name and `output_dir`. | `train.py`, `evaluate.py`, `predict.py` |
| `dataset` | Root, region path patterns, split files, crop size, `num_classes`, `ignore_index`. | `OpenEarthMapDataset`, tools |
| `model` | Architecture, encoder, pretrained weights, channels, classes. | `build_model` |
| `training` | Batch size, epochs, learning rate, optimizer, `weight_decay`, loss, `mix_precision`, `num_workers`. | `train.py`, `build_loss` |
| `metrics` | Names of metrics intended for reporting. | reference only |

`train.py` calls `validate_config`, which requires the `experiment`,
`dataset`, `model`, and `training` sections and enforces
`dataset.num_classes == model.num_classes`.

### `src/datasets/`

- `dataset_factory.py` — `build_dataset(config, split)` dispatches on
  `dataset.name`. `OpenEarthMap` is implemented; `IRSA_Map` and `LoveDA` raise
  `NotImplementedError`.
- `openearthmap_dataset.py` — `OpenEarthMapDataset`. Reads a split file,
  resolves each region with `filename.rsplit("_", 1)` (so multi-underscore
  regions such as `buenos_aires` resolve correctly), reads image/mask pairs
  with OpenCV, remaps raw labels `1-8` to training classes `0-7` (everything
  else becomes `ignore_index`), validates mask values, and applies
  Albumentations transforms.

Training transforms: random crop, horizontal/vertical flips, random 90°
rotation, ImageNet normalization, tensor conversion. Validation/test
transforms: center crop, ImageNet normalization, tensor conversion.

### `src/models/`

- `model_factory.py` — `build_model(config)` builds a
  `segmentation-models-pytorch` model. Names are lowercased; `unet`/`u-net` and
  `deeplabv3` are supported.

### `src/losses/`

- `loss_factory.py` — `build_loss(config)` returns cross-entropy, multiclass
  Dice, or `CEDiceLoss` (cross-entropy + Dice) per `training.loss`. All losses
  honor `dataset.ignore_index`.

### `src/metrics/`

- `segmentation_metrics.py` — `SegmentationMetrics` accumulates a confusion
  matrix and computes OA, mIoU, mF1, per-class IoU/F1, class support, and the
  full confusion matrix. Ignored pixels and out-of-range values are dropped
  before updating. Accepts NumPy arrays or PyTorch tensors of shape
  `[B, H, W]`.

### `src/engine/`

- `trainer.py` — `train_one_epoch(...)` runs the batch loop with optional
  `torch.amp` mixed precision and returns the average loss.
- `validator.py` — `validate_one_epoch(...)` runs evaluation, aggregates the
  loss, and returns the metrics dictionary.

### `src/utils/`

- `config.py` — `load_config(path)` parses a YAML file into a dict.
- `visualization.py` — OpenEarthMap class names and the official RGB palette,
  plus helpers to decode class-index masks to color (`decode_mask`),
  denormalize image tensors, overlay masks, and save prediction masks/panels.

The palette (training index → class → RGB):

| Index | Class | RGB |
| --- | --- | --- |
| 0 | Bareland | (128, 0, 0) |
| 1 | Rangeland | (0, 255, 36) |
| 2 | Developed space | (148, 148, 148) |
| 3 | Road | (255, 255, 255) |
| 4 | Tree | (34, 97, 38) |
| 5 | Water | (0, 69, 255) |
| 6 | Agriculture land | (75, 181, 73) |
| 7 | Building | (222, 31, 7) |

### `tools/oem/`

Diagnostic and data-preparation scripts for OpenEarthMap:

- `check_dataset.py` — structural/visual sanity check of image/mask pairs.
- `check_missing_files.py` — reports and filters out samples with missing
  image or mask files.
- `count_pixels.py` — counts raw mask values and their dataset percentages.
- `create_labeled_splits.py` — builds 60/10/30 train/val/test splits from the
  labeled train+val data (the official test split has no masks).
- `test_dataset_loader.py` — loads random samples through
  `OpenEarthMapDataset` and prints tensor shapes/dtypes/unique values.

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependency versions are not pinned, so installs may vary over time.

## Usage

Prepare labeled splits (once):

```bash
python tools/oem/create_labeled_splits.py --config configs/unet/unet_effb4_oem.yml
```

Train:

```bash
python train.py --config configs/unet/unet_effb4_oem.yml
```

Outputs are written under `experiment.output_dir`:
`checkpoints/last_checkpoint.pth`, `checkpoints/best_checkpoint.pth`, and
`logs/training_logs.json`.

Evaluate a checkpoint:

```bash
python evaluate.py \
  --config configs/unet/unet_effb4_oem.yml \
  --checkpoint experiments/exp_01_unet_effb4_oem/checkpoints/best_checkpoint.pth \
  --split test
```

`test.py` is a shortcut for evaluating the test split:

```bash
python test.py \
  --config configs/unet/unet_effb4_oem.yml \
  --checkpoint experiments/exp_01_unet_effb4_oem/checkpoints/best_checkpoint.pth
```

Predict on new imagery (a file or a directory):

```bash
python predict.py \
  --config configs/unet/unet_effb4_oem.yml \
  --checkpoint experiments/exp_01_unet_effb4_oem/checkpoints/best_checkpoint.pth \
  --input path/to/images \
  --panel
```

Colorized masks (`<name>_pred.png`) and optional panels (`<name>_panel.png`)
are saved to `<output_dir>/predictions` by default.

For imagery too large to fit in memory at once, enable sliding-window
inference. Overlapping tiles are run independently and their softmax
probabilities are averaged over the overlaps:

```bash
python predict.py \
  --config configs/unet/unet_effb4_oem.yml \
  --checkpoint experiments/exp_01_unet_effb4_oem/checkpoints/best_checkpoint.pth \
  --input path/to/images \
  --tile_size 512 --overlap 128
```

### Using the OpenEarthMap-SAR pretrained baseline

The official [OpenEarthMap-SAR](https://github.com/cliffbb/OpenEarthMap-SAR)
U-Net/EfficientNet-B4 weights can be used directly for evaluation and
inference via `configs/unet/unet_effb4_oem_pretrained.yml`. That checkpoint
differs from a checkpoint trained here in three ways, all handled by the
config and loader:

- **9-class head** (index 0 is a `background`/`unknown` class, 1-8 are the land
  cover classes). `model.pretrained_classes: 9` builds the 9-class head and the
  loader drops the leading background channel, so the model behaves as an
  8-class model aligned with training indices 0-7.
- **SCSE decoder attention** (`model.decoder_attention_type: scse`).
- A **bare `state_dict`** rather than this project's `{"model_state_dict": ...}`
  wrapper; `src/models/checkpoint.py` accepts either format.
- **`[0,1]` input scaling** (divide by 255), not ImageNet mean/std. This is set
  with `dataset.normalization: zero_one`; using `imagenet` normalization with
  these weights produces incoherent predictions.

```bash
# Evaluate on a labeled split
python evaluate.py \
  --config configs/unet/unet_effb4_oem_pretrained.yml \
  --checkpoint pretrain_weight/RGB_Real_5_u-efficientnet-b4.pth \
  --split test

# Predict on new optical imagery
python predict.py \
  --config configs/unet/unet_effb4_oem_pretrained.yml \
  --checkpoint pretrain_weight/RGB_Real_5_u-efficientnet-b4.pth \
  --input path/to/images --panel
```

These weights are for **optical (RGB)** imagery. The `dataset.normalization`
value must stay `zero_one` for them; switching it back to `imagenet` reproduces
the incoherent-prediction failure.

### Fine-tuning (warm start)

`train.py --init-weights <path>` loads a checkpoint into the model before
training so you can fine-tune from a pretrained model instead of starting from
the ImageNet encoder. It accepts either this project's checkpoints or a bare
`state_dict`, and the low-LR `unet_effb4_oem_finetune.yml` config matches the
pretrained architecture (SCSE + 9-class head) so the warm start loads cleanly.

```bash
python train.py \
  --config configs/unet/unet_effb4_oem_finetune.yml \
  --init-weights pretrain_weight/RGB_Real_5_u-efficientnet-b4.pth
```

Fine-tuning on a new domain (e.g. imagery unlike OpenEarthMap's regions)
requires labels for that domain; masks must use the raw OpenEarthMap encoding
(`0` = background/unknown, `1`-`8` = the land-cover classes) so the dataset
remaps them to training indices `0`-`7`.

## Testing

`tests/smoke_test.py` exercises the runtime paths with synthetic data (no
dataset or trained weights required): metrics, the visualization palette
helpers, the loss forward/backward pass, model construction, and both the
full-image and tiled inference paths. Checks that require
`segmentation_models_pytorch` are skipped (not failed) when it is not
installed.

```bash
python tests/smoke_test.py
```

## Current Configuration

`configs/unet/unet_effb4_oem.yml` describes:

- OpenEarthMap at `data/OpenEarthMap`, `<region>/images` and `<region>/labels`;
- 1024 × 1024 source imagery, 512 × 512 crops;
- eight output classes; ignored training target `255`;
- U-Net with an ImageNet-pretrained EfficientNet-B4 encoder;
- batch size 2 for 50 epochs;
- AdamW at learning rate `1e-4`, weight decay `0.01`;
- combined cross-entropy + Dice loss;
- mixed-precision training (used only on CUDA).

## Notes and Next Steps

1. `predict.py` runs full-image inference by default (reflect-padding inputs
   to a multiple of 32); use `--tile_size` for imagery too large for memory.
2. `src/datasets/irsa_dataset.py` and `loveda_dataset.py` are reserved stubs.
3. Dependency versions are unpinned; pin them for reproducible runs.
4. Consider a learning-rate scheduler and resume-from-checkpoint support.
5. `tests/smoke_test.py` covers inference, metrics, losses, and visualization;
   dataset mask-remapping/path-resolution tests would still be valuable.

## Reference

Xia et al., "OpenEarthMap: A Benchmark Dataset for Global High-Resolution Land
Cover Mapping," WACV 2023. Project page: https://open-earth-map.org/ ·
Example code: https://github.com/bao18/open_earth_map
