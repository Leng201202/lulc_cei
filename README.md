# LULC CEI

Semantic-segmentation project for land-use/land-cover (LULC) mapping. The
current experiment configuration targets the OpenEarthMap dataset and a U-Net
with an EfficientNet-B4 encoder.

The repository is under active development. Dataset inspection, pixel
statistics, model construction, loss construction, and segmentation metrics
are implemented. The training, validation, evaluation, and prediction
pipelines are still placeholders and cannot yet run end to end.

## Repository Structure

```text
lulc_cei/
├── configs/
│   ├── unet/
│   │   └── unet_effb4_oem.yml
│   └── unetformer/
├── data/
│   ├── OpenEarthMap/
│   ├── IRSA_Map/
│   └── LovaDA/
├── experiments/
├── src/
│   ├── datasets/
│   │   └── openearthmap_dataset.py
│   ├── engine/
│   │   ├── trainer.py
│   │   └── validator.py
│   ├── losses/
│   │   └── loss_factory.py
│   ├── metrics/
│   │   └── segmentation_metrics.py
│   ├── models/
│   │   ├── model_factory.py
│   │   └── unet.py
│   └── utils/
│       ├── config.py
│       └── visualization.py
├── tools/
│   ├── check_dataset.py
│   └── count_pixels.py
├── train.py
├── evaluate.py
├── predict.py
├── test.py
├── requirements.txt
└── .gitignore
```

Generated folders such as `.venv/`, `.git/`, and `__pycache__/` are local
environment or version-control metadata, not project modules.

## Folder and File Reference

### Root files

| Path | Status | Responsibility |
| --- | --- | --- |
| `README.md` | Implemented | Documents the repository, dataset, commands, and implementation status. |
| `requirements.txt` | Implemented | Lists the Python packages required for data loading, augmentation, modeling, metrics, and visualization. Versions are not pinned. |
| `.gitignore` | Implemented | Excludes datasets, virtual environments, caches, checkpoints, predictions, logs, and editor files from Git. |
| `train.py` | Empty | Intended command-line entry point for model training. |
| `evaluate.py` | Empty | Intended entry point for checkpoint evaluation. |
| `predict.py` | Empty | Intended entry point for inference on new imagery. |
| `test.py` | Empty | Reserved for tests or test-split execution; no behavior is currently defined. |

### `configs/`

Stores YAML experiment configurations.

- `configs/unet/unet_effb4_oem.yml` defines the current OpenEarthMap
  experiment, dataset metadata, U-Net architecture, training hyperparameters,
  and requested metrics.
- `configs/unetformer/` is empty and reserved for future UNetFormer
  configurations.

Configuration sections:

| Section | Purpose |
| --- | --- |
| `experiment` | Experiment name and output directory. |
| `dataset` | Dataset root, crop/image sizes, number of classes, and ignored target value. |
| `model` | Architecture, encoder, pretrained encoder weights, input channels, and output classes. |
| `training` | Batch size, epochs, learning rate, optimizer, loss, and mixed-precision flag. |
| `metrics` | Names of the metrics intended for reporting. |

Only some values are currently consumed by code. For example, the dataset
class uses `crop_size` and `ignore_index`, the model factory uses the `model`
section, and the loss factory uses `training.loss`. No trainer currently uses
the batch size, epochs, learning rate, optimizer, or mixed-precision settings.

### `data/`

Contains local datasets and is excluded from Git. Dataset contents are
intentionally not documented here.

### `experiments/`

Intended destination for checkpoints, metrics, logs, and generated outputs.
The current experiment directory is empty because no training pipeline has
been implemented.

### `src/`

- `src/__init__.py` marks the main source directory as a Python package.

### `src/datasets/`

- `src/datasets/__init__.py` marks the directory as a Python package.
- `src/datasets/openearthmap_dataset.py` is currently empty and is the intended location
  for `OpenEarthMapDataset`.

The current `OpenEarthMapDataset` implementation is instead located in
`src/models/unet.py`. It reads a split file, resolves image and label paths,
remaps mask values, applies Albumentations transforms, and returns an image
tensor with a `long` mask tensor.

Training transforms:

1. Random `crop_size × crop_size` crop.
2. Random horizontal and vertical flips.
3. Random 90-degree rotation.
4. ImageNet normalization.
5. Conversion to PyTorch tensors.

Validation/test transforms use a center crop, ImageNet normalization, and
tensor conversion.

### `src/models/`

- `src/models/model_factory.py` builds models from
  `segmentation-models-pytorch`. It currently supports U-Net and DeepLabV3.
- `src/models/unet.py` currently contains `OpenEarthMapDataset`; despite its filename, it
  does not define a U-Net model.
- `src/models/__init__.py` is an empty package marker.

The model factory expects lowercase names (`unet` or `deeplabv3`), but the
current configuration contains `UNet`. This mismatch must be normalized before
model construction will work.

### `src/losses/`

- `src/losses/loss_factory.py` provides:
  - cross-entropy loss;
  - multiclass Dice loss;
  - `CEDiceLoss`, which adds cross-entropy and Dice losses.
- `src/losses/__init__.py` is an empty package marker.

All losses use the dataset `ignore_index`, currently `255`.

### `src/metrics/`

- `src/metrics/segmentation_metrics.py` maintains a confusion matrix and computes:
  - overall accuracy (`OA`);
  - mean intersection over union (`mIoU`);
  - mean F1 score (`mF1`);
  - per-class IoU and F1;
  - the full confusion matrix.
- `src/metrics/__init__.py` is an empty package marker.

Predictions and targets may be NumPy arrays or PyTorch tensors with shape
`[batch, height, width]`. Ignored target pixels are removed before updating
the confusion matrix.

### `src/engine/`

- `src/engine/trainer.py` is empty and is intended to contain the epoch/batch training
  loop, optimizer updates, mixed precision, logging, and checkpoint saving.
- `src/engine/validator.py` is empty and is intended to run validation and aggregate
  losses and metrics.
- `src/engine/__init__.py` is an empty package marker.

### `src/utils/`

- `src/utils/config.py` implements `load_config()`, which parses a YAML file into a
  Python dictionary.
- `src/utils/visualization.py` is empty and is reserved for image, mask, prediction, and
  metric visualization.
- `src/utils/__init__.py` is an empty package marker.

### `tools/`

#### `check_dataset.py`

Performs a quick structural and visual-data sanity check:

- reads the selected split;
- supports both flat datasets and OpenEarthMap region folders;
- locates image/mask pairs by filename;
- loads a configurable number of samples;
- prints paths, shapes, mask types, and unique mask values;
- reports image/mask size mismatches.

```bash
python tools/check_dataset.py \
  --config configs/unet/unet_effb4_oem.yml \
  --split train \
  --num_samples 5
```

#### `count_pixels.py`

Counts each raw mask value and its dataset percentage:

- indexes masks under `<region>/labels/`;
- reads the requested split;
- counts values with NumPy;
- reports missing masks, total pixels, counts, and percentages;
- accepts `--max_files` for a quicker partial analysis.

Full training split:

```bash
python tools/count_pixels.py \
  --config configs/unet/unet_effb4_oem.yml \
  --split train
```

Quick 100-mask sample:

```bash
python tools/count_pixels.py \
  --config configs/unet/unet_effb4_oem.yml \
  --split train \
  --max_files 100
```

## Intended Processing Flow

```text
YAML configuration
        |
        v
OpenEarthMap split + region files
        |
        v
Dataset loading, mask remapping, augmentation
        |
        v
Segmentation model -> [B, 8, H, W] logits
        |
        +--> CE/Dice loss -> optimizer update
        |
        +--> argmax predictions -> confusion matrix -> OA/mIoU/mF1
        |
        v
Checkpoints, metrics, and visualizations
```

The configuration, dataset logic, model factory, loss factory, and metrics are
intended to connect in this order. The engine and root entry-point files still
need implementation to complete the flow.

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The virtual environment is active when the shell prompt begins with
`(.venv)`.

## Current Configuration

`configs/unet/unet_effb4_oem.yml` currently describes:

- OpenEarthMap at `data/OpenEarthMap`;
- 1,000 × 1,000 source imagery;
- 512 × 512 training crops;
- eight output classes;
- ignored training target `255`;
- U-Net with an ImageNet-pretrained EfficientNet-B4 encoder;
- batch size 2 for 50 epochs;
- Adam with learning rate `1e-4`;
- combined cross-entropy and Dice loss;
- mixed-precision training.

These are experiment intentions until the trainer is implemented.

## Known Implementation Gaps

1. `train.py`, `evaluate.py`, `predict.py`, and `test.py` are empty.
2. `src/engine/trainer.py` and `src/engine/validator.py` are empty.
3. `src/datasets/openearthmap_dataset.py` is empty; the dataset class is
   currently misplaced in `src/models/unet.py`.
4. The dataset class requires `image_dir`, `mask_dir`, and split keys that the
   YAML does not define. The diagnostic tools provide OpenEarthMap defaults,
   but the dataset class does not.
5. The dataset class extracts a region with `filename.split("_")[0]`. This
   fails for region names containing underscores, such as `buenos_aires`.
6. The model factory expects lowercase model names, while the YAML uses
   `UNet`.
7. `src/utils/visualization.py` is empty.
8. Dependency versions are not pinned, so installs may vary over time.

## Recommended Next Steps

1. Move and harden `OpenEarthMapDataset` under `src/datasets/`.
2. Resolve images through a filename index instead of parsing region names.
3. Normalize model names in the model factory.
4. Implement trainer and validator engines.
5. Implement the root train/evaluate/predict commands.
6. Add focused tests for mask remapping, path resolution, losses, and metrics.
