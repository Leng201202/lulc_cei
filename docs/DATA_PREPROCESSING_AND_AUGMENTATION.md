# Data Preprocessing and Augmentation

This document describes the preprocessing implemented by the current project,
not a proposed pipeline. The relevant code is in `src/datasets/` and the active
experiment choices are in `configs/cei_oem/` and `configs/cei_irsa/`.

## Data domains and loaders

| Domain | Config name | Loader | Typical layout |
| --- | --- | --- | --- |
| OpenEarthMap (OEM) | `OpenEarthMap` | `OpenEarthMapDataset` | `<region>/images`, `<region>/labels` |
| CEI | `CEI` | `OpenEarthMapDataset` | flat `images`, `masks` |
| IRSAMap | `IRSA_Map` | `IRSADataset` | `train/image`, `train/SegLabel_vwsbr`, corresponding `test/` folders |

Each split file contains one image filename per line. OEM region names are
derived by removing the final underscore-number component from the stem. CEI
and IRSA use flat directory paths from their configs.

## Preprocessing sequence

One sample follows this sequence:

```text
split filename
  -> resolve image and mask paths
  -> read with OpenCV
  -> image BGR to RGB
  -> mask to one channel
  -> validate raw label values
  -> map raw labels to zero-based training indices
  -> optionally mask IRSA nodata
  -> synchronized image/mask spatial transforms
  -> normalize image
  -> image tensor + torch.long mask tensor
```

Geometric transforms receive the image and target together. This preserves
pixel alignment, which is mandatory for semantic segmentation.

## Shared CEI taxonomy

The OEM-to-CEI and IRSA-to-CEI experiments both predict these channels:

| Output channel | CEI disk value | Class |
| ---: | ---: | --- |
| 0 | 1 | Rangeland |
| 1 | 2 | Agriculture |
| 2 | 3 | Tree |
| 3 | 4 | Water |
| 4 | 5 | Building |
| 5 | 6 | Road |
| 6 | 7 | Non-vegetated |

CEI disk value 0 is unlabeled and becomes internal `ignore_index` 255.

### OEM mapping

OEM raw values 1-8 are mapped as follows:

| OEM raw value | OEM class | CEI channel |
| ---: | --- | ---: |
| 1 | Bareland | 6 |
| 2 | Rangeland | 0 |
| 3 | Developed space | 6 |
| 4 | Road | 5 |
| 5 | Tree | 2 |
| 6 | Water | 3 |
| 7 | Agriculture land | 1 |
| 8 | Building | 4 |

Thus, Bareland and Developed space are intentionally merged into CEI
Non-vegetated. OEM raw value 0 becomes ignore.

### IRSA mapping and nodata

IRSAMap codes are converted with `IRSA_TO_CEI` in `taxonomy.py`. Cropland,
forest, grass/sparse, water subtypes, building, road, and sport surfaces map to
the closest CEI classes. IRSA code 0 is different from OEM/CEI zero: it mostly
represents unannotated bare or developed ground and therefore maps to
Non-vegetated.

IRSA also uses code 0 on near-black border padding. `NodataMasker` identifies
pixels for which `max(R, G, B) <= dataset.nodata_to_ignore` and changes their
mapped target to 255. The active IRSA configs use a threshold of 8.

All label schemes use a 256-entry NumPy lookup table. A mask with an unexpected
raw value raises an error instead of silently converting that value to ignore.

## Training transforms

The implemented training policy is:

1. constant-pad images to at least `crop_size x crop_size`;
2. fill image padding with zero and mask padding with `ignore_index`;
3. take a random crop of exactly `crop_size x crop_size`;
4. if `dataset.augment: true`, apply:
   - horizontal flip with probability 0.5;
   - vertical flip with probability 0.5;
   - random 90-degree rotation with probability 0.5;
5. normalize the image;
6. convert image and mask to PyTorch tensors.

The three optional augmentations are independent and can compose. The current
OEM-to-CEI five-model configs and IRSA-to-CEI config set `augment: false`, so
their stochastic spatial preprocessing is random cropping only. Changing this
flag changes the experimental protocol and should be recorded as a separate
run.

## Validation and test transforms

Evaluation is deterministic.

With `dataset.eval_mode: full`, the loader keeps the complete tile and pads its
height and width to multiples of 32. This accommodates encoders with five
downsampling stages. Padding in the target is 255, so it contributes to neither
loss nor metrics. Because full tiles can differ in size, the entry points force
evaluation batch size to one.

With any other `eval_mode`, the loader pads to `crop_size` and then takes a
center crop. This is faster and batchable, but discards content outside the
crop; results should not be compared directly with full-image evaluation.

## Normalization

| Config value | Operation | Intended use |
| --- | --- | --- |
| `imagenet` | `(x/255 - mean) / std`, mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)` | Models initialized from ImageNet encoders and trained in this project |
| `zero_one` | `x / 255` | External OEM-SAR baseline weights |

Normalization must match the checkpoint's training. A mismatch changes every
input feature and can make otherwise valid weights produce poor predictions.

## Tensor contract

For crop-based training with crop size 512, the loader returns:

```text
image: float tensor [3, 512, 512]
mask:  long tensor  [512, 512]
mask values: 0..N-1 or 255
```

After collation, a model receives `[B, 3, H, W]` and returns
`[B, N, H, W]` logits. Targets are class indices, not one-hot arrays.

## Split and integrity utilities

- `tools/oem/create_labeled_splits.py` creates labeled OEM train/val/test files.
- `tools/irsa/make_splits.py` creates IRSA split files and supports
  stratification.
- `tools/cei/make_test_split.py` creates the CEI test list.
- `tools/oem/check_dataset.py`, `check_missing_files.py`, and `count_pixels.py`
  inspect structure, missing pairs, and class distribution.

The training entry point does not set global Python, NumPy, PyTorch, or
DataLoader random seeds. Split-generation seeds do not make model training
fully deterministic; reproducible repeated trials require explicit runtime
seeding and deterministic-backend settings.
