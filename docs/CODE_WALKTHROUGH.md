# Code Walkthrough

This walkthrough follows the current code as it executes. It covers training,
evaluation, label harmonization across OpenEarthMap (OEM), IRSAMap, and CEI,
the five-model OEM-to-CEI experiment, and checkpoint loading.

For topic-focused descriptions, see:

- [Data preprocessing and augmentation](DATA_PREPROCESSING_AND_AUGMENTATION.md)
- [DL architecture and experimental setup](DL_ARCHITECTURE_AND_EXPERIMENTAL_SETUP.md)
- [Model training](MODEL_TRAINING.md)
- [Evaluation metrics](EVALUATION_METRICS.md)

## 1. Runtime overview

The main experiment trains a semantic-segmentation model on OEM labels remapped
into the seven-class CEI taxonomy, validates it on OEM, and evaluates the best
checkpoint on held-out CEI tiles.

```text
YAML config
   |
   +--> dataset factory --> image + raw mask
   |                         |
   |                         +--> RGB conversion
   |                         +--> label lookup table
   |                         +--> crop/pad/normalize/tensor
   |
   +--> model factory --> [B, 7, H, W] logits
   +--> loss factory  --> scalar loss
   |
   +--> train epoch --> backward --> optimizer step
   +--> validation  --> argmax --> confusion matrix --> OA/mIoU/mF1
   |
   +--> JSON log + last checkpoint + best-by-mIoU checkpoint
```

The primary training command is:

```bash
python train.py --config configs/cei_oem/train/unet_effb4_oem2cei.yml
```

The corresponding cross-domain CEI evaluation is:

```bash
python evaluate.py \
  --config configs/cei_oem/test/test_m2_uneteffb4_cei.yml \
  --checkpoint experiments/cei_exp01_oem2cei/checkpoints/best_checkpoint.pth \
  --split test --tta
```

## 2. Entry points

### `train.py`

`train.py` is the training orchestrator.

1. `load_config()` parses YAML.
2. `validate_config()` requires `experiment`, `dataset`, `model`, and
   `training`, checks that dataset/model class counts match, and validates
   `weight_decay`.
3. If `--init-weights` is supplied, encoder initialization is disabled before
   model construction; the supplied checkpoint will replace it.
4. The configured YAML is copied to `<output_dir>/config.yml`.
5. `select_device()` chooses CUDA, then Apple MPS, then CPU.
6. `build_dataset()` creates train and validation datasets.
7. PyTorch `DataLoader`s batch the data. Full-image validation uses batch size
   one because source tiles can have different spatial dimensions.
8. The model, loss, optimizer, optional scheduler, and CUDA gradient scaler are
   created.
9. Each epoch calls `train_one_epoch()` and `validate_one_epoch()`.
10. Epoch statistics are written to `logs/training_logs.json`.
11. `last_checkpoint.pth` is replaced each epoch. `best_checkpoint.pth` is
    replaced only when validation mIoU improves.

Supported optimizers are Adam and AdamW. The only implemented learning-rate
scheduler is cosine annealing; `none`, `constant`, or an omitted scheduler keeps
the learning rate fixed.

Checkpoint content is:

```python
{
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "metrics": validation_result,
    "config": config,
    "best_miou": best_miou,
}
```

`--init-weights` is a warm start: it loads model weights and creates a fresh
optimizer, epoch counter, log, and best-mIoU state. There is no full resume path
that restores the saved optimizer state.

### `evaluate.py` and `test.py`

`evaluate.py` reconstructs a model from a test config, disables pretrained
encoder initialization, loads a checkpoint, and runs `validate_one_epoch()` on
`train`, `val`, or `test`.

Project checkpoints contain their original model config. Before weight loading,
`check_architecture_matches()` compares model name, encoder name, and class
count, producing a readable error if the wrong test config is paired with a
checkpoint. External bare state dictionaries are validated later by strict
PyTorch weight loading.

`--tta` wraps the model in `FlipTTA`. It predicts the identity, horizontal
flip, vertical flip, and double flip, restores their orientations, and averages
the four logit maps. The output remains compatible with the configured loss and
with `argmax`.

Evaluation writes a JSON result containing the loss, aggregate and per-class
metrics, confusion matrix, checkpoint path, split, checkpoint epoch, and TTA
flag. `test.py` is only a wrapper that adds `--split test` when absent.

### `predict.py`

`predict.py` performs inference on one image or a directory. It uses the same
normalization selected by the config and supports:

- full-image inference with padding to a multiple of 32;
- overlapping sliding-window inference;
- four-way flip TTA;
- PNG or TIFF colorized masks;
- optional input/prediction panels.

Prediction does not compute metrics because it does not require reference
masks.

## 3. Configuration loading

`src/utils/config.py` uses `yaml.safe_load()` to parse the YAML file. Training
then validates the required top-level sections. Runtime behavior is driven by
four sections:

| Section | Main consumers | Purpose |
| --- | --- | --- |
| `experiment` | entry points | Run name and output directory |
| `dataset` | dataset factory/loaders | Paths, split files, taxonomy, crop, normalization, ignore behavior |
| `model` | model factory | Architecture, encoder, initialization, input/output channels |
| `training` | train/loss code | Batch, epochs, optimizer, LR, loss, AMP, workers |

The optional `metrics` list in YAML is descriptive. The validator currently
always computes OA, mIoU, mF1, per-class IoU/F1, class support, valid class
indices, and the confusion matrix.

## 4. Dataset construction and label harmonization

### `src/datasets/dataset_factory.py`

`build_dataset(config, split)` dispatches by `dataset.name`:

| Name | Class | Status |
| --- | --- | --- |
| `OpenEarthMap` | `OpenEarthMapDataset` | Implemented |
| `CEI` | `OpenEarthMapDataset` | Implemented; flat paths work because `<region>` is optional |
| `IRSA_Map` | `IRSADataset` | Implemented |
| `LoveDA` | none | Reserved; raises `NotImplementedError` |

### `src/datasets/taxonomy.py`

All training targets are zero-based class indices. `build_label_lut()` creates
a 256-entry lookup table so a complete mask is mapped in one NumPy operation.

The main seven-class CEI order is:

| Internal index | CEI disk ID | Class |
| ---: | ---: | --- |
| 0 | 1 | Rangeland |
| 1 | 2 | Agriculture |
| 2 | 3 | Tree |
| 3 | 4 | Water |
| 4 | 5 | Building |
| 5 | 6 | Road |
| 6 | 7 | Non-vegetated |

Available maps are `oem`, `oem_to_cei`, `cei`, and `irsa_to_cei`. In
`oem_to_cei`, OEM Bareland and Developed space both become CEI Non-vegetated.
In `irsa_to_cei`, IRSA two-digit thematic codes are projected into the same CEI
order. All unsupported pixels become `ignore_index` (normally 255), except IRSA
raw background 0, which is intentionally mapped to Non-vegetated.

### `src/datasets/openearthmap_dataset.py`

This loader serves OEM's region hierarchy and CEI's flat hierarchy.

For each split entry, it:

1. resolves image and mask paths, replacing `<region>` when present;
2. reads the image and mask with OpenCV;
3. converts BGR image data to RGB;
4. reduces a multi-channel mask to its first channel if necessary;
5. rejects raw mask values not allowed by the configured label map;
6. maps raw values to internal indices with the taxonomy lookup table;
7. applies synchronized image/mask transformations;
8. returns a float image tensor and a `torch.long` target tensor.

This class contains its own transform builder. Its policy matches the shared
builder used by IRSA: pad before cropping, optional flips/rotations for training,
and deterministic full-image or center-crop evaluation.

### `src/datasets/irsa_dataset.py`

`IRSADataset` handles IRSA's separate train/test directory trees. It delegates
non-I/O behavior to injectable collaborators:

- `LutLabelMapper` validates and maps raw codes;
- `NodataMasker` changes near-black border pixels to `ignore_index`;
- `build_segmentation_transforms` builds the split-specific transform pipeline.

The defaults are `label_map: irsa_to_cei` and `nodata_to_ignore: 8`. Thus, raw
IRSA background represents Non-vegetated ground unless the maximum RGB channel
is at most 8, in which case it is treated as nodata.

### `src/datasets/transforms.py`

Two normalization modes exist:

- `imagenet`: `(x / 255 - mean) / std` with ImageNet RGB statistics;
- `zero_one`: `x / 255`, required by the external OEM-SAR baseline.

For training, images are padded to at least `crop_size`, randomly cropped, and,
when `dataset.augment` is true, independently considered for horizontal flip,
vertical flip, and 90-degree rotation (each `p=0.5`). Image and mask geometric
operations are synchronized. For full evaluation, the image is padded to a
multiple of 32; mask padding is 255 and therefore excluded from loss/metrics.

## 5. Model construction

### `src/models/model_factory.py`

`build_model()` supports:

| Config name | Implementation | Key behavior |
| --- | --- | --- |
| `unet` / `u-net` | SMP U-Net | Configurable encoder and optional SCSE attention |
| `deeplabv3` | SMP DeepLabV3 | Configurable encoder |
| `upernet` | SMP UPerNet | Accepts timm encoders through `tu-...` names |
| `segformer` | SMP SegFormer | Used with MiT-B5 in the main comparison |
| `unetformer` | Local `UNetFormer` | timm CNN encoder; main and auxiliary outputs during training |
| `ftunetformer` aliases | Local `FTUNetFormer` | Fixed Swin-B encoder; one output |

Extra `model.encoder_params` are forwarded to SMP. This is used to set Swin's
input size for UPerNet. `dataset.num_classes` must equal `model.num_classes`.

`LeadingChannelDrop` adapts the external nine-channel OEM-SAR U-Net. The
external channel 0 is background; dropping it aligns the remaining eight logits
with this project's native OEM class indices.

### `src/models/unetformer.py`

UNetFormer obtains four encoder feature maps from a timm backbone. Its decoder
combines Global-Local Transformer Blocks, learned weighted skip fusion, and a
feature-refinement head. In training mode it returns `(main, auxiliary)`; in
evaluation mode it returns only the main full-resolution logit tensor.

### `src/models/ftunetformer.py`

FT-UNetFormer replaces the CNN encoder with an in-repository Swin-B
implementation and uses the same decoder family at a wider channel dimension.
When pretrained initialization is requested, timm Swin-B ImageNet weights are
mapped into the compatible backbone tensors. It always returns one output.

## 6. Loss and optimization

`src/losses/loss_factory.py` builds one of:

- cross-entropy;
- multiclass Dice;
- cross-entropy plus Dice;
- focal loss;
- focal plus Dice.

Every loss honors `ignore_index`. Cross-entropy-based options can receive a
`training.class_weights` list whose length must match `dataset.num_classes`.

`src/engine/trainer.py` switches the model to training mode and, for every
batch, moves tensors to the device, clears gradients, computes the forward loss,
backpropagates, and updates the optimizer. CUDA AMP is used only when requested,
CUDA is active, and a gradient scaler exists.

For a tuple/list model output, `compute_loss()` applies deep supervision:

```text
training loss = loss(main, target) + aux_weight * loss(aux, target)
```

The current UNetFormer config uses `aux_weight: 0.4`.

## 7. Validation and metrics

`src/engine/validator.py` uses evaluation mode and disables gradients. For each
batch it computes loss, converts logits to class predictions with `argmax` over
the channel dimension, and updates one dataset-level confusion matrix.

`src/metrics/segmentation_metrics.py` flattens predictions and targets, removes
ignored or invalid pixels, and accumulates an `N x N` matrix where rows are
reference classes and columns are predicted classes. It derives:

- OA from all diagonal pixels divided by all valid pixels;
- per-class IoU from `TP / (TP + FP + FN)`;
- per-class F1 from `2TP / (2TP + FP + FN)`;
- mIoU and mF1 as unweighted means across defined classes.

Undefined per-class values are serialized as JSON `null`. Class support is the
row sum of the confusion matrix.

## 8. Checkpoint loading

`src/models/checkpoint.py` accepts either a project checkpoint containing
`model_state_dict`, a dictionary containing `state_dict`, or a bare state
dictionary. Loading is strict. For a `LeadingChannelDrop` model, it first tries
the wrapper and then its `inner` model to support both prefixed project weights
and unprefixed external weights.

## 9. Current experiment routes

The main OEM comparison is configured under `configs/cei_oem/train/` and has
matching CEI test configs under `configs/cei_oem/test/`. The IRSA comparison is
under `configs/cei_irsa/`.

```text
OEM raw labels --oem_to_cei--> seven-class training --> OEM validation
                                                   \-> CEI test

IRSA raw labels --irsa_to_cei--> seven-class training --> IRSA test
                                                    \-> CEI test
```

This shared output taxonomy makes the OEM-trained and IRSA-trained models
directly comparable on the same CEI test set.
