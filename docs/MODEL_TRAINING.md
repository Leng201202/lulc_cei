# Model Training

## Starting a run

Run training from the repository root:

```bash
python train.py --config configs/cei_oem/train/unet_effb4_oem2cei.yml
```

To initialize from an existing project checkpoint or bare state dictionary:

```bash
python train.py \
  --config path/to/train_config.yml \
  --init-weights path/to/weights.pth
```

`--init-weights` performs a warm start, not a resume. Model weights are loaded,
but optimizer state, prior epoch number, scheduler state, log history, and prior
best mIoU are not restored.

## Initialization

Before the epoch loop, `train.py`:

1. parses and validates the config;
2. creates `<output_dir>/checkpoints` and `<output_dir>/logs`;
3. copies the supplied YAML to `<output_dir>/config.yml`;
4. selects CUDA, then MPS, then CPU;
5. builds train and validation datasets and loaders;
6. constructs the model and optionally loads warm-start weights;
7. builds and moves the loss to the device;
8. constructs Adam or AdamW and an optional cosine scheduler;
9. creates a CUDA gradient scaler when mixed precision is enabled.

Training batches are shuffled. CUDA loaders use pinned host memory. The number
of loader workers comes from `training.num_workers`, defaulting to zero.

## One training step

For a batch of images `x`, target masks `y`, and model `f`, the standard path is:

```text
optimizer.zero_grad()
logits = f(x)
loss = criterion(logits, y)
loss.backward()
optimizer.step()
```

Targets are `torch.long` tensors containing class indices. The model output is
an unnormalized logit tensor; softmax is not applied before the configured
loss.

When CUDA AMP is active, the forward pass and loss run inside float16 autocast,
and `GradScaler` scales the loss, applies a safe optimizer step, and updates its
scale. Setting `mix_precision: true` on MPS or CPU does not activate this CUDA
AMP branch; training falls back to ordinary precision.

## Deep supervision

UNetFormer returns `(main_logits, auxiliary_logits)` in training mode. Its batch
loss is:

```text
L = criterion(main_logits, target)
  + auxiliary_weight * criterion(auxiliary_logits, target)
```

The default auxiliary weight is 0.4, and the active UNetFormer config states it
explicitly. Models returning one tensor use only the ordinary criterion.

## Available losses

`training.loss` selects:

| Value | Objective | Optional settings |
| --- | --- | --- |
| `cross_entropy` | pixelwise multiclass cross-entropy | `class_weights` |
| `dice` | multiclass Dice loss | none |
| `ce_dice` | cross-entropy + Dice | `class_weights` affects CE |
| `focal` | multiclass focal loss | `focal_gamma`, default 2.0 |
| `focal_dice` | focal + Dice | `focal_gamma`, default 2.0 |

All objectives exclude target 255. `training.class_weights`, when present, must
contain exactly one numeric value per output class. The current OEM-to-CEI and
IRSA-to-CEI experiment configs use unweighted cross-entropy.

## Optimizer and learning-rate schedule

The optimizer configuration is read as:

```yaml
training:
  optimizer: AdamW       # or Adam
  learning_rate: 0.0001
  weight_decay: 0.000001
```

Supported scheduler settings are:

```yaml
scheduler: none          # also omitted or constant
```

or:

```yaml
scheduler: cosine
min_learning_rate: 0.0
```

Cosine annealing steps once after each validation epoch. The logged learning
rate is the rate used during that epoch, captured before the scheduler step.
The active primary configs use no scheduler.

## Epoch lifecycle

Each epoch performs:

```text
train_one_epoch
  -> average unweighted batch loss

validate_one_epoch
  -> average unweighted batch loss
  -> one accumulated confusion matrix
  -> OA, mIoU, mF1, per-class metrics

append JSON log
save last checkpoint
if validation mIoU improved: save best checkpoint
```

The training and validation loss values are arithmetic means of batch loss
values. They are not reweighted by batch size or number of valid pixels. The
confusion-matrix metrics, by contrast, aggregate all valid pixels before metric
calculation.

## Outputs

A typical run creates:

```text
experiments/<run>/
├── config.yml
├── checkpoints/
│   ├── last_checkpoint.pth
│   └── best_checkpoint.pth
└── logs/
    └── training_logs.json
```

Every log item includes epoch, learning rate, train/validation loss, validation
OA/mIoU/mF1, per-class IoU/F1, and class support. The full confusion matrix is
stored inside checkpoint metrics but is not copied into each training log item.

`last_checkpoint.pth` is written after every epoch. `best_checkpoint.pth` is
written when validation mIoU strictly exceeds the previous best value.

## Recommended run checks

Before a long run:

- verify that `dataset.num_classes` and `model.num_classes` agree;
- inspect label values and class distribution with the data utilities;
- confirm that normalization matches the initialization/checkpoint;
- confirm Swin `img_size` matches the training crop where required;
- run the corresponding smoke config under `configs/cei_oem/smoke/`;
- ensure the output directory belongs only to this run, because config, logs,
  and last/best checkpoint filenames are replaced in place.

During a run, compare training loss against validation loss and mIoU. Falling
training loss with stagnant or falling validation mIoU suggests overfitting;
overall accuracy alone can conceal failure on minority classes.

## Current limitations relevant to training

- There is no full checkpoint-resume command.
- Scheduler state is not stored in checkpoints.
- Global random seeds and deterministic backend settings are not configured.
- There is no early stopping or gradient clipping.
- The YAML `metrics` list does not alter validation calculations.

These points do not prevent training, but they should be stated when reporting
reproducibility or comparing interrupted and uninterrupted runs.
