# Evaluation Metrics

## Evaluation flow

`evaluate.py` rebuilds the configured architecture, strictly loads a checkpoint,
and passes the selected dataset split to `validate_one_epoch()`.

For each image or batch:

```text
logits [B, C, H, W]
  -> configured loss against target [B, H, W]
  -> argmax over C
  -> prediction [B, H, W]
  -> update dataset-level confusion matrix
```

Full-image mode uses batch size one because image dimensions can vary. The
final segmentation metrics are calculated once from all valid pixels, not
averaged image by image.

## Valid pixels and confusion matrix

Let `M[i, j]` be the number of pixels whose reference class is `i` and predicted
class is `j`. Rows are reference classes; columns are predictions.

Before accumulation, the metric code removes:

- targets equal to `ignore_index` (normally 255);
- targets outside `[0, num_classes - 1]`;
- predictions outside `[0, num_classes - 1]`.

Padding added for full-image compatibility is labeled 255, so it cannot improve
or reduce any reported metric.

For class `c`:

```text
TP_c = M[c, c]
FP_c = sum_i M[i, c] - TP_c
FN_c = sum_j M[c, j] - TP_c
support_c = sum_j M[c, j]
```

Class support is therefore the number of valid reference pixels for that class.

## Overall accuracy

Overall accuracy (OA) is:

```text
OA = sum_c TP_c / sum_i sum_j M[i, j]
```

OA answers: “What fraction of all valid pixels was labeled correctly?” It is
easy to understand but frequency-weighted. A dominant class can make OA high
even when small or rare classes perform poorly.

## Intersection over Union

Per-class Intersection over Union is:

```text
IoU_c = TP_c / (TP_c + FP_c + FN_c)
```

IoU penalizes both false alarms and missed pixels. Mean IoU is the unweighted
mean over classes for which the denominator is nonzero:

```text
mIoU = mean_c(IoU_c)
```

Because every defined class contributes equally, mIoU is the project's primary
model-selection metric. `train.py` saves a new best checkpoint only when
validation mIoU strictly improves.

## F1 score

Per-class F1 is:

```text
F1_c = 2 TP_c / (2 TP_c + FP_c + FN_c)
```

It is the harmonic mean of precision and recall. Mean F1 is an unweighted mean
over defined classes. For the same confusion counts, F1 and IoU are related by
`F1 = 2*IoU/(1+IoU)`, so F1 is numerically higher while ranking overlap in the
same order.

## Absent classes

When a class has neither reference nor predicted pixels, its IoU/F1 denominator
is zero. The implementation records that class as undefined (`null` in JSON)
and excludes it from `np.nanmean`. `valid_iou_classes` lists the class indices
included in mIoU.

A class absent from reference data but predicted by the model has false
positives, a nonzero denominator, and an IoU/F1 of zero. This is intentionally
penalized.

## Loss versus metrics

The configured loss is also reported, but it serves a different purpose:

- loss uses the full logit distribution and reflects confidence;
- OA, IoU, and F1 use the final `argmax` class only;
- loss is averaged over batches by the validator;
- metrics are computed from all accumulated valid pixel counts.

Consequently, lower loss usually but does not necessarily imply higher mIoU.
Use loss to diagnose optimization and mIoU/per-class scores to compare final
segmentation quality.

## Test-time augmentation

With `--tta`, evaluation predicts four orientations: original, horizontal
flip, vertical flip, and both flips. Flipped logits are restored to the source
orientation and averaged before loss and `argmax`.

TTA changes the inference procedure and must be reported alongside results. Do
not compare a TTA result with a non-TTA result as though only model weights
differed.

## Output schema

By default, evaluation writes
`<experiment.output_dir>/logs/<split>_metrics.json`. It contains:

```json
{
  "OA": 0.0,
  "mIoU": 0.0,
  "mF1": 0.0,
  "per_class_iou": [],
  "per_class_f1": [],
  "class_support": [],
  "valid_iou_classes": [],
  "confusion_matrix": [],
  "loss": 0.0,
  "checkpoint": "...",
  "split": "test",
  "epoch": 0,
  "tta": false
}
```

The numeric values above are placeholders illustrating the schema.

For the seven-class CEI experiments, array order is always:

1. Rangeland
2. Agriculture
3. Tree
4. Water
5. Building
6. Road
7. Non-vegetated

## Reporting recommendations

At minimum, report checkpoint epoch, source training domain, test domain, split,
full-image versus crop evaluation, TTA setting, OA, mIoU, mF1, all per-class IoU
values, and class support. Include the confusion matrix when discussing which
classes are confused.

For cross-domain analysis, compare:

- source-domain validation/test performance;
- the same checkpoint on CEI;
- the gap between them;
- class-specific changes, especially those affected by taxonomy merging or
  source-domain class imbalance.

The OEM-trained and IRSA-trained models are directly comparable on CEI only
because both loaders map their raw labels into the identical seven-class output
order before training.
