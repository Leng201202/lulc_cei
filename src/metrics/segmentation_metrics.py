import numpy as np


def _nan_to_none(values):
    return [None if np.isnan(value) else float(value) for value in values]


class SegmentationMetrics:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes),
            dtype=np.int64
        )

    def update(self, preds, targets):
        """
        preds:   numpy array or torch tensor, shape [B, H, W]
        targets: numpy array or torch tensor, shape [B, H, W]
        """

        if hasattr(preds, "detach"):
            preds = preds.detach().cpu().numpy()

        if hasattr(targets, "detach"):
            targets = targets.detach().cpu().numpy()

        preds = preds.reshape(-1)
        targets = targets.reshape(-1)

        if preds.shape[0] != targets.shape[0]:
            raise ValueError(
                "Predictions and targets must have the same number of pixels. "
                f"Got {preds.shape[0]} predictions and {targets.shape[0]} targets."
            )

        valid_mask = targets != self.ignore_index

        preds = preds[valid_mask]
        targets = targets[valid_mask]

        valid_class_mask = (
            (targets >= 0) & (targets < self.num_classes) &
            (preds >= 0) & (preds < self.num_classes)
        )
        preds = preds[valid_class_mask]
        targets = targets[valid_class_mask]

        indices = self.num_classes * targets + preds

        cm = np.bincount(
            indices,
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

        self.confusion_matrix += cm

    def compute(self):
        cm = self.confusion_matrix

        true_positive = np.diag(cm)
        false_positive = cm.sum(axis=0) - true_positive
        false_negative = cm.sum(axis=1) - true_positive

        total_correct = true_positive.sum()
        total_pixels = cm.sum()

        overall_accuracy = total_correct / total_pixels if total_pixels > 0 else np.nan

        iou_denominator = true_positive + false_positive + false_negative
        f1_denominator = 2 * true_positive + false_positive + false_negative

        iou = np.full(self.num_classes, np.nan, dtype=np.float64)
        f1 = np.full(self.num_classes, np.nan, dtype=np.float64)

        present_iou = iou_denominator > 0
        present_f1 = f1_denominator > 0

        iou[present_iou] = true_positive[present_iou] / iou_denominator[present_iou]
        f1[present_f1] = (2 * true_positive[present_f1]) / f1_denominator[present_f1]

        miou = np.nanmean(iou) if np.any(present_iou) else np.nan
        mf1 = np.nanmean(f1) if np.any(present_f1) else np.nan

        return {
            "OA": float(overall_accuracy) if not np.isnan(overall_accuracy) else None,
            "mIoU": float(miou) if not np.isnan(miou) else None,
            "mF1": float(mf1) if not np.isnan(mf1) else None,
            "per_class_iou": _nan_to_none(iou),
            "per_class_f1": _nan_to_none(f1),
            "class_support": cm.sum(axis=1).tolist(),
            "valid_iou_classes": np.where(present_iou)[0].tolist(),
            "confusion_matrix": cm.tolist(),
        }
