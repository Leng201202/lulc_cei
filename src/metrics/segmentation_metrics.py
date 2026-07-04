import numpy as np


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

        valid_mask = targets != self.ignore_index

        preds = preds[valid_mask]
        targets = targets[valid_mask]

        valid_class_mask = (targets >= 0) & (targets < self.num_classes)
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

        overall_accuracy = total_correct / (total_pixels + 1e-10)

        iou = true_positive / (
            true_positive + false_positive + false_negative + 1e-10
        )

        f1 = (2 * true_positive) / (
            2 * true_positive + false_positive + false_negative + 1e-10
        )

        miou = np.nanmean(iou)
        mf1 = np.nanmean(f1)

        return {
            "OA": float(overall_accuracy),
            "mIoU": float(miou),
            "mF1": float(mf1),
            "per_class_iou": iou.tolist(),
            "per_class_f1": f1.tolist(),
            "confusion_matrix": cm.tolist(),
        }