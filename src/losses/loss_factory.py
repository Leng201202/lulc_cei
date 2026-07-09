import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class CEDiceLoss(nn.Module):
    def __init__(self, ignore_index=255, weight=None):
        super().__init__()

        # ``weight`` is registered inside CrossEntropyLoss as a buffer, so a
        # later criterion.to(device) moves it alongside the module.
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weight)

        self.dice_loss = smp.losses.DiceLoss(
            mode="multiclass",
            ignore_index=ignore_index
        )

    def forward(self, outputs, masks):
        ce = self.ce_loss(outputs, masks)
        dice = self.dice_loss(outputs, masks)

        return ce + dice


class FocalDiceLoss(nn.Module):
    """Focal loss + Dice. Focal down-weights easy pixels, so it helps rare or
    frequently-misclassified classes without needing explicit class weights."""

    def __init__(self, ignore_index=255, gamma=2.0):
        super().__init__()
        self.focal_loss = smp.losses.FocalLoss(
            mode="multiclass",
            ignore_index=ignore_index,
            gamma=gamma,
        )
        self.dice_loss = smp.losses.DiceLoss(
            mode="multiclass",
            ignore_index=ignore_index,
        )

    def forward(self, outputs, masks):
        return self.focal_loss(outputs, masks) + self.dice_loss(outputs, masks)


def _build_class_weights(config):
    """Return a class-weight tensor from ``training.class_weights`` or None.

    Accepts a list of per-class floats (length must equal num_classes). The
    tensor is created on CPU; move the criterion to the device after building.
    """
    weights = config["training"].get("class_weights")
    if weights is None:
        return None

    tensor = torch.tensor([float(w) for w in weights], dtype=torch.float32)
    expected = config["dataset"]["num_classes"]
    if tensor.numel() != expected:
        raise ValueError(
            f"training.class_weights has {tensor.numel()} entries "
            f"but dataset.num_classes is {expected}."
        )
    return tensor


def build_loss(config):
    training_config = config["training"]
    dataset_config = config["dataset"]

    loss_name = training_config["loss"]
    ignore_index = dataset_config["ignore_index"]

    class_weights = _build_class_weights(config)

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)

    elif loss_name == "dice":
        return smp.losses.DiceLoss(
            mode="multiclass",
            ignore_index=ignore_index
        )

    elif loss_name == "ce_dice":
        return CEDiceLoss(ignore_index=ignore_index, weight=class_weights)

    elif loss_name == "focal":
        return smp.losses.FocalLoss(
            mode="multiclass",
            ignore_index=ignore_index,
            gamma=training_config.get("focal_gamma", 2.0),
        )

    elif loss_name == "focal_dice":
        return FocalDiceLoss(
            ignore_index=ignore_index,
            gamma=training_config.get("focal_gamma", 2.0),
        )

    else:
        raise ValueError(f"Unknown loss name: {loss_name}")
