import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class CEDiceLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.dice_loss = smp.losses.DiceLoss(
            mode="multiclass",
            ignore_index=ignore_index
        )

    def forward(self, outputs, masks):
        ce = self.ce_loss(outputs, masks)
        dice = self.dice_loss(outputs, masks)

        return ce + dice


def build_loss(config):
    training_config = config["training"]
    dataset_config = config["dataset"]

    loss_name = training_config["loss"]
    ignore_index = dataset_config["ignore_index"]

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(ignore_index=ignore_index)

    elif loss_name == "dice":
        return smp.losses.DiceLoss(
            mode="multiclass",
            ignore_index=ignore_index
        )

    elif loss_name == "ce_dice":
        return CEDiceLoss(ignore_index=ignore_index)

    else:
        raise ValueError(f"Unknown loss name: {loss_name}")