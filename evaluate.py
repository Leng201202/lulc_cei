import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.dataset_factory import build_dataset
from src.engine.validator import validate_one_epoch
from src.losses.loss_factory import build_loss
from src.models.checkpoint import load_model_weights
from src.models.model_factory import build_model
from src.utils.config import load_config
from train import select_device, validate_config


class FlipTTA(nn.Module):
    """Test-time augmentation over the four horizontal/vertical flips.

    Each flipped input is run through the model and its logits flipped back to
    the original orientation, then all four are averaged. Averaging in logit
    space keeps the output a valid logit map, so the loss and argmax used
    downstream behave exactly as for a single forward pass.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        identity = self.model(x)
        horizontal = torch.flip(self.model(torch.flip(x, dims=[3])), dims=[3])
        vertical = torch.flip(self.model(torch.flip(x, dims=[2])), dims=[2])
        both = torch.flip(self.model(torch.flip(x, dims=[2, 3])), dims=[2, 3])
        return (identity + horizontal + vertical + both) / 4.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Average predictions over horizontal/vertical flips (4-way TTA). "
             "Slower but usually a small metric gain.",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    validate_config(config)

    # Evaluation loads trained weights, so pretrained encoder initialization is
    # unnecessary and may trigger an avoidable network/cache dependency.
    config["model"]["encoder_weights"] = None

    dataset_config = config["dataset"]
    training_config = config["training"]

    device = select_device()
    print(f"Using device: {device}")

    dataset = build_dataset(config, split=args.split)
    # Full-image evaluation yields variable-size tensors that cannot be stacked,
    # so evaluate one image at a time; crop-mode eval can use the full batch.
    eval_batch_size = 1 if dataset_config.get("eval_mode", "full") == "full" \
        else training_config["batch_size"]
    dataloader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=training_config.get("num_workers", 0),
        pin_memory=device.type == "cuda",
    )

    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    load_model_weights(model, checkpoint)

    if args.tta:
        model = FlipTTA(model)
        print("Test-time augmentation: 4-way flip enabled")

    criterion = build_loss(config).to(device)
    result = validate_one_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        num_classes=dataset_config["num_classes"],
        ignore_index=dataset_config["ignore_index"],
    )
    result["checkpoint"] = args.checkpoint
    result["split"] = args.split
    result["epoch"] = checkpoint.get("epoch")
    result["tta"] = args.tta

    output_path = args.output
    if output_path is None:
        output_dir = config["experiment"]["output_dir"]
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        output_path = os.path.join(output_dir, "logs", f"{args.split}_metrics.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"{args.split} loss: {result['loss']:.4f}")
    print(f"{args.split} OA: {result['OA']}")
    print(f"{args.split} mIoU: {result['mIoU']}")
    print(f"{args.split} mF1: {result['mF1']}")
    print(f"Metrics saved to: {output_path}")


if __name__ == "__main__":
    main()
