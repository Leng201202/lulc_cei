import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from src.datasets.dataset_factory import build_dataset
from src.engine.validator import validate_one_epoch
from src.losses.loss_factory import build_loss
from src.models.model_factory import build_model
from src.utils.config import load_config
from train import select_device, validate_config


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain model_state_dict: {path}")
    return checkpoint


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
    dataloader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config.get("num_workers", 0),
        pin_memory=device.type == "cuda",
    )

    model = build_model(config).to(device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = build_loss(config)
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
