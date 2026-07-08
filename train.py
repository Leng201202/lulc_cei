import argparse
import json
import os
import shutil

import torch
from torch.utils.data import DataLoader

from src.datasets.dataset_factory import build_dataset
from src.engine.trainer import train_one_epoch
from src.engine.validator import validate_one_epoch
from src.losses.loss_factory import build_loss
from src.models.model_factory import build_model
from src.utils.config import load_config


def create_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_config(config):
    for section in ["experiment", "dataset", "model", "training"]:
        if section not in config:
            raise KeyError(f"Missing config section: {section}")

    dataset_config = config["dataset"]
    model_config = config["model"]
    training_config = config["training"]

    if dataset_config["num_classes"] != model_config["num_classes"]:
        raise ValueError(
            "dataset.num_classes and model.num_classes must match. "
            f"Got {dataset_config['num_classes']} and {model_config['num_classes']}."
        )

    weight_decay = training_config.get("weight_decay", 0.0)
    if weight_decay is None:
        training_config["weight_decay"] = 0.0
    elif isinstance(weight_decay, str):
        raise TypeError(
            "training.weight_decay must be numeric or null. "
            f"Got string value {weight_decay!r}."
        )


def build_optimizer(model, training_config):
    optimizer_name = training_config["optimizer"].lower()
    learning_rate = float(training_config["learning_rate"])
    weight_decay = float(training_config.get("weight_decay", 0.0) or 0.0)

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unknown optimizer: {training_config['optimizer']}")


def save_checkpoint(model, optimizer, epoch, metrics, path, config, best_miou):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
        "best_miou": best_miou,
    }
    torch.save(checkpoint, path)


def format_metric(value):
    return "nan" if value is None else f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    validate_config(config)

    experiment_config = config["experiment"]
    dataset_config = config["dataset"]
    training_config = config["training"]

    output_dir = experiment_config["output_dir"]
    create_output_dir(output_dir)

    shutil.copy(args.config, os.path.join(output_dir, "config.yml"))

    device = select_device()
    print(f"Using device: {device}")

    train_dataset = build_dataset(config, split="train")
    val_dataset = build_dataset(config, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config.get("num_workers", 0),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config.get("num_workers", 0),
        pin_memory=device.type == "cuda",
    )

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    model = build_model(config).to(device)
    criterion = build_loss(config)
    optimizer = build_optimizer(model, training_config)

    mixed_precision = training_config.get("mix_precision", False)
    scaler = None
    if mixed_precision and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    best_miou = -1.0
    logs = []

    epochs = training_config["epochs"]
    num_classes = dataset_config["num_classes"]
    ignore_index = dataset_config["ignore_index"]

    for epoch in range(1, epochs + 1):
        print("=" * 60)
        print(f"Epoch {epoch}/{epochs}")
        print("=" * 60)

        train_loss = train_one_epoch(
            model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            mixed_precision=mixed_precision,
        )
        val_result = validate_one_epoch(
            model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        val_loss = val_result["loss"]
        val_oa = val_result["OA"]
        val_miou = val_result["mIoU"]
        val_f1 = val_result["mF1"]

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation OA: {format_metric(val_oa)}")
        print(f"Validation mIoU: {format_metric(val_miou)}")
        print(f"Validation mF1: {format_metric(val_f1)}")

        log_item = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_OA": val_oa,
            "val_mIoU": val_miou,
            "val_mF1": val_f1,
            "per_class_iou": val_result["per_class_iou"],
            "per_class_f1": val_result["per_class_f1"],
            "class_support": val_result["class_support"],
        }
        logs.append(log_item)

        log_path = os.path.join(output_dir, "logs", "training_logs.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4)

        is_best = val_miou is not None and val_miou > best_miou
        if is_best:
            best_miou = val_miou

        last_checkpoint_path = os.path.join(
            output_dir,
            "checkpoints",
            "last_checkpoint.pth",
        )
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=val_result,
            path=last_checkpoint_path,
            config=config,
            best_miou=best_miou,
        )

        if is_best:
            best_checkpoint_path = os.path.join(
                output_dir,
                "checkpoints",
                "best_checkpoint.pth",
            )
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_result,
                path=best_checkpoint_path,
                config=config,
                best_miou=best_miou,
            )
            print(f"Best checkpoint saved at epoch {epoch} with mIoU: {best_miou:.4f}")

    print("=" * 60)
    print("Training completed.")
    print(f"Best mIoU: {best_miou:.4f}")
    print(f"Logs saved to: {log_path}")
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
