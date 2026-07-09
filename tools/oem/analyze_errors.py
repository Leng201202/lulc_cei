"""Per-class error analysis for a trained OpenEarthMap model.

Runs the model over a split and reports a row-normalized confusion matrix
(what each true class gets predicted as), then saves image/GT/prediction/error
panels for the tiles where the requested "focus" classes are most mispredicted.

Usage:
    python tools/oem/analyze_errors.py \
        --config configs/unet/unet_effb4_oem.yml \
        --checkpoint experiments/exp_01_unet_effb4_oem/checkpoints/best_checkpoint.pth
"""

import argparse
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.dataset_factory import build_dataset
from src.models.checkpoint import load_model_weights
from src.models.model_factory import build_model
from src.utils.config import load_config
from src.utils.visualization import OEM_CLASS_NAMES, decode_mask, denormalize_image
from train import select_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--focus", nargs="+", default=[0, 2, 1], type=int,
                        help="Class indices to save error panels for "
                             "(default: bareland=0, dev_surface=2, rangeland=1).")
    parser.add_argument("--panels-per-class", type=int, default=4)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config["model"]["encoder_weights"] = None
    num_classes = config["dataset"]["num_classes"]
    ignore_index = config["dataset"]["ignore_index"]

    device = select_device()
    out_dir = args.output or os.path.join(
        config["experiment"]["output_dir"], "analysis"
    )
    os.makedirs(out_dir, exist_ok=True)

    dataset = build_dataset(config, split=args.split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = build_model(config).to(device)
    load_model_weights(model, torch.load(args.checkpoint, map_location=device))
    model.eval()

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    # Per-sample: (error-count for each focus class) so we can rank worst tiles.
    focus = args.focus
    worst = {c: [] for c in focus}

    with torch.no_grad():
        for idx, (image, mask) in enumerate(loader):
            logits = model(image.to(device))
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy()
            gt = mask[0].numpy()

            valid = gt != ignore_index
            g = gt[valid].astype(np.int64)
            p = pred[valid].astype(np.int64)
            np.add.at(confusion, (g, p), 1)

            for c in focus:
                present = int((gt == c).sum())
                if present == 0:
                    continue
                wrong = int(((gt == c) & (pred != c)).sum())
                worst[c].append((wrong / present, present, idx))

    # ---- Confusion report (row-normalized: P(pred | true)) ----
    row_sum = confusion.sum(axis=1, keepdims=True).clip(min=1)
    norm = confusion / row_sum
    print("\nRow-normalized confusion  P(pred | true), split =", args.split)
    print("true \\ pred  " + " ".join(f"{n[:6]:>6s}" for n in OEM_CLASS_NAMES))
    for i, name in enumerate(OEM_CLASS_NAMES):
        row = " ".join(f"{norm[i, j]*100:6.1f}" for j in range(num_classes))
        print(f"{name[:11]:11s} {row}")

    print("\nTop confusions for focus classes:")
    for c in focus:
        order = np.argsort(-norm[c])
        conf = [f"{OEM_CLASS_NAMES[j]} {norm[c, j]*100:.0f}%"
                for j in order if j != c][:3]
        recall = norm[c, c] * 100
        print(f"  {OEM_CLASS_NAMES[c]:16s} recall {recall:4.1f}%  ->  "
              + ", ".join(conf))

    # ---- Error panels for the worst tiles of each focus class ----
    for c in focus:
        worst[c].sort(reverse=True)
        for rank, (frac, present, idx) in enumerate(worst[c][:args.panels_per_class]):
            image, mask = dataset[idx]
            with torch.no_grad():
                logits = model(image.unsqueeze(0).to(device))
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy()
            gt = mask.numpy()

            rgb = denormalize_image(image)
            gt_rgb = decode_mask(gt, ignore_index)
            pred_rgb = decode_mask(pred, ignore_index)

            # Error map: green = correct, red = wrong, black = ignore.
            err = np.zeros((*gt.shape, 3), dtype=np.uint8)
            valid = gt != ignore_index
            err[valid & (pred == gt)] = (40, 160, 40)
            err[valid & (pred != gt)] = (220, 40, 40)

            panel = np.concatenate([rgb, gt_rgb, pred_rgb, err], axis=1)
            name = OEM_CLASS_NAMES[c].replace(" ", "_").lower()
            fname = os.path.join(
                out_dir, f"err_{name}_{rank+1}_{dataset.file_names[idx]}.png"
            )
            cv2.imwrite(fname, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    print(f"\nPanels + report saved under: {out_dir}")
    print("Panel layout: [ image | ground truth | prediction | error map ]")


if __name__ == "__main__":
    main()
