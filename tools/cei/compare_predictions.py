"""Predict CEI tiles with a trained model and compare against the labels.

Produces image / ground-truth / prediction panels in the CEI palette, so the
model's output can be eyeballed against the hand-drawn labels. Each tile is
scored (per-tile mIoU over the classes present in its label), and by default the
tiles are ordered worst-first -- the failures are what you actually want to look
at.

Full-image inference at the tile's native resolution, optionally with 4-way flip
TTA to match how evaluate.py scores. The model and its checkpoint must agree; the
checkpoint's stored config records which architecture produced it.

Usage
-----
python tools/cei/compare_predictions.py \
    --config configs/cei_oem/test/test_m1_ftunetformer_cei.yml \
    --checkpoint experiments/cei_m1_ftunetformer_swinb/checkpoints/best_checkpoint.pth \
    --tta --num 12 --out experiments/cei_m1_ftunetformer_swinb/logs/cei_compare.png

    --order best      show the strongest tiles instead of the weakest
    --tiles 2,8,31    show specific tiles regardless of score
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.datasets.dataset_factory import build_dataset          # noqa: E402
from src.datasets.taxonomy import CEI_CLASS_COLORS, CEI_CLASS_NAMES  # noqa: E402
from src.models.checkpoint import load_model_weights            # noqa: E402
from src.models.model_factory import build_model                # noqa: E402
from src.utils.config import load_config                        # noqa: E402

IGNORE_INDEX = 255
LABEL_HEIGHT = 26


def decode(mask):
    """Map a class-index mask to a BGR color image in the CEI palette."""
    rgb = np.zeros((*mask.shape, 3), np.uint8)
    for index, color in enumerate(CEI_CLASS_COLORS):
        rgb[mask == index] = color
    rgb[mask == IGNORE_INDEX] = (0, 0, 0)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def denormalize(tensor):
    """Undo ImageNet normalization back to a BGR uint8 image for display."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    array = tensor.permute(1, 2, 0).cpu().numpy() * std + mean
    array = np.clip(array * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def per_tile_miou(prediction, target, num_classes):
    """Mean IoU over the classes actually present in this tile's label.

    Restricting to present classes keeps the score meaningful per tile -- a class
    absent from both the label and the prediction should not count as a perfect
    or a zero IoU and drag the tile's number around.
    """
    valid = target != IGNORE_INDEX
    prediction, target = prediction[valid], target[valid]
    ious = []
    for class_index in np.unique(target):
        p = prediction == class_index
        t = target == class_index
        union = np.logical_or(p, t).sum()
        if union:
            ious.append(np.logical_and(p, t).sum() / union)
    return float(np.mean(ious)) if ious else float("nan")


@torch.no_grad()
def predict(model, image, device, tta):
    logits = model(image.unsqueeze(0).to(device))
    if tta:
        for dims in ([3], [2], [2, 3]):
            flipped = torch.flip(model(torch.flip(image.unsqueeze(0).to(device), dims=dims)), dims=dims)
            logits = logits + flipped
    return torch.argmax(logits, dim=1)[0].cpu().numpy()


def banner(width, text):
    strip = np.full((LABEL_HEIGHT, width, 3), 30, np.uint8)
    cv2.putText(strip, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return strip


def legend(width):
    """A single row keying the CEI colors to class names."""
    strip = np.full((LABEL_HEIGHT + 8, width, 3), 245, np.uint8)
    x = 8
    for name, color in zip(CEI_CLASS_NAMES, CEI_CLASS_COLORS):
        bgr = tuple(int(c) for c in color[::-1])
        cv2.rectangle(strip, (x, 8), (x + 20, 26), bgr, -1)
        cv2.rectangle(strip, (x, 8), (x + 20, 26), (150, 150, 150), 1)
        cv2.putText(strip, name, (x + 24, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 0), 1, cv2.LINE_AA)
        x += 24 + 9 * len(name) + 14
    return strip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--num", type=int, default=12, help="How many tiles to show.")
    parser.add_argument("--order", choices=["worst", "best"], default="worst")
    parser.add_argument("--tiles", default=None,
                        help="Comma-separated tile numbers to show, overrides --order.")
    parser.add_argument("--cell", type=int, default=300, help="Panel cell size (px).")
    parser.add_argument("--out", default="cei_compare.png")
    args = parser.parse_args()

    config = load_config(args.config)
    config["model"]["encoder_weights"] = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = build_dataset(config, split="test")
    num_classes = config["dataset"]["num_classes"]

    model = build_model(config).to(device).eval()
    load_model_weights(model, torch.load(args.checkpoint, map_location=device))

    # Score every tile so the selection can be worst/best-first.
    print(f"scoring {len(dataset)} tiles ...", flush=True)
    scored = []
    for index in range(len(dataset)):
        image, target = dataset[index]
        prediction = predict(model, image, device, args.tta)
        score = per_tile_miou(prediction, target.numpy(), num_classes)
        scored.append((index, score))
        if (index + 1) % 25 == 0:
            print(f"  {index + 1}/{len(dataset)}", flush=True)

    if args.tiles:
        wanted = {int(t) for t in args.tiles.split(",")}
        chosen = [i for i, _ in scored
                  if int("".join(filter(str.isdigit, dataset.file_names[i]))) in wanted]
    else:
        ranked = sorted(scored, key=lambda pair: (pair[1] if pair[1] == pair[1] else 1e9),
                        reverse=(args.order == "best"))
        chosen = [index for index, _ in ranked[:args.num]]

    score_by_index = dict(scored)
    cell = args.cell
    rows = []
    for index in chosen:
        image, target = dataset[index]
        prediction = predict(model, image, device, args.tta)
        name = dataset.file_names[index]
        miou = score_by_index[index]

        triptych = [denormalize(image), decode(target.numpy()), decode(prediction)]
        triptych = [cv2.resize(p, (cell, cell), interpolation=cv2.INTER_NEAREST)
                    if k else cv2.resize(p, (cell, cell))
                    for k, p in enumerate(triptych)]
        labels = [f"{name}  mIoU {miou:.3f}", "ground truth", "prediction"]
        labelled = [np.vstack([banner(cell, text), panel])
                    for text, panel in zip(labels, triptych)]
        rows.append(np.hstack(labelled))

    grid = np.vstack(rows)
    grid = np.vstack([grid, legend(grid.shape[1])])
    cv2.imwrite(args.out, grid)

    shown = [score_by_index[i] for i in chosen]
    print(f"\nwrote {args.out}")
    print(f"  {len(chosen)} tiles ({args.order}-first), "
          f"per-tile mIoU {np.nanmin(shown):.3f}-{np.nanmax(shown):.3f}")
    print("  columns: image | ground truth | prediction")


if __name__ == "__main__":
    main()
