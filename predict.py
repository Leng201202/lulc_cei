import argparse
import os

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from src.utils.config import load_config
from src.utils.visualization import save_mask, save_prediction


IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def collect_images(input_path):
    """Return a sorted list of image paths from a file or directory input."""
    if os.path.isfile(input_path):
        return [input_path]

    if os.path.isdir(input_path):
        paths = []
        for name in sorted(os.listdir(input_path)):
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                paths.append(os.path.join(input_path, name))
        return paths

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def build_transform(config):
    """Normalize the same way the dataset does, matching the model weights."""
    from src.datasets.transforms import build_normalize

    return A.Compose([
        build_normalize(config),
        ToTensorV2(),
    ])


def pad_to_multiple(image, multiple=32):
    """Pad an ``[H, W, C]`` image on the bottom/right so both spatial
    dimensions are multiples of ``multiple``.

    Encoder downsampling requires input sizes divisible by 32; padding lets
    the network run on full-resolution imagery without cropping. The padding
    amounts are returned so predictions can be cropped back to the original
    size.
    """
    height, width = image.shape[:2]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple

    if pad_h == 0 and pad_w == 0:
        return image, (0, 0)

    padded = cv2.copyMakeBorder(
        image, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT_101
    )
    return padded, (pad_h, pad_w)


def read_image(image_path):
    """Read an image as RGB ``[H, W, 3]`` or raise if it cannot be read."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found or cannot be read: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@torch.no_grad()
def _infer_logits(model, image, transform, device):
    """Normalize, pad, and run the model, returning logits cropped back to the
    original image size as an ``[num_classes, H, W]`` tensor."""
    original_height, original_width = image.shape[:2]

    padded, _ = pad_to_multiple(image)
    tensor = transform(image=padded)["image"].unsqueeze(0).to(device)

    logits = model(tensor).squeeze(0)
    return logits[:, :original_height, :original_width]


def predict_full(model, image, transform, device):
    """Full-image inference. Returns an ``[H, W]`` class-index mask."""
    logits = _infer_logits(model, image, transform, device)
    return torch.argmax(logits, dim=0).cpu().numpy().astype(np.uint8)


@torch.no_grad()
def predict_tiled(model, image, transform, device, tile_size, overlap, num_classes):
    """Sliding-window inference for large imagery.

    Overlapping tiles are run independently and their softmax probabilities are
    accumulated into a full-size buffer, so seams are averaged rather than
    hard-cut. Returns an ``[H, W]`` class-index mask.
    """
    height, width = image.shape[:2]
    stride = max(1, tile_size - overlap)

    prob_sum = torch.zeros(
        (num_classes, height, width), dtype=torch.float32, device=device
    )
    count = torch.zeros((1, height, width), dtype=torch.float32, device=device)

    # Anchor rows/cols include a final tile flush against the bottom/right edge
    # so the whole image is covered even when it is not a multiple of stride.
    def anchors(length):
        if length <= tile_size:
            return [0]
        points = list(range(0, length - tile_size + 1, stride))
        if points[-1] != length - tile_size:
            points.append(length - tile_size)
        return points

    for top in anchors(height):
        for left in anchors(width):
            tile = image[top:top + tile_size, left:left + tile_size]
            logits = _infer_logits(model, tile, transform, device)
            probs = torch.softmax(logits, dim=0)

            th, tw = probs.shape[1], probs.shape[2]
            prob_sum[:, top:top + th, left:left + tw] += probs
            count[:, top:top + th, left:left + tw] += 1.0

    prob_sum /= count.clamp_min(1.0)
    return torch.argmax(prob_sum, dim=0).cpu().numpy().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to an image file or a directory of images.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory. Default: <experiment.output_dir>/predictions.",
    )
    parser.add_argument(
        "--panel",
        action="store_true",
        help="Also save an image+prediction side-by-side panel per input.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=None,
        help="Enable sliding-window inference with this tile size (pixels). "
             "Recommended for imagery too large to fit in memory at once.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=128,
        help="Tile overlap in pixels when --tile_size is set.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "tiff"],
        default="png",
        help="File format for the saved mask. Use tiff for lossless masks you "
             "intend to hand-correct in an image editor such as GIMP.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Average predictions over horizontal/vertical flips (4-way TTA). "
             "Slower, but usually a cleaner mask to start correcting from.",
    )
    args = parser.parse_args()

    if args.tile_size is not None and args.overlap >= args.tile_size:
        parser.error("--overlap must be smaller than --tile_size.")

    # Imported lazily so the inference helpers above can be used (and tested)
    # without pulling in segmentation_models_pytorch.
    from evaluate import FlipTTA
    from src.models.checkpoint import load_model_weights
    from src.models.model_factory import build_model
    from train import select_device, validate_config

    config = load_config(args.config)
    validate_config(config)

    # Trained weights are loaded below, so skip pretrained encoder downloads.
    config["model"]["encoder_weights"] = None
    ignore_index = config["dataset"]["ignore_index"]
    num_classes = config["dataset"]["num_classes"]

    device = select_device()
    print(f"Using device: {device}")

    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    load_model_weights(model, checkpoint)
    model.eval()

    if args.tta:
        # Wrap after loading weights: FlipTTA only averages logits, it holds none.
        model = FlipTTA(model)
        print("Test-time augmentation: 4-way flip enabled")

    output_dir = args.output
    if output_dir is None:
        output_dir = os.path.join(config["experiment"]["output_dir"], "predictions")
    os.makedirs(output_dir, exist_ok=True)

    transform = build_transform(config)
    image_paths = collect_images(args.input)
    print(f"Found {len(image_paths)} image(s) to predict.")

    for image_path in image_paths:
        image = read_image(image_path)

        if args.tile_size is not None:
            prediction = predict_tiled(
                model, image, transform, device,
                tile_size=args.tile_size,
                overlap=args.overlap,
                num_classes=num_classes,
            )
        else:
            prediction = predict_full(model, image, transform, device)

        stem = os.path.splitext(os.path.basename(image_path))[0]
        extension = "tif" if args.format == "tiff" else "png"
        mask_path = os.path.join(output_dir, f"{stem}_pred.{extension}")
        save_mask(mask_path, prediction, ignore_index=ignore_index)

        if args.panel:
            panel_path = os.path.join(output_dir, f"{stem}_panel.png")
            save_prediction(
                panel_path, image, prediction, ignore_index=ignore_index
            )

        print(f"Saved: {mask_path}")

    print("Predictions saved to:", output_dir)


if __name__ == "__main__":
    main()
