"""Visualization helpers for OpenEarthMap land-cover segmentation.

Masks handled here use the zero-based training class indices produced by
``OpenEarthMapDataset`` (0-7), plus the ignore index for unlabeled pixels.
Colors follow the official OpenEarthMap palette, shifted from the raw 1-8
label codes down to the 0-7 range used for training.
"""

import cv2
import numpy as np

# Class names in training-index order (raw OpenEarthMap code minus one).
OEM_CLASS_NAMES = [
    "Bareland",
    "Rangeland",
    "Developed space",
    "Road",
    "Tree",
    "Water",
    "Agriculture land",
    "Building",
]

# Official OpenEarthMap RGB palette for the eight land-cover classes.
OEM_CLASS_COLORS = [
    (128, 0, 0),      # Bareland
    (0, 255, 36),     # Rangeland
    (148, 148, 148),  # Developed space
    (255, 255, 255),  # Road
    (34, 97, 38),     # Tree
    (0, 69, 255),     # Water
    (75, 181, 73),    # Agriculture land
    (222, 31, 7),     # Building
]

# Color for ignored / unlabeled pixels.
IGNORE_COLOR = (0, 0, 0)


def decode_mask(mask, ignore_index=255):
    """Convert a class-index mask ``[H, W]`` into an RGB image ``[H, W, 3]``.

    Any pixel that is not a known class (including the ignore index) is
    rendered with ``IGNORE_COLOR`` so unlabeled regions stay visually distinct.
    """
    mask = np.asarray(mask)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[...] = IGNORE_COLOR

    for class_index, color in enumerate(OEM_CLASS_COLORS):
        rgb[mask == class_index] = color

    return rgb


def encode_mask(rgb, ignore_index=255):
    """Inverse of :func:`decode_mask`: RGB ``[H, W, 3]`` -> class-index ``[H, W]``.

    Every pixel is assigned the *nearest* palette color in RGB space, so a mask
    that came back from an image editor still decodes cleanly even if a soft or
    anti-aliased brush blended some pixels between two class colors.
    ``IGNORE_COLOR`` maps to ``ignore_index`` (it is a valid "unlabeled" paint
    color, not a class).

    Returns ``(mask, exact_pixels, total_pixels)``. ``exact_pixels`` counts the
    pixels that were already an exact palette color; a low ratio means the edit
    was made with a soft brush and the snapping did real work, which the caller
    can surface as a warning.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an [H, W, 3] RGB image, got shape {rgb.shape}.")

    # Candidate paint colors: the 8 classes, plus the ignore color at the end.
    palette = np.array([*OEM_CLASS_COLORS, IGNORE_COLOR], dtype=np.int16)
    targets = np.array(
        [*range(len(OEM_CLASS_COLORS)), ignore_index], dtype=np.uint8
    )

    # Editors leave only a handful of distinct colors behind, so resolve each
    # unique color once instead of doing a nearest-neighbor search per pixel.
    flat = rgb.reshape(-1, 3)
    colors, inverse = np.unique(flat, axis=0, return_inverse=True)

    # [num_colors, num_palette] squared distance, then pick the closest entry.
    distances = ((colors[:, None, :].astype(np.int32) - palette[None, :, :]) ** 2).sum(axis=2)
    nearest = distances.argmin(axis=1)

    color_to_index = targets[nearest]
    mask = color_to_index[inverse].reshape(rgb.shape[:2])

    # A distance of 0 means the color was already exactly on-palette.
    exact_colors = distances[np.arange(len(colors)), nearest] == 0
    exact_pixels = int(np.bincount(inverse, minlength=len(colors))[exact_colors].sum())

    return mask, exact_pixels, int(flat.shape[0])


def save_gimp_palette(path):
    """Write the OpenEarthMap palette as a GIMP ``.gpl`` file.

    Importing this into GIMP (Windows > Dockable Dialogs > Palettes) lets you
    pick class colors exactly, which is what keeps a corrected mask decodable
    back to class indices without any color drift.
    """
    lines = ["GIMP Palette", "Name: OpenEarthMap", "Columns: 3", "#"]
    for name, (r, g, b) in zip(OEM_CLASS_NAMES, OEM_CLASS_COLORS):
        lines.append(f"{r:3d} {g:3d} {b:3d}\t{name}")
    r, g, b = IGNORE_COLOR
    lines.append(f"{r:3d} {g:3d} {b:3d}\tUnlabeled (ignore)")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def denormalize_image(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Reverse ImageNet normalization on a ``[C, H, W]`` tensor or ``[H, W, C]``
    array and return an ``uint8`` RGB image ``[H, W, 3]``."""
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image, dtype=np.float32)

    # Accept CHW tensors (Albumentations/ToTensorV2 output) as well as HWC.
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    image = image * std + mean
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    return image


def overlay_mask(image, mask, alpha=0.5, ignore_index=255):
    """Blend a decoded color mask over an RGB ``uint8`` image."""
    color_mask = decode_mask(mask, ignore_index=ignore_index)
    blended = cv2.addWeighted(image, 1.0 - alpha, color_mask, alpha, 0.0)
    return blended


def _to_uint8_rgb(image):
    """Coerce an image (normalized tensor or plain array) to ``uint8`` RGB."""
    if hasattr(image, "detach"):
        return denormalize_image(image)

    image = np.asarray(image)
    if image.dtype != np.uint8:
        return denormalize_image(image)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


def save_prediction(path, image, prediction, ground_truth=None, ignore_index=255):
    """Save a side-by-side visualization panel to ``path``.

    The panel contains the input image, the predicted color mask, and -- when
    ``ground_truth`` is supplied -- the ground-truth color mask. Images are
    written with OpenCV, so RGB is converted to BGR before saving.
    """
    image = _to_uint8_rgb(image)

    panels = [image, decode_mask(prediction, ignore_index=ignore_index)]
    if ground_truth is not None:
        panels.append(decode_mask(ground_truth, ignore_index=ignore_index))

    panel = np.concatenate(panels, axis=1)
    cv2.imwrite(path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    return path


def save_mask(path, mask, ignore_index=255):
    """Save a single decoded color mask to ``path``."""
    color_mask = decode_mask(mask, ignore_index=ignore_index)
    cv2.imwrite(path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
    return path
