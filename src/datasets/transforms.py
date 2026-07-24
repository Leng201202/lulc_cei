"""Shared image normalization, selected by ``dataset.normalization`` in config.

The normalization must match what the model weights were trained with:

* ``imagenet`` -- ``(x/255 - mean) / std`` with ImageNet statistics. Correct

    step 1: old_value / 255
    step 2: (value - mean) / std
    
  for models with an ImageNet-pretrained encoder trained in this project.
* ``zero_one`` -- ``x / 255`` only, scaling to ``[0, 1]``. This is what the
    0   -> 0.0
    128 -> 0.5
    255 -> 1.0
  external OpenEarthMap-SAR baseline weights expect; using ImageNet statistics
  with those weights produces incoherent predictions.
"""

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

NORMALIZATIONS = {
    "imagenet": dict(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=255.0,
    ),
    "zero_one": dict(
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        max_pixel_value=255.0,
    ),
}


def get_normalization_name(config):
    """Return the configured normalization name, defaulting to ``imagenet``."""
    return config["dataset"].get("normalization", "imagenet")


def build_normalize(config):
    """Build the ``A.Normalize`` transform selected by the config."""
    name = get_normalization_name(config)
    if name not in NORMALIZATIONS:
        raise ValueError(
            f"Unknown dataset.normalization: {name!r}. "
            f"Expected one of {sorted(NORMALIZATIONS)}."
        )
    return A.Normalize(**NORMALIZATIONS[name])


def _pad_to(min_h, min_w, ignore_index, div=None):
    """Constant-border pad: image with zeros, mask with ignore_index, so any
    invented border pixel is excluded from loss and metrics."""
    return A.PadIfNeeded(
        min_height=min_h,
        min_width=min_w,
        pad_height_divisor=div,
        pad_width_divisor=div,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        mask_value=ignore_index,
    )


def build_segmentation_transforms(config, split, crop_size, ignore_index):
    """Build the albumentations pipeline for one split.

    Kept out of the dataset classes so the augmentation policy lives in one
    place and can change without touching dataset I/O. ``train`` pads up to the
    crop size, random-crops, and (opt-in via ``dataset.augment``) flips/rotates.
    Evaluation is deterministic: ``full`` scores the whole tile padded to a
    multiple of 32 for the encoder's 32x downsampling; ``crop`` center-crops.
    """
    normalize = build_normalize(config)
    dataset_config = config["dataset"]

    if split == "train":
        transforms = [
            _pad_to(crop_size, crop_size, ignore_index),
            A.RandomCrop(height=crop_size, width=crop_size),
        ]
        if dataset_config.get("augment", True):
            transforms += [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        transforms += [normalize, ToTensorV2()]
        return A.Compose(transforms)

    eval_mode = dataset_config.get("eval_mode", "full")
    if eval_mode == "full":
        return A.Compose(
            [_pad_to(None, None, ignore_index, div=32), normalize, ToTensorV2()]
        )
    return A.Compose([
        _pad_to(crop_size, crop_size, ignore_index),
        A.CenterCrop(height=crop_size, width=crop_size),
        normalize,
        ToTensorV2(),
    ])
