"""Shared image normalization, selected by ``dataset.normalization`` in config.

The normalization must match what the model weights were trained with:

* ``imagenet`` -- ``(x/255 - mean) / std`` with ImageNet statistics. Correct
  for models with an ImageNet-pretrained encoder trained in this project.
* ``zero_one`` -- ``x / 255`` only, scaling to ``[0, 1]``. This is what the
  external OpenEarthMap-SAR baseline weights expect; using ImageNet statistics
  with those weights produces incoherent predictions.
"""

import albumentations as A

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
