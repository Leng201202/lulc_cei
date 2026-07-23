"""Step 2 of the CEI label-correction workflow: corrected TIFF -> dataset label.

Run this after hand-correcting the `*_mask.tif` files that
``tools/cei/predict_for_gimp.py`` put in the review folder.

What it does, per mask:

1. Reads the corrected RGB TIFF you exported from GIMP.
2. Snaps every pixel to the nearest OpenEarthMap class color. Painting with the
   Pencil tool gives exact colors and this is a no-op; a soft/anti-aliased brush
   leaves blended in-between colors, and snapping recovers the intended class.
3. Writes a single-channel PNG label in the **raw OpenEarthMap encoding**:
   classes as codes 1-8 and unlabeled as 0. This is what ``OpenEarthMapDataset``
   expects on disk -- it remaps 1-8 down to training indices 0-7 and turns
   everything else into the ignore index.
4. Reports how much you actually changed, and flags anything suspicious (a mask
   that is mostly off-palette colors usually means it was exported to a lossy
   format such as JPEG, which destroys the class colors).

Example
-------
python tools/cei/import_from_gimp.py \
    --review outputs/cei_exp02/review \
    --output data/CEI_data/labels \
    --preview
"""

import argparse
import os
import sys

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.visualization import (  # noqa: E402
    OEM_CLASS_NAMES,
    decode_mask,
    encode_mask,
)


# Raw OpenEarthMap label encoding, as stored on disk.
UNLABELED_CODE = 0    # becomes ignore_index when the dataset loads it
FIRST_CLASS_CODE = 1  # training index 0 (Bareland) is stored as 1

# Below this share of exactly-on-palette pixels, the mask was almost certainly
# saved through a lossy format and the class colors can no longer be trusted.
EXACT_COLOR_WARN_RATIO = 0.98

IGNORE_INDEX = 255


def read_rgb(path):
    """Read an image as RGB ``[H, W, 3]``, dropping any alpha channel.

    GIMP can add an alpha channel on export; it carries no label information,
    so we discard it rather than fail.
    """
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")

    if image.ndim == 2:
        raise ValueError(
            f"{path} is single-channel. Expected the RGB color mask you edited "
            f"in GIMP, not an already-encoded label."
        )
    if image.shape[2] == 4:
        image = image[:, :, :3]

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_raw_label(mask, ignore_index=IGNORE_INDEX):
    """Training indices 0-7 -> raw on-disk codes 1-8, ignore -> 0."""
    raw = np.full(mask.shape, UNLABELED_CODE, dtype=np.uint8)
    is_class = mask != ignore_index
    raw[is_class] = mask[is_class] + FIRST_CLASS_CODE
    return raw


def collect_masks(review_dir):
    """Return the corrected `*_mask.tif` files, newest edits included.

    The pristine copies live in the `_original/` subfolder and are deliberately
    not picked up here.
    """
    if not os.path.isdir(review_dir):
        raise FileNotFoundError(f"Review directory does not exist: {review_dir}")

    names = sorted(
        name for name in os.listdir(review_dir)
        if name.lower().endswith((".tif", ".tiff")) and "_mask" in name
    )
    return [os.path.join(review_dir, name) for name in names]


def stem_of(mask_path):
    """`ms_001_0000_mask.tif` -> `ms_001_0000` (the original image's name)."""
    name = os.path.splitext(os.path.basename(mask_path))[0]
    return name[: -len("_mask")] if name.endswith("_mask") else name


def summarize_classes(mask, ignore_index=IGNORE_INDEX):
    """Return a `class name -> percentage of pixels` breakdown, largest first."""
    total = mask.size
    counts = []
    for index, name in enumerate(OEM_CLASS_NAMES):
        count = int((mask == index).sum())
        if count:
            counts.append((name, 100.0 * count / total))

    unlabeled = int((mask == ignore_index).sum())
    if unlabeled:
        counts.append(("Unlabeled", 100.0 * unlabeled / total))

    return sorted(counts, key=lambda item: item[1], reverse=True)


def main():
    parser = argparse.ArgumentParser(
        description="Convert GIMP-corrected color masks into OpenEarthMap-format "
                    "label PNGs for the CEI dataset."
    )
    parser.add_argument(
        "--review",
        type=str,
        required=True,
        help="The workspace folder created by tools/cei/predict_for_gimp.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to write the dataset label PNGs into.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Also save a colorized PNG of each imported label, so you can "
             "eyeball what actually landed in the dataset.",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    original_dir = os.path.join(args.review, "_original")

    mask_paths = collect_masks(args.review)
    if not mask_paths:
        raise SystemExit(f"No *_mask.tif files found in {args.review}")

    print(f"Importing {len(mask_paths)} corrected mask(s).\n")

    for mask_path in mask_paths:
        stem = stem_of(mask_path)
        rgb = read_rgb(mask_path)
        mask, exact_pixels, total_pixels = encode_mask(rgb, ignore_index=IGNORE_INDEX)

        print(f"{stem}")

        exact_ratio = exact_pixels / total_pixels
        if exact_ratio < EXACT_COLOR_WARN_RATIO:
            print(
                f"  WARNING: only {100 * exact_ratio:.1f}% of pixels were exactly "
                f"on-palette; the rest were snapped to the nearest class. Check "
                f"that you painted with the Pencil tool and exported to TIFF."
            )

        # How much did the hand-correction actually change? Compares against the
        # pristine draft this workspace was created with.
        original_path = os.path.join(original_dir, f"{stem}.tif")
        if os.path.isfile(original_path):
            original, _, _ = encode_mask(
                read_rgb(original_path), ignore_index=IGNORE_INDEX
            )
            if original.shape != mask.shape:
                print(f"  WARNING: size changed ({original.shape} -> {mask.shape}); "
                      f"the mask must stay the same size as the image. Skipped.")
                continue
            changed = int((original != mask).sum())
            print(f"  corrected: {changed:,} px ({100.0 * changed / total_pixels:.2f}%)")

        for name, percent in summarize_classes(mask):
            print(f"    {name:<18} {percent:5.1f}%")

        label_path = os.path.join(args.output, f"{stem}.png")
        cv2.imwrite(label_path, to_raw_label(mask))
        print(f"  -> {label_path}")

        if args.preview:
            preview_path = os.path.join(args.output, f"{stem}_preview.png")
            cv2.imwrite(
                preview_path,
                cv2.cvtColor(decode_mask(mask, ignore_index=IGNORE_INDEX),
                             cv2.COLOR_RGB2BGR),
            )
            print(f"  -> {preview_path}")

        print()

    print(f"Labels written to: {args.output}")
    print("These are raw OpenEarthMap codes (1-8, 0 = unlabeled) and can be read "
          "directly by OpenEarthMapDataset.")


if __name__ == "__main__":
    main()
