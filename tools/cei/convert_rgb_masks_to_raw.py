"""One-off fixup: convert RGB CEI-palette masks in data/CEI_data/masks into
the raw single-channel encoding the dataset loader expects.

The masks in data/CEI_data/masks turned out to be 3-channel RGB images
painted in the CEI class palette (the GIMP-review format tools/cei/
predict_for_gimp.py produces for hand-editing) instead of the raw
single-channel class-code format (0 = unlabeled, 1-7 = classes) that
OpenEarthMapDataset (used for both OEM and CEI) reads. This snaps every pixel
to the nearest CEI palette color -- the same algorithm
src/utils/visualization.py:encode_mask uses for the OEM palette,
parameterized here with src/datasets/taxonomy.CEI_CLASS_COLORS instead -- and
overwrites each mask with the raw-code version.

The original RGB file is backed up first (to --backup, default
data/CEI_data/masks_rgb_backup/) so nothing is destroyed if this needs to be
re-run or undone.

Usage
-----
python tools/cei/convert_rgb_masks_to_raw.py
python tools/cei/convert_rgb_masks_to_raw.py --root data/CEI_data --mask-dir masks
python tools/cei/convert_rgb_masks_to_raw.py --dry-run
"""

import argparse
import os
import shutil
import sys

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.datasets.taxonomy import CEI_CLASS_COLORS, CEI_CLASS_NAMES, CEI_IGNORE_COLOR  # noqa: E402

# Below this share of exactly-on-palette pixels, the source was probably a
# lossy format or a soft brush -- same threshold import_from_gimp.py uses.
EXACT_COLOR_WARN_RATIO = 0.98


def snap_to_cei_raw(rgb):
    """RGB [H, W, 3] -> raw CEI on-disk codes [H, W] uint8 (0 = unlabeled, 1-7 = classes).

    Every pixel is assigned the nearest palette color in RGB space, so
    anti-aliased or lossily-compressed edges still resolve to a real class.
    Returns (raw_mask, exact_pixels, total_pixels).
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an [H, W, 3] RGB image, got shape {rgb.shape}.")

    # Class colors map to on-disk codes 1..7 (internal index + 1); the ignore
    # color maps to 0, matching CEI's "0 = unlabeled" on-disk convention.
    palette = np.array([*CEI_CLASS_COLORS, CEI_IGNORE_COLOR], dtype=np.int16)
    targets = np.array([*range(1, len(CEI_CLASS_COLORS) + 1), 0], dtype=np.uint8)

    flat = rgb.reshape(-1, 3)
    colors, inverse = np.unique(flat, axis=0, return_inverse=True)

    distances = ((colors[:, None, :].astype(np.int32) - palette[None, :, :]) ** 2).sum(axis=2)
    nearest = distances.argmin(axis=1)

    color_to_code = targets[nearest]
    raw_mask = color_to_code[inverse].reshape(rgb.shape[:2])

    exact_colors = distances[np.arange(len(colors)), nearest] == 0
    exact_pixels = int(np.bincount(inverse, minlength=len(colors))[exact_colors].sum())

    return raw_mask, exact_pixels, int(flat.shape[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/CEI_data")
    parser.add_argument("--mask-dir", default="masks")
    parser.add_argument("--backup", default=None,
                        help="Where to copy the original RGB files before "
                             "overwriting. Default: <root>/masks_rgb_backup/")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would happen without writing anything.")
    args = parser.parse_args()

    mask_root = os.path.join(args.root, args.mask_dir)
    backup_root = args.backup or os.path.join(args.root, "masks_rgb_backup")
    names = sorted(f for f in os.listdir(mask_root) if f.lower().endswith((".tif", ".tiff")))
    if not names:
        raise SystemExit(f"No .tif files found under {mask_root}")

    if not args.dry_run:
        os.makedirs(backup_root, exist_ok=True)

    converted, already_raw, warnings = 0, 0, []
    class_totals = np.zeros(len(CEI_CLASS_COLORS) + 1, dtype=np.int64)  # index 0 = unlabeled

    for name in names:
        path = os.path.join(mask_root, name)
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            warnings.append(f"{name}: unreadable, skipped")
            continue

        if image.ndim == 2:
            # Already single-channel -- nothing to convert.
            already_raw += 1
            class_totals += np.bincount(image.ravel(), minlength=len(class_totals))
            continue

        if image.shape[2] == 4:
            image = image[:, :, :3]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        raw_mask, exact_pixels, total_pixels = snap_to_cei_raw(rgb)
        exact_ratio = exact_pixels / total_pixels
        if exact_ratio < EXACT_COLOR_WARN_RATIO:
            warnings.append(f"{name}: only {100 * exact_ratio:.1f}% of pixels were "
                            f"exactly on-palette (lossy source?)")

        class_totals += np.bincount(raw_mask.ravel(), minlength=len(class_totals))

        if not args.dry_run:
            shutil.copy2(path, os.path.join(backup_root, name))
            cv2.imwrite(path, raw_mask)
        converted += 1

    print(f"{'[DRY RUN] ' if args.dry_run else ''}"
          f"{converted} RGB mask(s) converted to raw CEI codes, "
          f"{already_raw} already single-channel (left alone).")
    if not args.dry_run and converted:
        print(f"Originals backed up to: {backup_root}")
    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for warning in warnings:
            print(f"  {warning}")

    labeled_total = class_totals[1:].sum()
    print("\nClass distribution across all masks:")
    for index, name in enumerate(CEI_CLASS_NAMES):
        count = int(class_totals[index + 1])
        share = 100.0 * count / labeled_total if labeled_total else 0.0
        flag = "   <-- ABSENT" if count == 0 else ""
        print(f"  {name:<16} {share:5.1f}%{flag}")
    if class_totals[0]:
        print(f"  {'Unlabeled (0)':<16} {100.0 * class_totals[0] / class_totals.sum():5.1f}%")


if __name__ == "__main__":
    main()
