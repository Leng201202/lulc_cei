"""Build the CEI test split file from whichever tiles currently have labels.

The CEI test set is defined as tiles 1-100. Labeling happens in batches, so this
script scans the masks folder, keeps every labeled tile whose number falls in the
range, and writes the split file the eval config reads. Re-run it after each
labeling batch instead of editing the list by hand.

What it does, step by step:

1. Find every mask in ``<root>/masks`` and read its tile number.
2. Keep the ones inside ``--start`` .. ``--end`` (default 1..100).
3. Check each one has a matching image, and that its label values are legal
   CEI class ids (0-7) -- a mask that fails is reported and left out, because it
   would crash training later anyway.
4. Report blank/nodata tiles (uniform captures with no bright pixels). These are
   kept by default; pass ``--exclude-blank`` to drop them.
5. Write one image filename per line to the output split file.

Example
-------
python tools/cei/make_test_split.py
python tools/cei/make_test_split.py --start 1 --end 100 --exclude-blank
"""

import argparse
import os
import re
import sys

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.datasets.taxonomy import CEI_CLASS_NAMES  # noqa: E402


# CEI labels are class ids: 0 = Unlabeled (ignore), 1-7 = the seven classes.
VALID_LABEL_VALUES = set(range(8))

# A blank/nodata capture is uniformly dark with no bright pixels anywhere. Mean
# brightness is the wrong test -- a real tile covered by a lake is also dark, but
# still contains bright shoreline/road pixels. The 99th percentile separates
# them cleanly (blank tiles sit near 55; real tiles are 128+).
BLANK_P99_THRESHOLD = 100


def tile_number(name):
    """Return the trailing number in a filename, or None if there isn't one."""
    match = re.search(r"_(\d+)$", os.path.splitext(name)[0])
    return int(match.group(1)) if match else None


def collect_labeled_tiles(root, mask_dir, image_dir, mask_suffix):
    """Return ``{tile_number: image_filename}`` for every readable labeled tile.

    Masks that have no matching image, or that contain values outside the CEI
    range, are skipped and reported -- better to catch them here than to have
    the dataset raise mid-training.
    """
    mask_root = os.path.join(root, mask_dir)
    if not os.path.isdir(mask_root):
        raise FileNotFoundError(f"Masks folder not found: {mask_root}")

    tiles, problems = {}, []

    for mask_name in sorted(os.listdir(mask_root)):
        # Reverse the "<stem><suffix>.tif" naming to recover the image filename.
        stem, extension = os.path.splitext(mask_name)
        if mask_suffix and stem.endswith(mask_suffix):
            stem = stem[: -len(mask_suffix)]
        image_name = f"{stem}{extension}"

        number = tile_number(image_name)
        if number is None:
            continue

        if not os.path.isfile(os.path.join(root, image_dir, image_name)):
            problems.append(f"{mask_name}: no matching image ({image_name})")
            continue

        mask = cv2.imread(os.path.join(mask_root, mask_name), cv2.IMREAD_UNCHANGED)
        if mask is None:
            problems.append(f"{mask_name}: unreadable")
            continue
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        bad_values = sorted(set(np.unique(mask).tolist()) - VALID_LABEL_VALUES)
        if bad_values:
            problems.append(f"{mask_name}: invalid label values {bad_values}")
            continue

        tiles[number] = image_name

    return tiles, problems


def is_blank(image_path):
    """True if the image has no bright pixels at all (a nodata capture)."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return False
    return float(np.percentile(image, 99)) < BLANK_P99_THRESHOLD


def main():
    parser = argparse.ArgumentParser(
        description="Build the CEI test split from the currently labeled tiles."
    )
    parser.add_argument("--root", default="data/CEI_data", help="CEI dataset root.")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--mask-dir", default="masks")
    parser.add_argument(
        "--mask-suffix",
        default="",
        help="Suffix on mask filenames relative to image names (none by default).",
    )
    parser.add_argument("--start", type=int, default=1, help="First tile number.")
    parser.add_argument("--end", type=int, default=100, help="Last tile number.")
    parser.add_argument(
        "--output",
        default=None,
        help="Split file to write. Default: <root>/test_split.txt",
    )
    parser.add_argument(
        "--exclude-blank",
        action="store_true",
        help="Drop blank/nodata tiles. They are kept by default.",
    )
    args = parser.parse_args()

    tiles, problems = collect_labeled_tiles(
        args.root, args.mask_dir, args.image_dir, args.mask_suffix
    )

    if problems:
        print(f"Skipped {len(problems)} unusable mask(s):")
        for problem in problems:
            print(f"  {problem}")
        print()

    in_range = sorted(n for n in tiles if args.start <= n <= args.end)
    out_of_range = sorted(n for n in tiles if not (args.start <= n <= args.end))
    if out_of_range:
        print(f"Ignored {len(out_of_range)} labeled tile(s) outside "
              f"{args.start}-{args.end}: {out_of_range}\n")

    # Blank tiles are reported either way, so the choice stays visible.
    blanks = [
        n for n in in_range
        if is_blank(os.path.join(args.root, args.image_dir, tiles[n]))
    ]
    if blanks:
        state = "EXCLUDED" if args.exclude_blank else "kept (use --exclude-blank to drop)"
        print(f"Blank/nodata tiles detected: {blanks} -> {state}\n")
    if args.exclude_blank:
        in_range = [n for n in in_range if n not in blanks]

    if not in_range:
        raise SystemExit("No labeled tiles found in range -- nothing to write.")

    output_path = args.output or os.path.join(args.root, "test_split.txt")
    with open(output_path, "w", encoding="utf-8") as handle:
        for number in in_range:
            handle.write(tiles[number] + "\n")

    # Class coverage: a class with no pixels is dropped from mIoU, so it is worth
    # knowing before you trust the number.
    totals = np.zeros(8, dtype=np.int64)
    for number in in_range:
        mask_name = f"{os.path.splitext(tiles[number])[0]}{args.mask_suffix}.tif"
        mask = cv2.imread(os.path.join(args.root, args.mask_dir, mask_name),
                          cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        totals += np.bincount(mask.ravel(), minlength=8)

    labeled_total = totals[1:8].sum()
    print(f"Wrote {len(in_range)} tile(s) to {output_path}")
    print(f"Coverage of tiles {args.start}-{args.end}: "
          f"{len(in_range)}/{args.end - args.start + 1} labeled "
          f"({100.0 * len(in_range) / (args.end - args.start + 1):.0f}%)")
    print("\nClass distribution:")
    for index, class_name in enumerate(CEI_CLASS_NAMES):
        count = int(totals[index + 1])
        share = 100.0 * count / labeled_total if labeled_total else 0.0
        flag = "   <-- ABSENT" if count == 0 else ""
        print(f"  {class_name:<16} {share:5.1f}%{flag}")
    if totals[0]:
        print(f"  {'Unlabeled (0)':<16} {100.0 * totals[0] / totals.sum():5.1f}%")


if __name__ == "__main__":
    main()
