"""Build train/val/test split files for IRSAMap.

IRSAMap ships no split files and no validation set -- only ``train/`` (4617
tiles) and ``test/`` (912). This carves a validation set out of the training
tiles and writes all three lists.

What it does, step by step:

1. List every image in ``train/image`` and ``test/image``.
2. Check each has a mask in the chosen ``SegLabel_`` folder, and that the mask
   contains only known IRSA codes -- an unknown code would otherwise be remapped
   to ignore silently, quietly deleting part of a class.
3. Split the training tiles into train/val with a fixed seed, so the split is
   reproducible across machines and reruns.
4. Report the CEI-scheme class distribution of each split, so a class that is
   scarce or missing in validation is visible before training starts.

Sport (code 34) is only 0.16% of pixels and appears in 328 of 4617 tiles, so a
naive random split can leave validation with almost none. Pass ``--stratify`` to
force tiles containing the rarest classes to be spread across both splits.

Usage
-----
python tools/irsa/make_splits.py
python tools/irsa/make_splits.py --val-fraction 0.1 --stratify
"""

import argparse
import os
import random
import sys

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.datasets.taxonomy import CEI_CLASS_NAMES, IRSA_TO_CEI  # noqa: E402


VALID_CODES = set(IRSA_TO_CEI)

# Classes rare enough that a random split can under-serve them. Used only when
# --stratify is passed.
RARE_CODES = [34, 23]  # sport, water subtype c


def list_tiles(root, split_dir, image_dir, mask_dir):
    """Return (usable filenames, problems) for one on-disk split."""
    image_root = os.path.join(root, split_dir, image_dir)
    mask_root = os.path.join(root, split_dir, mask_dir)

    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"Image folder not found: {image_root}")
    if not os.path.isdir(mask_root):
        raise FileNotFoundError(f"Mask folder not found: {mask_root}")

    names, problems = [], []

    for name in sorted(os.listdir(image_root)):
        if not name.lower().endswith((".png", ".tif", ".tiff", ".jpg")):
            continue
        if not os.path.isfile(os.path.join(mask_root, name)):
            problems.append(f"{name}: no mask in {mask_dir}")
            continue
        names.append(name)

    return names, problems


def scan_codes(root, split_dir, mask_dir, names, sample=None):
    """Return per-tile code sets and the overall code histogram.

    Reading every mask is the slow part, so ``--sample`` allows a quick pass
    during development. The default reads all of them, because an unknown code
    hiding in one tile is exactly what this is meant to catch.
    """
    mask_root = os.path.join(root, split_dir, mask_dir)
    targets = names if sample is None else names[:sample]

    histogram = np.zeros(256, dtype=np.int64)
    per_tile = {}
    problems = []

    for index, name in enumerate(targets):
        mask = cv2.imread(os.path.join(mask_root, name), cv2.IMREAD_UNCHANGED)
        if mask is None:
            problems.append(f"{name}: unreadable mask")
            continue
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        present = set(np.unique(mask).tolist())
        unknown = sorted(present - VALID_CODES)
        if unknown:
            problems.append(f"{name}: unknown label codes {unknown}")
            continue

        per_tile[name] = present
        histogram += np.bincount(mask.ravel(), minlength=256)

        if (index + 1) % 500 == 0:
            print(f"    scanned {index + 1}/{len(targets)}", flush=True)

    return per_tile, histogram, problems


def split_train_val(names, per_tile, val_fraction, seed, stratify):
    """Split tile names into (train, val), optionally spreading rare classes."""
    rng = random.Random(seed)
    shuffled = list(names)
    rng.shuffle(shuffled)

    target_val = int(round(len(shuffled) * val_fraction))

    if not stratify:
        return shuffled[target_val:], shuffled[:target_val]

    # Take the same fraction of the tiles containing each rare class first, so
    # validation is guaranteed to contain some of them, then fill the rest.
    val, taken = [], set()
    for code in RARE_CODES:
        holders = [n for n in shuffled if code in per_tile.get(n, ())]
        quota = max(1, int(round(len(holders) * val_fraction)))
        for name in holders:
            if len(taken) >= target_val:
                break
            if name not in taken and quota > 0:
                val.append(name)
                taken.add(name)
                quota -= 1

    for name in shuffled:
        if len(taken) >= target_val:
            break
        if name not in taken:
            val.append(name)
            taken.add(name)

    train = [n for n in shuffled if n not in taken]
    return train, val


def cei_distribution(root, split_dir, mask_dir, names, sample=400):
    """CEI-scheme class shares for a list of tiles (sampled for speed)."""
    mask_root = os.path.join(root, split_dir, mask_dir)
    totals = np.zeros(len(CEI_CLASS_NAMES), dtype=np.int64)

    for name in names[:sample]:
        mask = cv2.imread(os.path.join(mask_root, name), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        counts = np.bincount(mask.ravel(), minlength=256)
        for code, class_index in IRSA_TO_CEI.items():
            totals[class_index] += int(counts[code])

    return totals


def write_split(path, names):
    with open(path, "w", encoding="utf-8") as handle:
        for name in names:
            handle.write(name + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/IRSAMap")
    parser.add_argument("--image-dir", default="image")
    parser.add_argument("--mask-dir", default="SegLabel_vwsbr",
                        help="Combined label folder. vwsbr puts road above "
                             "building where they overlap; rvwsb reverses that.")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify", action="store_true",
                        help="Spread tiles containing rare classes across the "
                             "train/val split instead of splitting purely at random.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Scan only the first N masks per split (development).")
    args = parser.parse_args()

    print(f"IRSAMap root: {args.root}  (masks: {args.mask_dir})\n")

    # ---------------------------------------------------------------- train/val
    print("train/")
    train_names, problems = list_tiles(args.root, "train", args.image_dir, args.mask_dir)
    print(f"  {len(train_names)} tiles with masks")
    print("  scanning label codes ...", flush=True)
    per_tile, train_hist, scan_problems = scan_codes(
        args.root, "train", args.mask_dir, train_names, args.sample
    )
    problems += scan_problems
    usable = [n for n in train_names if n in per_tile]

    train_split, val_split = split_train_val(
        usable, per_tile, args.val_fraction, args.seed, args.stratify
    )

    # -------------------------------------------------------------------- test
    print("\ntest/")
    test_names, test_problems = list_tiles(args.root, "test", args.image_dir, args.mask_dir)
    print(f"  {len(test_names)} tiles with masks")
    print("  scanning label codes ...", flush=True)
    test_per_tile, test_hist, test_scan_problems = scan_codes(
        args.root, "test", args.mask_dir, test_names, args.sample
    )
    problems += test_problems + test_scan_problems
    test_split = [n for n in test_names if n in test_per_tile]

    if problems:
        print(f"\nSkipped {len(problems)} unusable tile(s):")
        for problem in problems[:20]:
            print(f"  {problem}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")

    # ------------------------------------------------------------------- write
    outputs = [
        ("train_split.txt", train_split, "train"),
        ("val_split.txt", val_split, "train"),
        ("test_split.txt", test_split, "test"),
    ]
    print()
    for filename, names, source in outputs:
        write_split(os.path.join(args.root, filename), names)
        print(f"wrote {filename:<16} {len(names):>5} tiles  (from {source}/)")

    # ------------------------------------------------- codes present on disk
    print("\nLabel codes found (train):")
    for code in sorted(VALID_CODES):
        count = int(train_hist[code])
        share = 100.0 * count / max(train_hist.sum(), 1)
        flag = "   <-- ABSENT" if count == 0 else ""
        print(f"  {code:<4} -> CEI {IRSA_TO_CEI[code]} "
              f"{CEI_CLASS_NAMES[IRSA_TO_CEI[code]]:<16} {share:6.2f}%{flag}")

    # -------------------------------------------- per-split class distribution
    print("\nCEI-scheme class distribution per split (sampled):")
    print(f"  {'CLASS':<16} {'train':>9} {'val':>9} {'test':>9}")
    distributions = [
        cei_distribution(args.root, "train", args.mask_dir, train_split),
        cei_distribution(args.root, "train", args.mask_dir, val_split),
        cei_distribution(args.root, "test", args.mask_dir, test_split),
    ]
    shares = [100.0 * d / max(d.sum(), 1) for d in distributions]
    for index, class_name in enumerate(CEI_CLASS_NAMES):
        row = "  ".join(f"{s[index]:8.2f}%" for s in shares)
        warn = "   <-- scarce in val" if shares[1][index] < 0.05 else ""
        print(f"  {class_name:<16} {row}{warn}")

    print("\nNote: these shares treat background (code 0) as Non-vegetated. The "
          "loader additionally sends near-black nodata to ignore, so the "
          "Non-vegetated share seen during training is slightly lower.")


if __name__ == "__main__":
    main()
