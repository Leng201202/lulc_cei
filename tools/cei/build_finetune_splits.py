"""Build the fixed val set and nested diverse training subsets for the
OEM -> CEI few-shot fine-tuning study.

The existing data/CEI_data/test_split.txt (maesuai_1..10) is reused as-is for
the fixed CEI test set -- untouched by this script, and excluded from every
split it builds, so training/validation data can never leak into the number
every fine-tuned checkpoint is finally scored on.

Of the 102 remaining labeled tiles: 12 go to a fixed validation set (used for
early stopping), and the other 90 become the training pool. Nested subsets of
that pool -- sizes 1, 5, 10, 20, 30, 50, 75, 90 -- are built by a greedy
farthest-point ordering in each tile's per-class pixel-share space: the first
tile is the most class-balanced one (highest label entropy), and each next
tile is the one most different in land-cover composition from everything
already picked. Taking a prefix of that ordering for subset size N means
smaller subsets are still as diverse as N tiles can be, and subsets are
strictly nested (the size-5 set is the size-10 set's first 5 tiles, etc.) by
construction.

Small subsets (N < --oversample-target) get their split file's tile names
repeated so one training epoch still sees a reasonable number of (randomly
cropped and augmented) samples -- e.g. N=1 repeated 60x means each epoch draws
60 different random 512x512 crops/flips/rotations of that one image, not one.
No dataset code changes needed: OpenEarthMapDataset already treats every split
file line as one independent sample.

Before allocating val/pool, tiles containing any below-average-share class are
given a head start into the val set (up to --guarantee-per-class each) so the
validation signal isn't blind to rare classes -- the same idea already applied
when the CEI test set itself was built.

Usage
-----
python tools/cei/build_finetune_splits.py
python tools/cei/build_finetune_splits.py --sizes 1 5 10 20 30 50 75 90
python tools/cei/build_finetune_splits.py --val-size 12 --seed 42
"""

import argparse
import os
import sys

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.datasets.taxonomy import CEI_CLASS_NAMES  # noqa: E402

NUM_CLASSES = len(CEI_CLASS_NAMES)
RARE_SHARE_THRESHOLD = 0.10   # a class averaging under 10% of pixels counts as rare
TILE_PRESENCE_THRESHOLD = 0.02  # a tile "contains" a class above 2% of its pixels


def class_shares(mask_path):
    """Per-class pixel-share vector (length NUM_CLASSES) for one raw CEI mask."""
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    counts = np.bincount(mask.ravel(), minlength=NUM_CLASSES + 1)
    total = counts[1:].sum()
    return counts[1:] / total if total else np.zeros(NUM_CLASSES)


def load_pool(root, image_dir, mask_dir, exclude):
    names = sorted(
        f for f in os.listdir(os.path.join(root, image_dir))
        if f.lower().endswith(".tif") and f not in exclude
    )
    shares = np.stack([class_shares(os.path.join(root, mask_dir, n)) for n in names])
    return names, shares


def guaranteed_val_split(names, shares, val_size, seed, guarantee_per_class):
    """Reserve a few tiles per rare class into val first, then fill randomly."""
    rng = np.random.default_rng(seed)
    mean_share = shares.mean(axis=0)
    rare_classes = [c for c in range(NUM_CLASSES) if mean_share[c] < RARE_SHARE_THRESHOLD]

    remaining = set(range(len(names)))
    val_idx = []

    for c in rare_classes:
        candidates = [i for i in remaining if shares[i, c] > TILE_PRESENCE_THRESHOLD]
        rng.shuffle(candidates)
        for i in candidates[:guarantee_per_class]:
            if len(val_idx) >= val_size:
                break
            val_idx.append(i)
            remaining.discard(i)

    rest = list(remaining)
    rng.shuffle(rest)
    cursor = 0
    while len(val_idx) < val_size and cursor < len(rest):
        val_idx.append(rest[cursor])
        cursor += 1
    pool_idx = rest[cursor:]

    return val_idx, pool_idx


def diverse_order(pool_idx, shares, seed):
    """Greedy farthest-point ordering in class-share space (see module docstring)."""
    pool_idx = list(pool_idx)
    vectors = shares[pool_idx]

    eps = 1e-9
    entropy = -(vectors * np.log(vectors + eps)).sum(axis=1)
    start = int(np.argmax(entropy))

    selected = [start]
    remaining = set(range(len(pool_idx))) - {start}
    min_dist = np.linalg.norm(vectors - vectors[start], axis=1)
    min_dist[start] = -1

    while remaining:
        best = max(remaining, key=lambda i: min_dist[i])
        selected.append(best)
        remaining.discard(best)
        new_dist = np.linalg.norm(vectors - vectors[best], axis=1)
        min_dist = np.minimum(min_dist, new_dist)
        min_dist[best] = -1

    return [pool_idx[i] for i in selected]


def write_split(path, names):
    with open(path, "w", encoding="utf-8") as handle:
        for name in names:
            handle.write(name + "\n")


def report_coverage(label, names, root, mask_dir):
    totals = np.zeros(NUM_CLASSES, dtype=np.int64)
    for name in names:
        mask = cv2.imread(os.path.join(root, mask_dir, name), cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        totals += np.bincount(mask.ravel(), minlength=NUM_CLASSES + 1)[1:]
    total = totals.sum()
    missing = [CEI_CLASS_NAMES[i] for i in range(NUM_CLASSES) if totals[i] == 0]
    shares = ", ".join(f"{CEI_CLASS_NAMES[i]} {100*totals[i]/total:.1f}%"
                       for i in range(NUM_CLASSES) if totals[i] > 0)
    print(f"  {label:<10} {len(names):3d} tiles -- {shares}")
    if missing:
        print(f"             MISSING: {', '.join(missing)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/CEI_data")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--mask-dir", default="masks")
    parser.add_argument("--test-split", default="test_split.txt",
                        help="Existing fixed test split to exclude from val/pool.")
    parser.add_argument("--val-size", type=int, default=12)
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[1, 5, 10, 20, 30, 50, 75, 90],
                        help="Nested training subset sizes (must be <= pool size).")
    parser.add_argument("--oversample-target", type=int, default=60,
                        help="Repeat small subsets' filenames so each epoch sees "
                             "roughly this many samples.")
    parser.add_argument("--guarantee-per-class", type=int, default=1,
                        help="Rare-class tiles reserved into val before random fill.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/CEI_data/finetune_splits")
    args = parser.parse_args()

    test_path = os.path.join(args.root, args.test_split)
    with open(test_path, "r", encoding="utf-8") as handle:
        test_names = {line.strip() for line in handle if line.strip()}
    print(f"Excluding {len(test_names)} existing test tile(s) from {test_path}.")

    names, shares = load_pool(args.root, args.image_dir, args.mask_dir, test_names)
    print(f"{len(names)} tile(s) available for val + training pool.")

    if max(args.sizes) > len(names) - args.val_size:
        raise SystemExit(f"Largest requested size {max(args.sizes)} exceeds the "
                         f"available pool ({len(names) - args.val_size} after val).")

    val_local, pool_local = guaranteed_val_split(
        names, shares, args.val_size, args.seed, args.guarantee_per_class
    )
    val_names = [names[i] for i in val_local]
    ordered_pool = diverse_order(pool_local, shares, args.seed)
    pool_names = [names[i] for i in ordered_pool]

    os.makedirs(args.out, exist_ok=True)
    write_split(os.path.join(args.out, "val.txt"), val_names)
    write_split(os.path.join(args.out, "pool.txt"), pool_names)

    for size in sorted(args.sizes):
        subset = pool_names[:size]
        repeat = max(1, args.oversample_target // size)
        oversampled = subset * repeat
        write_split(os.path.join(args.out, f"train_{size}.txt"), oversampled)
        print(f"train_{size}.txt: {size} unique tile(s) x{repeat} = "
              f"{len(oversampled)} lines/epoch")

    print(f"\nWritten to {args.out}/\n")
    print("Class coverage:")
    report_coverage("test", sorted(test_names), args.root, args.mask_dir)
    report_coverage("val", val_names, args.root, args.mask_dir)
    for size in sorted(args.sizes):
        report_coverage(f"train_{size}", pool_names[:size], args.root, args.mask_dir)


if __name__ == "__main__":
    main()
