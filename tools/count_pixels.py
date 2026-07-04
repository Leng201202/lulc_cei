import os
import argparse
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import Counter


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def read_split(split_path):
    with open(split_path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def index_region_masks(root):
    """Index masks stored as <root>/<region>/labels/<filename>."""
    index = {}
    supported_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    for current_root, _, files in os.walk(root):
        if os.path.basename(current_root) != "labels":
            continue

        for filename in files:
            stem, extension = os.path.splitext(filename)
            if extension.lower() not in supported_extensions:
                continue

            path = os.path.join(current_root, filename)
            index[filename] = path
            index[stem] = path

    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--max_files", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    dataset_config = config["dataset"]
    root = dataset_config["root"]
    mask_dir_name = dataset_config.get("mask_dir")
    split_dir = os.path.join(root, dataset_config.get("split_dir", ""))
    split_file = dataset_config.get(f"{args.split}_split", f"{args.split}.txt")
    split_path = os.path.join(split_dir, split_file)

    names = read_split(split_path)
    if args.max_files is not None:
        names = names[:args.max_files]

    if mask_dir_name and "<region>" not in mask_dir_name:
        mask_dir = os.path.join(root, mask_dir_name)
        mask_index = None
    else:
        mask_dir = "<region>/labels"
        mask_index = index_region_masks(root)

    counter = Counter()
    missing_count = 0

    print("=" * 50)
    print("Counting Mask Pixel Values")
    print("=" * 50)
    print("Split:", args.split)
    print("Mask dir:", mask_dir)
    print("Total files:", len(names))
    print()

    for name in tqdm(names):
        if mask_index is None:
            mask_path = os.path.join(mask_dir, name)
        else:
            stem = os.path.splitext(name)[0]
            mask_path = mask_index.get(name) or mask_index.get(stem)

        if mask_path is None or not os.path.exists(mask_path):
            print("Missing mask:", name)
            missing_count += 1
            continue

        mask = np.array(Image.open(mask_path))
        values, counts = np.unique(mask, return_counts=True)

        for v, c in zip(values, counts):
            counter[int(v)] += int(c)

    print()
    print("=" * 50)
    print("Pixel Count Summary")
    print("=" * 50)

    total = sum(counter.values())
    print("Missing masks:", missing_count)
    print("Total pixels:", f"{total:,}")

    for value in sorted(counter.keys()):
        count = counter[value]
        percent = count / total * 100
        print(f"Value {value}: {count:,} pixels ({percent:.4f}%)")


if __name__ == "__main__":
    main()
