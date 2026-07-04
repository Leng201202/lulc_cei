"""Python script to check the dataset for a semantic segmentation task."""

import argparse
import os
import numpy as np
from PIL import Image
import yaml


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def read_split(split_path):
    with open(split_path, 'r') as f:
        names = [line.strip() for line in f.readlines()]
    return names

def find_file(folder, name):
    """
    Supports names with or without extension.
    Example:
    train.txt has abc_001
    actual file may be abc_001.tif or abc_001.png
    """
    if os.path.splitext(name)[1]:  # name has extension
        path = os.path.join(folder, name)
        return path if os.path.exists(path) else None
    
    possible_extensions = ['.tif','.tiff', '.png', '.jpg', '.jpeg']
    for ext in possible_extensions:
        path = os.path.join(folder, name + ext)
        if os.path.exists(path):
            return path
    return None


def index_region_files(root, folder_name):
    """Index files stored as <root>/<region>/<folder_name>/<filename>."""
    index = {}
    possible_extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}

    for current_root, _, files in os.walk(root):
        if os.path.basename(current_root) != folder_name:
            continue

        for filename in files:
            stem, extension = os.path.splitext(filename)
            if extension.lower() not in possible_extensions:
                continue

            path = os.path.join(current_root, filename)
            index[filename] = path
            index[stem] = path

    return index


def main():
    parser = argparse.ArgumentParser();
    parser.add_argument(
        "--config",type=str,help="Path to the config file"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Path to the split file"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5
    )
    args = parser.parse_args()
    config = load_config(args.config)
    dataset_config = config["dataset"]
    root = dataset_config["root"]
    image_dir_name = dataset_config.get("image_dir")
    mask_dir_name = dataset_config.get("mask_dir")
    split_dir = os.path.join(root, dataset_config.get("split_dir", ""))
    split_file = dataset_config.get(f"{args.split}_split", f"{args.split}.txt")
    split_path = os.path.join(split_dir, split_file)
    names = read_split(split_path)

    if image_dir_name and mask_dir_name:
        image_dir = os.path.join(root, image_dir_name)
        mask_dir = os.path.join(root, mask_dir_name)
        image_index = None
        mask_index = None
    else:
        image_dir = "<region>/images"
        mask_dir = "<region>/labels"
        image_index = index_region_files(root, "images")
        mask_index = index_region_files(root, "labels")

    print("=" * 50)
    print("Dataset Check")
    print("=" * 50)
    print("Dataset root:", root)
    print("Image dir:", image_dir)
    print("Mask dir:", mask_dir)
    print("Split path:", split_path)
    print("Total samples in split:", len(names))
    print()
    all_mask_values = set()
    for i, name in enumerate(names[:args.num_samples]):

        if image_index is None:
            image_path = find_file(image_dir, name)
            mask_path = find_file(mask_dir, name)
        else:
            stem = os.path.splitext(name)[0]
            image_path = image_index.get(name) or image_index.get(stem)
            mask_path = mask_index.get(name) or mask_index.get(stem)

        print("-" * 50)

        print(f"Sample {i + 1}: {name}")

        if image_path is None:

            print("Image not found")

            continue

        if mask_path is None:

            print("Mask not found")

            continue

        image = np.array(Image.open(image_path).convert("RGB"))

        mask = np.array(Image.open(mask_path))

        unique_values = np.unique(mask)

        all_mask_values.update(unique_values.tolist())

        print("Image path:", image_path)

        print("Mask path:", mask_path)

        print("Image shape:", image.shape)

        print("Mask shape:", mask.shape)

        print("Mask dtype:", mask.dtype)

        print("Unique mask values:", unique_values)

        if image.shape[:2] != mask.shape[:2]:
            print("Warning: image and mask size do not match")

    print()
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    print("Unique mask values found:", sorted(all_mask_values))
    expected_classes = list(range(config["dataset"]["num_classes"]))
    print("Expected class IDs:", expected_classes)
    print("Ignore index:", config["dataset"]["ignore_index"])

if __name__ == "__main__":
    main()
