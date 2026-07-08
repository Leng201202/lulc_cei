import argparse
import os
import random

import yaml


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_names(path):
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def resolve_path(root, pattern, filename):
    stem = os.path.splitext(filename)[0]
    region = stem.rsplit("_", 1)[0]
    folder = pattern.replace("<region>", region)
    return os.path.join(root, folder, filename)


def write_names(path, names):
    with open(path, "w", encoding="utf-8") as file:
        for name in names:
            file.write(f"{name}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_config = config["dataset"]

    root = dataset_config["root"]
    image_dir = dataset_config["image_dir"]
    mask_dir = dataset_config["mask_dir"]

    # The official test split has no masks, so use labeled train and val data.
    source_files = ["train.txt", "val.txt"]
    all_names = []

    for source_file in source_files:
        all_names.extend(read_names(os.path.join(root, source_file)))

    # Remove duplicate names while keeping their original order.
    all_names = list(dict.fromkeys(all_names))

    complete_names = []
    for name in all_names:
        image_path = resolve_path(root, image_dir, name)
        mask_path = resolve_path(root, mask_dir, name)

        if os.path.isfile(image_path) and os.path.isfile(mask_path):
            complete_names.append(name)

    random.Random(args.seed).shuffle(complete_names)

    total = len(complete_names)
    train_count = round(total * 0.60)
    val_count = round(total * 0.10)

    train_names = complete_names[:train_count]
    val_names = complete_names[train_count:train_count + val_count]
    test_names = complete_names[train_count + val_count:]

    output_splits = {
        "train_split_60.txt": train_names,
        "val_split_10.txt": val_names,
        "test_split_30.txt": test_names,
    }

    for filename, names in output_splits.items():
        write_names(os.path.join(root, filename), names)

    print("Complete labeled samples:", total)
    print("Train (60%):", len(train_names))
    print("Validation (10%):", len(val_names))
    print("Test (30%):", len(test_names))
    print("Random seed:", args.seed)


if __name__ == "__main__":
    main()
