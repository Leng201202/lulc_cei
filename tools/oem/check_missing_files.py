import os
import argparse
import yaml


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_split(split_path):
    with open(split_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def resolve_region_path(root, pattern, filename):
    stem = os.path.splitext(filename)[0]
    region = stem.rsplit("_", 1)[0]
    folder = pattern.replace("<region>", region)
    return os.path.join(root, folder, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument(
        "--input",
        type=str,
        help="Input filename. Default: <split>.txt"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output filename. Default: <split>_filtered.txt"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    dataset_config = config["dataset"]

    root = dataset_config["root"]
    image_dir = dataset_config["image_dir"]
    mask_dir = dataset_config["mask_dir"]

    split_file = args.input or f"{args.split}.txt"
    split_path = os.path.join(root, split_file)

    names = read_split(split_path)

    missing_images = []
    missing_masks = []
    complete_names = []

    for name in names:
        image_path = resolve_region_path(root, image_dir, name)
        mask_path = resolve_region_path(root, mask_dir, name)

        if not os.path.exists(image_path):
            missing_images.append(image_path)

        if not os.path.exists(mask_path):
            missing_masks.append(mask_path)

        if os.path.exists(image_path) and os.path.exists(mask_path):
            complete_names.append(name)

    output_file = args.output or f"{args.split}_filtered.txt"
    output_path = os.path.join(root, output_file)

    if os.path.abspath(output_path) == os.path.abspath(split_path):
        raise ValueError("Output must not overwrite the original split file.")

    with open(output_path, "w", encoding="utf-8") as f:
        for name in complete_names:
            f.write(f"{name}\n")

    print("=" * 50)
    print("Missing File Check")
    print("=" * 50)
    print("Split:", args.split)
    print("Total samples:", len(names))
    print("Complete pairs:", len(complete_names))
    print("Missing images:", len(missing_images))
    print("Missing masks:", len(missing_masks))
    print("Filtered split:", output_path)

    if missing_images:
        print("\nFirst missing images:")
        for path in missing_images[:20]:
            print(path)

    if missing_masks:
        print("\nFirst missing masks:")
        for path in missing_masks[:20]:
            print(path)


if __name__ == "__main__":
    main()
