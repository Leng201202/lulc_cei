"""Convert OpenEarthMap class-index masks to RGB palette images.

Raw OpenEarthMap masks use:
    0 = unlabeled / ignore
    1-8 = land-cover classes

Training masks produced by this project use:
    0-7 = land-cover classes
    255 = unlabeled / ignore
"""

import argparse
from pathlib import Path

from PIL import Image


RAW_PALETTE = {
    0: (0, 0, 0),          # Unlabeled (ignore)
    1: (128, 0, 0),        # Bareland
    2: (0, 255, 36),       # Rangeland
    3: (148, 148, 148),    # Developed space
    4: (255, 255, 255),    # Road
    5: (34, 97, 38),       # Tree
    6: (0, 69, 255),       # Water
    7: (75, 181, 73),      # Agriculture land
    8: (222, 31, 7),       # Building
}

TRAINING_PALETTE = {
    0: (128, 0, 0),        # Bareland
    1: (0, 255, 36),       # Rangeland
    2: (148, 148, 148),    # Developed space
    3: (255, 255, 255),    # Road
    4: (34, 97, 38),       # Tree
    5: (0, 69, 255),       # Water
    6: (75, 181, 73),      # Agriculture land
    7: (222, 31, 7),       # Building
}

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def read_mask(path):
    """Read a mask as a single-channel PIL image."""
    mask = Image.open(path)
    if mask.mode not in ("L", "I;16", "I"):
        mask = mask.getchannel(0)
    return mask


def colorize_mask(mask, label_format, ignore_index=255, strict=False):
    """Convert a single-channel label mask to an RGB PIL image."""
    palette = RAW_PALETTE if label_format == "raw" else TRAINING_PALETTE

    if label_format == "training":
        valid_values = set(palette) | {ignore_index}
    else:
        valid_values = set(palette)

    if hasattr(mask, "get_flattened_data"):
        unique_values = set(mask.get_flattened_data())
    else:
        unique_values = set(mask.getdata())
    invalid_values = sorted(unique_values - valid_values)
    if strict and invalid_values:
        expected = sorted(valid_values)
        raise ValueError(
            f"Invalid mask values {invalid_values}. "
            f"Expected values for {label_format!r}: {expected}."
        )

    if mask.mode != "L":
        mask = mask.point(lambda value: value if value in valid_values else 0).convert("L")

    color_palette = [0] * (256 * 3)
    for value, color in palette.items():
        color_palette[value * 3:value * 3 + 3] = color
    if label_format == "training" and 0 <= ignore_index <= 255:
        color_palette[ignore_index * 3:ignore_index * 3 + 3] = RAW_PALETTE[0]

    mask = mask.point(lambda value: value if value in valid_values else 0)
    mask.putpalette(color_palette)
    return mask.convert("RGB"), invalid_values


def iter_mask_paths(input_path):
    """Yield supported mask files from a file or directory input."""
    if input_path.is_file():
        yield input_path
        return

    for path in sorted(input_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def output_path_for(mask_path, input_path, output_path):
    """Resolve the destination path for one converted mask."""
    if input_path.is_file():
        if output_path is None:
            return mask_path.with_name(f"{mask_path.stem}_rgb.png")
        if output_path.suffix:
            return output_path
        return output_path / f"{mask_path.stem}_rgb.png"

    output_root = output_path or input_path.with_name(f"{input_path.name}_rgb")
    relative_path = mask_path.relative_to(input_path).with_suffix(".png")
    return output_root / relative_path


def convert_file(mask_path, input_path, output_path, label_format, ignore_index, strict, overwrite):
    destination = output_path_for(mask_path, input_path, output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output exists, use --overwrite to replace it: {destination}")

    mask = read_mask(mask_path)
    rgb, invalid_values = colorize_mask(
        mask,
        label_format=label_format,
        ignore_index=ignore_index,
        strict=strict,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(destination)
    return destination, invalid_values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert OpenEarthMap mask images to RGB palette images."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input mask file or directory of mask files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output file or directory. Defaults to '<mask>_rgb.png' for one file "
            "or '<input_dir>_rgb/' for a directory."
        ),
    )
    parser.add_argument(
        "--label-format",
        choices=["raw", "training"],
        default="raw",
        help="Use 'raw' for masks with values 0-8, or 'training' for values 0-7 plus ignore.",
    )
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=255,
        help="Ignore value for --label-format training.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a mask contains values outside the selected label format.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    mask_paths = list(iter_mask_paths(input_path))
    if not mask_paths:
        raise FileNotFoundError(f"No supported mask files found in: {input_path}")

    converted_count = 0
    files_with_invalid_values = []
    total_count = len(mask_paths)
    for index, mask_path in enumerate(mask_paths, start=1):
        destination, invalid_values = convert_file(
            mask_path=mask_path,
            input_path=input_path,
            output_path=args.output,
            label_format=args.label_format,
            ignore_index=args.ignore_index,
            strict=args.strict,
            overwrite=args.overwrite,
        )
        converted_count += 1
        if invalid_values:
            files_with_invalid_values.append((mask_path, invalid_values))
        print(f"[{index}/{total_count}] {mask_path} -> {destination}")

    print(f"Converted {converted_count} mask(s).")
    if files_with_invalid_values:
        print("Warning: invalid values were rendered as black:")
        for path, values in files_with_invalid_values:
            print(f"  {path}: {values}")


if __name__ == "__main__":
    main()
