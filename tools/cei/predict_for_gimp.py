"""Step 1 of the CEI label-correction workflow: predict, then lay out a GIMP workspace.

The trained model gives us a *draft* label for each CEI image. Those drafts are
good but not perfect, so we hand-correct them in GIMP and keep the result as
ground truth. This script produces everything GIMP needs, in one flat folder:

    <output>/
        <stem>_image.png       the source image  (reference layer, never edited)
        <stem>_mask.tif        the draft mask    <-- THIS is the file you edit
        _original/<stem>.tif   pristine copy of the draft, for the change report
        oem_palette.gpl        the 8 class colors, to import into GIMP
        HOW_TO_EDIT.md         step-by-step editing instructions

The mask is written as an RGB TIFF in the official OpenEarthMap palette. TIFF is
lossless, so the exact class colors survive the round trip through GIMP -- a
lossy format such as JPEG would blur them into unrecognizable in-between colors.

When you are done editing, run ``tools/cei/import_from_gimp.py`` to turn the
corrected masks back into dataset labels.

Example
-------
python tools/cei/predict_for_gimp.py \
    --config configs/unet/oem/unet_effb4_oem_v2.yml \
    --checkpoint experiments/exp_02_unet_effb4_oem_paper/checkpoints/best_checkpoint.pth \
    --input data/CEI_data \
    --output outputs/cei_exp02/review \
    --tta
"""

import argparse
import os
import re
import sys

import cv2
import torch

# Make the repository root importable so this tool can reuse the project modules
# (it lives two levels down, in tools/cei/).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from predict import (  # noqa: E402
    build_transform,
    collect_images,
    predict_full,
    predict_tiled,
    read_image,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.visualization import save_gimp_palette, save_mask  # noqa: E402


def tile_index(image_path):
    """Return the trailing number in a filename, or ``None`` if there isn't one.

    CEI tiles are named ``<area>_<n>.tif`` (``maesuai_85.tif``), so this is what
    ``--start-index`` / ``--end-index`` filter on when you add a new batch of
    tiles and only want to predict the new ones.
    """
    match = re.search(r"_(\d+)$", os.path.splitext(os.path.basename(image_path))[0])
    return int(match.group(1)) if match else None


def select_images(image_paths, start_index=None, end_index=None):
    """Keep only the tiles whose trailing number falls in ``[start, end]``.

    Files with no trailing number are kept only when no range was requested --
    a range asks for specific tiles, and an unnumbered file cannot satisfy it.
    Returns ``(selected, skipped_unnumbered)``.
    """
    if start_index is None and end_index is None:
        return list(image_paths), []

    selected, unnumbered = [], []
    for path in image_paths:
        index = tile_index(path)
        if index is None:
            unnumbered.append(path)
            continue
        if start_index is not None and index < start_index:
            continue
        if end_index is not None and index > end_index:
            continue
        selected.append(path)

    # Sort numerically so progress reads 85, 86, ... 112 rather than 85, 100, 101.
    selected.sort(key=tile_index)
    return selected, unnumbered


HOW_TO_EDIT = """# How to correct a predicted mask in GIMP

Each image here comes as a pair:

* `<stem>_image.png` -- the CEI aerial image. Reference only; never edit it.
* `<stem>_mask.tif`  -- the model's draft label. **This is the file you edit.**

## One-time setup

1. Open GIMP.
2. Load the class colors: **Windows > Dockable Dialogs > Palettes**, right-click
   in the list, **Import Palette...**, choose *Palette file*, and select
   `oem_palette.gpl` from this folder. You now have the 8 class colors plus
   "Unlabeled (ignore)" and can click one to make it the foreground color.
3. Pick the **Pencil** tool (`N`), not the Paintbrush. The pencil has hard edges.
   The paintbrush anti-aliases, which invents blended colors that are not real
   classes. (Those get snapped back to the nearest class on import, so nothing
   breaks -- but the boundary ends up a pixel or two off from what you drew.)

## Editing one mask

1. **File > Open** -> `<stem>_mask.tif`.
2. **File > Open as Layers** -> `<stem>_image.png`. The aerial image lands on top.
3. In the **Layers** panel, drag the image layer **below** the mask layer, then
   lower the *mask* layer's **Opacity** to about 60%. You can now see the aerial
   image through the mask and spot where the model got it wrong.
4. Select the mask layer, pick a class color from the palette, and paint the
   corrections with the Pencil. Zoom in (`+`) for boundaries.
5. Set the mask layer's Opacity back to 100%.
6. **File > Export As...**, keep the name `<stem>_mask.tif`, and overwrite.
   In the TIFF export dialog, LZW compression is fine. Do **not** "Save As" an
   `.xcf` and stop there -- the import step reads the `.tif`.

## Class colors

| Class | Color (RGB) |
| --- | --- |
| Bareland | 128, 0, 0 |
| Rangeland | 0, 255, 36 |
| Developed space | 148, 148, 148 |
| Road | 255, 255, 255 |
| Tree | 34, 97, 38 |
| Water | 0, 69, 255 |
| Agriculture land | 75, 181, 73 |
| Building | 222, 31, 7 |
| Unlabeled (ignore) | 0, 0, 0 |

Paint a region black ("Unlabeled") when you genuinely cannot tell what it is.
Those pixels are excluded from the loss instead of teaching the model a guess.

## When you are done

Run the import step to convert the corrected masks into dataset labels:

    python tools/cei/import_from_gimp.py --review <this folder> --output data/CEI_data/labels
"""


def main():
    parser = argparse.ArgumentParser(
        description="Predict CEI images with a trained model and lay out a GIMP "
                    "workspace for hand-correcting the masks."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Trained weights. Use the best checkpoint of your strongest run.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="A CEI image file, or a directory of them.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to create the GIMP workspace in.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="4-way flip test-time augmentation. Slower, but a cleaner draft "
             "mask means less hand-correcting.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Only predict tiles whose trailing number is >= this. Use it when "
             "you add a new batch (e.g. --start-index 85) so already-corrected "
             "tiles are left alone.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Only predict tiles whose trailing number is <= this.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-predict tiles that already have a mask in the output folder. "
             "Off by default, so a re-run never destroys hand-corrections.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=None,
        help="Sliding-window inference for imagery too large to fit in memory. "
             "CEI tiles are 1024x1024 and fit fine, so leave this unset.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=128,
        help="Tile overlap in pixels when --tile_size is set.",
    )
    args = parser.parse_args()

    if args.tile_size is not None and args.overlap >= args.tile_size:
        parser.error("--overlap must be smaller than --tile_size.")
    if (args.start_index is not None and args.end_index is not None
            and args.start_index > args.end_index):
        parser.error("--start-index must not be greater than --end-index.")

    # Work out exactly which tiles to run *before* loading the model, so a range
    # that matches nothing fails in a second instead of after GPU setup.
    image_paths = collect_images(args.input)
    print(f"Found {len(image_paths)} image(s) in {args.input}")

    image_paths, unnumbered = select_images(
        image_paths, args.start_index, args.end_index
    )
    if unnumbered:
        print(f"Skipped {len(unnumbered)} file(s) with no trailing number "
              f"(an index range cannot select them), e.g. "
              f"{os.path.basename(unnumbered[0])}")

    # Never silently clobber a mask the user may have already corrected.
    if not args.overwrite:
        kept = []
        already_done = []
        for path in image_paths:
            stem = os.path.splitext(os.path.basename(path))[0]
            if os.path.exists(os.path.join(args.output, f"{stem}_mask.tif")):
                already_done.append(stem)
            else:
                kept.append(path)
        if already_done:
            print(f"Skipped {len(already_done)} tile(s) that already have a mask "
                  f"in the output folder (pass --overwrite to redo them).")
        image_paths = kept

    if not image_paths:
        raise SystemExit("No images left to predict after filtering.")

    selected_indices = [tile_index(p) for p in image_paths]
    if all(index is not None for index in selected_indices):
        print(f"Predicting {len(image_paths)} tile(s): "
              f"{min(selected_indices)}..{max(selected_indices)}")
    else:
        print(f"Predicting {len(image_paths)} image(s).")

    # Imported here rather than at module import time so the heavy model stack is
    # only pulled in when the tool actually runs.
    from evaluate import FlipTTA
    from src.models.checkpoint import load_model_weights
    from src.models.model_factory import build_model
    from train import select_device, validate_config

    config = load_config(args.config)
    validate_config(config)

    # We load trained weights below, so don't waste time downloading the
    # pretrained encoder we are about to overwrite.
    config["model"]["encoder_weights"] = None
    ignore_index = config["dataset"]["ignore_index"]
    num_classes = config["dataset"]["num_classes"]

    device = select_device()
    print(f"Using device: {device}")

    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    load_model_weights(model, checkpoint)
    model.eval()

    reported_miou = checkpoint.get("best_miou") if isinstance(checkpoint, dict) else None
    if reported_miou is not None:
        print(f"Checkpoint mIoU (validation): {reported_miou:.4f}")

    if args.tta:
        model = FlipTTA(model)
        print("Test-time augmentation: 4-way flip enabled")

    # _original/ keeps an untouched copy of every draft mask, so the import step
    # can report exactly how many pixels you changed by hand.
    original_dir = os.path.join(args.output, "_original")
    os.makedirs(original_dir, exist_ok=True)

    transform = build_transform(config)

    for position, image_path in enumerate(image_paths, start=1):
        image = read_image(image_path)

        if args.tile_size is not None:
            prediction = predict_tiled(
                model, image, transform, device,
                tile_size=args.tile_size,
                overlap=args.overlap,
                num_classes=num_classes,
            )
        else:
            prediction = predict_full(model, image, transform, device)

        stem = os.path.splitext(os.path.basename(image_path))[0]

        # The editable draft mask, and the reference image beside it.
        mask_path = os.path.join(args.output, f"{stem}_mask.tif")
        save_mask(mask_path, prediction, ignore_index=ignore_index)
        save_mask(os.path.join(original_dir, f"{stem}.tif"), prediction,
                  ignore_index=ignore_index)

        image_copy_path = os.path.join(args.output, f"{stem}_image.png")
        cv2.imwrite(image_copy_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        print(f"  [{position}/{len(image_paths)}] {os.path.basename(mask_path)}")

    save_gimp_palette(os.path.join(args.output, "oem_palette.gpl"))
    with open(os.path.join(args.output, "HOW_TO_EDIT.md"), "w", encoding="utf-8") as handle:
        handle.write(HOW_TO_EDIT)

    print(f"\nGIMP workspace ready: {args.output}")
    print("Next: read HOW_TO_EDIT.md, correct the *_mask.tif files in GIMP, "
          "then run tools/cei/import_from_gimp.py")


if __name__ == "__main__":
    main()
