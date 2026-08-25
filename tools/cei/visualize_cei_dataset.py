"""Visualize CEI dataset tiles: image | color mask | overlay, side by side.

Masks under ``data/CEI_data/masks`` store raw CEI class ids (0-7, single
channel); this tool renders them with the CEI palette and writes one panel PNG
per tile, plus an optional combined contact sheet. It also prints the per-tile
class distribution, which is the quickest way to sanity-check a label.

Usage:
    # 6 random tiles (seeded), panels + contact sheet
    python tools/cei/visualize_cei_dataset.py --num 6 --sheet sheet.png

    # specific tiles
    python tools/cei/visualize_cei_dataset.py --tiles maesuai_3 maesuai_42

    # every labeled tile (112 panels)
    python tools/cei/visualize_cei_dataset.py --all
"""

import argparse
import os
import random
import sys

import cv2
import numpy as np

# Raw CEI ids as stored on disk: 0 = unlabeled, 1-7 = classes. This is the
# authoring palette (data/raw/review masks are painted in exactly these RGBs),
# identical to the matching OpenEarthMap class colors.
CEI_CLASSES = [
    (0, "Unlabeled",     (0, 0, 0)),
    (1, "Rangeland",     (0, 255, 36)),
    (2, "Agriculture",   (75, 181, 73)),
    (3, "Tree",          (34, 97, 38)),
    (4, "Water",         (0, 69, 255)),
    (5, "Building",      (222, 31, 7)),
    (6, "Road",          (255, 255, 255)),
    (7, "Non-vegetated", (128, 0, 0)),
]

HEADER = 44      # per-panel caption strip, px
LEGEND = 92      # legend strip below the panels, px
FONT = cv2.FONT_HERSHEY_SIMPLEX


def build_palette_lut():
    """256-entry RGB LUT so a whole mask colorizes in one indexing op."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for class_id, _, color in CEI_CLASSES:
        lut[class_id] = color
    return lut


PALETTE_LUT = build_palette_lut()


def colorize_mask(mask):
    """Raw CEI id mask [H, W] -> RGB [H, W, 3] via the palette."""
    return PALETTE_LUT[mask]


def class_shares(mask):
    """Return {class_id: fraction} for the ids present in the mask."""
    counts = np.bincount(mask.ravel(), minlength=256)
    total = counts.sum()
    return {cid: counts[cid] / total for cid, _, _ in CEI_CLASSES if counts[cid]}


def caption_strip(width, text):
    strip = np.full((HEADER, width, 3), 245, np.uint8)
    cv2.putText(strip, text, (12, HEADER - 15), FONT, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    return strip


def legend_strip(width, shares):
    """Legend of all 8 classes; entries present in this tile show their share."""
    strip = np.full((LEGEND, width, 3), 245, np.uint8)
    columns = 4
    cell_w = width // columns
    for index, (class_id, name, color) in enumerate(CEI_CLASSES):
        row, column = divmod(index, columns)
        x, y = 12 + column * cell_w, 14 + row * 40
        cv2.rectangle(strip, (x, y), (x + 26, y + 26), color[::-1], -1)
        cv2.rectangle(strip, (x, y), (x + 26, y + 26), (120, 120, 120), 1)
        share = shares.get(class_id)
        label = f"{name} {100 * share:.1f}%" if share else name
        # Absent classes are greyed out so presence is readable at a glance.
        text_color = (30, 30, 30) if share else (170, 170, 170)
        cv2.putText(strip, label, (x + 34, y + 20), FONT, 0.55, text_color, 1, cv2.LINE_AA)
    return strip


def tile_panel(image_bgr, mask, alpha):
    """Compose captioned image | mask | overlay panels plus the legend."""
    color_bgr = colorize_mask(mask)[..., ::-1]
    overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, color_bgr, alpha, 0.0)

    columns = []
    for caption, panel in (("image", image_bgr), ("mask", color_bgr), ("overlay", overlay)):
        columns.append(np.concatenate([caption_strip(panel.shape[1], caption), panel], axis=0))
    body = np.concatenate(columns, axis=1)
    return np.concatenate([body, legend_strip(body.shape[1], class_shares(mask))], axis=0)


def select_tiles(root, image_dir, args):
    names = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(os.path.join(root, image_dir))
        if f.lower().endswith(".tif")
    )
    if args.tiles:
        missing = [t for t in args.tiles if t not in names]
        if missing:
            sys.exit(f"Tiles not found under {root}/{image_dir}: {missing}")
        return args.tiles
    if args.all:
        return names
    rng = random.Random(args.seed)
    return sorted(rng.sample(names, min(args.num, len(names))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="data/CEI_data")
    parser.add_argument("--image_dir", default="images")
    parser.add_argument("--mask_dir", default="masks")
    parser.add_argument("--mask_suffix", default="",
                        help="Suffix on mask filenames relative to image names "
                             "(none by default).")
    parser.add_argument("--tiles", nargs="+", default=None,
                        help="Specific tile names, e.g. maesuai_3 maesuai_42.")
    parser.add_argument("--num", type=int, default=6,
                        help="Random sample size when --tiles/--all not given.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all", action="store_true", help="Render every tile.")
    parser.add_argument("--out", default="outputs/cei_dataset_viz",
                        help="Output directory for per-tile panel PNGs.")
    parser.add_argument("--alpha", type=float, default=0.45,
                        help="Mask opacity in the overlay panel.")
    parser.add_argument("--sheet", default=None,
                        help="Also write one combined contact sheet PNG (this name, inside --out).")
    parser.add_argument("--scale", type=float, default=0.4,
                        help="Downscale factor for the contact sheet only.")
    args = parser.parse_args()

    tiles = select_tiles(args.root, args.image_dir, args)
    os.makedirs(args.out, exist_ok=True)

    name_by_id = {cid: name for cid, name, _ in CEI_CLASSES}
    sheet_rows = []
    print(f"{'tile':<16}" + "".join(f"{name_by_id[c]:>15}" for c in range(8)))
    for tile in tiles:
        image = cv2.imread(os.path.join(args.root, args.image_dir, tile + ".tif"))
        mask = cv2.imread(os.path.join(args.root, args.mask_dir,
                                       tile + args.mask_suffix + ".tif"),
                          cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            sys.exit(f"Could not read image or mask for tile {tile!r}.")
        if mask.ndim == 3:
            mask = mask[..., 0]
        bad = set(np.unique(mask)) - {c for c, _, _ in CEI_CLASSES}
        if bad:
            sys.exit(f"{tile}: mask contains non-CEI values {sorted(bad)}.")

        shares = class_shares(mask)
        print(f"{tile:<16}" + "".join(
            f"{100 * shares.get(c, 0):>14.1f}%" if shares.get(c) else f"{'-':>15}"
            for c in range(8)))

        panel = tile_panel(image, mask, args.alpha)
        path = os.path.join(args.out, f"{tile}_panel.png")
        cv2.imwrite(path, panel)
        if args.sheet:
            sheet_rows.append(cv2.resize(panel, None, fx=args.scale, fy=args.scale,
                                         interpolation=cv2.INTER_AREA))

    print(f"\n{len(tiles)} panel(s) written to {args.out}")
    if args.sheet:
        sheet_path = os.path.join(args.out, args.sheet)
        cv2.imwrite(sheet_path, np.concatenate(sheet_rows, axis=0))
        print(f"Contact sheet: {sheet_path}")


if __name__ == "__main__":
    main()
