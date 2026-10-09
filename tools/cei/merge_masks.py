"""Merge the per-tile GeoTIFF masks of one area into a single mosaic.

``predict.py`` writes one mask per tile (``maesuai_1.tif``, ``maesuai_2.tif``,
...). This script glues them back together with :func:`rasterio.merge.merge` so
the whole area can be opened as one raster in QGIS or handed to an area
statistics script.

What it does, step by step:

1. Collect every file matching ``--pattern`` inside ``--input-dir`` and sort them
   by tile number (so ``maesuai_2`` comes before ``maesuai_10``). The output file
   itself is skipped, so re-running after a previous merge is safe.
2. Check the tiles are georeferenced. Merge places each tile by its geotransform,
   so tiles without one would all land on top of each other at the origin and the
   mosaic would come out the size of a single tile. That case fails loudly here
   instead of silently producing a wrong map.
3. Merge them (nearest-neighbour, the only correct resampling for class ids) and
   write the mosaic with the CRS of the first tile, LZW compression and tiling.

Example
-------
python tools/cei/merge_masks.py
python tools/cei/merge_masks.py --input-dir outputs/predictions --pattern "maesuai_*.tif"
python tools/cei/merge_masks.py --nodata 0 --method first --overwrite
"""

import argparse
import glob
import os
import re
import sys

import rasterio
from rasterio.merge import merge


DEFAULT_INPUT_DIR = os.path.join("data", "masks")
DEFAULT_PATTERN = "maesuai_*.tif"
DEFAULT_OUTPUT_NAME = "maesuai_full_prediction.tif"


def tile_sort_key(path):
    """Sort by the trailing tile number so _2 comes before _10."""
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r"(\d+)$", stem)
    return (0, int(match.group(1))) if match else (1, 0), stem


def collect_tiles(input_dir, pattern, output_path):
    """Every file matching the pattern, minus the output file itself."""
    paths = glob.glob(os.path.join(input_dir, pattern))
    output_abs = os.path.abspath(output_path)
    paths = [p for p in paths if os.path.abspath(p) != output_abs]
    return sorted(paths, key=tile_sort_key)


def check_georeferenced(paths):
    """Refuse to merge tiles that carry no geotransform.

    rasterio.merge positions each tile by its transform. With the identity
    transform every tile sits at (0, 0) and the mosaic silently collapses to one
    tile, so this is a hard error rather than a warning.
    """
    ungeoreferenced = []
    for path in paths:
        with rasterio.open(path) as src:
            if src.transform.is_identity or src.crs is None:
                ungeoreferenced.append(path)
    if ungeoreferenced:
        sample = "\n  ".join(os.path.basename(p) for p in ungeoreferenced[:5])
        more = len(ungeoreferenced) - 5
        if more > 0:
            sample += f"\n  ... and {more} more"
        raise SystemExit(
            f"{len(ungeoreferenced)} of {len(paths)} tiles have no CRS/geotransform:\n"
            f"  {sample}\n\n"
            "Merging them would stack every tile at the origin and produce a mosaic "
            "the size of a single tile. Re-export the tiles with their georeferencing "
            "(e.g. gdal_translate -a_srs / -a_ullr, or copy the transform from the "
            "source imagery), then run this again."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-tile GeoTIFF masks into one mosaic with rasterio.merge."
    )
    parser.add_argument(
        "--input-dir", default=DEFAULT_INPUT_DIR,
        help=f"Folder holding the tiles (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--pattern", default=DEFAULT_PATTERN,
        help=f"Glob pattern for the tiles (default: {DEFAULT_PATTERN}).",
    )
    parser.add_argument(
        "--output", default=None,
        help=f"Output GeoTIFF (default: <input-dir>/{DEFAULT_OUTPUT_NAME}).",
    )
    parser.add_argument(
        "--method", default="first", choices=["first", "last", "min", "max"],
        help="How overlapping pixels are resolved (default: first).",
    )
    parser.add_argument(
        "--nodata", type=float, default=None,
        help="Nodata value for gaps and for skipping pixels while merging. "
             "Masks use 0 for unlabeled, so --nodata 0 keeps labeled pixels from "
             "being overwritten by a neighbour's unlabeled border. "
             "Default: whatever the tiles declare.",
    )
    parser.add_argument(
        "--compress", default="LZW",
        help="GeoTIFF compression (default: LZW). Use NONE to disable.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise SystemExit(f"Input folder not found: {args.input_dir}")

    output_path = args.output or os.path.join(args.input_dir, DEFAULT_OUTPUT_NAME)
    if os.path.exists(output_path) and not args.overwrite:
        raise SystemExit(
            f"Output already exists: {output_path}\nPass --overwrite to replace it."
        )

    paths = collect_tiles(args.input_dir, args.pattern, output_path)
    if not paths:
        raise SystemExit(
            f"No tiles matched {args.pattern!r} in {args.input_dir}"
        )
    print(f"Found {len(paths)} tiles in {args.input_dir}")
    print(f"  first: {os.path.basename(paths[0])}   last: {os.path.basename(paths[-1])}")

    check_georeferenced(paths)

    sources = [rasterio.open(p) for p in paths]
    try:
        mosaic, out_transform = merge(
            sources, method=args.method, nodata=args.nodata
        )
        profile = sources[0].profile.copy()
    finally:
        for src in sources:
            src.close()

    profile.update(
        driver="GTiff",
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        count=mosaic.shape[0],
        transform=out_transform,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER",
    )
    if args.compress.upper() == "NONE":
        profile.pop("compress", None)
    else:
        profile.update(compress=args.compress)
    if args.nodata is not None:
        profile.update(nodata=args.nodata)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mosaic)

    height, width = mosaic.shape[1], mosaic.shape[2]
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Wrote {output_path}")
    print(f"  {width} x {height} px, {mosaic.shape[0]} band(s), "
          f"{profile['dtype']}, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
