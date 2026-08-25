"""Show the confusion matrix from a metrics JSON written by evaluate.py.

evaluate.py already prints the matrix and saves a heatmap after every test, so
this is for revisiting a saved result: reprint it, dump raw counts, or redraw the
PNG. The rendering itself lives in src/utils/confusion.py, shared with evaluate.py.

Usage
-----
python tools/show_confusion.py experiments/<run>/logs/cei_test_tta.json
python tools/show_confusion.py <metrics.json> --counts
python tools/show_confusion.py <metrics.json> --heatmap out.png
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.confusion import (  # noqa: E402
    format_confusion,
    resolve_class_names,
    save_confusion_heatmap,
    top_confusions,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", help="Metrics JSON written by evaluate.py.")
    parser.add_argument("--normalize", choices=["row", "col", "global", "both", "count"],
                        default="row",
                        help="row=recall (default), col=precision, global=pixel "
                             "share, both=doubly-stochastic (rows+cols sum to 1), "
                             "count=raw pixels.")
    parser.add_argument("--float", dest="fraction", action="store_true",
                        help="Show 0-1 fractions instead of percentages.")
    parser.add_argument("--counts", action="store_true",
                        help="Alias for --normalize count.")
    parser.add_argument("--heatmap", default=None, help="Also write a PNG heatmap.")
    parser.add_argument("--decimals", type=int, default=2,
                        help="Decimal places for percentage cells (default: 2).")
    parser.add_argument("--all-cells", action="store_true",
                        help="Annotate every heatmap cell, even values that "
                             "round to 0.0 (normally left blank).")
    parser.add_argument("--top", type=int, default=6,
                        help="How many worst confusions to list (0 to skip).")
    args = parser.parse_args()
    normalize = "count" if args.counts else args.normalize

    with open(args.metrics, "r", encoding="utf-8") as handle:
        result = json.load(handle)

    matrix = result.get("confusion_matrix")
    if not matrix:
        raise SystemExit(
            f"{args.metrics} has no confusion_matrix. It was probably written by "
            "an older run -- re-run evaluate.py to regenerate it."
        )

    names = resolve_class_names(len(matrix))

    print(args.metrics)
    print(f"  split {result.get('split')}  epoch {result.get('epoch')}  "
          f"tta {result.get('tta')}")
    print(f"  OA {result.get('OA'):.4f}  mIoU {result.get('mIoU'):.4f}  "
          f"mF1 {result.get('mF1'):.4f}\n")

    print(format_confusion(matrix, names, normalize=normalize, fraction=args.fraction,
                           decimals=args.decimals))

    if args.top:
        print("\n  Worst confusions:")
        for share, true_name, predicted in top_confusions(matrix, names, args.top):
            print(f"    {true_name:<15} -> {predicted:<15} {share:5.1f}%")

    if args.heatmap:
        save_confusion_heatmap(matrix, args.heatmap, names, normalize=normalize,
                               decimals=args.decimals, annotate_all=args.all_cells)
        print(f"\nheatmap written to {args.heatmap}")


if __name__ == "__main__":
    main()
