"""Show the confusion matrix from a metrics JSON written by evaluate.py.

The matrix is already saved -- evaluate.py stores everything
``SegmentationMetrics.compute()`` returns -- but it is not printed, and raw pixel
counts are hard to read. This prints it row-normalised, which answers the
question that usually matters: *when a class was truly X, what did the model
call it?*

Rows are ground truth, columns are predictions, so the diagonal is recall and
every off-diagonal cell is a specific confusion. Ignored pixels never enter the
matrix, so rows sum to 100%.

Usage
-----
python tools/show_confusion.py experiments/cei_exp01_oem2cei/logs/cei_test_tta.json
python tools/show_confusion.py <metrics.json> --counts        # raw pixel counts
python tools/show_confusion.py <metrics.json> --heatmap out.png
"""

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.datasets.taxonomy import CEI_CLASS_NAMES  # noqa: E402


def short(name, width=5):
    return name[:width]


def print_matrix(matrix, names, counts):
    """Print the confusion matrix, row-normalised unless ``counts`` is set."""
    totals = matrix.sum(axis=1, keepdims=True)
    shown = matrix if counts else 100.0 * matrix / np.maximum(totals, 1)

    header = " " * 17 + "".join(f"{short(n):>8}" for n in names)
    print(header)
    print(" " * 17 + "-" * (8 * len(names)))

    for index, name in enumerate(names):
        cells = []
        for column in range(len(names)):
            value = shown[index, column]
            if counts:
                text = f"{int(value):>8d}"
            elif index == column:
                text = f"{value:>7.1f}*"          # diagonal = recall
            elif value >= 0.05:
                text = f"{value:>8.1f}"
            else:
                text = f"{'.':>8}"               # keep near-zero cells quiet
            cells.append(text)
        print(f"  {name:<15}" + "".join(cells))

    if not counts:
        print("\n  rows = ground truth, columns = prediction, values = % of the "
              "true class\n  * diagonal = recall")


def save_heatmap(matrix, names, path):
    """Write a row-normalised heatmap PNG. Uses cv2 so no new dependency."""
    import cv2

    normalised = 100.0 * matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    cell, pad = 64, 130
    size = cell * len(names)
    # Right/bottom margin so the last column header and row label are not clipped.
    margin = 70
    canvas = np.full((size + pad + margin, size + pad + margin, 3), 255, np.uint8)

    for row in range(len(names)):
        for column in range(len(names)):
            value = normalised[row, column]
            # White -> dark blue with increasing value.
            shade = int(255 - min(value, 100) * 2.2)
            colour = (255, shade, shade) if row == column else (shade, shade, 255)
            y, x = pad + row * cell, pad + column * cell
            cv2.rectangle(canvas, (x, y), (x + cell, y + cell), colour, -1)
            cv2.rectangle(canvas, (x, y), (x + cell, y + cell), (200, 200, 200), 1)
            if value >= 0.5:
                text_colour = (0, 0, 0) if value < 55 else (255, 255, 255)
                cv2.putText(canvas, f"{value:.0f}", (x + 14, y + 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_colour, 1, cv2.LINE_AA)

    for index, name in enumerate(names):
        cv2.putText(canvas, short(name, 11), (4, pad + index * cell + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, short(name, 8), (pad + index * cell + 4, pad - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "true \\ predicted", (4, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(path, canvas)
    print(f"\nheatmap written to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", help="Metrics JSON written by evaluate.py.")
    parser.add_argument("--counts", action="store_true",
                        help="Show raw pixel counts instead of row percentages.")
    parser.add_argument("--heatmap", default=None, help="Also write a PNG heatmap.")
    parser.add_argument("--top", type=int, default=6,
                        help="How many worst confusions to list (0 to skip).")
    args = parser.parse_args()

    with open(args.metrics, "r", encoding="utf-8") as handle:
        result = json.load(handle)

    if "confusion_matrix" not in result:
        raise SystemExit(
            f"{args.metrics} has no confusion_matrix. It was probably written by "
            "an older run -- re-run evaluate.py to regenerate it."
        )

    matrix = np.array(result["confusion_matrix"], dtype=np.int64)
    names = CEI_CLASS_NAMES[:matrix.shape[0]]

    print(f"{args.metrics}")
    print(f"  split {result.get('split')}  epoch {result.get('epoch')}  "
          f"tta {result.get('tta')}")
    print(f"  OA {result.get('OA'):.4f}  mIoU {result.get('mIoU'):.4f}  "
          f"mF1 {result.get('mF1'):.4f}\n")

    print_matrix(matrix, names, args.counts)

    if args.top:
        pairs = []
        for row in range(len(names)):
            total = matrix[row].sum()
            if not total:
                continue
            for column in range(len(names)):
                if row != column:
                    pairs.append((100.0 * matrix[row, column] / total,
                                  names[row], names[column]))
        print(f"\n  Worst confusions:")
        for share, true_name, predicted in sorted(pairs, reverse=True)[:args.top]:
            print(f"    {true_name:<15} predicted as {predicted:<15} {share:5.1f}%")

    if args.heatmap:
        save_heatmap(matrix, names, args.heatmap)


if __name__ == "__main__":
    main()
