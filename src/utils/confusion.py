"""Render a confusion matrix as text or a heatmap PNG.

Shared by ``evaluate.py`` (which emits one automatically after every test) and
``tools/show_confusion.py`` (which regenerates one from a saved metrics JSON), so
the formatting lives in exactly one place.

The matrix is what ``SegmentationMetrics.compute()`` returns: rows are ground
truth, columns are predictions, so the diagonal is recall and every off-diagonal
cell is a specific confusion. Ignored pixels never enter it, so rows sum to 100%.
"""

import numpy as np

from src.datasets.taxonomy import CEI_CLASS_NAMES
from src.utils.visualization import OEM_CLASS_NAMES


def resolve_class_names(num_classes):
    """Best-effort class names for a matrix of the given size.

    The two label schemes this project trains on have distinct sizes -- CEI is 7
    classes, native OpenEarthMap is 8 -- so the count alone selects the right
    names. Anything else falls back to numeric labels rather than guessing wrong.
    """
    if num_classes == len(CEI_CLASS_NAMES):
        return list(CEI_CLASS_NAMES)
    if num_classes == len(OEM_CLASS_NAMES):
        return list(OEM_CLASS_NAMES)
    return [f"class {i}" for i in range(num_classes)]


def _short(name, width=5):
    return name[:width]


def _sinkhorn(matrix, iterations=1000, tol=1e-9):
    """Doubly-stochastic normalisation via Sinkhorn-Knopp iteration.

    Alternately rescales rows then columns to sum to 1; for a non-negative matrix
    with no empty row or column this converges to a matrix whose rows *and*
    columns both sum to 1. A single division cannot do this -- see the note in
    ``normalize_confusion``.

    Note what it means: the result is neither recall nor precision. It rebalances
    both axes to a uniform marginal, so it shows class-agreement with the effect
    of class frequency removed from *both* the true and predicted sides. The
    diagonal is a balanced agreement score, not a rate you can quote as recall.
    """
    result = matrix.astype(np.float64).copy()
    row_totals, col_totals = result.sum(axis=1), result.sum(axis=0)
    if (row_totals == 0).any() or (col_totals == 0).any():
        empty = sorted(set(np.where(row_totals == 0)[0]) | set(np.where(col_totals == 0)[0]))
        raise ValueError(
            f"Doubly-stochastic normalisation is undefined: class index(es) "
            f"{empty} have no true or no predicted pixels (an all-zero row or "
            f"column cannot be scaled to sum to 1). Use row/col/global instead."
        )
    for _ in range(iterations):
        result = result / result.sum(axis=1, keepdims=True)   # rows -> 1
        result = result / result.sum(axis=0, keepdims=True)   # cols -> 1
        if abs(result.sum(axis=1) - 1.0).max() < tol:
            break
    return result


def normalize_confusion(matrix, mode="row"):
    """Return a float-normalised copy of the matrix.

    * ``"row"``    -- divide each row by its true-class total; rows sum to 1,
                      diagonal is recall.
    * ``"col"``    -- divide each column by its predicted total; columns sum to
                      1, diagonal is precision.
    * ``"global"`` -- divide everything by the grand total; the whole matrix
                      sums to 1.
    * ``"both"``   -- Sinkhorn iteration; rows AND columns both sum to 1
                      (doubly-stochastic). See ``_sinkhorn`` for the caveat.
    * ``"count"``  -- raw integer counts, no normalisation.

    A single division makes rows OR columns sum to 1, never both, unless the
    model predicts exactly as many pixels of each class as truly exist. ``both``
    reaches double-normalisation by iteration instead, at the cost of the numbers
    no longer being recall or precision.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if mode == "count":
        return matrix
    if mode == "row":
        return matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    if mode == "col":
        return matrix / np.maximum(matrix.sum(axis=0, keepdims=True), 1)
    if mode == "global":
        return matrix / max(matrix.sum(), 1)
    if mode == "both":
        return _sinkhorn(matrix)
    raise ValueError(
        f"Unknown normalize mode {mode!r}: use row/col/global/both/count."
    )


def format_confusion(matrix, names=None, normalize="row", fraction=False):
    """Return the confusion matrix as a printable string.

    ``normalize`` selects row / col / global / count (see ``normalize_confusion``).
    ``fraction`` shows 0-1 floats instead of percentages (ignored for counts).
    """
    matrix = np.asarray(matrix, dtype=np.int64)
    names = names or resolve_class_names(matrix.shape[0])
    counts = normalize == "count"
    data = normalize_confusion(matrix, normalize)
    if not counts and not fraction:
        data = data * 100.0

    lines = [" " * 17 + "".join(f"{_short(n):>8}" for n in names),
             " " * 17 + "-" * (8 * len(names))]
    for row, name in enumerate(names):
        cells = []
        for column in range(len(names)):
            value = data[row, column]
            if counts:
                cells.append(f"{int(value):>8d}")
            else:
                near_zero = value < (0.0005 if fraction else 0.05)
                if near_zero and row != column:
                    cells.append(f"{'.':>8}")
                else:
                    text = f"{value:.3f}" if fraction else f"{value:.1f}"
                    cells.append(f"{text + ('*' if row == column else ''):>8}")
        lines.append(f"  {name:<15}" + "".join(cells))

    if not counts:
        axis = {"row": "rows sum to %s (diagonal = recall)",
                "col": "columns sum to %s (diagonal = precision)",
                "global": "whole matrix sums to %s",
                "both": "rows AND columns sum to %s (doubly-stochastic; "
                        "diagonal is balanced agreement, not recall/precision)"}[normalize]
        unit = "1" if fraction else "100%"
        lines.append(f"\n  rows = truth, columns = prediction; {axis % unit}")
    return "\n".join(lines)


def top_confusions(matrix, names=None, k=6):
    """Return the ``k`` largest off-diagonal (true -> predicted, share) triples."""
    matrix = np.asarray(matrix, dtype=np.int64)
    names = names or resolve_class_names(matrix.shape[0])
    pairs = []
    for row in range(len(names)):
        total = matrix[row].sum()
        if not total:
            continue
        for column in range(len(names)):
            if row != column and matrix[row, column]:
                pairs.append((100.0 * matrix[row, column] / total,
                              names[row], names[column]))
    return sorted(pairs, reverse=True)[:k]


def save_confusion_heatmap(matrix, path, names=None, normalize="row", decimals=1):
    """Write a heatmap PNG under the given normalization. cv2 only, no plotting dep.

    ``normalize`` is one of row / col / global / both / count (see
    ``normalize_confusion``). ``decimals`` controls the cell labels -- they are
    shown as floats (never rounded to whole numbers), so 8.7 reads 8.7 not 8.

    Shading is scaled to the matrix's own largest cell, so the strongest cell is
    always dark regardless of the normalization's absolute range (a global matrix
    tops out near 30%, a row matrix near 100%). Only cells that would round to a
    nonzero label at this precision are annotated, to keep the grid readable.
    """
    import cv2

    matrix = np.asarray(matrix, dtype=np.int64)
    names = names or resolve_class_names(matrix.shape[0])
    data = normalize_confusion(matrix, normalize)
    normalised = data if normalize == "count" else 100.0 * data

    peak = float(normalised.max()) or 1.0
    floor = 0.5 * (10.0 ** -decimals)  # smallest value that is nonzero at this precision
    font, fscale = cv2.FONT_HERSHEY_SIMPLEX, 0.4
    cell, pad, margin = 64, 130, 70
    size = cell * len(names)
    canvas = np.full((size + pad + margin, size + pad + margin, 3), 255, np.uint8)

    for row in range(len(names)):
        for column in range(len(names)):
            value = normalised[row, column]
            intensity = min(value / peak, 1.0)          # relative, not absolute
            shade = int(255 - intensity * 175)
            color = (255, shade, shade) if row == column else (shade, shade, 255)
            y, x = pad + row * cell, pad + column * cell
            cv2.rectangle(canvas, (x, y), (x + cell, y + cell), color, -1)
            cv2.rectangle(canvas, (x, y), (x + cell, y + cell), (200, 200, 200), 1)
            if value >= floor:
                text = f"{value:.{decimals}f}"
                (tw, th), _ = cv2.getTextSize(text, font, fscale, 1)
                tx, ty = x + (cell - tw) // 2, y + (cell + th) // 2
                text_color = (0, 0, 0) if intensity < 0.55 else (255, 255, 255)
                cv2.putText(canvas, text, (tx, ty), font, fscale, text_color, 1, cv2.LINE_AA)

    for index, name in enumerate(names):
        cv2.putText(canvas, _short(name, 11), (4, pad + index * cell + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, _short(name, 8), (pad + index * cell + 4, pad - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "true \\ predicted", (4, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(path, canvas)
    return path
