"""Class taxonomies and the label mappings between them.

Two 8-class label spaces are in play:

* **OpenEarthMap (OEM)** -- the dataset we train on. On disk, labels are raw
  codes ``1-8`` (``0`` = unlabeled). Training index = raw code - 1 (``0-7``).
* **CEI** -- the target land-cover scheme we evaluate on. On disk, labels are
  class ids ``1-7`` (``0`` = unlabeled / ignore). Training index = id - 1
  (``0-6``).

CEI is OEM with **Bareland and Developed space merged into one "Non-vegetated"
class**; every other class shares the same colour. So an OEM label can be mapped
into the CEI scheme losslessly except for that (intended) merge. This module is
the single source of truth for that mapping -- the dataset loader, the palette,
and the label tools all import ``OEM_TO_CEI`` / the CEI palette from here.

Internal (0-based) index convention used everywhere in training:

    CEI internal index ``i``  <->  CEI on-disk id ``i + 1``

so turning a model prediction back into a CEI label file is just ``+ 1``.
"""

# --- CEI taxonomy (target scheme), in internal-index order (0-6) --------------
# Index i here corresponds to CEI on-disk id (i + 1).
CEI_CLASS_NAMES = [
    "Rangeland",      # id 1
    "Agriculture",    # id 2
    "Tree",           # id 3
    "Water",          # id 4
    "Building",       # id 5
    "Road",           # id 6
    "Non-vegetated",  # id 7  (OEM Bareland + Developed space)
]

# CEI RGB palette. The seven colours are identical to the corresponding OEM
# classes, so a CEI-trained model's output uses the same colours a reviewer sees.
CEI_CLASS_COLORS = [
    (0, 255, 36),     # Rangeland
    (75, 181, 73),    # Agriculture
    (34, 97, 38),     # Tree
    (0, 69, 255),     # Water
    (222, 31, 7),     # Building
    (255, 255, 255),  # Road
    (128, 0, 0),      # Non-vegetated
]

# Colour for ignored / unlabeled pixels (CEI id 0).
CEI_IGNORE_COLOR = (0, 0, 0)


# --- OEM -> CEI class mapping -------------------------------------------------
# Indexed by OEM *training index* (0-7, i.e. raw code - 1); value is the CEI
# *internal index* (0-6). Bareland (0) and Developed space (2) both fold into
# Non-vegetated (6); every other class maps one-to-one.
#
#   OEM idx  OEM class          -> CEI idx  CEI class
#   0        Bareland              6         Non-vegetated
#   1        Rangeland             0         Rangeland
#   2        Developed space       6         Non-vegetated
#   3        Road                  5         Road
#   4        Tree                  2         Tree
#   5        Water                 3         Water
#   6        Agriculture land      1         Agriculture
#   7        Building              4         Building
OEM_TO_CEI = [6, 0, 6, 5, 2, 3, 1, 4]


def build_label_lut(label_map, ignore_index=255):
    """Return a 256-entry uint8 lookup table: raw on-disk value -> internal index.

    Any raw value not covered by the mapping (including ``0`` = unlabeled) is
    sent to ``ignore_index``, so a single ``lut[mask]`` both remaps and masks in
    one vectorized step. ``label_map`` selects the scheme:

    * ``"oem"``        raw ``1-8`` -> ``0-7``           (native OEM training)
    * ``"oem_to_cei"`` raw ``1-8`` -> CEI ``0-6``       (OEM labels, CEI scheme)
    * ``"cei"``        raw ``1-7`` -> ``0-6``           (native CEI labels)
    """
    import numpy as np

    if label_map == "oem":
        pairs = {raw: raw - 1 for raw in range(1, 9)}
    elif label_map == "oem_to_cei":
        pairs = {raw: OEM_TO_CEI[raw - 1] for raw in range(1, 9)}
        # if irsa write under this for config or rule of class.
    elif label_map == "cei":
        pairs = {raw: raw - 1 for raw in range(1, 8)}
    else:
        raise ValueError(
            f"Unknown dataset.label_map: {label_map!r}. "
            f"Expected one of 'oem', 'oem_to_cei', 'cei'."
        )

    lut = np.full(256, ignore_index, dtype=np.uint8)
    for raw_value, internal_index in pairs.items():
        lut[raw_value] = internal_index

    # allowed_raw is the set of on-disk values we expect to see; 0 (unlabeled)
    # is always allowed. Anything else means a corrupt/mislabeled mask.
    allowed_raw = set(pairs) | {0}
    num_classes = len(set(pairs.values()))
    return lut, allowed_raw, num_classes
