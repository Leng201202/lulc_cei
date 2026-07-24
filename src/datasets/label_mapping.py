"""Label-mapping collaborators for segmentation datasets.

These decouple *how raw on-disk labels become training indices* from *how a
dataset reads files*. A dataset holds a :class:`LabelMapper` and, optionally, a
:class:`MaskRefiner`, and calls them -- it does not know about lookup tables or
nodata heuristics. That keeps each piece single-purpose and swappable:

* :class:`LabelMapper`      -- abstraction: raw mask -> class-index mask.
* :class:`LutLabelMapper`   -- the concrete mapper, driven by ``label_map``.
* :class:`MaskRefiner`      -- abstraction: adjust a converted mask using the
                               image (e.g. reclassify nodata).
* :class:`NodataMasker`     -- the concrete refiner used by IRSAMap.

Because these are plain, injectable objects, a dataset can be given a different
mapping or refinement rule without editing the dataset class, and each collaborator
can be unit-tested on a small array in isolation.
"""

from abc import ABC, abstractmethod

import numpy as np

from src.datasets.taxonomy import build_label_lut


class LabelMapper(ABC):
    """Maps a raw on-disk mask to a zero-based class-index mask."""

    @abstractmethod
    def __call__(self, mask):
        """Return the class-index mask for ``mask`` (a 2-D uint array)."""

    @property
    @abstractmethod
    def num_classes(self):
        """Number of distinct training classes this mapper produces."""


class LutLabelMapper(LabelMapper):
    """Lookup-table mapper configured by a ``label_map`` name.

    Validates that every raw value is one the scheme expects, so a mask that
    does not match the chosen scheme fails loudly instead of silently having
    unmapped pixels dropped to ignore.
    """

    def __init__(self, label_map, ignore_index=255):
        self.label_map = label_map
        self.ignore_index = ignore_index
        self._lut, self._allowed_raw, self._num_classes = build_label_lut(
            label_map, ignore_index=ignore_index
        )

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def allowed_raw(self):
        return self._allowed_raw

    def __call__(self, mask):
        unexpected = sorted(set(np.unique(mask).tolist()) - self._allowed_raw)
        if unexpected:
            raise ValueError(
                f"Unexpected raw label values {unexpected}. "
                f"label_map={self.label_map!r} allows {sorted(self._allowed_raw)}."
            )
        return self._lut[mask]


class MaskRefiner(ABC):
    """Refines an already-converted mask using the source image."""

    @abstractmethod
    def __call__(self, converted_mask, image):
        """Return a possibly-adjusted copy of ``converted_mask``."""


class NodataMasker(MaskRefiner):
    """Send near-black pixels to ignore.

    IRSAMap's background (code 0) maps to Non-vegetated because bareland is left
    unlabeled -- but the near-black border padding of the source imagery carries
    the same code and is not ground. Only the image distinguishes them, so this
    refiner reclassifies pixels at or below ``threshold`` brightness as ignore.
    """

    def __init__(self, threshold, ignore_index=255):
        self.threshold = threshold
        self.ignore_index = ignore_index

    def __call__(self, converted_mask, image):
        nodata = image.max(axis=2) <= self.threshold
        return np.where(nodata, self.ignore_index, converted_mask).astype(
            converted_mask.dtype
        )
