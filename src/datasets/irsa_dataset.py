"""IRSAMap dataset loader.

A self-contained dataset for IRSAMap that follows SOLID boundaries: the class
does one thing -- turn a split of filenames into (image, mask) tensors -- and
delegates the parts that are not file I/O to injected collaborators:

* label remapping      -> :class:`LabelMapper`   (src/datasets/label_mapping.py)
* nodata reclassifying -> :class:`MaskRefiner`    (same module)
* augmentation policy  -> ``build_segmentation_transforms`` (src/datasets/transforms.py)

So the augmentation policy, the label scheme, and the nodata rule can each change
-- or be swapped for a test double -- without editing this class (open/closed,
dependency inversion). The collaborators are constructed from config by default,
which is why ``build_dataset`` can keep calling ``IRSADataset(config, split=...)``.

On-disk layout (train and test are separate directory trees)::

    data/IRSAMap/
        train/image/<id>.png
        train/SegLabel_vwsbr/<id>.png     # combined mask, road-over-building
        train/SegLabel_rvwsb/<id>.png     # combined mask, building-over-road
        test/image/<id>.png
        test/SegLabel_vwsbr/<id>.png

The two ``SegLabel_`` folders are not copies: they encode a different overlap
priority (169 of 200 sampled tiles differ). ``vwsbr`` is the default.

IRSA specifics, applied as config defaults so they are not retyped per config:

* ``label_map`` -> ``"irsa_to_cei"`` (two-digit IRSA codes into the CEI scheme).
* ``nodata_to_ignore`` -> ``8``: background (code 0) maps to Non-vegetated because
  bareland is left unlabeled, but near-black border padding under that code is
  nodata, not ground. Set to ``null`` in config to disable.
"""

import os

import cv2
from torch.utils.data import Dataset

from src.datasets.label_mapping import LutLabelMapper, NodataMasker
from src.datasets.transforms import build_segmentation_transforms

# Distinguishes "caller did not pass a refiner" (build the IRSA default) from
# "caller passed None" (no refinement at all).
_DEFAULT = object()

IRSA_DEFAULTS = {
    "label_map": "irsa_to_cei",
    "nodata_to_ignore": 8,
}


class IRSADataset(Dataset):
    """Load IRSAMap image/mask pairs, remapped into the CEI scheme.

    Collaborators may be injected (for testing or alternative schemes); when
    omitted they are built from config:

    * ``label_mapper``    -- a :class:`LabelMapper`; default ``LutLabelMapper``
      for ``dataset.label_map``.
    * ``mask_refiner``    -- a :class:`MaskRefiner` applied after mapping;
      default ``NodataMasker`` when ``dataset.nodata_to_ignore`` is set, else none.
    * ``transform_factory`` -- ``(config, split, crop_size, ignore_index) ->
      albumentations pipeline``; default ``build_segmentation_transforms``.
    """

    def __init__(
        self,
        config,
        split="train",
        label_mapper=None,
        mask_refiner=_DEFAULT,
        transform_factory=build_segmentation_transforms,
    ):
        self.config = config
        self.split = split

        dataset_config = config["dataset"]
        for key, value in IRSA_DEFAULTS.items():
            dataset_config.setdefault(key, value)

        self.root = dataset_config["root"]
        self.image_dir = dataset_config["image_dir"]
        self.mask_dir = dataset_config["mask_dir"]
        self.crop_size = dataset_config["crop_size"]
        self.ignore_index = dataset_config["ignore_index"]

        self.label_mapper = label_mapper or self._default_label_mapper(dataset_config)
        self.mask_refiner = (
            self._default_mask_refiner(dataset_config)
            if mask_refiner is _DEFAULT
            else mask_refiner
        )

        split_file = dataset_config[f"{split}_split"]
        self.file_names = self._read_split(os.path.join(self.root, split_file))

        self.transform = transform_factory(
            config, split, self.crop_size, self.ignore_index
        )

    # ------------------------------------------------- default collaborators
    def _default_label_mapper(self, dataset_config):
        label_map = dataset_config["label_map"]
        if not str(label_map).startswith("irsa"):
            raise ValueError(
                f"IRSADataset expects an IRSA label_map (e.g. 'irsa_to_cei'), "
                f"got {label_map!r}. Use OpenEarthMapDataset for OEM/CEI, or "
                f"inject a label_mapper explicitly."
            )
        return LutLabelMapper(label_map, ignore_index=self.ignore_index)

    def _default_mask_refiner(self, dataset_config):
        threshold = dataset_config["nodata_to_ignore"]
        if threshold is None:
            return None
        return NodataMasker(threshold, ignore_index=self.ignore_index)

    # --------------------------------------------------------------- helpers
    def _read_split(self, split_path):
        """Read one image filename per line, ignoring blanks and comments."""
        if not os.path.isfile(split_path):
            raise FileNotFoundError(
                f"Split file not found: {split_path}. Generate it with "
                f"tools/irsa/make_splits.py."
            )
        with open(split_path, "r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle]
        return [n for n in names if n and not n.startswith("#")]

    def _read_image(self, filename):
        path = os.path.join(self.root, self.image_dir, filename)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {path}")
        # OpenCV reads BGR; pretrained models and the normalization expect RGB.
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _read_mask(self, filename):
        path = os.path.join(self.root, self.mask_dir, filename)
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Mask not found or unreadable: {path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask

    # ----------------------------------------------------------- Dataset API
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        """Return one normalized image tensor and its integer class mask."""
        filename = self.file_names[index]

        image = self._read_image(filename)
        mask = self.label_mapper(self._read_mask(filename))
        if self.mask_refiner is not None:
            mask = self.mask_refiner(mask, image)

        # Image and mask go through together so spatial transforms stay aligned.
        transformed = self.transform(image=image, mask=mask)
        return transformed["image"], transformed["mask"].long()
