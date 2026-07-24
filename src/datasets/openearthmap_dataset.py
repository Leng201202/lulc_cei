import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.datasets.transforms import build_normalize
from src.datasets.taxonomy import build_label_lut


class OpenEarthMapDataset(Dataset):
    """Load region-structured image/mask pairs for semantic segmentation.

    The split file supplies image filenames. Each filename is used to infer its
    region, locate the corresponding image and label, and return an augmented
    image tensor together with a zero-based class-index mask.

    The same loader serves OpenEarthMap and any dataset sharing its layout
    (``<region>/images`` + ``<region>/labels``, ``.tif`` masks) -- for example
    the CEI test set. Only the label encoding differs, selected by
    ``dataset.label_map`` (see :func:`src.datasets.taxonomy.build_label_lut`):

    * ``"oem"``        native OpenEarthMap 8-class training (default)
    * ``"oem_to_cei"`` OEM imagery remapped into the 7-class CEI scheme
    * ``"cei"``        native CEI labels (ids ``1-7``, ``0`` = ignore)
    """

    def __init__(self, config, split="train"):
        self.config = config
        self.split = split

        # Dataset paths and preprocessing settings are kept under one config
        # section so the same class can be reused for train/validation/test.
        dataset_config = config["dataset"]

        self.root = dataset_config["root"]
        self.image_dir = dataset_config["image_dir"]
        self.mask_dir = dataset_config["mask_dir"]

        # Some datasets name the mask after the image but not identically --
        # CEI stores "maesuai_1.tif" alongside the mask "maesuai_1_label.tif".
        # Empty for OpenEarthMap, where image and mask share a filename.
        self.mask_suffix = dataset_config.get("mask_suffix", "")

        self.crop_size = dataset_config["crop_size"]
        self.ignore_index = dataset_config["ignore_index"]

        # Brightness at or below which a pixel counts as nodata (image border
        # padding) rather than real ground. Only meaningful for datasets whose
        # background maps to a real class -- IRSAMap. Unset for OEM and CEI, so
        # their behaviour is unchanged.
        self.nodata_to_ignore = dataset_config.get("nodata_to_ignore")

        # How to translate the raw on-disk label values into training indices.
        # Defaults to plain OpenEarthMap so existing configs keep working.
        self.label_map = dataset_config.get("label_map", "oem")
        self._label_lut, self._allowed_raw, self._num_label_classes = build_label_lut(
            self.label_map, ignore_index=self.ignore_index
        )

        # For example, split="train" selects dataset.train_split.
        split_file = dataset_config[f"{split}_split"]
        split_path = os.path.join(self.root, split_file)

        self.file_names = self._read_split(split_path)

        self.transform = self._build_transform(split)

    def _read_split(self, split_path):
        """Read non-empty sample filenames from a split text file."""
        with open(split_path, "r", encoding="utf-8") as f:
            file_names = [line.strip() for line in f.readlines() if line.strip()]
        return file_names

    def _resolve_region_path(self, pattern, filename, suffix=""):
        """
        Your OpenEarthMap structure:
        data/OpenEarthMap/<region>/images/<filename>
        data/OpenEarthMap/<region>/labels/<filename>

        Example:
        filename = aachen_1.tif
        region = aachen

        pattern = <region>/images
        path = data/OpenEarthMap/aachen/images/aachen_1.tif

        A pattern without "<region>" (e.g. CEI's flat "images"/"masks" folders)
        is used as-is. ``suffix`` is inserted before the extension for datasets
        whose labels are named after the image but not identically (e.g.
        "tile_1.tif" -> "tile_1_label.tif"); empty for both OEM and CEI.
        """
        # Remove the image number to get the region name.
        # Example: santa_rosa_10.tif becomes santa_rosa.
        region = os.path.splitext(filename)[0].rsplit("_", 1)[0]
        folder = pattern.replace("<region>", region)

        if suffix:
            stem, extension = os.path.splitext(filename)
            filename = f"{stem}{suffix}{extension}"

        return os.path.join(self.root, folder, filename)

    def _convert_mask(self, mask, mask_path, image=None):
        """Translate a raw on-disk mask into a training class-index mask.

        The lookup table built from ``dataset.label_map`` maps each raw value to
        its training index (or ``ignore_index`` for unlabeled/unmapped values),
        so this both remaps classes and masks unlabeled pixels in one step.

        Raw values outside the expected set (e.g. a corrupt mask, or an 8-class
        OEM label loaded with the 7-class ``cei`` map) raise, so a mismatched
        ``label_map`` fails loudly instead of silently discarding pixels.

        ``image`` is only needed when ``nodata_to_ignore`` is set; see below.
        """
        unexpected = sorted(set(np.unique(mask).tolist()) - self._allowed_raw)
        if unexpected:
            raise ValueError(
                f"Unexpected raw label values in {mask_path}: {unexpected}. "
                f"label_map={self.label_map!r} allows raw values "
                f"{sorted(self._allowed_raw)}."
            )

        converted = self._label_lut[mask]

        # IRSAMap needs this; OEM and CEI leave nodata_to_ignore unset and skip it.
        #
        # IRSA's background code 0 maps to Non-vegetated because the annotators
        # left bareland unlabeled. But ~14% of that background is the near-black
        # border padding of the source imagery, which is not ground at all. Both
        # carry the same mask value, so the only way to tell them apart is to
        # look at the pixels: a black image pixel under background is nodata.
        #
        # Without this, a model learns "black region -> Non-vegetated" and then
        # confidently mislabels the blank CEI captures (maesuai_1, maesuai_5).
        if self.nodata_to_ignore is not None and image is not None:
            nodata = image.max(axis=2) <= self.nodata_to_ignore
            converted = np.where(nodata, self.ignore_index, converted).astype(
                converted.dtype
            )

        return converted

    def _build_transform(self, split):
        """Build split-specific image and mask transformations."""
        # Normalization must match the model weights (see src/datasets/transforms).
        normalize = build_normalize(self.config)
        dataset_config = self.config["dataset"]

        # Constant padding: mask padded with ignore_index so the border never
        # contributes to loss/metrics; image padded with zeros (masked out).
        def pad_to(min_h, min_w, div=None):
            return A.PadIfNeeded(
                min_height=min_h,
                min_width=min_w,
                pad_height_divisor=div,
                pad_width_divisor=div,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=self.ignore_index,
            )

        if split == "train":
            # OpenEarthMap tiles are not a uniform size (e.g. rio tiles are
            # 406px), so smaller-than-crop images are padded up before cropping.
            transforms = [
                pad_to(self.crop_size, self.crop_size),
                # Albumentations applies geometric ops identically to image and
                # mask, preserving pixel-level alignment.
                A.RandomCrop(height=self.crop_size, width=self.crop_size),
            ]
            # The OpenEarthMap paper trains with random cropping only; extra
            # flip/rotate augmentation is opt-in via dataset.augment.
            if dataset_config.get("augment", True):
                transforms += [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ]
            transforms += [normalize, ToTensorV2()]
            return A.Compose(transforms)

        # Evaluation is deterministic. "full" (paper protocol) keeps the whole
        # image, padding only up to the next multiple of 32 so the encoder's
        # 32x downsampling divides evenly; the padding is ignored in metrics.
        # "crop" center-crops to crop_size (faster, but discards image borders).
        eval_mode = dataset_config.get("eval_mode", "full")
        if eval_mode == "full":
            return A.Compose([pad_to(None, None, div=32), normalize, ToTensorV2()])
        return A.Compose([
            pad_to(self.crop_size, self.crop_size),
            A.CenterCrop(height=self.crop_size, width=self.crop_size),
            normalize,
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        """Return one normalized image tensor and its integer class mask."""
        filename = self.file_names[index]

        image_path = self._resolve_region_path(self.image_dir, filename)
        mask_path = self._resolve_region_path(
            self.mask_dir, filename, self.mask_suffix
        )

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found or cannot be read: {image_path}")

        # OpenCV reads color channels as BGR, while pretrained vision models
        # and the normalization statistics below expect RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Mask not found or cannot be read: {mask_path}")

        # Segmentation labels must be a single class-index channel.
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # The LUT guarantees the output is in range, so validation happens on the
        # raw values inside _convert_mask (which knows what label_map expects).
        # The image goes along too, so nodata_to_ignore can tell real background
        # apart from black border padding.
        mask = self._convert_mask(mask, mask_path, image=image)

        # Pass image and mask together to keep random spatial transforms synced.
        transformed = self.transform(image=image, mask=mask)

        image = transformed["image"]
        # Cross-entropy-style segmentation losses require torch.long targets.
        mask = transformed["mask"].long()

        return image, mask
