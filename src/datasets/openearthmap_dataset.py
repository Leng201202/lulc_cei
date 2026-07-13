import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.datasets.transforms import build_normalize


class OpenEarthMapDataset(Dataset):
    """Load OpenEarthMap image/mask pairs for semantic segmentation.

    The split file supplies image filenames. Each filename is used to infer its
    region, locate the corresponding image and label, and return an augmented
    image tensor together with a zero-based class-index mask.
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

        self.crop_size = dataset_config["crop_size"]
        self.ignore_index = dataset_config["ignore_index"]

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

    def _resolve_region_path(self, pattern, filename):
        """
        Your OpenEarthMap structure:
        data/OpenEarthMap/<region>/images/<filename>
        data/OpenEarthMap/<region>/labels/<filename>

        Example:
        filename = aachen_1.tif
        region = aachen

        pattern = <region>/images
        path = data/OpenEarthMap/aachen/images/aachen_1.tif
        """
        # Remove the image number to get the region name.
        # Example: santa_rosa_10.tif becomes santa_rosa.
        region = os.path.splitext(filename)[0].rsplit("_", 1)[0]
        folder = pattern.replace("<region>", region)
        return os.path.join(self.root, folder, filename)

    def _convert_mask(self, mask):
        """
        Raw OpenEarthMap mask values:
        0 = ignore / unlabeled
        1–8 = land-cover classes

        Training mask values:
        255 = ignore
        0–7 = land-cover classes
        """
        # Initialize every pixel as ignored so unexpected raw values are not
        # Because the model output channels are indexed from zero:
        # accidentally treated as trainable classes.
        new_mask = np.full(mask.shape, self.ignore_index, dtype=np.uint8)

        # Loss functions generally expect class IDs to start at zero. and valid pixels is class-id 1-8 from dataset
        valid_pixels = (mask >= 1) & (mask <= 8)
        new_mask[valid_pixels] = mask[valid_pixels] - 1

        return new_mask

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
        mask_path = self._resolve_region_path(self.mask_dir, filename)

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

        mask = self._convert_mask(mask)

        allowed_values = set(range(8)) | {self.ignore_index}
        unique_values = set(np.unique(mask).tolist())
        invalid_values = sorted(unique_values - allowed_values)
        if invalid_values:
            raise ValueError(
                f"Invalid processed mask values in {mask_path}: {invalid_values}. "
                f"Expected classes 0-7 or ignore index {self.ignore_index}."
            )

        # Pass image and mask together to keep random spatial transforms synced.
        transformed = self.transform(image=image, mask=mask)

        image = transformed["image"]
        # Cross-entropy-style segmentation losses require torch.long targets.
        mask = transformed["mask"].long()

        return image, mask
