import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class OpenEarthMapDataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        self.split = split

        dataset_config = config["dataset"]

        self.root = dataset_config["root"]
        self.image_dir = dataset_config["image_dir"]
        self.mask_dir = dataset_config["mask_dir"]

        self.crop_size = dataset_config["crop_size"]
        self.ignore_index = dataset_config["ignore_index"]

        split_file = dataset_config[f"{split}_split"]
        split_path = os.path.join(self.root, split_file)

        self.file_names = self._read_split(split_path)

        self.transform = self._build_transform(split)

    def _read_split(self, split_path):
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
        region = filename.split("_")[0]
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
        new_mask = np.full(mask.shape, self.ignore_index, dtype=np.uint8)

        valid_pixels = (mask >= 1) & (mask <= 8)
        new_mask[valid_pixels] = mask[valid_pixels] - 1

        return new_mask

    def _build_transform(self, split):
        if split == "train":
            return A.Compose([
                A.RandomCrop(
                    height=self.crop_size,
                    width=self.crop_size
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
                ToTensorV2()
            ])

        else:
            return A.Compose([
                A.CenterCrop(
                    height=self.crop_size,
                    width=self.crop_size
                ),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        filename = self.file_names[index]

        image_path = self._resolve_region_path(self.image_dir, filename)
        mask_path = self._resolve_region_path(self.mask_dir, filename)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found or cannot be read: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Mask not found or cannot be read: {mask_path}")

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        mask = self._convert_mask(mask)

        transformed = self.transform(image=image, mask=mask)

        image = transformed["image"]
        mask = transformed["mask"].long()

        return image, mask