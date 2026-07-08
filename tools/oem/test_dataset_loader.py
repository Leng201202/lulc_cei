import random
import sys
from pathlib import Path


# Add the project folder so Python can import "src".
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.datasets.openearthmap_dataset import OpenEarthMapDataset


config = load_config(PROJECT_ROOT / "configs/unet/unet_effb4_oem.yml")

# Use the full dataset path when this file runs from another folder.
config["dataset"]["root"] = str(PROJECT_ROOT / config["dataset"]["root"])

dataset = OpenEarthMapDataset(config, split="train")

sample_count = min(10, len(dataset))
sample_indexes = random.sample(range(len(dataset)), sample_count)

print("Dataset length:", len(dataset))
print("Testing random samples:", sample_indexes)

passed = 0
failed = 0

for number, index in enumerate(sample_indexes, start=1):
    print(f"\nSample {number} (dataset index {index})")
    try:
        image, mask = dataset[index]
    except (FileNotFoundError, ValueError) as error:
        failed += 1
        print("FAILED:", error)
        continue

    passed += 1
    print("PASSED")
    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)
    print("Mask shape:", mask.shape)
    print("Mask dtype:", mask.dtype)
    print("Mask unique values:", mask.unique())

print(f"\nResult: {passed} passed, {failed} failed")
