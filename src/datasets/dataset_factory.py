from src.datasets.irsa_dataset import IRSADataset
from src.datasets.openearthmap_dataset import OpenEarthMapDataset


def build_dataset(config, split="train"):
    dataset_name = config["dataset"]["name"]

    # OpenEarthMap and CEI share one loader: they differ only in directory
    # layout and label encoding, both already config options. _resolve_region_path
    # uses a directory pattern as-is when it has no <region> token, cv2 reads
    # .png and .tif alike, and dataset.label_map handles the remapping.
    if dataset_name in ("OpenEarthMap", "CEI"):
        return OpenEarthMapDataset(config, split=split)

    # IRSAMap uses the same loading path but carries dataset-specific conventions
    # (label_map, the nodata rule). IRSADataset is a thin subclass that supplies
    # those defaults; see src/datasets/irsa_dataset.py.
    elif dataset_name == "IRSA_Map":
        return IRSADataset(config, split=split)

    elif dataset_name == "LoveDA":
        raise NotImplementedError("LoveDADataset is not implemented yet.")
    raise ValueError(f"Unknown dataset name: {dataset_name}")
