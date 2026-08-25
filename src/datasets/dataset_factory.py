from torch.utils.data import ConcatDataset

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

    # Joint OEM + CEI training (e.g. OEM's full pool alongside a handful of
    # CEI tiles as a "replay" mix). See build_mix_dataset below.
    elif dataset_name == "OEM_CEI_Mix":
        return build_mix_dataset(config, split=split)

    elif dataset_name == "LoveDA":
        raise NotImplementedError("LoveDADataset is not implemented yet.")
    raise ValueError(f"Unknown dataset name: {dataset_name}")


def build_mix_dataset(config, split):
    """Concatenate two (or more) independently-configured sub-datasets.

    ``dataset.sources`` is a list of per-source overrides (root, image_dir,
    mask_dir, label_map, name, and its own train/val/test split filename);
    every other ``dataset`` key (crop_size, num_classes, ignore_index,
    normalization, augment, eval_mode, ...) is shared across all sources, so
    architecture/preprocessing stay identical regardless of which source a
    sample came from -- only the data location and label encoding differ.

    A source that has no ``<split>_split`` key is simply skipped for that
    split. That is how a mix can train on OEM+CEI together while validating
    and testing on CEI alone: the OEM source only defines ``train_split``.

    Reuses build_dataset (and so OpenEarthMapDataset) unchanged for each
    source -- this function only assembles the pieces.
    """
    dataset_config = config["dataset"]
    shared = {k: v for k, v in dataset_config.items() if k not in ("sources", "name")}

    datasets = []
    for source in dataset_config["sources"]:
        if f"{split}_split" not in source:
            continue
        merged = {**shared, **source}
        datasets.append(build_dataset({"dataset": merged}, split=split))

    if not datasets:
        raise ValueError(
            f"No dataset.sources entry defines a '{split}_split' for split={split!r}."
        )
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
