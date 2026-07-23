from src.datasets.openearthmap_dataset import OpenEarthMapDataset


def build_dataset(config, split="train"):
    dataset_name = config["dataset"]["name"]

    # CEI shares OpenEarthMap's on-disk layout (<region>/images + <region>/labels,
    # .tif masks); only the label encoding differs, and that is handled by
    # dataset.label_map. So both names use the same loader.
    if dataset_name in ("OpenEarthMap", "CEI"):
        return OpenEarthMapDataset(config, split=split)
    elif dataset_name == "IRSA_Map":
        raise NotImplementedError("IRSA_MapDataset is not implemented yet.")
    elif dataset_name == "LoveDA":
        raise NotImplementedError("LoveDADataset is not implemented yet.")
    raise ValueError(f"Unknown dataset name: {dataset_name}")
