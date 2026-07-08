from src.datasets.openearthmap_dataset import OpenEarthMapDataset


def build_dataset(config, split="train"):
    dataset_name = config["dataset"]["name"]

    if dataset_name == "OpenEarthMap":
        return OpenEarthMapDataset(config, split=split)
    elif dataset_name == "IRSA_Map":
        raise NotImplementedError("IRSA_MapDataset is not implemented yet.")
    elif dataset_name == "LoveDA":
        raise NotImplementedError("LoveDADataset is not implemented yet.")
    raise ValueError(f"Unknown dataset name: {dataset_name}")
