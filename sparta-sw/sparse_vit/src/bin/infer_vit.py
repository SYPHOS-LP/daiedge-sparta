from typing import List, Dict, Tuple

import os
import toml
import json
import argparse

import torch

from torch.utils.data import DataLoader

from sparse_vit.dataset import ImageDataset

from sparse_vit.model import build_vit_base_model
from sparse_vit.training import infer_epoch
from sparse_vit.training.utils import validate_device, model_load


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg_path",
        type=str,
        help="Path to model configuration `.toml` file.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the `.pth` trained model.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--data_path",
        action="store",
        dest="data_path",
        help="Path to instance to generate predictions for.",
    )

    group.add_argument(
        "--dataset_path",
        action="store",
        dest="dataset_path",
        help="Path to dataset with instances to generate predictions for.",
    )

    parser.add_argument(
        "--out",
        type=str,
        help="Path where the results will be saved.",
    )

    args = parser.parse_args()

    return args


def parse_dataset(dataset_path: str) -> List[str]:
    """
    Load dataset.
    """
    paths = [os.path.join(dataset_path, p) for p in os.listdir(dataset_path)]

    return paths


def initialize_map_datasets(paths: List[str], cfg: Dict) -> Tuple[Dict, Dict]:
    """
    Initialize Map Dataset.
    """
    dataset = ImageDataset(paths, labels=None, transform="basic")

    loader = DataLoader(dataset, **cfg["dataset"]["dataloader"])

    return dataset, loader


def main():
    """
    Loads dataset, dataloaders, trains, validates and saves weights.
    """
    args = parse_args()

    with open(args.cfg_path, "r") as f:
        cfg = toml.load(f)

    # Configure device
    device = validate_device(cfg["device"])

    if device.type == "cpu":
        torch.set_num_interop_threads(4)
        torch.set_num_threads(4)

    print(device)

    # Prepare inference data
    if args.data_path:
        paths = [args.data_path]

    elif args.dataset_path:
        paths = parse_dataset(args.dataset_path)

    _, data_loader = initialize_map_datasets(paths, cfg)

    build_vit_kw = {}
    build_vit_kw.update(cfg["infer"]["model"]["embed"])
    build_vit_kw.update(cfg["infer"]["model"]["encoder"])
    build_vit_kw.update(cfg["infer"]["model"]["task"])
    build_vit_kw.update(cfg["infer"]["model"]["params"])

    model = build_vit_base_model(**build_vit_kw).to(device.type)

    infer_kw = {"task": "classification"}

    preds = infer_epoch(
        data_loader=data_loader, model=model, device=device, infer_kw=infer_kw
    )

    results = {path: pred for path, pred in zip(paths, preds)}

    with open(args.out, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
