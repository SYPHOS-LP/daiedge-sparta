from typing import List, Dict, Tuple

import toml
import json
import argparse

import torch
import torchvision

from torch.utils.data import DataLoader

from sparse_vit.dataset import ImageDatasetInfer
from sparse_vit.dataset.transforms import _img_transform_basic

from sparse_vit.model import build_vit_base_model, build_vit_embed_head_layers
from sparse_vit.training.training import log_multiclass__classification
from sparse_vit.training.utils import validate_device


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg_path",
        type=str,
        help="Path to model configuration `.toml` file.",
    )

    parser.add_argument(
        "--out",
        type=str,
        help="Path where the results will be saved.",
    )

    args = parser.parse_args()

    return args


def _load_cifar10_test(data_path):
    """
    Load CIFAR10 test set.
    """
    return torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=_img_transform_basic()
    )


def initialize_map_datasets(cfg):
    """
    Initialize Map Dataset.
    """
    if cfg["dataset"]["name"] == "cifar10":
        data = _load_cifar10_test(cfg["dataset"]["path"])

    elif cfg["dataset"]["name"] == "dir":
        data = ImageDatasetInfer.from_dir(cfg["dataset"]["path"])

    loader_kw = cfg["dataset"].get("dataloader")

    if loader_kw:
        dataloader = DataLoader(data, **loader_kw)
    else:
        dataloader = None

    return data, dataloader


def infer_epoch_cifar10(data_loader, model, device, infer_kw):
    """
    Performs inference on a CIFAR10 provided by the dataloader.

    Parameters
    ----------
    data_loader : torch.utils.data.Dataset
        A `pytorch` dataset.
    model : torch.nn.Models
        A `pytorch` model.
    device : torch.device
        Device to validate on i.e. 'cpu' or 'cuda'.
    infer_kw : Dict
        Any additional model inference keyword arguments.

    Returns
    -------
    all_preds: List
        Model predictions for input samples.
    """
    # switch into eval mode
    model.eval()

    all_preds = []

    for idx, (image, _) in enumerate(data_loader):

        # zero gradients
        with torch.no_grad():

            with torch.autocast(device.type, dtype=torch.bfloat16):
                # mount data to device
                image = image.to(device.type)
                logits = model(image)

            preds, _ = log_multiclass__classification(logits, label=None)

        all_preds.extend(preds.tolist())

    return all_preds


def infer_epoch_dir(data_loader, model, device, infer_kw):
    """
    Performs inference on a `ImageDatasetInfer` instance wrapped by the dataloader.

    Parameters
    ----------
    data_loader : torch.utils.data.Dataset
        A `pytorch` dataset.
    model : torch.nn.Models
        A `pytorch` model.
    device : torch.device
        Device to validate on i.e. 'cpu' or 'cuda'.
    infer_kw : Dict
        Any additional model inference keyword arguments.

    Returns
    -------
    all_preds: List
        Model predictions for input samples.
    """
    # switch into eval mode
    model.eval()

    all_preds = []

    for idx, image in enumerate(data_loader):

        # zero gradients
        with torch.no_grad():

            with torch.autocast(device.type, dtype=torch.bfloat16):
                # mount data to device
                image = image.to(device.type)
                logits = model(image)

            preds, _ = log_multiclass__classification(logits, label=None)

        all_preds.extend(preds.tolist())

    return all_preds


def main():
    """
    Loads dataset, dataloaders, trains, validates and saves weights.
    """
    args = parse_args()

    with open(args.cfg_path, "r") as f:
        cfg = toml.load(f)

    # Configure device
    device = validate_device(cfg["device"])
    print(device)

    data, data_loader = initialize_map_datasets(cfg)

    build_vit_kw = {}
    build_vit_kw.update(cfg["infer"]["model"]["embed"])
    build_vit_kw.update(cfg["infer"]["model"]["encoder"])
    build_vit_kw.update(cfg["infer"]["model"]["task"])
    build_vit_kw.update(cfg["infer"]["model"]["params"])

    if cfg["infer"]["model"]["load"]["mode"] == "all":
        model = build_vit_base_model(**build_vit_kw).to(device.type)

    elif cfg["infer"]["model"]["load"]["mode"] == "embed_head":
        embed, heads = build_vit_embed_head_layers(**build_vit_kw)
        embed = embed.to(device.type)
        heads = heads.to(device.type)

    # If we load only `embed` and `heads` layers we need to add
    # a separate function that executes `embed`, sends to device, 
    # gets result and executes `heads`

    infer_kw = {"task": "classification"}

    if cfg["dataset"]["name"] == "cifar10":
        preds = infer_epoch_cifar10(
            data_loader=data_loader, model=model, device=device, infer_kw=infer_kw
        )

    elif cfg["dataset"]["name"] == "dir":
        preds = infer_epoch_dir(
            data_loader=data_loader, model=model, device=device, infer_kw=infer_kw
        )

    if cfg["dataset"]["name"] == "cifar10":
        results = {str(idx): pred for idx, pred in enumerate(preds)}

    elif cfg["dataset"]["name"] == "dir":
        results = {path: pred for path, pred in zip(data.paths, preds)}

    with open(args.out, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
