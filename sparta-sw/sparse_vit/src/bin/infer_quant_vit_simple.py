from typing import List, Dict, Tuple

import os
import toml
import json
import argparse

import torch
import torchvision

try:
    import wandb
except ImportError:
    print("`wandb` was not found. `wandb` functionalities will be disabled.")

from torch.utils.data import DataLoader

from sparse_vit.dataset import ImageDatasetInfer
from sparse_vit.dataset.transforms import _img_transform_basic

from sparse_vit.model import build_vit_base_model, build_vit_embed_head_layers
from sparse_vit.training.training import log_multiclass__classification
from sparse_vit.training.utils import validate_device
from sparse_vit.training import train_epoch, validate_epoch, log_model_stats_epoch

from sparse_vit.quantization.brevitas_quant import (
    _iter_target_layers,
    _get_parent_and_attr,
    _replace_with_brevitas,
    _setup_dyadic_residuals,
    fuse_rms_norms,
)

from sparse_vit.quantization.conversion import (
    patch_embedding_layer,
    patch_encoder_residuals,
    patch_mha_softmax,
)

from sparse_vit.quantization.utils import _densify_pruning_state_dict


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
        default="./out_results.json",
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


def init_loss(loss_type: str = "ce", **loss_kw) -> torch.nn.Module:
    """
    Initialize loss class instance.
    """
    LOSS_CLS = {
        "ce": torch.nn.CrossEntropyLoss,
        "bce": torch.nn.BCEWithLogitsLoss,
    }

    loss_cls = LOSS_CLS.get(loss_type)

    if loss_cls is None:
        raise ValueError(
            ("Given `loss_type` argument is not available. " "Choose from {}").format(
                tuple(LOSS_CLS.keys())
            )
        )

    return loss_cls(**loss_kw)


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


def build_quant_model(
    cfg,
    wbits,
    abits,
    weight_per_tensor=False,
    weight_scale_po2=False,
    weight_weight_po2=False,
    act_scale_po2=False,
    fuse_rms_norm=False,
    encoder_only=False,
):
    build_kw = {}
    build_kw.update(cfg["infer"]["model"]["embed"])
    build_kw.update(cfg["infer"]["model"]["encoder"])
    build_kw.update(cfg["infer"]["model"]["task"])

    if fuse_rms_norm is True:
        build_kw["norm_kw"] = {"elementwise_affine": False}

    # Architecture only — the checkpoint's own state_dict provides all values,
    # so there is no need to load any intermediate pretrained/pruned model.
    build_kw["params"] = "init"
    build_kw["params_kw"] = {}

    model = build_vit_base_model(**build_kw)

    if fuse_rms_norm is True:
        fuse_rms_norms(model, encoder_only=encoder_only)

    for name, module in list(_iter_target_layers(model)):

        parent, attr = _get_parent_and_attr(model, name)

        is_heads = name.startswith("heads")
        is_embed = name.startswith("embed")

        if encoder_only is True and (is_heads or is_embed):
            continue

        setattr(
            parent,
            attr,
            _replace_with_brevitas(
                module,
                wbits=wbits,
                abits=abits,
                per_tensor=weight_per_tensor,
                wpo2_s=weight_scale_po2,
                wpo2_w=weight_weight_po2,
                apo2=act_scale_po2,
            ),
        )

    if encoder_only is False:
        patch_embedding_layer(model, abits=abits, po2=act_scale_po2)

    patch_encoder_residuals(model, abits=abits, po2=act_scale_po2)
    patch_mha_softmax(model, abits=abits, po2=act_scale_po2)

    return model


def _load_ckpt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _resolve_state(ckpt):
    """
    Resolve state.
    """
    if isinstance(ckpt, dict):
        for key in ("state", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if all(isinstance(v, torch.Tensor) for v in list(ckpt.values())[:3]):
            return ckpt
    return ckpt


def main():
    """
    Loads dataset, dataloaders, trains, validates and saves weights.
    """
    args = parse_args()

    with open(args.cfg_path, "r") as f:
        cfg = toml.load(f)

    ckpt = _load_ckpt(cfg["model"]["path"])
    state = _resolve_state(ckpt)

    wbits = cfg["quant"]["bits"]
    abits = cfg["quant"]["abits"]
    weight_per_tensor = cfg["quant"]["weight_per_tensor"]
    weight_scale_po2 = cfg["quant"]["weight_scale_po2"]
    weight_weight_po2 = cfg["quant"]["weight_weight_po2"]
    act_scale_po2 = cfg["quant"]["act_scale_po2"]
    fuse_rms_norm = cfg["quant"]["fuse_rms_norm"]

    # Configure device
    device = validate_device(cfg["device"])
    print(device)

    data, data_loader = initialize_map_datasets(cfg)

    loss_compute = init_loss(loss_type="ce")

    model = build_quant_model(
        cfg=cfg,
        wbits=wbits,
        abits=abits,
        weight_per_tensor=weight_per_tensor,
        weight_scale_po2=weight_scale_po2,
        weight_weight_po2=weight_weight_po2,
        act_scale_po2=act_scale_po2,
        fuse_rms_norm=fuse_rms_norm,
    ).to(device.type)

    state, _ = _densify_pruning_state_dict(state)

    missing, unexpected = model.load_state_dict(state, strict=False)

    model.eval()
    model.to(device.type)

    _setup_dyadic_residuals(model)

    # Use `wandb` logger if specified
    if cfg["use_wandb"] is True:

        os.environ["WANDB_MODE"] = cfg["wandb"]["WANDB_MODE"]

        wandb.init(**cfg["wandb"]["init"], config=cfg)
        wandb.run.name = cfg["wandb"]["name"]
        wandb.watch(model)

    infer_kw = {"task": "classification"}
    log_kw = cfg.get("other", {}).get("log", {})

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

    _ = validate_epoch(
        data_loader=data_loader,
        model=model,
        epoch=1,
        device=device,
        loss_compute=loss_compute,
        valid_kw={"epochs": 1},
    )

    log_metrics = log_model_stats_epoch(
        data_loader=data_loader,
        model=model,
        device=device,
        log_kw=log_kw,
    )

    with open(args.out, "w") as f:
        json.dump(log_metrics, f, indent=4)


if __name__ == "__main__":
    main()
