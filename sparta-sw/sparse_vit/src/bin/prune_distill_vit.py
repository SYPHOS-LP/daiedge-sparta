from typing import List

import os
import toml
import argparse

import torch
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    print("`wandb` was not found. `wandb` functionalities will be disabled.")

from sparse_vit.dataset import load_tiny_imagenet, load_cifar10, load_imagenet
from sparse_vit.dataset import (
    collate_cifar10_train_fn,
    collate_tiny_imagenet_train_fn,
    collate_tiny_imagenet_valid_fn,
    collate_imagenet_train_fn,
    collate_imagenet_valid_fn,
)
from sparse_vit.dataset import ImageDataset

from sparse_vit.model import build_vit_base_model
from sparse_vit.training import (
    train_distill_epoch,
    validate_epoch,
    log_model_stats_epoch,
)
from sparse_vit.training.utils import (
    validate_device,
    model_save,
    freeze_weights,
    init_lr_scheduler_cosine_w_warmup,
)

# pruning
from sparse_vit.pruning.dispatcher import apply_pruning
from sparse_vit.pruning.utils import (
    resolve_prune_amounts,
    parse_training_epochs,
    parse_learning_rates,
)
from sparse_vit.pruning.utils import (
    _build_zero_masks,
    _apply_zero_masks,
    _remove_prune,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg_path",
        type=str,
        help="Path to training configuration `.toml` file.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="./runs/",
        help="Path to directory to store model.",
    )

    args = parser.parse_args()

    return args


def initialize_map_datasets(cfg):
    """
    Initialize Map Dataset.
    """
    LOADERS = {
        "cifar10": load_cifar10,
        "tiny-imagenet": load_tiny_imagenet,
        "imagenet": load_imagenet,
    }

    DATALOADER_TRAIN_KW = {
        "cifar10": {"collate_fn": collate_cifar10_train_fn},
        "tiny-imagenet": {"collate_fn": collate_tiny_imagenet_train_fn},
        "imagenet": {"collate_fn": collate_imagenet_train_fn},
    }

    DATALOADER_VALID_KW = {
        "cifar10": {},
        "tiny-imagenet": {"collate_fn": collate_tiny_imagenet_valid_fn},
        "imagenet": {"collate_fn": collate_imagenet_valid_fn},
    }

    dataloader_kw = {"train": {}, "valid": {}}

    if cfg["dataset"]["format"] == "predefined":

        name = cfg["dataset"]["name"]
        data_path = cfg["dataset"]["path"]
        transform = cfg["dataset"]["transform"]

        loader = LOADERS[name]
        dataloader_kw["train"].update(DATALOADER_TRAIN_KW.get(name, {}))
        dataloader_kw["valid"].update(DATALOADER_VALID_KW.get(name, {}))

        train_dset, valid_dset = loader(data_path, transform=transform)

    elif cfg["dataset"]["format"] == "json":

        data_path = cfg["dataset"]["path"]

        train_dset = ImageDataset.from_json(cfg["dataset"]["json"]["train"], data_path)
        valid_dset = ImageDataset.from_json(cfg["dataset"]["json"]["valid"], data_path)

    elif cfg["dataset"]["format"] == "dir":

        train_dset = ImageDataset.from_dir(cfg["dataset"]["dir"]["train"])
        valid_dset = ImageDataset.from_dir(cfg["dataset"]["dir"]["valid"])

    data = {"train": train_dset, "valid": valid_dset}

    dataloader = {
        split: DataLoader(
            dataset, **cfg["dataset"]["dataloader"][split], **dataloader_kw[split]
        )
        for split, dataset in data.items()
    }

    return data, dataloader


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


def calculate_class_weights(labels: List[str], num_classes: int) -> np.ndarray:
    """
    Determine class weights to scale loss for class imbalance.
    """
    counts = np.bincount(labels, minlength=num_classes)

    _eps = 1e-2

    freqs = counts / counts.sum()
    weight = 1.0 / (freqs + _eps)

    return weight


def get_unique_run_id():
    """
    Get unique run id. Use wandb run_id if exists, else return a six letter hash.
    """
    if getattr(wandb, "run", None):
        return getattr(wandb.run, "id", None)


def main():
    """
    Loads dataset, dataloaders, trains, validates and saves weights.
    """
    args = parse_args()

    with open(args.cfg_path, "r") as f:
        cfg = toml.load(f)

    # Initialize torch `Dataset` and `DataLoader` instances
    data, loader = initialize_map_datasets(cfg)

    # Configure device
    device = validate_device(cfg["device"])

    if device.type == "cpu":
        torch.set_num_interop_threads(4)
        torch.set_num_threads(4)

    print(device)

    # Prepare teacher model kws
    build_teacher_vit_kw = {}
    build_teacher_vit_kw.update(cfg["train"]["teacher"]["model"]["embed"])
    build_teacher_vit_kw.update(cfg["train"]["teacher"]["model"]["encoder"])
    build_teacher_vit_kw.update(cfg["train"]["teacher"]["model"]["task"])
    build_teacher_vit_kw.update(cfg["train"]["teacher"]["model"]["params"])

    t_model = build_vit_base_model(**build_teacher_vit_kw).to(device.type)

    # Prepare student model kws
    build_student_vit_kw = {}
    build_student_vit_kw.update(cfg["train"]["model"]["embed"])
    build_student_vit_kw.update(cfg["train"]["model"]["encoder"])
    build_student_vit_kw.update(cfg["train"]["model"]["task"])
    build_student_vit_kw.update(cfg["train"]["model"]["params"])

    s_model = build_vit_base_model(**build_student_vit_kw).to(device.type)

    if cfg["train"]["layers"]["freeze"] is True:
        freeze_weights(t_model)

    # NOTE: We can parametrize loss, optimizer and scheduler if required
    loss_compute = init_loss(**cfg["train"]["loss"])

    # For pruning we need to aling training epochs, learning rates with pruning schedule
    warump_epochs_per_step, cosine_epochs_per_step = parse_training_epochs(cfg)

    lr_per_step = parse_learning_rates(cfg)

    all_epochs = sum(warump_epochs_per_step) + sum(cosine_epochs_per_step)

    # Use `wandb` logger if specified
    if cfg["use_wandb"] is True:

        os.environ["WANDB_MODE"] = cfg["wandb"]["WANDB_MODE"]

        wandb.init(**cfg["wandb"]["init"], config=cfg)
        wandb.run.name = cfg["wandb"]["name"]
        wandb.watch(s_model)

    # make model savepath if not there
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    kw = {
        "batch_size": cfg["dataset"]["dataloader"]["train"]["batch_size"],
        "epochs": all_epochs,
        "gradient_clip": 5,
        "num_classes": cfg["train"]["model"]["task"]["num_classes"],
        "dstl_type": cfg["other"].get("distill", {}).get("dstl_type", "soft"),
        "model_mode": cfg["train"]["model"]["encoder"].get("mode", "vit"),
    }

    log_kw = cfg.get("other", {}).get("log", {})

    # Get pruning configuration parameters
    prune_amounts = resolve_prune_amounts(**cfg["prune"]["schedule"])
    prune_method = cfg["prune"]["method"]["prune_method"]
    prune_method_kw = cfg["prune"]["method"]["prune_method_kw"]

    finetune = cfg["prune"]["finetune"]
    onlylast = cfg["prune"].get("onlylast", False)

    if prune_method == "sparsegpt":
        prune_method_kw.update({"device": device, "data_loader": loader["train"]})

    e = 0

    # pruning & finetuning step
    for i, (amount, warmup_epochs, cosine_epochs, lr) in enumerate(
        zip(prune_amounts, warump_epochs_per_step, cosine_epochs_per_step, lr_per_step)
    ):

        # apply pruning and finetune if required
        s_model = apply_pruning(
            model=s_model, amount=amount, prune_method=prune_method, **prune_method_kw
        )

        # finetuning step
        if finetune is True:

            # Remove `lr` to not add it twice
            opt_kw = dict(cfg["train"]["optim"])
            opt_kw.pop("lr")

            # Initialize optimizer and scheduler
            optimizer = optim.AdamW(s_model.parameters(), lr=lr, **opt_kw)

            total_epochs = warmup_epochs + cosine_epochs

            scheduler = init_lr_scheduler_cosine_w_warmup(
                optimizer=optimizer,
                cosine_epochs=cosine_epochs,
                warmup_epochs=warmup_epochs,
            )

            # onlylast allows weight regrowing
            if onlylast is False or i == (len(prune_amounts) - 1):

                # Apply zero masks to avoid regrowing during finetuning
                layers, masks = _build_zero_masks(s_model)
                s_model = _apply_zero_masks(s_model, layers=layers, masks=masks)

            for epoch in range(e, e + total_epochs):

                if cfg["dataset"]["name"] == "imagenet":
                    data["train"].set_epoch(epoch)

                _ = train_distill_epoch(
                    data_loader=loader["train"],
                    s_model=s_model,
                    t_model=t_model,
                    loss_compute=loss_compute,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    device=device,
                    training_kw=kw,
                )

                _ = validate_epoch(
                    data_loader=loader["valid"],
                    model=s_model,
                    epoch=epoch,
                    device=device,
                    loss_compute=loss_compute,
                    valid_kw=kw,
                )

            e += total_epochs

            if onlylast is False or i == (len(prune_amounts) - 1):

                # Enforce masks to finalized weights
                _remove_prune(layers)

        else:
            epoch = i

            _ = validate_epoch(
                data_loader=loader["valid"],
                model=s_model,
                epoch=epoch,
                device=device,
                loss_compute=loss_compute,
                valid_kw=kw,
            )

        _ = log_model_stats_epoch(
            data_loader=loader["valid"],
            model=s_model,
            epoch=epoch,
            device=device,
            log_kw=log_kw,
        )

    model_name = cfg["out"]["model_name"]

    _id = get_unique_run_id()

    model_name = (model_name + "-" + _id) if _id is not None else model_name

    save_path = os.path.join(args.out, "{}-ep{}.pth".format(model_name, epoch))

    model_save(s_model, optimizer, scheduler, epoch, save_path)


if __name__ == "__main__":
    main()
