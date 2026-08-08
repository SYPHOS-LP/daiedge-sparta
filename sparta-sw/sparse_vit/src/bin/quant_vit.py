import os
import sys
import glob
import json
import argparse
import toml
import torch

from torch.utils.data import DataLoader, Subset

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    import wandb
except ImportError:
    print("`wandb` was not found. `wandb` functionalities will be disabled.")

from sparse_vit.dataset import ImageDataset
from sparse_vit.dataset import load_cifar10, load_tiny_imagenet
from sparse_vit.dataset import (
    collate_cifar10_train_fn,
    collate_tiny_imagenet_train_fn,
    collate_tiny_imagenet_valid_fn,
)

from sparse_vit.model import build_vit_base_model
from sparse_vit.training.utils import validate_device

from sparse_vit.quantization import quantize_vit
from sparse_vit.quantization.utils import save_results_csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_path",
        type=str,
        help="Path to quantization & finetuning configuration `.toml` file.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="./runs/quantized",
        help="Path to directory to store quantized models.",
    )

    return parser.parse_args()


def _get_dataset_subset(dataset, max_samples: int) -> Subset:
    """
    Get dataset subset.
    """
    max_samples = int(max_samples)
    max_samples = max(1, max_samples)
    subset_indices = list(range(min(max_samples, len(dataset))))
    return Subset(dataset, subset_indices)


def initialize_map_datasets(cfg):
    """
    Initialize Map Dataset.
    """
    LOADERS = {
        "cifar10": load_cifar10,
        "tiny-imagenet": load_tiny_imagenet,
    }

    DATALOADER_TRAIN_KW = {
        "cifar10": {"collate_fn": collate_cifar10_train_fn},
        "tiny-imagenet": {"collate_fn": collate_tiny_imagenet_train_fn},
    }

    DATALOADER_VALID_KW = {
        "cifar10": {},
        "tiny-imagenet": {"collate_fn": collate_tiny_imagenet_valid_fn},
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

    # Select dataset samples
    max_eval_samples = cfg.get("eval", {}).get("max_samples")

    if max_eval_samples is not None:
        data["valid"] = _get_dataset_subset(data["valid"], max_eval_samples)

    max_train_samples = cfg.get("train", {}).get("max_samples")

    if max_train_samples is not None:
        data["train"] = _get_dataset_subset(data["train"], max_train_samples)

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


def resolve_model_paths(model_path: str):
    """
    Validate and resolve model paths.
    """
    if model_path is None:
        raise ValueError(
            "No model path provided. Set --model_path or cfg['model']['path']."
        )

    model_path = os.path.expanduser(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    if os.path.isfile(model_path):
        return [model_path]

    paths = sorted(glob.glob(os.path.join(model_path, "*.pth")))
    if not paths:
        raise FileNotFoundError(f"No .pth files found in directory: {model_path}")

    return paths


def get_unique_run_id():
    """
    Get unique run id. Use wandb run_id if exists, else return a six letter hash.
    """
    if getattr(wandb, "run", None):
        return getattr(wandb.run, "id", None)


def _build_model_kw(cfg: dict) -> dict:
    """
    Helper function for creating configuration file
    """
    params = cfg["train"]["model"]["params"]
    embed = cfg["train"]["model"]["embed"]
    encoder = cfg["train"]["model"]["encoder"]
    task = cfg["train"]["model"]["task"]

    build_vit_kw = {}
    build_vit_kw.update(embed)
    build_vit_kw.update(encoder)
    build_vit_kw.update(task)
    build_vit_kw.update(params)

    return build_vit_kw


def main():
    """
    Loads coonfiguration file, dataloaders, config files and calls `quantize_vit` function.
    """
    args = parse_args()
    with open(args.cfg_path, "r") as f:
        cfg = toml.load(f)

    if "quant" not in cfg:
        raise KeyError(
            "Missing [quant] section in config. "
            f"You passed '{args.cfg_path}', which may be a pruning config. "
            "Use cfgs/quant_vit.toml (or another quantization config)."
        )

    model_path = args.model_path or cfg.get("model", {}).get("path")
    model_paths = resolve_model_paths(model_path)
    quant_scope = cfg["quant"]["quant_scope"]

    if quant_scope not in {"all_conv2d", "pruned_conv2d"}:
        raise ValueError(
            "quant scope must be one of: all, pruned_linear, all_conv2d, pruned_conv2d"
        )
    eval_verbose = bool(cfg["quant"]["eval_verbose"])
    eval_log_interval = cfg["quant"]["eval_log_interval"]
    eval_log_interval = max(1, int(eval_log_interval))

    cfg_device = cfg["device"]
    quant_device = validate_device(cfg_device)
    print(quant_device)

    loss_compute = init_loss(**cfg["train"]["loss"])

    data, loaders = initialize_map_datasets(cfg)

    build_vit_kw = _build_model_kw(cfg)

    # Use `wandb` logger if specified
    if cfg["use_wandb"] is True:

        os.environ["WANDB_MODE"] = cfg["wandb"]["WANDB_MODE"]

        wandb.init(**cfg["wandb"]["init"], config=cfg)
        wandb.run.name = cfg["wandb"]["name"]

    run_id = get_unique_run_id()

    out_cfg = cfg.get("out", {})
    out_dir = out_cfg.get("model_dir") or args.out
    model_name = out_cfg.get("model_name")

    os.makedirs(out_dir, exist_ok=True)

    results = quantize_vit(
        model_paths=model_paths,
        quant_device=quant_device,
        build_vit_kw=build_vit_kw,
        loss_compute=loss_compute,
        eval_verbose=eval_verbose,
        eval_log_interval=eval_log_interval,
        quant_scope=quant_scope,
        loaders=loaders,
        cfg=cfg,
        out_dir=out_dir,
        run_id=run_id,
        model_name=model_name,
    )

    report_path = os.path.join(out_dir, "quant_results.json")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    csv_report_path = save_results_csv(results, out_dir)

    print("\n" + "#" * 80)
    print(f"Done. Wrote quantization report: {report_path}")
    print(f"Done. Wrote quantization report: {csv_report_path}")


if __name__ == "__main__":
    main()
