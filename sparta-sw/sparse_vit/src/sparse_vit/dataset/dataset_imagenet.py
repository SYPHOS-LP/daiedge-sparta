"""
This module holds functions for loading the `imagenet` dataset
as provided officialy by HuggingFace here:

    `https://huggingface.co/datasets/ILSVRC/imagenet-1k`

Download and unzip the dataset.
"""

from typing import Tuple, List

import torch
import torch.nn.functional as F

from datasets import load_dataset
from datasets import Dataset, IterableDataset
from huggingface_hub import snapshot_download
from pathlib import Path

from .transforms import (
    _img_imagenet_transform_basic,
    _img_imagenet_transform_augment,
    _img_imagenet_transform_basic_dct,
    _img_imagenet_transform_augment_dct,
    apply_transform_factory,
    cutmix_or_mixup,
)


def get_imagenet_split_files(data_dir: str | Path, split: str) -> list[Path]:
    """
    Get imagenet split files from folder.
    """
    data_dir = Path(data_dir).expanduser().resolve()

    files = sorted((data_dir / "data").glob(f"{split}-*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No Parquet shards found for split {split!r} " f"under {data_dir / 'data'}"
        )

    return files


def download_imagenet_split(split: str, target_dir: str | Path, **kwargs) -> list[str]:
    """
    Download only the Parquet shards belonging to one ImageNet split.

    Parameters
    ----------
    split : str
        The name of the target split to download ("train", "validation", "test").
    target_dir : str | Path
        Directory into which the HF repository files are downloaded.
    **kwargs
        Keyword arguments to be passed to the `snapshot_download` function, such
        as `revision`, `token` and `max_workers`

    Returns
    -------
    List
        A list of downloaded Parquet shard paths.
    """
    IMAGENET_REPO = "ILSVRC/imagenet-1k"

    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split!r}")

    target_dir = Path(target_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=IMAGENET_REPO,
        repo_type="dataset",
        local_dir=str(target_dir),
        allow_patterns=[
            f"data/{split}-*.parquet",
        ],
        **kwargs,
    )

    parquet_files = get_imagenet_split_files(target_dir, split)

    return parquet_files


def download_imagenet_splits(target_dir: str, **kwargs) -> None:
    """
    Download all ImageNet splits.
    """
    # target_dir = "/home/anousias/Documents/Datasets/Imagenet"
    # _ = download_imagenet_split(
    #     split="validation", target_dir=target_dir, max_workers=8
    # )

    splits = ("train", "validation", "test")

    for split in splits:
        _ = download_imagenet_split(split=split, target_dir=target_dir, **kwargs)


def load_imagenet_split(
    split: str, data_dir: str | Path, **kwargs
) -> Dataset | IterableDataset:
    """
    Load an already-downloaded ImageNet split from local Parquet files.

    No Hugging Face Hub access occurs here.

    Parameters
    ----------
    data_dir : str
        Directory into which the repository files are downloaded.
    split : str
        The name of the target split to download ("train", "validation", "test").
    **kwargs
        Keyword arguments to be passed to the `load_dataset` function, such
        as `streaming`, `filters` and `cache_dir`.

    Parameters
    ----------
    Dataset, IterableDataset
        A `Dataset` or `IterableDataset` instance.
    """
    COLUMNS = ["image", "label"]

    files = get_imagenet_split_files(data_dir, split)

    data_files = {split: [str(f) for f in files]}

    return load_dataset(
        "parquet", data_files=data_files, split=split, columns=COLUMNS, **kwargs
    )


def _load_class_labels() -> List:
    """
    Load class labels as an ordered list from index to class name.
    """
    from .dataset_imagenet_classes import IMAGENET2012_CLASSES

    return list(IMAGENET2012_CLASSES.values())


def load_imagenet(data_dir: str, transform: str | None = "augment") -> Tuple:
    """
    Load the `ILSVRC/imagenet-1k"` dataset.

    Parameters
    ----------
    data_dir : str
        The path to the directory of the downloaded `ILSVRC/imagenet-1k"` dataset
        in parquet format from HuggingFace.
    transform : str | None
        A transformation to be applied to the output dataset. If None
        the dataset is returned with the `with_format('torch')` method
        applied.

    Returns
    -------
    Tuple
        A tuple of dataset splits.
    """
    TRANSFORMS = [
        "torch",
        "basic",
        "augment",
        "basic-dct",
        "augment-dct",
    ]

    transforms = {
        "basic": _img_imagenet_transform_basic,
        "augment": _img_imagenet_transform_augment,
        "basic-dct": _img_imagenet_transform_basic_dct,
        "augment-dct": _img_imagenet_transform_augment_dct,
    }

    if transform not in TRANSFORMS:
        raise ValueError(f"Input `transform` is not valid. Choose from {TRANSFORMS}")

    cls_labels = _load_class_labels()

    train_dset = load_imagenet_split(data_dir=data_dir, split="train", streaming=True)
    # valid_dset = load_imagenet_split(data_dir=data_dir, split="validation")
    valid_dset = load_imagenet_split(
        data_dir=data_dir, split="validation", streaming=True
    )

    # shuffle training data via a shuffle buffer
    train_dset = train_dset.shuffle(seed=42, buffer_size=10000)

    if transform == "torch":
        train_dset = train_dset.with_format("torch")
        valid_dset = valid_dset.with_format("torch")

    else:
        func_train = apply_transform_factory(transforms.get(transform), batched=False)

        if transform.endswith("-dct"):
            func_valid = apply_transform_factory(
                transforms.get("basic-dct"), batched=False
            )
        else:
            func_valid = apply_transform_factory(transforms.get("basic"), batched=False)

        train_dset = train_dset.map(func_train)

        # for validation set apply basic transformations (no augmentations)
        # valid_dset.set_transform(func_valid)
        # for test set apply basic transformations (no augmentations)
        valid_dset = valid_dset.map(func_valid)

    return train_dset, valid_dset


def collate_imagenet_train_fn(batch):
    """
    Collate function to return tuple instead of dicts.
    """
    trns = cutmix_or_mixup(num_classes=1000)

    imgs = torch.stack([rec["image"] for rec in batch])
    lbls = torch.LongTensor([rec["label"] for rec in batch])

    lbls = F.one_hot(lbls, num_classes=1000).float()

    return trns(imgs, lbls)


def collate_imagenet_valid_fn(batch):
    """
    Collate function to return tuple instead of dicts.
    """
    imgs = torch.stack([rec["image"] for rec in batch])
    lbls = torch.LongTensor([rec["label"] for rec in batch])

    return (imgs, lbls)
