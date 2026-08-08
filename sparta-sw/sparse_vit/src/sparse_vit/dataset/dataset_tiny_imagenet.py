"""
This module holds functions for loading the `tiny-imagenet` dataset
as provided officialy by Stanford here:

    `wget http://cs231n.stanford.edu/tiny-imagenet-200.zip`

Download and unzip the dataset.
"""

from typing import Dict, Tuple

import os
import torch
import torch.nn.functional as F

from datasets import load_dataset
from datasets import Dataset, Image, Features

from .transforms import (
    _img_transform_augment,
    _img_transform_basic,
    _img_transform_augment_deit,
    _img_transform_basic_dct,
    _img_transform_augment_dct,
    _img_transform_augment_deit_dct,
    apply_transform_factory,
    cutmix_or_mixup,
)


def _load_class_labels(wnids_path: str, words_path: str) -> Dict:
    """
    Load class labels as a map from `wnid` to class name.

    Parameters
    ----------
    wnids_path : str
        Path to `wnids` file with wordnet-ids included in tiny-imagenet.
    words_path : str
        Path to `words` file with wordnet-ids-to-labels mapping.

    Returns
    -------
    Dict
        Mapping from wordnet-ids in tiny-imagenet mapped to labels.
    """
    with open(wnids_path, "r") as f:
        wnids = [line.strip() for line in f if line]

    with open(words_path, "r") as f:
        wnid_to_cls = dict(line.strip().split("\t") for line in f if line)

    return {wnid: wnid_to_cls.get(wnid) for wnid in wnids}


def _parse_annotation_line(line: str) -> Tuple[str, str]:
    """
    Parse line from `tiny-imagenet-200` validation annotation file.
    """
    img_file, label, *_ = line.split()

    return img_file.lower(), label


def load_annotations(path: str) -> Dict:
    """
    Path to `tiny-imagenet-200` validation annotation file.
    """
    with open(path, "r") as f:
        annotations = dict([_parse_annotation_line(line) for line in f if line])

    return annotations


def _parse_train_dataset(train_dset: Dataset) -> Dataset:
    """
    Transform train dataset.
    """

    def _extract_label_from_path(rec):
        return {"label": rec["image"]["path"].replace("\\", "/").split("/")[-3]}

    # Disable decoding to transform dataset and enable.
    train_dset = train_dset.cast_column("image", Image(decode=False))
    train_dset = train_dset.map(_extract_label_from_path)
    train_dset = train_dset.cast_column("image", Image(decode=True))
    train_dset = train_dset.class_encode_column("label")

    return train_dset


def _parse_valid_dataset(
    valid_dset: Dataset, annotations: Dict, features: Features
) -> Dataset:
    """
    Transform validation dataset.
    """

    labels = features.get("label")

    if labels is None:
        raise ValueError("A `ClassLabel` instance is not found in `features`.")

    def _assign_label_from_annotations(rec):

        img_path = os.path.basename(rec["image"]["path"]).lower()

        return {"label": labels.str2int(annotations[img_path])}

    valid_dset = valid_dset.cast_column("image", Image(decode=False))
    valid_dset = valid_dset.map(_assign_label_from_annotations, features=features)
    valid_dset = valid_dset.cast_column("image", Image(decode=True))

    return valid_dset


def load_tiny_imagenet(data_dir: str, transform: str | None = "augment") -> Tuple:
    """
    Load the Stanford `tiny-imagenet-200` dataset.

    Parameters
    ----------
    data_dir : str
        The path to the directory of the downloaded `tiny-imagenet-200` dataset.
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
        "augment-deit",
        "basic-dct",
        "augment-dct",
        "augment-deit-dct",
    ]

    transforms = {
        "basic": _img_transform_basic,
        "augment": _img_transform_augment,
        "augment-deit": _img_transform_augment_deit,
        "basic-dct": _img_transform_basic_dct,
        "augment-dct": _img_transform_augment_dct,
        "augment-deit-dct": _img_transform_augment_deit_dct,
    }

    if transform not in TRANSFORMS:
        raise ValueError(f"Input `transform` is not valid. Choose from {TRANSFORMS}")

    wnids_path = os.path.join(data_dir, "wnids.txt")
    words_path = os.path.join(data_dir, "words.txt")

    wnid_to_cls = _load_class_labels(wnids_path, words_path)

    valid_ann = load_annotations(os.path.join(data_dir, "val", "val_annotations.txt"))

    train_dset = load_dataset("imagefolder", data_dir=data_dir, split="train")
    train_dset = _parse_train_dataset(train_dset)

    features = train_dset.features

    valid_dset = load_dataset("imagefolder", data_dir=data_dir, split="validation")
    valid_dset = _parse_valid_dataset(valid_dset, valid_ann, features=features)

    if transform == "torch":
        train_dset = train_dset.with_format("torch")
        valid_dset = valid_dset.with_format("torch")

    else:
        func_train = apply_transform_factory(transforms.get(transform))

        if transform.endswith("-dct"):
            func_valid = apply_transform_factory(transforms.get("basic-dct"))
        else:
            func_valid = apply_transform_factory(transforms.get("basic"))

        train_dset.set_transform(func_train)

        # for validation set apply basic transformations (no augmentations)
        valid_dset.set_transform(func_valid)

    return train_dset, valid_dset


def collate_tiny_imagenet_train_fn(batch):
    """
    Collate function to return tuple instead of dicts.
    """
    trns = cutmix_or_mixup(num_classes=200)

    imgs = torch.stack([rec["image"] for rec in batch])
    lbls = torch.LongTensor([rec["label"] for rec in batch])

    lbls = F.one_hot(lbls, num_classes=200).float()

    return trns(imgs, lbls)


def collate_tiny_imagenet_valid_fn(batch):
    """
    Collate function to return tuple instead of dicts.
    """
    imgs = torch.stack([rec["image"] for rec in batch])
    lbls = torch.LongTensor([rec["label"] for rec in batch])

    return (imgs, lbls)
