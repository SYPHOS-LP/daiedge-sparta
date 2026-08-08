from typing import Tuple

import torchvision
import torch.nn.functional as F

from torch.utils.data import default_collate

from .transforms import (
    _img_transform_augment,
    _img_transform_basic,
    _img_transform_augment_deit,
    _img_transform_basic_dct,
    _img_transform_augment_dct,
    _img_transform_augment_deit_dct,
    cutmix_or_mixup,
)


def load_cifar10(data_dir: str = "./data", transform: str = "augment") -> Tuple:
    """
    Load the `CIFAR10` dataset.
    """
    TRANSFORMS = [
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

    if transform.endswith("-dct"):
        trns_valid = transforms["basic-dct"]()
    else:
        trns_valid = transforms["basic"]()

    trns_train = transforms[transform]()

    train_dset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=trns_train
    )

    valid_dset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=trns_valid
    )

    return train_dset, valid_dset


def collate_cifar10_train_fn(batch):
    """
    Collate function to add cutmix or mixup.
    """
    trns = cutmix_or_mixup(num_classes=10)

    imgs, lbls = default_collate(batch)

    lbls = F.one_hot(lbls, num_classes=10).float()

    return trns(imgs, lbls)
