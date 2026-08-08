from typing import List

import os
import glob

from PIL import Image

from torch.utils.data import Dataset

from .utils import _from_dir, _from_json
from .transforms import (
    _img_transform_augment,
    _img_transform_basic,
    _img_transform_augment_deit,
)


class ImageDatasetInfer(Dataset):
    """
    Constructs a simple dataset class for inference.
    """

    def __init__(self, paths: List[str]):
        """
        Initializes a `torch` dataset class for inference.
        """
        super().__init__()

        self.paths = paths
        self.transform = _img_transform_basic()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        """
        Used by the dataloader to form batches and parallelize fetching.

        Parameters
        ----------
        index : int
            Indicates a sample of the dataset

        Returns
        -------
        array : torch.tensor
            Input image tensor.
        """
        img = Image.open(self.paths[index])

        array = self.transform(img)

        return array

    @classmethod
    def from_dir(cls, data_dir):
        """
        Initialize `ImageDataset` instance from path to rood diretory.
        """
        exts = ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")

        paths = [
            path
            for ext in exts
            for path in glob.glob(os.path.join(data_dir, "*", f"*.{ext}"))
        ]

        return cls(paths=paths)


class ImageDataset(Dataset):
    """
    Constructs the ImageDataset.
    """

    def __init__(self, paths, labels: List | None, transform=None):
        """
        Initializes a `torch` dataset class.

        Parameters
        ----------
        paths : List[str]
            A list of image paths.
        labels : List[int]
            A list integers indicating the target class.
        num_classes : int
            Number of classes
        transform : str (Optional)
            Transformations to be applied on the input image data
        """
        super().__init__()

        self.paths = paths
        self.labels = labels if labels is not None else [0] * len(self.paths)

        if transform == "basic":
            self.transform = _img_transform_basic()
        elif transform == "augment":
            self.transform = _img_transform_augment()
        elif transform == "augment-deit":
            self.transform = _img_transform_augment_deit()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        """
        Used by the dataloader to form batches and parallelize fetching.

        Parameters
        ----------
        index : int
            Indicates a sample of the dataset

        Returns
        -------
        array : torch.tensor
            Input image tensor.
        label : torch.tensor
            Output classification label.
        """
        # NOTE:
        # The ViT model expects images that follow the (c, h, w) order.
        # Make sure that the image is correctly returned.
        img = Image.open(self.paths[index])

        array = self.transform(img)

        label = self.labels[index]

        return array, label

    @classmethod
    def from_json(cls, json_path, data_dir):
        """
        Initialize `ImageDataset` instance from `.json` file.
        """
        paths, labels = _from_json(json_path, data_dir)

        return cls(paths=paths, labels=labels)

    @classmethod
    def from_dir(cls, data_dir):
        """
        Initialize `ImageDataset` instance from path to rood diretory.
        """
        paths, labels = _from_dir(data_dir)

        return cls(paths=paths, labels=labels)
