from typing import Tuple, List

import os
import glob
import json


def _from_json(json_path: str, data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Extract paths and labels from `.json` file of image records, where each
    record has the following schema:
    ```
    [
        {
            "name": "img-name",
            "label": "img-label"
        },
    ]
    ```

    Parameters
    ----------
    json_path : str
        Path to the json file where the dataset is defined.
    data_dir : str
        Path to the folder that holds dataset images.

    Returns
    -------
    Tuple[List[str], List[int]]
        A tuple of lists with image paths and image labels.
    """
    with open(json_path, "r") as f:
        records = json.load(f)

    paths = [os.path.join(data_dir, record["name"] + ".png") for record in records]
    labels = [record["label"] for record in records]

    return paths, labels


def _from_dir(data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Extract paths and labels from images inside a directory with the image
    label inferred from the image name.

    Parameters
    ----------
    data_dir : str
        Path to the folder that holds dataset images.

    Returns
    -------
    Tuple[List[str], List[int]]
        A tuple of lists with image paths and image labels.
    """
    paths = glob.glob(os.path.join(data_dir, "*", "*"))

    labels = [0 if "benign" in path else 1 for path in paths]

    return paths, labels
