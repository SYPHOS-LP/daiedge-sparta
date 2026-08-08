import toml
import argparse

import torch

from sparse_vit.model import build_vit_base_model
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


def main():
    """
    Loads dataset, dataloaders, trains, validates and saves weights.
    """
    args = parse_args()

    with open(args.cfg_path, "r") as f:
        cfg = toml.load(f)

    device = validate_device(cfg["device"])

    build_vit_kw = {}
    build_vit_kw.update(cfg["infer"]["model"]["embed"])
    build_vit_kw.update(cfg["infer"]["model"]["encoder"])
    build_vit_kw.update(cfg["infer"]["model"]["task"])

    model = build_vit_base_model(**build_vit_kw).to(device.type)

    print(model)


if __name__ == "__main__":
    main()
