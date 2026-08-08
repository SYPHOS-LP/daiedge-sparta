from typing import Dict, List, Any

import os
import toml
import copy
import operator
import argparse
import itertools
import subprocess

from functools import reduce


def parse_args():
    """
    Parse input arguments with `argparse`.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg_path",
        type=str,
        help="Path to training configuration `.toml` file.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="./out/",
        help="Path to directory to store results and trained models.",
    )

    args = parser.parse_args()

    return args


def find_with_path(cfg: Dict, path: str) -> Any:
    """
    Find element in configuration
    """
    return reduce(operator.getitem, path.split("."), cfg)


def set_with_path(cfg: Dict, path: str, value: Any) -> None:
    """
    Set element in configuration
    """
    *path, last = path.split(".")

    for bit in path:
        cfg = cfg.setdefault(bit, {})

    if isinstance(cfg[last], dict) and isinstance(value, dict):
        cfg[last].update(value)
    else:
        cfg[last] = value


def build_config_files(base_cfg: Dict, hyperparams: List[Dict]) -> List[Dict]:
    """
    Create a set of config files based on input hyperparams
    """
    all_paths = [h_param["paths"] for h_param in hyperparams]

    all_vals = itertools.product(*[h_param["vals"] for h_param in hyperparams])

    cfgs = []

    for vals in all_vals:

        _cfg = copy.deepcopy(base_cfg)

        for paths, val in zip(all_paths, vals):
            for path in paths:
                set_with_path(_cfg, path, val)

        cfgs.append(_cfg)

    return cfgs


def main():
    """
    Main function.
    """
    args = parse_args()

    with open(args.cfg_path, "r") as f:
        cfg = toml.load(f)

    with open(cfg["opt"]["cfg_path"], "r") as f:
        cfg_ml = toml.load(f)

    # Parse hyperparameter fields to be tuned
    hyperparams = cfg["opt"]["params"]

    cfgs = build_config_files(cfg_ml, hyperparams)

    print("There will be executed {} training sessions.".format(len(cfgs)))

    TMP_PATH = "./cfg_tmp.toml"
    # Execute script (results will be logged in wandb)
    for _cfg in cfgs:

        with open(TMP_PATH, "w") as tmp:
            toml.dump(_cfg, tmp)

        out = subprocess.run(
            [
                "python",
                cfg["opt"]["exec"],
                "--cfg_path",
                tmp.name,
            ],
            stdout=subprocess.PIPE,
        )

        if out.stderr:
            print("Script `{}` failed.".format(cfg["exec"]))
            return None

        try:
            os.remove(TMP_PATH)
        except Exception:
            pass


if __name__ == "__main__":
    main()
