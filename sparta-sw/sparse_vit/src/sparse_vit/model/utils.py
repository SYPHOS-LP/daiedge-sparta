from typing import Dict, List

from collections import OrderedDict
from torch import nn


def init_model_weights(model: nn.Module, **kwargs) -> None:
    """
    Initializes model weights.
    """
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


def _remove_branch(path: str, branch: str, b_type: str = "branch") -> str:
    """
    Remove branch or consecutive branches from path.
    """
    if b_type == "root":
        _branch = branch + "."

    elif b_type == "branch":
        _branch = "." + branch + "."

    elif b_type == "leaf":
        _branch = "." + branch
    else:
        raise ValueError("Given `b_type` is not valid")

    before, found, after = path.partition(_branch)

    if not found:
        return before

    if b_type == "root":
        return ".".join([after])

    elif b_type == "branch":
        return ".".join([before, after])

    elif b_type == "leaf":
        return ".".join([before])


def _replace_branch(path: str, branch: str, repl: str, b_type: str = "branch") -> str:
    """
    Replace branch or consecutive branches from path.
    """
    if b_type == "root":
        _branch = branch + "."

    elif b_type == "branch":
        _branch = "." + branch + "."

    elif b_type == "leaf":
        _branch = "." + branch
    else:
        raise ValueError("Given `b_type` is not valid")

    before, found, after = path.partition(_branch)

    if not found:
        return before

    if b_type == "root":
        return ".".join([repl, after])

    elif b_type == "branch":
        return ".".join([before, repl, after])

    elif b_type == "leaf":
        return ".".join([before, repl])


def replace_keys_ordered_dict(state_dict: OrderedDict, repl: Dict) -> OrderedDict:
    """
    Replace keys in state dict with a replacement map.
    """
    return OrderedDict(
        (repl[key], val) if key in repl else (key, val)
        for key, val in state_dict.items()
    )


def path_in_branch(path: str, branch: str) -> bool:
    """
    Chen if `path` belongs to a specific `branch`.

    Example
    -------
    branch = "node1.node2"

    path_in_branch("node1.node2", branch) is True
    path_in_branch("node1.node2.node3", branch) is True
    """
    return path == branch or path.startswith(branch + ".")


def leaf_in_path(path: str, leaf: str) -> bool:
    """
    Chen if `leaf` belongs to a `path`.

    Example
    -------
    path = "node1.node2"

    leaf_in_path("node2", path) is True
    """
    return path == leaf or path.endswith("." + leaf)
