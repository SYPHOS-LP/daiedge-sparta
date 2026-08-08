from typing import List, Dict


from torch import nn

from .magnitude import prune_unstructured_w_magnitude
from .sparsegpt import prune_unstructured_w_sparsegpt

# from .movement import replace_with_masked_modules


def apply_pruning(
    model: nn.Module, amount: float, prune_method: str, **prune_method_kw
) -> nn.Module:
    """
    A function to dispatch to different pruning methods.
    """

    if prune_method == "magnitude":
        return prune_unstructured_w_magnitude(model, amount, **prune_method_kw)

    elif prune_method == "sparsegpt":
        return prune_unstructured_w_sparsegpt(model, amount, **prune_method_kw)

    # if prune_method == "movement":
    #     return replace_with_masked_modules(model)
    else:
        raise ValueError("Unknown prune method: {}".format(prune_method))
