from typing import List

from collections import OrderedDict
from torch import nn

from ..utils import (
    replace_keys_ordered_dict,
    _remove_branch,
    _replace_branch,
)


def load_pretrained_vit_weights_torch(
    model: nn.Module,
    weights: str,
    base_model: str = "vit_b_16",
    filter_branch: List = [],
) -> OrderedDict:
    """
    Preprocess `state_dict` of `pytorch` ViT pretrained weights.

    Parameters
    ----------
    model : nn.Module
        An initialized model to load vit pretrained weights.
    weights : str
        A string representing pretrained weights to load for given base models
        represented by the `ViT_B_16_Weights` and `ViT_B_32_Weights` classes.
    base_model : str
        Base vision transformer model to get weights from. `vit_b_16` and
        `vit_b_32` are supported.
    filter_branch : List
        Filter out branches from loading weights.

    Returns
    -------
    None
        Loads pretrained weights.
    """
    from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights

    weight_clss = {
        "vit_b_16": ViT_B_16_Weights,
        "vit_b_32": ViT_B_32_Weights,
    }

    FILTER = {}

    PATH_REPLACE = {
        "class_token": "embed.class_token",
        "conv_proj.weight": "embed.conv_proj.weight",
        "conv_proj.bias": "embed.conv_proj.bias",
        "encoder.pos_embedding": "embed.pos_embedding",
        "encoder.ln.weight": "heads.ln.weight",
        "encoder.ln.bias": "heads.ln.bias",
    }

    BRANCH_REPLACE = {
        "ln_1": "ln_mha",
        "ln_2": "ln_mlp",
        "self_attention": "mha",
    }

    try:
        weight_cls = weight_clss[base_model]
    except KeyError:
        raise ValueError(f"Given `weights` argument is not valid.")

    try:
        _w = getattr(weight_cls, weights)
    except AttributeError:
        raise ValueError(
            (
                f"`{weights}` argument for `{base_model}` model is not valid. "
                f"Select from {tuple(weight_cls.__members__)}"
            )
        )

    state_dict = _w.get_state_dict()

    # Filter paths
    new_dict = OrderedDict((k, v) for k, v in state_dict.items() if k not in FILTER)

    # Direct replacement for embedding layers
    new_dict = replace_keys_ordered_dict(new_dict, PATH_REPLACE)

    # Remove `layers` from path
    new_dict = OrderedDict(
        ((_remove_branch(k, "layers"), v) if k.startswith("encoder") else (k, v))
        for k, v in new_dict.items()
    )

    # Replace branches in encoder
    for old_b, new_b in BRANCH_REPLACE.items():
        new_dict = OrderedDict(
            (
                (_replace_branch(k, old_b, new_b), v)
                if k.startswith("encoder")
                else (k, v)
            )
            for k, v in new_dict.items()
        )

    # Finally filter out branches
    filter_branch = set(filter_branch)

    new_dict = OrderedDict(
        (k, v) for k, v in new_dict.items() if k not in filter_branch
    )

    try:
        missing, unexpected = model.load_state_dict(new_dict, strict=False)

        print("Missing layers:", missing)
        print("Unexpected layers:", unexpected)
    except Exception as e:
        print("An error has occured when loading the model: {}".format(e))
