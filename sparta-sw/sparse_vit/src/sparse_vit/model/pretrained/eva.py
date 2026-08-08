from typing import List

import torch

from collections import OrderedDict
from torch import nn

from ..utils import replace_keys_ordered_dict, _replace_branch, _remove_branch


def load_pretrained_vit_weights_eva02(
    model: nn.Module, weights: str, filter_branch: List = []
) -> OrderedDict:
    """
    Preprocess `state_dict` of `pytorch` ViT pretrained weights.

    Parameters
    ----------
    model : nn.Module
        An initialized model to load vit pretrained weights.
    weights : str
        A string pointing to the saved model `bin` file.
    filter_branch : List
        Filter out branches from loading weights.

    Returns
    -------
    None
        Loads pretrained weights.
    """
    FILTER = {
        "norm.weight",
        "norm.bias",
        "head.weight",
        "head.bias",
    }

    PATH_REPLACE = {
        "cls_token": "embed.class_token",
        "pos_embed": "embed.pos_embedding",
        "patch_embed.proj.weight": "embed.conv_proj.weight",
        "patch_embed.proj.bias": "embed.conv_proj.bias",
        "encoder.ln.weight": "heads.ln.weight",
        "encoder.ln.bias": "heads.ln.bias",
    }

    BRANCH_REPLACE = {
        "attn": "mha",
        "q_proj": "w_q",
        "k_proj": "w_k",
        "v_proj": "w_v",
        "proj": "out_proj",
        "norm1": "ln_mha",
        "norm2": "ln_mlp",
        "fc1_g": "gate_proj",
        "fc1_x": "up_proj",
        "fc2": "down_proj",
    }

    state_dict = torch.load(weights)

    # 1. Keep only `visual.trunk` related layers
    new_dict = OrderedDict(
        (k, v) for k, v in state_dict.items() if k.startswith("visual.trunk")
    )

    # 2. Remove `visual.trunk` prefic
    new_dict = OrderedDict(
        (_remove_branch(k, "visual.trunk", b_type="root"), v)
        for k, v in new_dict.items()
    )

    # 3. Filter paths
    new_dict = OrderedDict((k, v) for k, v in new_dict.items() if k not in FILTER)

    # 4. Direct replacement
    new_dict = replace_keys_ordered_dict(new_dict, PATH_REPLACE)

    # 5. Rename `blocks` to `encoder`
    new_dict = OrderedDict(
        (_replace_branch(k, "blocks", "encoder", b_type="root"), v)
        for k, v in new_dict.items()
    )

    # 6. Rename `i` to `encoder_layer_i`
    for idx in range(12):

        new_dict = OrderedDict(
            (
                (
                    _replace_branch(
                        k, f"{idx}", f"encoder_layer_{idx}", b_type="branch"
                    ),
                    v,
                )
                if k.startswith(f"encoder.{idx}")
                else (k, v)
            )
            for k, v in new_dict.items()
        )

    # 7. Replace branches in encoder
    for old_b, new_b in BRANCH_REPLACE.items():
        new_dict = OrderedDict(
            (
                (_replace_branch(k, old_b, new_b), v)
                if k.startswith("encoder")
                else (k, v)
            )
            for k, v in new_dict.items()
        )

    # Finally filter out input branches
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
