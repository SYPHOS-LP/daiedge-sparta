from typing import List

import re


def assign_pruning_ratio_levels_across_modules(module: str, target: float) -> float:
    """
    Gather modules to assign pruning ratio.
    """
    # Per module ratio
    EMBED_RATIO = 0.0
    HEADS_RATIO = 0.0
    QKVP_RATIO = 1.00
    MLP_RATIO = 1.00

    # Per layer ratio
    LAYER_0_11_RATIO = 1.00
    LAYER_1_10_RATIO = 1.00

    try:
        if module.startswith("embed."):
            return EMBED_RATIO * target

        if module.startswith("heads."):
            return HEADS_RATIO * target

        if module.startswith("encoder"):

            # Get layer number
            layer_num = int(
                re.search("^encoder.encoder_layer_([0-9]{1,2})*", module).group(1)
            )

            if layer_num in (0, 1):
                layer_num_mul = LAYER_0_11_RATIO
            else:
                layer_num_mul = LAYER_1_10_RATIO

            # Get type of layer
            if any([module.endswith(sfx) for sfx in ("w_q", "w_k", "w_v", "out_proj")]):
                layer_type_mul = QKVP_RATIO
            else:
                layer_type_mul = MLP_RATIO

            return layer_num_mul * layer_type_mul * target
    except Exception:
        print("An error has occured when estimating pruning ratio.")

    return target
