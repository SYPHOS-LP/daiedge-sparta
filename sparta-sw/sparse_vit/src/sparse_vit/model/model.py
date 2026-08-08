from typing import Dict

import torch

from torch import nn
from collections import OrderedDict

from .embedding import EmbeddingPatchifyLinear
from .encoder import ViTEncoderLayer
from .tasks import ClassifierHeadToken, ClassifierHeadDeiT
from .utils import init_model_weights
from .pretrained import (
    load_pretrained_vit_weights_torch,
    load_pretrained_vit_weights_eva02,
)

from ..training.utils import model_load, model_load_submodule


def build_vit_base_model(
    embed_type: str = "rgb",
    img_size: int = 224,
    p_size: int = 16,
    channels: int = 3,
    use_cls: bool = True,
    use_dstl: bool = False,
    use_regs: int = 0,
    pe_type: str = "learnable",
    pe_type_kw: Dict = {},
    embed_kw: Dict = {},
    N: int = 12,
    mode: str = "vit",
    h: int = 8,
    d_in: int = 512,
    d_hid: int | None = 1024,
    activation: str = "relu",
    dropout: float | None = 0.1,
    bias: bool = True,
    ffn_type: str = "mlp",
    dropout_attn: float | None = 0.1,
    attn_type: str = "self",
    norm_type: str = "layernorm",
    norm_kw: Dict = {},
    num_classes: int = 1000,
    params: str | None = None,
    params_kw: Dict = {},
) -> nn.Module:
    """
    Builder function for a Vision Transformer model.

    Parameters
    ----------
    embed_type : str
        Type of the embedding layer. Choose between `rgb` and `dct`.
    img_size : int
        Size of input image. We assume the image is of same `h` and `w`.
    p_size : int
        Size of patch size. Default is 16x16.
    channels : int
        Number of channels.
    use_cls : bool
        Whether to use a CLS token.
    use_dstl : bool
        Whether to use a distillation token.
    use_regs : int
        Whether to add register tokens.
    pe_type : {'sincos', 'learnable', 'haar'}
        Type of positional embeddings.
    pe_type_kw : Dict
        Additional keyword arguments to be passed to the positional embeddings.
    embed_kw : Dict
        Additional keyword arguments to be passed to the embeding layer.
    N : int
        Number of layers.
    d_in : int
        Input dimension for first linear layer.
    d_hid : int
        Output dimension for first linear layer.
    activation : {'relu', 'silu', 'gelu'}
        Activation functions.
    dropout : float | None
        Dropout probability for between layers.
    bias : bool
        Whether to use bias in the embedding's cnn layer and the encoder's ffn layers.
    ffn_type : str
        Type of FFN layer to use. Choose from {'mlp', 'glu'}.
    dropout_attn : float
        Dropout probability applied on the attention weights.
    attn_type : {'self', 'sparsemax', 'entmax', 'evsoftmax', 'multimax', 'linear'}
        Type of attention to apply.
    norm_type : {'layernorm', 'rms_norm', 'dynamic_tanh', 'linear_clip'}
        Type of normalization layer to apply.
    norm_kw : Dict
        Keyword arguments to be passed to the normalization layer.
    num_classes : int
        Number of output classes.
    params : {'init', 'pretrained'}
        How to handle parameter initialization.
    params_kw : Dict
        Keyword arguments to be passed to parameter initialization function.

    Returns
    -------
    nn.Module
        A vanilla Vision Transformer model.
    """
    _embed_kw = {
        "d": d_in,
        "img_size": img_size,
        "p_size": p_size,
        "channels": channels,
        "bias": bias,
        "use_cls": use_cls,
        "use_dstl": use_dstl,
        "use_regs": use_regs,
        "pe_type": pe_type,
        "pe_type_kw": pe_type_kw,
    }
    _embed_kw.update(embed_kw)

    encoder_kw = {
        "h": h,
        "d_in": d_in,
        "d_hid": d_hid,
        "activation": activation,
        "dropout": dropout,
        "bias": bias,
        "dropout_attn": dropout_attn,
        "attn_type": attn_type,
        "norm_type": norm_type,
        "norm_kw": norm_kw,
        "ffn_type": ffn_type,
    }

    task_kw = {
        "d": d_in,
        "num_classes": num_classes,
        "dropout": dropout,
        "bias": bias,
        "norm_type": norm_type,
        "norm_kw": norm_kw,
    }

    # Initialize embedding layer and task-specific layer
    if embed_type == "rgb":
        embed = EmbeddingPatchifyLinear(**_embed_kw)

    else:
        raise ValueError("Choose `embed_type` from ('rgb', 'cdt')")

    if mode in "vit":
        heads = ClassifierHeadToken(**task_kw)

    elif mode == "deit":
        heads = ClassifierHeadDeiT(**task_kw)

    else:
        raise ValueError("Choose between ('vit', 'deit').")

    model = VisionTransformer(N=N, encoder_kw=encoder_kw, embed=embed, heads=heads)

    # initialize parameters with Glorot or load pretrained
    if params == "init":
        return model

    elif params == "load":
        model = model_load(model, **params_kw)

    elif params == "pretrained":
        load_pretrained_vit_weights_torch(model, **params_kw)

    elif params == "eva02":
        load_pretrained_vit_weights_eva02(model, **params_kw)

    else:
        raise ValueError(
            "Arg `params` is not valid. Choose from ('init', 'pretrained')"
        )

    return model


def build_vit_embed_head_layers(
    embed_type: str = "rgb",
    img_size: int = 224,
    p_size: int = 16,
    channels: int = 3,
    use_cls: bool = True,
    use_dstl: bool = False,
    use_regs: int = 0,
    pe_type: str = "learnable",
    pe_type_kw: Dict = {},
    embed_kw: Dict = {},
    mode: str = "vit",
    d_in: int = 512,
    dropout: float | None = 0.1,
    bias: bool = True,
    norm_type: str = "layernorm",
    norm_kw: Dict = {},
    num_classes: int = 1000,
    params: str | None = None,
    params_kw: Dict = {},
    **kwargs,
) -> nn.Module:
    """
    Builder function for the embedding and head layers of a Vision Transformer.

    Parameters
    ----------
    embed_type : str
        Type of the embedding layer. Choose between `rgb` and `dct`.
    img_size : int
        Size of input image. We assume the image is of same `h` and `w`.
    p_size : int
        Size of patch size. Default is 16x16.
    channels : int
        Number of channels.
    use_cls : bool
        Whether to use a CLS token.
    use_dstl : bool
        Whether to use a distillation token.
    use_regs : int
        Whether to add register tokens.
    pe_type : {'sincos', 'learnable', 'haar'}
        Type of positional embeddings.
    pe_type_kw : Dict
        Additional keyword arguments to be passed to the positional embeddings.
    embed_kw : Dict
        Additional keyword arguments to be passed to the embeding layer.
    d_in : int
        Input dimension for first linear layer.
    dropout : float | None
        Dropout probability for between layers.
    bias : bool
        Whether to use bias in the embedding's cnn layer and the encoder's ffn layers.
    norm_type : {'layernorm', 'rms_norm', 'dynamic_tanh', 'linear_clip'}
        Type of normalization layer to apply.
    norm_kw : Dict
        Keyword arguments to be passed to the normalization layer.
    num_classes : int
        Number of output classes.
    params : {'init', 'pretrained'}
        How to handle parameter initialization.
    params_kw : Dict
        Keyword arguments to be passed to parameter initialization function.

    Returns
    -------
    nn.Module
        A vanilla Vision Transformer model.
    """
    _embed_kw = {
        "d": d_in,
        "img_size": img_size,
        "p_size": p_size,
        "channels": channels,
        "bias": bias,
        "use_cls": use_cls,
        "use_dstl": use_dstl,
        "use_regs": use_regs,
        "pe_type": pe_type,
        "pe_type_kw": pe_type_kw,
    }
    _embed_kw.update(embed_kw)

    task_kw = {
        "d": d_in,
        "num_classes": num_classes,
        "dropout": dropout,
        "bias": bias,
        "norm_type": norm_type,
        "norm_kw": norm_kw,
    }

    # Initialize embedding layer and task-specific layer
    if embed_type == "rgb":
        embed = EmbeddingPatchifyLinear(**_embed_kw)

    if mode in "vit":
        heads = ClassifierHeadToken(**task_kw)

    elif mode == "deit":
        heads = ClassifierHeadDeiT(**task_kw)

    embed = model_load_submodule(embed, submodule="embed", **params_kw)
    heads = model_load_submodule(heads, submodule="heads", **params_kw)

    return embed, heads


class VisionTransformer(nn.Module):
    """
    A class to represent a full ViT model.
    """

    def __init__(
        self, N: int, encoder_kw: Dict, embed: nn.Module, heads: nn.Module
    ) -> None:
        """
        Initializer for the `VisionTransformer` instance.

        Parameters
        ----------
        N : int
            Number of encoder layers in the ViT architecture.
        encoder_kw : Dict
            Arguments to initialize a list of `ViTEncoderLayer` layers.
        embed : nn.Module
            An initialized embedding layer.
        heads : nn.Module
            An initialized task head layer.
        """
        super(VisionTransformer, self).__init__()

        self.encoder = nn.Sequential(
            OrderedDict(
                [
                    (f"encoder_layer_{i}", ViTEncoderLayer(**encoder_kw))
                    for i in range(N)
                ]
            )
        )

        self.embed = embed
        self.heads = heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method.
        """
        emb = self.embed(x)

        emb = self.encoder(emb)

        return self.heads(emb)
