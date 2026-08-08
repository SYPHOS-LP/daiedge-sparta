from typing import Dict

import torch

from torch import nn

from .mha import MultiHeadAttention
from .ffn import FeedForwardNetwork, GatedFeedForwardNetwork
from .norms import DynamicTanh, LinearClip


class ViTEncoderLayer(nn.Module):
    """
    A class representing a single ViT Layer.
    """

    NORMS = {
        "layernorm": nn.LayerNorm,
        "rms_norm": nn.RMSNorm,
        "dynamic_tanh": DynamicTanh,
        "linear_clip": LinearClip,
    }

    def __init__(
        self,
        h: int = 8,
        d_in: int = 512,
        d_hid: int | None = 1024,
        activation: str = "relu",
        dropout: float | None = 0.1,
        bias: bool = True,
        dropout_attn: float | None = 0.1,
        attn_type: str = "self",
        norm_type: str = "layernorm",
        norm_kw: Dict = {},
        ffn_type: str = "mlp",
        **kwargs
    ) -> None:
        """
        Initializer for a ViTEncoder layer.
        """
        super(ViTEncoderLayer, self).__init__()

        d_hid = d_hid if d_hid is not None else d_in * 4

        self.ln_mha = self.NORMS[norm_type](d_in, **norm_kw)
        self.ln_mlp = self.NORMS[norm_type](d_in, **norm_kw)

        self.mha = MultiHeadAttention(
            h=h,
            d=d_in,
            dropout_out=dropout,
            dropout_attn=dropout_attn,
            attn_type=attn_type,
            bias=bias,
        )

        if ffn_type == "mlp":
            self.mlp = FeedForwardNetwork(
                d_in=d_in,
                d_hid=d_hid,
                activation=activation,
                dropout=dropout,
                bias=bias,
            )
        elif ffn_type == "glu":
            self.mlp = GatedFeedForwardNetwork(
                d_in=d_in,
                d_hid=d_hid,
                activation=activation,
                dropout=dropout,
                bias=bias,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward step of a ViT Encoder.

        Parameters
        ----------
        x : torch.Tensor
            Features as tensor of shape (b, n, d).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (b, n, d).
        """
        _x = self.ln_mha(x)

        _x = self.mha(_x)

        x_res = _x + x

        _y = self.ln_mlp(x_res)

        _y = self.mlp(_y)

        return _y + x_res
