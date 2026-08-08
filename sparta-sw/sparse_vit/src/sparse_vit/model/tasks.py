from typing import Dict

import torch

from torch import nn

from .norms import DynamicTanh, LinearClip

NORMS = {
    "layernorm": nn.LayerNorm,
    "rms_norm": nn.RMSNorm,
    "dynamic_tanh": DynamicTanh,
    "linear_clip": LinearClip,
}


class ClassifierHeadToken(nn.Module):
    """
    A class representing a classifier head that estimates class labels
    using a specialized CLS token.
    """

    def __init__(
        self,
        d: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
        norm_type: str = "layernorm",
        norm_kw: Dict = {},
        bias: bool = False,
        **kwargs,
    ):
        """
        Initializer for `ClassifierHeadLinear`.

        Parameters
        ----------
        d : int
            The embedding dimension.
        num_classes : int
            Number of output classes.
        dropout_out : float
            Dropout probability applied on the final projection layer.
        """
        super(ClassifierHeadToken, self).__init__()

        self.ln = NORMS[norm_type](d, **norm_kw)

        self.dropout = nn.Dropout(dropout)

        self.head = nn.Linear(d, num_classes, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given input (b, n, d) get special CLS token and apply final projection to
        map embeddings to number of classes.
        """
        # Apply final layernorm
        x_out = self.ln(x)

        # (b, n, d) -> (b, d)
        x_out = torch.select(x_out, dim=1, index=0)

        x_out = self.dropout(x_out)

        # (b, d) -> (b, c)
        return self.head(x_out)


class ClassifierHeadDeiT(nn.Module):
    """
    A class representing a classifier head following the DeiT formulation,
    and includes two heads, a CLS token and a DSTL token.

    During training the CLS token is used for learning by the class labels
    and the DSTL token is used for learning by a teacher model. During
    inference, the two tokens are lately combined.
    """

    def __init__(
        self,
        d: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
        norm_type: str = "layernorm",
        norm_kw: Dict = {},
        bias: bool = False,
        **kwargs,
    ):
        """
        Initializer for `ClassifierHeadLinear`.

        Parameters
        ----------
        d : int
            The embedding dimension.
        num_classes : int
            Number of output classes.
        dropout_out : float
            Dropout probability applied on the final projection layer.
        """
        super(ClassifierHeadDeiT, self).__init__()

        self.ln = NORMS[norm_type](d, **norm_kw)

        self.dropout = nn.Dropout(dropout)

        self.head = nn.Linear(d, num_classes, bias=bias)
        self.head_dstl = nn.Linear(d, num_classes, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given input (b, n, d) get special CLS and DSTL tokens and apply final projection to
        map embeddings to number of classes.
        """
        # Apply final layernorm
        x_out = self.ln(x)

        # CLS token: (b, n, d) -> (b, d)
        x_cls = torch.select(x_out, dim=1, index=0)
        x_cls = self.dropout(x_cls)
        x_cls = self.head(x_cls)

        # DSTL token: (b, n, d) -> (b, d)
        x_dstl = torch.select(x_out, dim=1, index=1)
        x_dstl = self.dropout(x_dstl)
        x_dstl = self.head_dstl(x_dstl)

        # `self.training` flag inherited from `nn.Module`
        if self.training:
            return x_cls, x_dstl
        else:
            return (x_cls + x_dstl) / 2.0


class ClassifierHeadAverage(nn.Module):
    """
    A class representing a classifier head that estimates class labels
    via averaging all input tokens across the `n` dimension.
    """

    def __init__(
        self,
        d: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
        norm_type: str = "layernorm",
        norm_kw: Dict = {},
    ):
        """
        Initializer for `ClassifierHeadLinear`.

        Parameters
        ----------
        d : int
            The embedding dimension.
        num_classes : int
            Number of output classes.
        dropout_out : float
            Dropout probability applied on the final projection layer.
        """
        super(ClassifierHeadAverage, self).__init__()

        self.ln = NORMS[norm_type](d, **norm_kw)

        self.dropout = nn.Dropout(dropout)

        self.head = nn.Linear(d, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given input (b, n, d) estimate final class label by averaging over the `n`
        dimension and apply final projection to map embeddings to number of classes.
        """
        # Apply final layernorm
        x_out = self.ln(x)

        # (b, n, d) -> (b, d)
        x_out = torch.mean(x, dim=1)

        x_out = self.dropout(x_out)

        # (b, d) -> (b, c)
        return self.head(x_out)
