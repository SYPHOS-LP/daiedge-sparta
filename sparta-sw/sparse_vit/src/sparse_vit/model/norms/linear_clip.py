import torch
from torch import nn


class LinearClip(nn.Module):
    """
    Class to represent a `LinearClip` layer for point-wise normalization (also DynamicHardtanh).

            x = hardtanh(alpha * x) * w + b

    In the `https://arxiv.org/abs/2512.10938`paper, that investigates layernorm alternatives
    there are multple variants of linearclip, which apply non-linear transformations on x
    before clipping such as `logquad_clip` and `logsign_clip`.

    In addition in this paper a `shift` learnable parameter is also applied to x before clipping.
    This may change our results especially when the `bias` term is removed.

    NOTE: Consider testing those as well.
    """

    def __init__(
        self,
        normalized_shape: int,
        init_alpha: float = 0.5,
        elementwise_affine: bool = True,
        bias: bool = False,
    ):
        """
        Initializer for Linear Clip class instance.

        Parameters
        ----------
        normalized_shape : int
            Normalize over the last dimension which is expected to be of that size.
        init_alpha : float
            Alpha parameter init value inside `hardtanh` layer.
        elementwise_affine : bool
            If `True`, apply learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases).
        bias : bool
            If `False`, skip the additive bias (only relevant if `elementwise_affine`
            is `True`). Default: `True`.
        """
        super(LinearClip, self).__init__()

        self.normalized_shape = normalized_shape
        self.init_alpha = init_alpha

        self.elementwise_affine = elementwise_affine
        self.bias = bias

        self.alpha = nn.Parameter(torch.ones(normalized_shape) * init_alpha)

        if self.elementwise_affine is True:
            self.w = nn.Parameter(torch.ones(normalized_shape))

            if self.bias is True:
                self.b = nn.Parameter(torch.zeros(normalized_shape))
            else:
                self.register_parameter("b", None)
        else:
            self.register_parameter("w", None)
            self.register_parameter("b", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method for `LinearClip` instance.
        """
        # NOTE: self.shift is ommited
        x = self.alpha * x
        x = torch.clip(x, -1, 1)

        if self.elementwise_affine is False:
            return x

        if self.bias is False:
            x = x * self.w
        else:
            x = x * self.w + self.b

        return x

    def extra_repr(self):
        return "normalized_shape={}, alpha_init_value={},channels_last={}".format(
            self.normalized_shape, self.init_alpha, self.channels_last
        )
