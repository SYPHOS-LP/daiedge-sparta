import torch
import torch.nn as nn


class DynamicTanh(nn.Module):
    """
    Class to represent a `DynamicTanh` layer for point-wise normalization.

            x = tanh(alpha * x) * w + b
    """

    def __init__(
        self,
        normalized_shape: int,
        init_alpha: float = 0.5,
        elementwise_affine: bool = True,
        bias: bool = False,
    ) -> None:
        """
        Initializer for Dynamic Tanh class instance.

        Parameters
        ----------
        normalized_shape : int
            Normalize over the last dimension which is expected to be of that size.
        init_alpha : float
            Alpha parameter init value inside `tanh` layer.
        elementwise_affine : bool
            If `True`, apply learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases).
        bias : bool
            If `False`, skip the additive bias (only relevant if `elementwise_affine`
            is `True`). Default: `True`.
        """
        super(DynamicTanh, self).__init__()

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
        Forward method for `DynamicTanh` instance.
        """
        x = torch.tanh(self.alpha * x)

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
