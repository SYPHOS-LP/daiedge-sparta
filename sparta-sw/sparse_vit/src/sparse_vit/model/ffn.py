import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    """
    Class to represent a two-layer MLP feed-forward neural network.

                x_out = s(W_1*x + b_1) * W_2 + b_2

    NOTE: Based on the original paper and the google ViT code dropout
    is applied after every dense layer (after the activation).

    https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py#L66
    """

    def __init__(
        self,
        d_in: int = 256,
        d_hid: int | None = 1024,
        d_out: int | None = None,
        activation: str = "relu",
        dropout: float | None = 0.1,
        bias: bool = True,
    ) -> None:
        """
        Initialization for `FeedForwardNetwork` class.

        Parameters
        ----------
        d_in : int
            Input dimension for first linear layer.
        d_hid : int
            Output dimension for first linear layer.
        d_out : int | None
            Output dimension for second linear layer. If not specified
            the `d_in` is used.
        activation : {'relu', 'silu', 'gelu'}
            Activation functions.
        dropout : float | None
            Dropout probability for between layers.
        """
        super(FeedForwardNetwork, self).__init__()

        activations = {
            "relu": nn.ReLU,
            "relu6": nn.ReLU6,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
        }

        d_hid = d_hid if d_hid is not None else d_in * 4
        d_out = d_out if d_out is not None else d_in

        self.linear_1 = nn.Linear(d_in, d_hid, bias=bias)
        self.linear_2 = nn.Linear(d_hid, d_out, bias=bias)

        activation_cls = activations[activation]

        self.activation = activation_cls()

        self.dropout_h = nn.Dropout(dropout)
        self.dropout_o = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.linear_1(x)

        x = self.activation(x)

        x = self.dropout_h(x)

        x = self.linear_2(x)

        x = self.dropout_o(x)

        return x


class GatedFeedForwardNetwork(nn.Module):
    """
    Class to represent a modern Gated Linear Units as feed-forward neural network.

                x_out = s(W_1*x + b_1) * W_2 + b_2

    NOTE: Based on the paper `GLU Variants Improve Transformer` (https://arxiv.org/pdf/2002.05202)
    """

    def __init__(
        self,
        d_in: int = 256,
        d_hid: int | None = 1024,
        d_out: int | None = None,
        activation: str = "relu",
        dropout: float | None = 0.1,
        bias: bool = False,
    ):
        """ """
        super(GatedFeedForwardNetwork, self).__init__()

        activations = {
            "relu": nn.ReLU,
            "relu6": nn.ReLU6,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
        }

        d_hid = d_hid if d_hid is not None else d_in * 2
        d_out = d_out if d_out is not None else d_in

        self.gate_proj = nn.Linear(d_in, d_hid, bias=bias)
        self.up_proj = nn.Linear(d_in, d_hid, bias=bias)
        self.down_proj = nn.Linear(d_hid, d_out, bias=bias)

        activation_cls = activations[activation]

        self.activation = activation_cls()
        self.dropout_o = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x_gate = self.gate_proj(x)

        x_up = self.up_proj(x)

        x_o = self.down_proj(self.activation(x_gate) * x_up)

        x_o = self.dropout_o(x_o)

        return x_o
