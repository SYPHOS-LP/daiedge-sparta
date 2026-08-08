from typing import Tuple, List

import torch

from torch import nn

BLOCKS_VAR_8x8 = [(6, 6), (22, 22), (36, 36)]
BLOCKS_VAR_16x16 = [(b * 4, b * 4) for (b, _) in BLOCKS_VAR_8x8]


class LinearBlockedDiagonal(nn.Module):
    """
    Applies an affine linear transformation to the incoming data: :`y = xA^T + b`
    but enforces a block structure to the weight matrix A in {d_in, d_out}:

    ```
    A = [B     ...       ]
        [   B  ...       ]
        [      ...       ]
        [      ...  B    ]
        [      ...      B]
    ```

    A class to represent a linear layer with weights organized diagonally
    as blocks. If the block-size == 1, then the layer becomes a diagonal
    operator.
    """

    def __init__(self, d_in: int, d_out: int, b_size: int, bias: bool = False):
        """
        Initializer for `LinearBlockedDiagonal` class.

        Parameters
        ----------
        d_in : int
            The size of the input features of the linear layer.
        d_out : int
            The size of the output features of the linear layer.
        b_size : int
            The size of the block of the block-diagonal layer. This
            should divide exactly `d_in` and `d_out`.
        bias : bool
            Whether a bias vector will be included.
        """
        super(LinearBlockedDiagonal, self).__init__()

        assert (d_in % b_size == 0) and (d_out % b_size == 0)

        self.n_blocks = d_in // b_size
        self.b_size = b_size

        self.weight = nn.Parameter(torch.empty(self.n_blocks, b_size, b_size))

        if bias:
            self.bias = nn.Parameter(torch.empty(self.n_blocks, b_size))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters.
        """
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method.
        """
        _size = x.size()

        # (b, n, d) -> (b, n, n_b, d_b)
        x = x.view(*_size[:-1], self.n_blocks, self.b_size)

        # each block has its own weight matrix
        y = torch.einsum("...bi, boi->...bo", x, self.weight)

        if self.bias is not None:
            y = y + self.bias

        return y.reshape(*_size[:-1], self.n_blocks * self.b_size)


class LinearBlockedDiagGeneral(nn.Module):
    """
    Applies an affine linear transformation to the incoming data: :`y = xA^T + b`
    but enforces a block structure to the weight matrix A in {d_in, d_out} with
    arbitrary block sizes (B_1, B_2, ..., B_n):

    ```
    A = [B_1       ...            ]
        [     B_2  ...            ]
        [          ...            ]
        [          ...  B_n-1     ]
        [          ...         B_n]
    ```

    A class to represent a linear layer with weights organized diagonally
    as blocks. If the block-size == 1, then the layer becomes a diagonal
    operator.
    """

    def __init__(
        self, d_in: int, d_out: int, b_sizes: List[Tuple[int, int]], bias: bool = False
    ):
        """
        Initializer for `LinearBlockedDiagGeneral` class.

        Parameters
        ----------
        d_in : int
            The size of the input features of the linear layer.
        d_out : int
            The size of the output features of the linear layer.
        b_sizes : List[Tuple[int, int]]
            The size of each block of the block-diagonal layer. The sizes
            should add up in a way to match `d_in` and `d_out`.
        bias : bool
            Whether a bias vector will be included.
        """
        super(LinearBlockedDiagGeneral, self).__init__()

        self.bs_in, self.bs_out = zip(*b_sizes)

        if not (d_in == sum(self.bs_in) and d_out == sum(self.bs_out)):
            raise ValueError(
                "Defined block sizes must match the overall in and out dimensions."
            )

        self.n_blocks = len(b_sizes)

        self.blocks = nn.ModuleList(
            [nn.Linear(b_in, b_out, bias=bias) for (b_in, b_out) in b_sizes]
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters.
        """
        for m in self.blocks:
            m.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method.
        """
        xs = torch.split(x, self.bs_in, dim=-1)

        ys = [block(x_i) for block, x_i in zip(self.blocks, xs)]

        return torch.cat(ys, dim=-1)
