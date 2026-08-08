from typing import Dict

import torch

from torch import nn

from .pes import pe_haar_2d, pe_sincos_2d, pe_learnable_2d


class EmbeddingPatchifyLinear(nn.Module):
    """
    A class to represent a ViT embedding step of splitting into patches
    and applying a linear layer on top.

    This layer applies patchification & linear layer via a 2D CNN with
    a kernel size and stride size equal to patch size.
    """

    def __init__(
        self,
        d: int = 256,
        img_size: int = 224,
        p_size: int = 16,
        channels: int = 3,
        bias: bool = False,
        use_cls: bool = True,
        use_dstl: bool = False,
        use_regs: int = 0,
        pe_type: str = "learnable",
        pe_type_kw: Dict = {},
        **kwargs,
    ) -> None:
        """
        Initialzer for the Embedding layer.

        Parmaters
        ---------
        d : int
            Embedding dimension.
        img_size : int
            Size of input image. We assume the image is of same `h` and `w`.
        p_size : int
            Size of patch size. Default is 16x16.
        channels : int
            Number of channels.
        bias : bool
            Whether the CNN layer has bias terms.
        use_cls : bool
            Whether to use a CLS token.
        use_dstl : bool
            Whether to use a distillation token for training in DeiT style.
        use_regs : int
            Number of additional register tokens to use as proposed in paper
            `Vision Transormers Need Registers`.
        pe_type : {'sincos', 'learnable', 'haar', 'none'}
            Type of positional embeddings.
        """
        super(EmbeddingPatchifyLinear, self).__init__()

        self.d = d
        self.img_w = img_size
        self.img_h = img_size
        self.p_size = p_size
        self.channels = channels

        self.use_cls = use_cls
        self.use_dstl = use_dstl
        self.use_regs = use_regs
        self.pe_type = pe_type

        self.n_p = (self.img_h // p_size) * (self.img_w // p_size)

        h = self.img_h // p_size

        if h == 7:
            scales = [1 / 4]

        elif h == 14:
            scales = [1 / 8]

        self.conv_proj = nn.Conv2d(
            channels,
            d,
            kernel_size=p_size,
            stride=p_size,
            padding_mode="zeros",
            bias=bias,
        )

        self.class_token = nn.Parameter(torch.empty(1, 1, d))

        if self.use_dstl:
            self.dstl_token = nn.Parameter(torch.empty(1, 1, d))

        if self.use_regs > 0:
            self.reg_tokens = nn.Parameter(torch.empty(1, use_regs, d))

        pes = {"sincos": pe_sincos_2d, "learnable": pe_learnable_2d, "haar": pe_haar_2d}

        pes_kwargs = {
            "sincos": {},
            "learnable": {
                "d": d,
                "n": self.n_p,
                "add_cls": use_cls,
                "add_dstl": use_dstl,
                "add_regs": use_regs,
            },
            "haar": {
                "d": d,
                "h": h,
                "dim": 32,
                "n": 32,
                "scales": scales,
                "apply_kron": "per_scale",
                "add_cls": use_cls,
                "add_dstl": use_dstl,
                "add_regs": use_regs,
            },
        }

        pe_func = pes.get(pe_type)
        pe_kwargs = pes_kwargs.get(pe_type, {})
        pe_kwargs.update(pe_type_kw)

        if pe_type != "none":
            pe = pe_func(**pe_kwargs)

        # Use `register_buffer` to declare `Tensor` which is not a parameter, as part of the module state.
        if pe_type in ("haar", "sincos"):
            self.register_buffer("pos_embedding", pe, persistent=True)
        elif pe_type == "learnable":
            self.register_parameter("pos_embedding", pe)

        # Initialize parameters for class token, dstl token and register tokens.
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset parameters.
        """
        # Reset `conv_proj` parameters
        self.conv_proj.reset_parameters()

        # Initialze learnable parameters
        if self.pe_type == "learnable":
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Initialize parameters for class token, dstl token and register tokens.
        nn.init.trunc_normal_(self.class_token, std=0.02)

        if self.use_dstl:
            nn.init.trunc_normal_(self.dstl_token, std=0.02)

        if self.use_regs > 0:
            nn.init.trunc_normal_(self.reg_tokens, std=0.02)

    def add_tokens(
        self, x: torch.Tensor, b: int, use_cls: bool, use_dstl: bool, use_regs: int
    ) -> torch.Tensor:
        """ """
        # (b, m, d) -> (b, n, d)   where   `n = n_p + 1`
        xs = []

        if use_cls is True:
            xs.append(self.class_token.expand(b, -1, -1))

        if use_dstl is True:
            xs.append(self.dstl_token.expand(b, -1, -1))

        xs.append(x)

        if use_regs > 0:
            xs.append(self.reg_tokens.expand(b, -1, -1))

        return torch.cat(xs, dim=1)

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        """
        Performs the following actions on an input image:
          * Apply patchification
          * Apply linear projection
          * Concat CLS embedding
          * Add positional embeddings
        """
        b, _, _, _ = x_img.size()

        # (b, c, h, w) -> (b, d, p, p)
        x = self.conv_proj(x_img)

        # (b, d, p, p) -> (b, d, n_p) -> (b, n_p, d)  where   `n_p = p * p`
        x = x.view(b, self.d, self.n_p).transpose(1, 2)

        # (b, n_p, d) -> (b, n, d)
        x = self.add_tokens(x, b, self.use_cls, self.use_dstl, self.use_regs)

        # (b, n, d) -> (b, n, d)
        if self.pe_type != "none":
            x = x + self.pos_embedding

        return x
