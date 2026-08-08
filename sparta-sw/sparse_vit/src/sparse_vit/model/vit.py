from typing import Dict

import torch

from torch import nn

from .encoder import ViTEncoderLayer
from .embedding import EmbeddingPatchifyLinear
from .tasks import ClassifierHeadAverage, ClassifierHeadToken


class VisionTransformer(nn.Module):
    """
    This class represents a full Vision Transformer (ViT) model.

    The final model includes:
    - Embedding layer
    - ViT Encoder layers
    - Task head
    """

    def __init__(
        self,
        img_size: int = 224,
        p_size: int = 16,
        channels: int = 3,
        use_cls: bool = True,
        pe_type: str = "learnable",
        N: int = 12,
        h: int = 8,
        d_in: int = 512,
        d_hid: int | None = 1024,
        activation: str = "relu",
        dropout: float | None = 0.1,
        dropout_attn: float | None = 0.1,
        num_classes: int = 1000,
    ) -> None:
        """
        Initializer for a full VisionTransformer instance.
        """
        super(VisionTransformer, self).__init__()

        self.N = N

        embed_kw = {
            "d": d_in,
            "img_size": img_size,
            "p_size": p_size,
            "channels": channels,
            "use_cls": use_cls,
            "pe_type": pe_type,
        }

        encoder_kw = {
            "h": h,
            "d_in": d_in,
            "d_hid": d_hid,
            "activation": activation,
            "dropout": dropout,
            "dropout_attn": dropout_attn,
        }

        task_kw = {
            "d": d_in,
            "num_classes": num_classes,
            "dropout": dropout,
        }

        self.embed_layer = EmbeddingPatchifyLinear(**embed_kw)

        self.transformer = nn.Sequential(
            [ViTEncoderLayer(**encoder_kw) for _ in range(N)]
        )

        self.task_layer = ClassifierHeadAverage(**task_kw)

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        """
        Apply a VisionTransformer to input `x_img` tensor.
        """
        # (b, c, h, w) -> (b, n, d)
        x = self.embed_layer(x_img)

        # (b, n, d) -> (b, n, d)
        x = self.transformer(x)

        # (b, n, d) -> (b, c)
        return self.task_layer(x)
