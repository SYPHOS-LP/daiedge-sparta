import copy
import torch

from torch import nn, topk
from entmax import Sparsemax, Entmax15

from .embedding import pe_haar_2d


def topk_attention(x):
    """ """
    raise NotImplementedError


class EvSoftmax(torch.nn.Module):
    """
    A class representing an `Evidential Softmax` or `EvSoftmax` activation
    layer as defined in the paper: `https://arxiv.org/pdf/2110.14182`.

    The implementation is taken from here:

    `https://github.com/sisl/EvSoftmax/blob/main/evsoftmax.py`
    """

    EPS = 1e-6

    def __init__(self, dim: int = -1):
        """
        Initializer for `EvSoftmax` layer.
        """
        super(EvSoftmax, self).__init__()
        self.dim = dim

    def evsoftmax(
        self, x: torch.Tensor, dim: int, training: bool = True
    ) -> torch.Tensor:
        """
        Method to apply `evsoftmax` to input tensor.
        """
        mask = x < torch.mean(x, dim=dim, keepdim=True)

        mask_offset = torch.ones(x.shape, device=x.device, dtype=x.dtype)
        mask_offset[mask] = self.EPS if training else 0.0

        probs_unnormalized = x.softmax(dim=dim) * mask_offset

        probs = probs_unnormalized / torch.sum(
            probs_unnormalized, dim=dim, keepdim=True
        )

        return probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.evsoftmax(x, self.dim, self.training)


class MultiMax(nn.Module):
    """
    A class representing a `MultiMax` activation layer as defined in
    the paper: `https://arxiv.org/pdf/2406.01189v1`.

    The implementation is taken from here:

    `https://github.com/ZhouYuxuanYX/MultiMax/blob/main/vision_transformer.py`
    """

    def __init__(self):
        """
        Initializer for `MultiMax` layer.
        """
        super(MultiMax, self).__init__()
        self.ranges = nn.Parameter(torch.zeros((2)), requires_grad=True)
        self.ts = nn.Parameter(torch.zeros((2)), requires_grad=True)
        self.t = nn.Parameter(torch.tensor(1.0))
        # self.b = nn.Parameter(torch.tensor(0.0))

        torch.nn.init.trunc_normal_(self.ranges, std=0.02)
        torch.nn.init.trunc_normal_(self.ts, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multimax to inputs x.
        """
        x = self.t * x
        x = (
            x
            + self.ts[0] * torch.relu(self.ranges[0] - x)
            + self.ts[1] * torch.relu(x - self.ranges[1])
        )

        return x.softmax(dim=-1)


class MultiHeadAttention(nn.Module):
    """
    A class representing a Multi-Head Attention layer.
    """

    def __init__(
        self,
        h: int = 8,
        d: int = 512,
        dropout_out: float | None = 0.1,
        dropout_attn: float | None = 0.1,
        norm_type: str | bool = "layer_norm",
        attn_type: str = "self",
        bias: bool = False,
        **kwargs,
    ) -> None:
        """
        Initializer for the MultiHeadAttention layer.

        Parameters
        ----------
        h : int
            Number of heads
        d : int
            The vertex embeddings dimension.
        dropout_out : float
            Dropout probability applied on the final projection layer.
        dropout_attn : float
            Dropout probability applied on the attention weights.
        norm_type : {'layer_norm', 'rms_norm'} or False
            Type of normalization to apply to Q, K matrices.
        attn_type : {'self', 'sparsemax', 'entmax', 'evsoftmax', 'multimax', 'linear', 'linear-relu6'}
            Whether to apply vanilla self-attention or attention using sparsemax.
        """
        super(MultiHeadAttention, self).__init__()

        assert d % h == 0

        self.h = h
        self.d_k = d_k = d // h
        self.scale = d_k ** (-0.5)

        self.norm_type = norm_type
        self.attn_type = attn_type

        # Initialize linear mappings
        self.w_q = nn.Linear(d, d, bias=bias)
        self.w_k = nn.Linear(d, d, bias=bias)
        self.w_v = nn.Linear(d, d, bias=bias)

        # NOTE: QK normalization layers are not included in original ViT implementation
        # norm_q, norm_k = self._initialize_norm_layers(d_k, norm_type=norm_type)

        # self.norm_qs = nn.ModuleList([copy.deepcopy(norm_q) for _ in range(self.h)])
        # self.norm_ks = nn.ModuleList([copy.deepcopy(norm_k) for _ in range(self.h)])

        # Initialize activation function
        if self.attn_type == "self":
            self.activation = nn.Softmax(dim=-1)

        elif self.attn_type == "sparsemax":
            self.activation = Sparsemax(dim=-1)

        elif self.attn_type == "entmax":
            self.activation = Entmax15(dim=-1)

        elif self.attn_type == "evsoftmax":
            self.activation = EvSoftmax(dim=-1)

        elif self.attn_type == "multimax":
            self.activation = MultiMax()

        elif self.attn_type == "rela":
            self.activation = nn.ReLU()

        elif self.attn_type == "rela6":
            self.activation = nn.ReLU6()

        elif self.attn_type == "topk":
            self.activation = topk_attention

        elif self.attn_type == "linear":
            self.activation_q = nn.ReLU()
            self.activation_k = nn.ReLU()

        elif self.attn_type == "linear-relu6":
            self.activation_q = nn.ReLU6()
            self.activation_k = nn.ReLU6()

        else:
            raise ValueError("Given `attn_type` is not valid.")

        # Initialize output linear layers
        self.out_proj = nn.Linear(d, d, bias=bias)

        self.dropout_out = dropout_out
        self.dropout_attn = dropout_attn

        self.register_load_state_dict_pre_hook(self._preload_hook)

    def vanilla_multi_head_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        dropout_attn: float,
    ) -> torch.Tensor:
        """
        Vanilla multi-head attention.
        """
        # Estimate qk.T matrix, softmax
        A = torch.matmul(q, k.transpose(-1, -2)) * scale

        # Calculate attention weights
        A = self.activation(A)

        # Apply dropout on attention weights
        A = nn.functional.dropout(A, p=dropout_attn)

        # Apply attention weights on values
        return torch.matmul(A, v)

    def linear_multi_head_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        eps: float = 1.0e-8,
    ) -> torch.Tensor:
        """
        Linear multi-head attention.

        https://github.com/mit-han-lab/efficientvit/blob/master/efficientvit/models/nn/ops.py#L582
        """
        _q = self.activation_q(q * scale**0.5)
        _k = self.activation_k(k * scale**0.5)

        # implementation #1
        _kv = torch.matmul(_k.transpose(-1, -2), v)

        numerator = _q @ _kv
        denominator = _q @ _k.sum(dim=-2, keepdim=True).transpose(-2, -1)

        out = numerator / (denominator + eps)

        # implementation #2
        # ones = torch.ones_like(v[..., :1])
        # _v = torch.cat([v, ones], dim=-1)
        # _kv = (_k.transpose(-2, -1) @ _v)

        # out = _q @ _kv
        # out = out[..., :-1] / (out[..., -1:] + eps)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward step of multi-head attention.

        Parameters
        ----------
        x : torch.Tensor
            Features as tensor of shape (b, n, d).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (b, n, d).
        """
        b = x.size(0)

        # Apply linear and reshape
        q = self.w_q(x).view(b, -1, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(b, -1, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(b, -1, self.h, self.d_k).transpose(1, 2)

        if self.attn_type == "linear" or self.attn_type == "linear-relu6":
            _x = self.linear_multi_head_attention(q=q, k=k, v=v, scale=self.scale)

        else:
            _x = self.vanilla_multi_head_attention(
                q=q, k=k, v=v, scale=self.scale, dropout_attn=self.dropout_attn
            )

        # Reshape (concat) across heads
        _x = _x.transpose(1, 2).contiguous().view(b, -1, self.h * self.d_k)

        # Apply final projection
        out = self.out_proj(_x)

        return nn.functional.dropout(out, p=self.dropout_attn)

    def _preload_hook(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        Preload hook to handle fuzed qkv matrices from pretrained vit weights.
        """
        w_key = prefix + "in_proj_weight"
        b_key = prefix + "in_proj_bias"

        if w_key in state_dict:
            w = state_dict.pop(w_key)
            w_q, w_k, w_v = w.chunk(3, dim=0)
            state_dict[prefix + "w_q.weight"] = w_q
            state_dict[prefix + "w_k.weight"] = w_k
            state_dict[prefix + "w_v.weight"] = w_v

        if b_key in state_dict:
            b = state_dict.pop(b_key)
            b_q, b_k, b_v = b.chunk(3, dim=0)
            state_dict[prefix + "w_q.bias"] = b_q
            state_dict[prefix + "w_k.bias"] = b_k
            state_dict[prefix + "w_v.bias"] = b_v

    def _initialize_norm_layers(self, d, norm_type):
        """
        Initialize norm layers.
        """
        if norm_type and norm_type not in ("layer_norm", "rms_norm"):
            raise ValueError(
                (
                    "Given `norm_type` argument is not valid. "
                    "Choose from ('layer_norm', 'rms_norm')."
                )
            )

        layer_cls = {
            False: nn.Identity,
            "layer_norm": nn.LayerNorm,
            "rms_norm": nn.RMSNorm,
        }
        layer_kws = {
            "layer_norm": {"elementwise_affine": False, "bias": False},
        }

        norm_q = layer_cls[norm_type](d, **layer_kws.get(norm_type, {}))
        norm_k = layer_cls[norm_type](d, **layer_kws.get(norm_type, {}))

        return norm_q, norm_k

    def _apply_norm_across_heads(self, norms, x):
        """
        Apply each `LayerNorm` across heads
        """
        return torch.stack(
            [
                norm(_x)
                for norm, _x in zip(norms, (x[:, i, ...] for i in range(self.h)))
            ],
            dim=1,
        )
