import torch
import torch.nn as nn
import brevitas.nn as qnn

from brevitas.quant_tensor import QuantTensor
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

from .utils import _dequant, _batch_frexp
from .logging import _LOG_RANGES, _RANGE_STATS, _range_update


class QuantRMSNorm(nn.Module):
    """
    A class to represent a quantized RMSNorm.
    """

    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        act_quant: Int8ActPerTensorFloat = Int8ActPerTensorFloat,
        input_quant: bool = False,
        output_quant: bool = True,
        weight_quant: Int8WeightPerTensorFloat | None = None,
    ):
        """

        Parameters
        ----------
        normalized_shape : int
            Input shape from an expected input of size.
        elementwise_affine : bool
            Use learnable per-element affine parameters initialized to ones (for weights).
        act_quant : Int8ActPerTensorFloat

        """
        super().__init__()

        self.normalized_shape = normalized_shape
        self.eps = eps

        self.rms_norm = nn.RMSNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )

        self.input_quant = (
            qnn.QuantIdentity(act_quant=act_quant) if input_quant else None
        )
        self.output_quant = (
            qnn.QuantIdentity(act_quant=act_quant) if output_quant else None
        )

        if elementwise_affine is True and weight_quant is not None:
            self.weight_quant = weight_quant
        else:
            self.weight_quant = None

    def forward(self, x):
        """
        Forward pass of `QuantRMSNorm`.
        """
        if isinstance(x, QuantTensor):
            x = x.value

        if self.input_quant is not None:
            x = self.input_quant(x)

        if self.weight_quant is None:
            x = self.rms_norm(x)

        else:
            if self.rms_norm.weight is None:
                raise RuntimeError(
                    "gamma_quantizer was supplied, but " "elementwise_affine=False."
                )

            quant_weight = self.weight_quant(self.rms_norm.weight)

            x = nn.functional.rms_norm(
                x, self.normalized_shape, weight=quant_weight, eps=self.eps
            )

        if self.output_quant is not None:
            x = self.output_quant(x)

        return x


def _int_isqrt(x: torch.Tensor, n_halv: int, n_doubl: int) -> torch.Tensor:
    """
    1/sqrt(x) using only Where + Mul + Add — no Log, Pow, or Sqrt ops.

    Finds the bit-position k = floor(log2(x)) by iterative halving/doubling:
      - halve x_n while >= 2  (absorb 1/sqrt(2) into scale each step)
      - double x_n while < 1  (absorb sqrt(2) into scale each step)
    After the loops x_n ∈ [1, 2) and scale = 2^(-k/2).
    A minimax quadratic gives 1/sqrt(x_n), multiplied by scale gives 1/sqrt(x),
    then one Newton-Raphson step refines to ~0.002 % error.

    n_halv/n_doubl are set automatically by _apply_loop_bounds() from observed ranges.
    ONNX ops: Where, Mul, Add, Sub only.
    """
    _INV_SQRT2 = 0.7071067811865476
    _SQRT2 = 1.4142135623730951

    x_n = x.clamp(min=1e-30)
    if _LOG_RANGES:
        _range_update("isqrt_x", x_n)
    scale = torch.ones_like(x_n)

    for _ in range(n_halv):
        mask = x_n >= 2.0
        x_n = torch.where(mask, x_n * 0.5, x_n)
        scale = torch.where(mask, scale * _INV_SQRT2, scale)

    for _ in range(n_doubl):
        mask = x_n < 1.0
        x_n = torch.where(mask, x_n * 2.0, x_n)
        scale = torch.where(mask, scale * _SQRT2, scale)

    # Quadratic minimax for 1/sqrt(x_n) on [1, 2), exact at x_n ∈ {1, 1.5, 2}
    poly = 0.1482 * x_n * x_n - 0.7375 * x_n + 1.5893
    y0 = poly * scale

    # One Newton-Raphson step: y*(1.5 - 0.5*x*y²) → ~0.002 % error
    return y0 * (1.5 - 0.5 * x * y0 * y0)


class IntRMSNorm(nn.Module):
    """
    Integer-domain RMSNorm using polynomial _int_isqrt approximation.
    No mean subtraction and no bias (matches the RMSNorm contract).
    Output is requantized via QuantIdentity.
    Loop counts are set automatically by _apply_loop_bounds() after calibration.
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-8,
        isqrt_n_halv: int = 14,
        isqrt_n_doubl: int = 17,
        act_quant=Int8ActPerTensorFloat,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.isqrt_n_halv = isqrt_n_halv
        self.isqrt_n_doubl = isqrt_n_doubl
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.output_quant = qnn.QuantIdentity(
            act_quant=act_quant,
            return_quant_tensor=True,
        )

    def forward(self, x):
        x_f = x.value if hasattr(x, "value") else x
        mean_sq = (x_f * x_f).mean(dim=-1, keepdim=True)
        y = (
            x_f
            * _int_isqrt(mean_sq + self.eps, self.isqrt_n_halv, self.isqrt_n_doubl)
            * self.weight
        )
        return self.output_quant(y)


def _int_inv(x: torch.Tensor, n_halv: int, n_doubl: int) -> torch.Tensor:
    """
    1/x using only Where + Mul + Add — no Div or Reciprocal ops.

    Normalises x to [1, 2) by iterative halving/doubling, applies a
    quadratic minimax polynomial, scales back, then one Newton-Raphson step.

    n_halv/n_doubl are set automatically by _apply_loop_bounds() from observed ranges.
    ONNX ops: Where, Mul, Add, Sub only.
    """
    x_n = x.clamp(min=1e-30)
    if _LOG_RANGES:
        _range_update("inv_x", x_n)
    scale = torch.ones_like(x_n)

    for _ in range(n_halv):
        mask = x_n >= 2.0
        x_n = torch.where(mask, x_n * 0.5, x_n)
        scale = torch.where(mask, scale * 0.5, scale)

    for _ in range(n_doubl):
        mask = x_n < 1.0
        x_n = torch.where(mask, x_n * 2.0, x_n)
        scale = torch.where(mask, scale * 2.0, scale)

    # Quadratic minimax for 1/x_n on [1, 2), exact at x_n ∈ {1, 1.5, 2}
    poly = 0.3333 * x_n * x_n - 1.5 * x_n + 2.1667
    y0 = poly * scale

    # Newton-Raphson step: y*(2 - x*y) → ~0.001 % error
    return y0 * (2.0 - x * y0)


class IntLinearAttnNorm(nn.Module):
    """
    Integer-domain denominator normalization for linear attention.

    Replaces the float division  out = numerator / (denominator + eps)
    with                         out = numerator * _int_inv(denominator + eps)
    so the QONNX graph has no Div nodes — only Where + Mul + Add.
    Loop counts are set automatically by _apply_loop_bounds() after calibration.
    """

    def __init__(self, inv_n_halv: int = 10, inv_n_doubl: int = 3, eps: float = 1e-8):
        super().__init__()
        self.inv_n_halv = inv_n_halv
        self.inv_n_doubl = inv_n_doubl
        self.eps = eps

    def forward(self, numerator, denominator):
        return numerator * _int_inv(
            denominator + self.eps, self.inv_n_halv, self.inv_n_doubl
        )


class QuantMatMul(nn.Module):
    """
    Integer-domain matrix multiply for QuantTensors.
    Computes C = A @ B as: c_int32 = a_int8 @ b_int8, then rescales by scale_a * scale_b,
    then requantizes the output to int8.
    Falls back to float matmul if either input is not a QuantTensor.
    """

    def __init__(self, output_quant=Int8ActPerTensorFloat):
        super().__init__()
        self.output_quant_module = qnn.QuantIdentity(
            act_quant=output_quant,
            return_quant_tensor=True,
        )

    def forward(self, a, b):
        if (
            hasattr(a, "int")
            and hasattr(a, "scale")
            and hasattr(b, "int")
            and hasattr(b, "scale")
        ):
            a_int = a.int().to(torch.int32)
            b_int = b.int().to(torch.int32)
            c_float = torch.matmul(a_int.float(), b_int.float()) * (a.scale * b.scale)
        else:
            a_f = a.value if hasattr(a, "value") else a
            b_f = b.value if hasattr(b, "value") else b
            c_float = torch.matmul(a_f, b_f)
        return self.output_quant_module(c_float)


class DyadicResidualAdd(nn.Module):
    """
    Integer residual addition following the I-ViT dyadic fixed-point approach.

    During calibration (float path):
      - align_a and align_b observe the ranges of the two input branches.
      - out_quant observes the range of their sum and calibrates the output scale.

    After setup_dyadic():
      - Each branch is requantized from its own scale to s_out using
        integer multiply + right-shift (batch_frexp decomposition).
      - The two requantized integers are added and clipped to INT8.
      - No float arithmetic on the hardware critical path.
    """

    def __init__(self, bit: int = 8, act_quant=Int8ActPerTensorFloat):
        super().__init__()
        self.bit = bit
        self.n = 2 ** (bit - 1) - 1
        self.align_a = qnn.QuantIdentity(act_quant=act_quant, return_quant_tensor=True)
        self.align_b = qnn.QuantIdentity(act_quant=act_quant, return_quant_tensor=True)
        self.out_quant = qnn.QuantIdentity(
            act_quant=act_quant, return_quant_tensor=True
        )
        self.register_buffer("m_a", torch.ones(1, dtype=torch.long))
        self.register_buffer("e_a", torch.zeros(1, dtype=torch.float))
        self.register_buffer("m_b", torch.ones(1, dtype=torch.long))
        self.register_buffer("e_b", torch.zeros(1, dtype=torch.float))
        self.register_buffer("scale_out", torch.ones(1, dtype=torch.float))
        self.register_buffer("_ready", torch.tensor(False))

    def setup_dyadic(self):
        """Call once after calibration to freeze the dyadic coefficients."""
        s_a = self.align_a.act_quant.scale().detach().float()
        s_b = self.align_b.act_quant.scale().detach().float()
        s_out = self.out_quant.act_quant.scale().detach().float()
        self.scale_out.copy_(s_out)
        m_a, e_a = _batch_frexp((s_a / s_out).view(1))
        m_b, e_b = _batch_frexp((s_b / s_out).view(1))
        self.m_a.copy_(m_a)
        self.e_a.copy_(e_a)
        self.m_b.copy_(m_b)
        self.e_b.copy_(e_b)
        self._ready.fill_(True)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_f = _dequant(a)
        b_f = _dequant(b)

        if not self._ready.item() or self.training:
            # Calibration path and QAT training: STE gradients flow through QuantIdentity
            qa = self.align_a(a_f)
            qb = self.align_b(b_f)
            return self.out_quant(_dequant(qa) + _dequant(qb))

        # Dyadic integer path — inference only (matches I-ViT fixedpoint_mul)
        s_a = self.align_a.act_quant.scale()
        s_b = self.align_b.act_quant.scale()
        a_int = torch.round(a_f / s_a)
        b_int = torch.round(b_f / s_b)
        a_req = torch.round(a_int * self.m_a.float() / (2.0 ** self.e_a.float()))
        b_req = torch.round(b_int * self.m_b.float() / (2.0 ** self.e_b.float()))
        out_int = torch.clamp(a_req + b_req, -(self.n + 1), self.n)
        return out_int * self.scale_out
