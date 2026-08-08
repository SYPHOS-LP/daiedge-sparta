import types
import torch
import torch.nn as nn
import brevitas.nn as qnn

from brevitas.quant import (
    Int8ActPerTensorFloat,
    Int8WeightPerTensorFloat,
    Int8WeightPerChannelFloat,
    Uint8ActPerTensorFloat,
    Int32Bias,
)

from brevitas.quant import (
    Int8WeightPerTensorFixedPoint,
    Int8ActPerTensorFixedPoint,
    Uint8ActPerTensorFixedPoint,
)
from brevitas.quant.fixed_point import Int8WeightPerChannelFixedPoint

try:
    from .pot4_weight_per_tensor_fixed_point import PoT4WeightPerTensorFixedPoint
except ImportError:
    print("`brevitas-utils` is not available.")

from .models import (
    QuantRMSNorm,
    IntRMSNorm,
    DyadicResidualAdd,
    IntLinearAttnNorm,
)
from .utils import _dequant


class Int4WeightPerChannelFloat(Int8WeightPerChannelFloat):
    bit_width = 4


class Int4WeightPerTensorFloat(Int8WeightPerTensorFloat):
    bit_width = 4


class Int4ActPerTensorFloat(Int8ActPerTensorFloat):
    bit_width = 4


class Uint4ActPerTensorFloat(Uint8ActPerTensorFloat):
    bit_width = 4


class Int4WeightPerChannelFixedPoint(Int8WeightPerChannelFixedPoint):
    bit_width = 4


class Int4WeightPerTensorFixedPoint(Int8WeightPerTensorFixedPoint):
    bit_width = 4


class Int4ActPerTensorFixedPoint(Int8ActPerTensorFixedPoint):
    bit_width = 4


class Uint4ActPerTensorFixedPoint(Uint8ActPerTensorFixedPoint):
    bit_width = 4


def _get_weight_quantization_cls(
    bits: int, per_tensor: bool = False, po2_s: bool = False, po2_w: bool = True
):
    """
    Get weight quantization class.
    """
    weight_quant_clss = {
        (4, True, False, False): Int4WeightPerTensorFloat,
        (8, True, False, False): Int8WeightPerTensorFloat,
        (4, False, False, False): Int4WeightPerChannelFloat,
        (8, False, False, False): Int8WeightPerChannelFloat,
        (4, True, True, False): Int4WeightPerTensorFixedPoint,
        (8, True, True, False): Int8WeightPerTensorFixedPoint,
        (4, False, True, False): Int4WeightPerChannelFixedPoint,
        (8, False, True, False): Int8WeightPerChannelFixedPoint,
        (4, True, True, True): PoT4WeightPerTensorFixedPoint,
    }

    return weight_quant_clss[(bits, per_tensor, po2_s, po2_w)]


def _get_activation_quantization_cls(
    bits: int, unsigned: bool = False, po2: bool = False
):
    """
    Get acitvation quantization class.
    """
    act_quant_clss = {
        (4, False, False): Int4ActPerTensorFloat,
        (8, False, False): Int8ActPerTensorFloat,
        (4, True, False): Uint4ActPerTensorFloat,
        (8, True, False): Uint8ActPerTensorFloat,
        (4, False, True): Int4ActPerTensorFixedPoint,
        (8, False, True): Int8ActPerTensorFixedPoint,
        (4, True, True): Uint4ActPerTensorFixedPoint,
        (8, True, True): Uint8ActPerTensorFixedPoint,
    }

    return act_quant_clss[bits, unsigned, po2]


def linear_to_qlinear(
    module: nn.Module,
    wbits: int,
    abits: int,
    per_tensor: bool = True,
    wpo2_s: bool = True,
    wpo2_w: bool = True,
    apo2: bool = True,
) -> qnn.QuantLinear:
    """
    Convert `Linear` to `QuantLinear`.
    """
    weight_quant = _get_weight_quantization_cls(
        wbits, per_tensor=per_tensor, po2_s=wpo2_s, po2_w=wpo2_w
    )
    input_quant = _get_activation_quantization_cls(abits, po2=apo2)

    device = module.weight.device
    dtype = module.weight.dtype

    qmod = qnn.QuantLinear(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=module.bias is not None,
        weight_bit_width=wbits,
        weight_quant=weight_quant,
        input_bit_width=abits,
        input_quant=input_quant,
        bias_quant=Int32Bias,
        return_quant_tensor=True,
    ).to(device=device, dtype=dtype)

    with torch.no_grad():
        qmod.weight.data.copy_(module.weight.data)

        if module.bias is not None:
            qmod.bias.data.copy_(module.bias.data)

        weight_mask = (module.weight.data != 0).to(dtype)

        qmod.register_buffer("weight_mask", weight_mask)
        qmod.weight.data.mul_(weight_mask)

    return qmod


def relu_to_qrelu(abits: int, apo2: bool = True) -> qnn.QuantReLU:
    """
    Convert `ReLU` to `QuantReLU`.
    """
    input_quant = _get_activation_quantization_cls(abits, unsigned=True, po2=apo2)

    qrelu = qnn.QuantReLU(act_quant=input_quant, return_quant_tensor=True)

    return qrelu


def conv_to_qconv(
    module: nn.Module,
    wbits: int,
    abits: int,
    per_tensor: bool = True,
    wpo2_s: bool = True,
    wpo2_w: bool = True,
    apo2: bool = True,
    device: torch.device | str | None = None,
) -> qnn.QuantConv2d:
    """
    Convert `Conv` to `QConv`
    """
    weight_quant = _get_weight_quantization_cls(
        wbits, per_tensor=per_tensor, po2_s=wpo2_s, po2_w=wpo2_w
    )
    input_quant = _get_activation_quantization_cls(abits, po2=apo2)

    device = module.weight.device
    dtype = module.weight.dtype

    qmod = qnn.QuantConv2d(
        in_channels=module.in_channels,
        out_channels=module.out_channels,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=module.bias is not None,
        padding_mode=module.padding_mode,
        weight_bit_width=wbits,
        weight_quant=weight_quant,
        input_bit_width=abits,
        input_quant=input_quant,
        bias_quant=Int32Bias,
        return_quant_tensor=True,
    ).to(device=device, dtype=dtype)

    with torch.no_grad():
        qmod.weight.data.copy_(module.weight.data)

        if module.bias is not None:
            qmod.bias.data.copy_(module.bias.data)

        weight_mask = (module.weight.data != 0).to(dtype)

        qmod.register_buffer("weight_mask", weight_mask)
        qmod.weight.data.mul_(weight_mask)

    return qmod


def rmsnorm_to_qrmsnorm(
    module: nn.Module,
    wbits: int,
    abits: int,
    per_tensor: bool = True,
    wpo2_s: bool = True,
    wpo2_w: bool = True,
    apo2: bool = True,
    device: torch.device | str | None = None,
) -> QuantRMSNorm:
    """
    Convert `RMSNorm` to `QuantRMSNorm`.
    """
    weight_quant = _get_weight_quantization_cls(
        wbits, per_tensor=per_tensor, po2_s=wpo2_s, po2_w=wpo2_w
    )
    act_quant = _get_activation_quantization_cls(abits, po2=apo2)

    device = module.weight.device if module.elementwise_affine else device

    eps = module.eps if module.eps is not None else 1e-6
    normalized_shape = module.normalized_shape
    elementwise_affine = module.elementwise_affine

    qmod = QuantRMSNorm(
        normalized_shape=normalized_shape,
        eps=eps,
        elementwise_affine=elementwise_affine,
        act_quant=act_quant,
        weight_quant=weight_quant,
        input_quant=True,
        output_quant=True,
    ).to(device)

    with torch.no_grad():
        if module.elementwise_affine is True and module.weight is not None:
            qmod.rms_norm.weight.data.copy_(module.weight.data)

    return qmod


def rmsnorm_to_irmsnorm(
    module: nn.Module,
    wbits: int,
    abits: int,
    per_tensor: bool = True,
    wpo2_s: bool = True,
    wpo2_w: bool = True,
    apo2: bool = True,
) -> IntRMSNorm:
    """ """
    raise NotImplementedError()


def patch_embedding_layer(model: nn.Module, abits: int = 8, po2: bool = True) -> None:
    """
    A function to patch the model's embedding layer
    """
    from sparse_vit.model.embedding import EmbeddingPatchifyLinear

    device = next(model.parameters()).device

    act_quant = _get_activation_quantization_cls(bits=abits, po2=po2)

    for _, module in model.named_modules():
        if not isinstance(module, EmbeddingPatchifyLinear):
            continue
        if not getattr(module, "use_cls", False):
            continue

        module.quant_cls = qnn.QuantIdentity(
            act_quant=act_quant, return_quant_tensor=False
        ).to(device)

        module.quant_pe = qnn.QuantIdentity(
            act_quant=act_quant, return_quant_tensor=False
        ).to(device)

        module.res_pe = DyadicResidualAdd(bit=abits, act_quant=act_quant).to(device)

        def patched_forward(self, x_img):
            b, _, _, _ = x_img.size()
            x = self.conv_proj(x_img)
            x = x.view(b, self.d, self.n_p).transpose(1, 2)

            cls = self.quant_cls(self.class_token)
            xs = [cls.expand(b, -1, -1)]

            if self.use_dstl:
                xs.append(self.dstl_token.expand(b, -1, -1))
            xs.append(_dequant(x))

            if self.use_regs > 0:
                xs.append(self.reg_tokens.expand(b, -1, -1))
            x = torch.cat(xs, dim=1)

            if self.pe_type != "none":
                pe = self.quant_pe(self.pos_embedding)
                x = self.res_pe(x, pe)
            return x

        module.forward = types.MethodType(patched_forward, module)


def patch_encoder_residuals(model: nn.Module, abits: int = 8, po2: bool = True) -> None:
    """
    Patch the encoder layers of a ViT model.
    """
    from sparse_vit.model.encoder import ViTEncoderLayer

    device = next(model.parameters()).device

    act_quant = _get_activation_quantization_cls(bits=abits, po2=po2)

    for name, module in model.named_modules():
        if not isinstance(module, ViTEncoderLayer):
            continue

        module.quant_input = qnn.QuantIdentity(
            act_quant=act_quant, return_quant_tensor=True
        ).to(device)

        module.res1 = DyadicResidualAdd(bit=abits, act_quant=act_quant).to(device)
        module.res2 = DyadicResidualAdd(bit=abits, act_quant=act_quant).to(device)

        # Re-quantizes the float output of res1 before passing to the norm
        module.quant_pre_mlp = qnn.QuantIdentity(
            act_quant=act_quant, return_quant_tensor=True
        ).to(device)

        def patched_forward(self, x):

            x_q = self.quant_input(x)

            _x = self.ln_mha(x_q)
            _x = self.mha(_x)

            x_res = self.res1(x_q, _x)

            _y = self.ln_mlp(self.quant_pre_mlp(x_res))
            _y = self.mlp(_y)

            return self.res2(x_res, _y)

        module.forward = types.MethodType(patched_forward, module)


def patch_mha_softmax(
    model: nn.Module, abits: int = 8, use_int_norm: bool = False, po2: bool = True
) -> None:
    """
    Patch the `MultiHeadAttention` layers of a ViT model.
    """
    from sparse_vit.model.mha import MultiHeadAttention

    device = next(model.parameters()).device
    act_quant = _get_activation_quantization_cls(abits, po2=po2)

    for _, module in model.named_modules():
        if not isinstance(module, MultiHeadAttention):
            continue

        module.quant_q = qnn.QuantIdentity(
            act_quant=act_quant, return_quant_tensor=True
        ).to(device)
        module.quant_k = qnn.QuantIdentity(
            act_quant=act_quant, return_quant_tensor=True
        ).to(device)
        module.quant_v = qnn.QuantIdentity(
            act_quant=act_quant, return_quant_tensor=True
        ).to(device)

        if module.attn_type == "linear":
            module.quant_out = qnn.QuantIdentity(
                act_quant=act_quant, return_quant_tensor=True
            ).to(device)

            if use_int_norm:
                module.linear_attn_norm = IntLinearAttnNorm().to(device)

            def patched_forward_linear(self, x):
                b = x.size(0)

                q = self.w_q(x)
                q_f = (
                    (q.value if hasattr(q, "value") else q)
                    .view(b, -1, self.h, self.d_k)
                    .transpose(1, 2)
                )
                q_f = self.quant_q(q_f)
                q_f = q_f.value if hasattr(q_f, "value") else q_f

                k = self.w_k(x)
                k_f = (
                    (k.value if hasattr(k, "value") else k)
                    .view(b, -1, self.h, self.d_k)
                    .transpose(1, 2)
                )
                k_f = self.quant_k(k_f)
                k_f = k_f.value if hasattr(k_f, "value") else k_f

                v = self.w_v(x)
                v_f = (
                    (v.value if hasattr(v, "value") else v)
                    .view(b, -1, self.h, self.d_k)
                    .transpose(1, 2)
                )
                v_f = self.quant_v(v_f)
                v_f = v_f.value if hasattr(v_f, "value") else v_f

                _q = self.activation_q(q_f * self.scale**0.5)
                _k = self.activation_k(k_f * self.scale**0.5)
                _kv = torch.matmul(_k.transpose(-1, -2), v_f)

                numerator = torch.matmul(_q, _kv)
                denominator = torch.matmul(
                    _q, _k.sum(dim=-2, keepdim=True).transpose(-2, -1)
                )

                if hasattr(self, "linear_attn_norm"):
                    out = self.linear_attn_norm(numerator, denominator)
                else:
                    out = numerator / (denominator + 1e-8)

                out = self.quant_out(out)
                out_f = out.value if hasattr(out, "value") else out
                out_f = (
                    out_f.transpose(1, 2).contiguous().view(b, -1, self.h * self.d_k)
                )
                return self.out_proj(out_f)

            module.forward = types.MethodType(patched_forward_linear, module)
