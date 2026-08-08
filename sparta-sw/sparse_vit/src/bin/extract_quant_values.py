"""
extract_quant_values.py  —  export a Brevitas QAT checkpoint's quantization
values (per-layer int weights/biases + scale/zero_point/bit_width, plus every
activation quantizer's scale/zero_point/bit_width) to a plain .txt file for
loading onto the FPGA.

Requires the model architecture (via --cfg_path) since the raw state_dict
only holds latent float weights — the actual INT4/INT8 snapped grid is only
available by rebuilding the Brevitas layers and calling quant_weight() /
quant_bias(), same as inspect_quant.py's "full model" mode.

Usage:
    python src/bin/extract_quant_values.py \
        --ckpt_path runs/quantized/vit-base-cifar10-pruned-quant-brevitas-ptq-w4-a8-qat.pth \
        --cfg_path cfgs/quant_vit.toml \
        --out runs/quantized/vit-base-cifar10-pruned-quant-brevitas-ptq-w4-a8-qat.quant_values.txt
"""

import os
import sys
import argparse
import toml
import math
import torch
import torch.nn as nn

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import brevitas.nn as qnn

from sparse_vit.model import build_vit_base_model
from sparse_vit.quantization.brevitas_quant import (
    _iter_target_layers,
    _get_parent_and_attr,
    _replace_with_brevitas,
    _setup_dyadic_residuals,
    fuse_rms_norms,
)

from sparse_vit.quantization.conversion import (
    patch_embedding_layer,
    patch_encoder_residuals,
    patch_mha_softmax,
)

from sparse_vit.quantization.utils import _densify_pruning_state_dict

from sparse_vit.quantization.models import (
    QuantRMSNorm,
    DyadicResidualAdd,
)

# ── checkpoint / state-dict helpers ─────────────────────────────────────────


def _load_ckpt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _resolve_state(ckpt):
    if isinstance(ckpt, dict):
        for key in ("state", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if all(isinstance(v, torch.Tensor) for v in list(ckpt.values())[:3]):
            return ckpt
    return ckpt


# ── model building ──────────────────────────────────────────────────────────


def build_quant_model(
    cfg,
    wbits,
    abits,
    weight_per_tensor=False,
    weight_scale_po2=False,
    weight_weight_po2=False,
    act_scale_po2=False,
    fuse_rms_norm=False,
    encoder_only=False,
):
    """
    Build quantized model.
    """
    build_kw = {}
    build_kw.update(cfg["train"]["model"]["embed"])
    build_kw.update(cfg["train"]["model"]["encoder"])
    build_kw.update(cfg["train"]["model"]["task"])

    if fuse_rms_norm is True:
        build_kw["norm_kw"] = {"elementwise_affine": False}

    # Architecture only — the checkpoint's own state_dict provides all values,
    # so there is no need to load any intermediate pretrained/pruned model.
    build_kw["params"] = "init"
    build_kw["params_kw"] = {}

    model = build_vit_base_model(**build_kw)

    if fuse_rms_norm is True:
        fuse_rms_norms(model, encoder_only=encoder_only)

    for name, module in list(_iter_target_layers(model)):

        parent, attr = _get_parent_and_attr(model, name)

        is_heads = name.startswith("heads")
        is_embed = name.startswith("embed")

        if encoder_only is True and (is_heads or is_embed):
            continue

        setattr(
            parent,
            attr,
            _replace_with_brevitas(
                module,
                wbits=wbits,
                abits=abits,
                per_tensor=weight_per_tensor,
                wpo2_s=weight_scale_po2,
                wpo2_w=weight_weight_po2,
                apo2=act_scale_po2,
            ),
        )

    if encoder_only is False:
        patch_embedding_layer(model, abits=abits, po2=act_scale_po2)

    patch_encoder_residuals(model, abits=abits, po2=act_scale_po2)
    patch_mha_softmax(model, abits=abits, po2=act_scale_po2)

    return model, build_kw["channels"], build_kw["img_size"]


# ── formatting helpers ───────────────────────────────────────────────────────


def _flat(t):
    return t.detach().reshape(-1).tolist()


def _fmt_floats(vals):
    return " ".join(f"{v:.10g}" for v in vals)


def _fmt_ints(vals):
    return " ".join(str(int(v)) for v in vals)


def _fmt_shifts(vals):
    """For power-of-two scales: the right-shift amount k such that scale == 2**-k
    (real_value ~= int_value >> k). Only meaningful when the scale was actually
    produced by a po2-restricted quantizer -- see the `po2` flag on the callers."""
    return " ".join(f"{-math.log2(v):.0f}" for v in vals)


def _write_int_matrix(f, values_2d):
    for row in values_2d:
        f.write(_fmt_ints(row) + "\n")


def _write_quant_tensor_block(f, prefix, qt, po2=False):
    """qt: brevitas IntQuantTensor (has .value/.scale/.zero_point/.bit_width/.signed/.int())."""
    shape = tuple(qt.value.shape)
    f.write(f"{prefix}.shape        : {shape}\n")
    f.write(f"{prefix}.bit_width    : {int(round(qt.bit_width.item()))}\n")
    f.write(f"{prefix}.signed       : {bool(qt.signed)}\n")
    f.write(f"{prefix}.scale        : {_fmt_floats(_flat(qt.scale))}\n")
    if po2 is True:
        f.write(
            f"{prefix}.shift        : {_fmt_shifts(_flat(qt.scale))}  (right-shift k, scale == 2**-k)\n"
        )
    f.write(f"{prefix}.zero_point   : {_fmt_floats(_flat(qt.zero_point))}\n")

    ivals = qt.int(float_datatype=False)
    if ivals.ndim >= 2:
        rows = ivals.reshape(ivals.shape[0], -1)
        f.write(
            f"{prefix}.int          : ({rows.shape[0]} rows x {rows.shape[1]} vals)\n"
        )
        _write_int_matrix(f, rows.tolist())
    else:
        f.write(f"{prefix}.int          : {_fmt_ints(_flat(ivals))}\n")


def _write_act_quant_block(f, prefix, scale, zero_point, bit_width, signed, po2=False):
    f.write(f"{prefix}.bit_width    : {int(round(bit_width.item()))}\n")
    f.write(f"{prefix}.signed       : {bool(signed)}\n")
    f.write(f"{prefix}.scale        : {_fmt_floats(_flat(scale))}\n")
    if po2 is True:
        f.write(
            f"{prefix}.shift        : {_fmt_shifts(_flat(scale))}  (right-shift k, scale == 2**-k)\n"
        )
    f.write(f"{prefix}.zero_point   : {_fmt_floats(_flat(zero_point))}\n")


def _write_dyadic_block(f, m):
    """m: a DyadicResidualAdd — writes the frozen integer multiply+shift coefficients
    actually used on the inference path (setup_dyadic() must have been called)."""
    f.write(f"bit_width           : {m.bit}\n")
    f.write(f"ready               : {bool(m._ready.item())}\n")
    f.write(f"branch_a.scale      : {m.align_a.act_quant.scale().item():.10g}\n")
    f.write(
        f"branch_a.bit_width  : {int(round(m.align_a.act_quant.bit_width().item()))}\n"
    )
    f.write(f"branch_b.scale      : {m.align_b.act_quant.scale().item():.10g}\n")
    f.write(
        f"branch_b.bit_width  : {int(round(m.align_b.act_quant.bit_width().item()))}\n"
    )
    f.write(f"out.scale           : {m.out_quant.act_quant.scale().item():.10g}\n")
    f.write(
        f"out.bit_width       : {int(round(m.out_quant.act_quant.bit_width().item()))}\n"
    )
    f.write(f"m_a                 : {int(m.m_a.item())}\n")
    f.write(f"e_a                 : {int(round(m.e_a.item()))}\n")
    f.write(f"m_b                 : {int(m.m_b.item())}\n")
    f.write(f"e_b                 : {int(round(m.e_b.item()))}\n")
    f.write(f"scale_out           : {m.scale_out.item():.10g}\n")


def _write_dyadic_hardware_spec(f):
    """Write the bit-accurate contract for consuming dyadic residual values."""
    f.write("=" * 80 + "\n")
    f.write("DYADIC RESIDUAL HARDWARE SPECIFICATION\n")
    f.write("=" * 80 + "\n\n")
    f.write("dyadic_spec_version : 1\n")
    f.write("input_type          : signed integer at the corresponding branch scale\n")
    f.write("zero_point          : 0\n")
    f.write("multiplier_type     : positive signed INT32 constant\n")
    f.write("product_type        : signed INT40 minimum\n")
    f.write("shift_type          : non-negative integer arithmetic right shift\n")
    f.write("rounding_mode       : round-to-nearest, ties-to-even\n")
    f.write("saturate_stage      : after aligned branch addition\n\n")
    f.write("Operation for every DYADIC block:\n")
    f.write("  product_a = signed_int40(a_int) * signed_int32(m_a)\n")
    f.write("  product_b = signed_int40(b_int) * signed_int32(m_b)\n")
    f.write("  aligned_a = round_to_nearest_even(product_a / 2^e_a)\n")
    f.write("  aligned_b = round_to_nearest_even(product_b / 2^e_b)\n")
    f.write("  sum_wide  = aligned_a + aligned_b\n")
    f.write("  out_int   = saturate(sum_wide, -2^(bit_width-1),\n")
    f.write("                       2^(bit_width-1)-1)\n\n")
    f.write("Scale contract:\n")
    f.write("  real_a      = a_int * branch_a.scale\n")
    f.write("  real_b      = b_int * branch_b.scale\n")
    f.write("  real_output = out_int * scale_out\n")
    f.write(
        "  scale_out is tensor metadata; do not multiply by it on the FPGA unless\n"
    )
    f.write("  a floating-point output is explicitly required.\n\n")
    f.write("Branch mapping:\n")
    f.write("  encoder.encoder_layer_i.res1:\n")
    f.write("    branch_a = transformer block input / skip branch\n")
    f.write("    branch_b = multi-head-attention output\n")
    f.write("  encoder.encoder_layer_i.res2:\n")
    f.write("    branch_a = res1 output / skip branch\n")
    f.write("    branch_b = MLP output\n")
    f.write("  embed.res_pe:\n")
    f.write("    branch_a = token embeddings\n")
    f.write("    branch_b = positional embeddings\n\n")
    f.write("The input integers must already use branch_a.scale and branch_b.scale.\n")
    f.write(
        "If an upstream tensor uses a different scale, requantize it first or fuse\n"
    )
    f.write("that conversion into the branch multiplier and shift.\n\n")


# ── extraction ───────────────────────────────────────────────────────────────


def _extract(model, f, weight_scale_po2=False, act_scale_po2=False):
    sep = "=" * 80

    f.write(sep + "\n")
    f.write("QUANTIZED LINEAR / CONV2D LAYERS  (weight, bias, input activation)\n")
    f.write(sep + "\n\n")

    n_layers = 0
    for name, module in model.named_modules():
        if not isinstance(module, (qnn.QuantLinear, qnn.QuantConv2d)):
            continue

        kind = type(module).__name__
        f.write("-" * 80 + "\n")
        f.write(f"LAYER: {name}  ({kind})\n")
        f.write("-" * 80 + "\n")

        qw = module.quant_weight()
        _write_quant_tensor_block(f, "weight", qw, po2=weight_scale_po2)

        if module.bias is not None:
            qb = module.quant_bias()
            _write_quant_tensor_block(f, "bias", qb)

        _write_act_quant_block(
            f,
            "input",
            module.input_quant.scale(),
            module.input_quant.zero_point(),
            module.input_quant.bit_width(),
            module.input_quant.is_signed,
            po2=act_scale_po2,
        )
        f.write("\n")
        n_layers += 1

    f.write(sep + "\n")
    f.write("RMSNORM MODULES  (gamma weight + input/output activation quantizers)\n")
    f.write(sep + "\n\n")

    covered = set()  # id()s of QuantIdentity submodules already reported below,
    # so the generic scan at the end doesn't duplicate them.

    n_norms = 0
    for name, module in model.named_modules():
        if not isinstance(module, QuantRMSNorm):
            continue

        f.write("-" * 80 + "\n")
        f.write(f"NORM: {name}\n")
        f.write("-" * 80 + "\n")

        if module.weight_quant is not None:

            gamma_qt = module.weight_quant.act_quant(module.rms.weight)
            _write_quant_tensor_block(f, "gamma", gamma_qt, po2=act_scale_po2)
            covered.add(id(module.weight_quant))

        if module.input_quant is not None:
            _write_act_quant_block(
                f,
                "input",
                module.input_quant.act_quant.scale(),
                module.input_quant.act_quant.zero_point(),
                module.input_quant.act_quant.bit_width(),
                module.input_quant.act_quant.is_signed,
                po2=act_scale_po2,
            )
            covered.add(id(module.input_quant))

        if module.output_quant is not None:
            _write_act_quant_block(
                f,
                "output",
                module.output_quant.act_quant.scale(),
                module.output_quant.act_quant.zero_point(),
                module.output_quant.act_quant.bit_width(),
                module.output_quant.act_quant.is_signed,
                po2=act_scale_po2,
            )
            covered.add(id(module.output_quant))

        f.write("\n")
        n_norms += 1

    _write_dyadic_hardware_spec(f)

    f.write(sep + "\n")
    f.write("DYADIC RESIDUAL ADDS  (frozen integer multiply+shift coefficients)\n")
    f.write(sep + "\n\n")

    n_dyadic = 0
    for name, module in model.named_modules():
        if not isinstance(module, DyadicResidualAdd):
            continue

        f.write("-" * 80 + "\n")
        f.write(f"DYADIC: {name}\n")
        f.write("-" * 80 + "\n")
        _write_dyadic_block(f, module)
        f.write("\n")

        covered.add(id(module.align_a))
        covered.add(id(module.align_b))
        covered.add(id(module.out_quant))
        n_dyadic += 1

    f.write(sep + "\n")
    f.write("OTHER STANDALONE ACTIVATION QUANTIZERS  (embed, mha q/k/v/out, ...)\n")
    f.write(sep + "\n\n")

    n_acts = 0
    for name, module in model.named_modules():
        if not isinstance(module, qnn.QuantIdentity):
            continue
        if id(module) in covered:
            continue

        f.write("-" * 80 + "\n")
        f.write(f"ACT: {name}\n")
        f.write("-" * 80 + "\n")
        _write_act_quant_block(
            f,
            "act",
            module.act_quant.scale(),
            module.act_quant.zero_point(),
            module.act_quant.bit_width(),
            module.act_quant.is_signed,
            po2=act_scale_po2,
        )
        f.write("\n")
        n_acts += 1

    return n_layers, n_norms, n_dyadic, n_acts


# ── main ─────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Export a Brevitas QAT checkpoint's quantization values to a .txt file for FPGA."
    )
    p.add_argument(
        "--ckpt_path",
        required=True,
        help="Path to .pth checkpoint",
    )
    p.add_argument(
        "--cfg_path",
        required=True,
        help="Path to quant .toml config (architecture)",
    )
    p.add_argument(
        "--wbits",
        type=int,
        default=None,
        help="Weight bits (auto-read from quant_meta if omitted)",
    )
    p.add_argument(
        "--abits",
        type=int,
        default=None,
        help="Activation bits (auto-read from quant_meta if omitted)",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output .txt path (default: alongside checkpoint)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    ckpt = _load_ckpt(args.ckpt_path)
    state = _resolve_state(ckpt)
    meta = ckpt.get("quant_meta", {}) if isinstance(ckpt, dict) else {}

    wbits = args.wbits or meta.get("bits", 8)
    abits = args.abits or meta.get("act_bits", 8)

    weight_per_tensor = meta.get("weight_granularity", "per_channel") == "per_tensor"
    weight_scale_po2 = bool(meta.get("weight_scale_po2", False))
    weight_weight_po2 = bool(meta.get("weight_weight_po2", True))
    act_scale_po2 = bool(meta.get("act_scale_po2", False))

    print(f"Checkpoint : {args.ckpt_path}")
    print(f"Config     : {args.cfg_path}")
    print(f"Bit widths : W{wbits}A{abits}")
    print(
        f"Weight granularity : {'per_tensor' if weight_per_tensor else 'per_channel'}"
    )
    print(f"Weight scheme : {'po2' if weight_weight_po2 else 'uniform'}")
    print(
        f"Scale type : weight={'po2' if weight_scale_po2 else 'float'} act={'po2' if act_scale_po2 else 'float'}"
    )

    with open(args.cfg_path) as fh:
        cfg = toml.load(fh)

    fuse_rms_norm = bool(cfg["quant"].get("fuse_rms_norm", False))
    encoder_only = bool(cfg["quant"].get("encoder_only", False))

    print("Building model architecture...")
    model, channels, img_size = build_quant_model(
        cfg,
        wbits=wbits,
        abits=abits,
        weight_per_tensor=weight_per_tensor,
        weight_scale_po2=weight_scale_po2,
        weight_weight_po2=weight_weight_po2,
        act_scale_po2=act_scale_po2,
        fuse_rms_norm=fuse_rms_norm,
        encoder_only=encoder_only,
    )

    state, _ = _densify_pruning_state_dict(state)

    print("Loading checkpoint state_dict...")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  missing keys (first 5): {missing[:5]}")

    model.eval()

    # Freeze/refresh the dyadic residual-add coefficients from whatever scales
    # actually ended up in this checkpoint (idempotent — safe even if the
    # training pipeline already called this before saving).
    _setup_dyadic_residuals(model)

    # quant_bias() needs a cached input scale from an actual forward pass.
    for module in model.modules():
        if (
            isinstance(module, (qnn.QuantLinear, qnn.QuantConv2d))
            and module.bias is not None
        ):
            module.bias_quant.cache_inference_quant_bias = True

    with torch.no_grad():
        model(torch.randn(1, channels, img_size, img_size))

    out_path = args.out
    if out_path is None:
        stem, _ = os.path.splitext(args.ckpt_path)
        out_path = stem + ".quant_values.txt"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    print(f"Extracting quantization values -> {out_path}")
    with open(out_path, "w") as f:
        f.write(f"# source checkpoint : {os.path.abspath(args.ckpt_path)}\n")
        f.write(f"# config             : {os.path.abspath(args.cfg_path)}\n")
        f.write(f"# bit widths         : W{wbits}A{abits}\n")
        f.write(
            f"# weight granularity : {'per_tensor' if weight_per_tensor else 'per_channel'}\n"
        )
        f.write(
            f"# scale type         : weight={'po2' if weight_scale_po2 else 'float'} act={'po2' if act_scale_po2 else 'float'}\n"
        )
        f.write(f"# quant_meta         : {meta}\n\n")

        n_layers, n_norms, n_dyadic, n_acts = _extract(
            model, f, weight_scale_po2=weight_scale_po2, act_scale_po2=act_scale_po2
        )

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(
        f"Done. {n_layers} quantized layers, {n_norms} RMSNorm modules, "
        f"{n_dyadic} dyadic residual adds, {n_acts} other activation quantizers."
    )
    print(f"Wrote {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
