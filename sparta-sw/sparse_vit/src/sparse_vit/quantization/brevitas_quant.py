import inspect
import math

import torch
import torch.nn as nn
import brevitas.nn as qnn

from brevitas.graph.gptq import gptq_mode
from brevitas.graph.calibrate import calibration_mode

from brevitas.export import export_qonnx

from sparse_vit.quantization.utils import evaluate_model, eval_sparsity
from sparse_vit.quantization.conversion import (
    linear_to_qlinear,
    conv_to_qconv,
    relu_to_qrelu,
    rmsnorm_to_qrmsnorm,
    rmsnorm_to_irmsnorm,
    patch_embedding_layer,
    patch_encoder_residuals,
    patch_mha_softmax,
)
from sparse_vit.quantization.models import (
    IntLinearAttnNorm,
    IntRMSNorm,
    DyadicResidualAdd,
)

from sparse_vit.quantization.utils import _dequant, _batch_frexp
from sparse_vit.quantization.logging import (
    _LOG_RANGES,
    _RANGE_STATS,
    range_logging_mode,
    print_range_stats,
)

# ---------------------------------------------------------------------------
# Range logging — enabled inside range_logging_mode(), zero overhead otherwise.
# Used to size the loop bounds of the integer-domain norm ops (use_int_norm=True).
# ---------------------------------------------------------------------------


def _iter_target_layers(model: nn.Module):
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d, nn.RMSNorm)):
            continue
        yield name, module


def _get_parent_and_attr(root: nn.Module, dotted_name: str):
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _replace_with_brevitas(
    module: nn.Module,
    wbits: int,
    abits: int,
    use_int_norm: bool = False,
    per_tensor: bool = False,
    wpo2_s: bool = True,
    wpo2_w: bool = True,
    apo2: bool = True,
    device: torch.device | str = torch.device("cpu"),
):
    """
    Replace input module with a brevitas module.
    """
    if isinstance(module, nn.Linear):
        return linear_to_qlinear(
            module,
            wbits=wbits,
            abits=abits,
            per_tensor=per_tensor,
            wpo2_s=wpo2_s,
            wpo2_w=wpo2_w,
            apo2=apo2,
        )

    elif isinstance(module, nn.Conv2d):
        return conv_to_qconv(
            module,
            wbits=wbits,
            abits=abits,
            per_tensor=per_tensor,
            wpo2_s=wpo2_s,
            wpo2_w=wpo2_w,
            apo2=apo2,
        )

    elif isinstance(module, nn.ReLU):
        return relu_to_qrelu(abits=abits, apo2=apo2)

    elif isinstance(module, nn.RMSNorm):

        if use_int_norm is False:
            return rmsnorm_to_qrmsnorm(
                module,
                wbits=wbits,
                abits=abits,
                per_tensor=per_tensor,
                wpo2_s=wpo2_s,
                wpo2_w=wpo2_w,
                apo2=apo2,
                device=device,
            )

        elif use_int_norm is True:
            return rmsnorm_to_irmsnorm(
                module,
                wbits=wbits,
                abits=abits,
                per_tensor=per_tensor,
                wpo2_s=wpo2_s,
                wpo2_w=wpo2_w,
                apo2=apo2,
            )

    else:
        raise TypeError(
            f"Unsupported module type for Brevitas replacement: {type(module)}"
        )


def _check_quantization(model: nn.Module):
    """
    Check quantization
    """
    for name, module in model.named_modules():
        if isinstance(module, qnn.QuantLinear):
            w_float = module.weight.data
            w_quant = module.quant_weight().value
            unique_vals = w_quant.unique().numel()
            mse = (w_float - w_quant).pow(2).mean().item()
            print(
                f"[quant_check] {name}: unique_vals={unique_vals} (max 256 for INT8) | quant_mse={mse:.6e}"
            )
            break


def _check_scales(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, qnn.QuantIdentity):
            try:
                scale = module.act_quant.scale()
                calibrated = abs(scale.item() - 0.0078125) > 1e-8
                print(
                    f"[scale_check] {name}: scale={scale.item():.6e} | calibrated={'YES' if calibrated else 'NO'}"
                )
            except Exception as e:
                print(f"[scale_check] {name}: could not read scale -- {e}")
            break


def _sanity_check(fp32_model: nn.Module, int8_model: nn.Module, loader, device):
    fp32_model.eval()
    int8_model.eval()
    images, labels = next(iter(loader["val"]))
    images = images.to(device)

    with torch.no_grad():
        out_fp32 = fp32_model(images)
        out_int8 = int8_model(images)

    out_int8 = _dequant(out_int8)

    diff = (out_fp32 - out_int8).abs().mean().item()
    print(f"[sanity_check] Output diff fp32 vs int8: {diff:.6e}")
    print(f"[sanity_check] fp32 preds: {out_fp32.argmax(-1)[:8].tolist()}")
    print(f"[sanity_check] int8 preds: {out_int8.argmax(-1)[:8].tolist()}")


def _calibrate(model: nn.Module, calib_loader, device, batches: int, verbose: bool):
    if batches <= 0:
        return
    model.eval()
    print(f"[Brevitas-PTQ] Starting calibration for {batches} batch(es).")
    seen = 0
    with calibration_mode(model):
        with torch.no_grad():
            for images, _ in calib_loader["train"]:
                images = images.to(device)
                _ = model(images)
                seen += 1
                if verbose:
                    print(f"[Brevitas-PTQ] Calibration batch {seen}/{batches} done.")
                if seen >= batches:
                    break


def _gptq_calibrate(
    model: nn.Module, calib_loader, device, batches: int, verbose: bool
):
    """ """
    if batches <= 0:
        return
    model.eval()

    with calibration_mode(model):
        with torch.no_grad():
            seen = 0
            for images, _ in calib_loader["train"]:
                images = images.to(device)
                model(images)
                seen += 1
                if seen >= batches:
                    break

    with gptq_mode(model, use_quant_activations=True) as gptq:
        for i in range(gptq.num_layers):
            with torch.no_grad():
                seen = 0
                for images, _ in calib_loader["train"]:
                    if seen >= batches:
                        break
                    images = images.to(device)
                    model(images)
                    seen += 1
            gptq.update()
            for _, m in model.named_modules():
                if hasattr(m, "weight_mask"):
                    m.weight.data.mul_(m.weight_mask)
            if verbose:
                print(f"[Brevitas-GPTQ] Layer {i + 1}/{gptq.num_layers} done.")


def _setup_dyadic_residuals(model: nn.Module):
    """Freeze dyadic coefficients in every DyadicResidualAdd after calibration."""
    for _, module in model.named_modules():
        if isinstance(module, DyadicResidualAdd):
            module.setup_dyadic()


def _fuse_norm_into_linears(norm: nn.Module, linears: list) -> bool:
    """
    Fold an RMSNorm's per-channel gamma into the weight of every Linear that
    consumes its output directly, then drop gamma entirely.

    y = (x_norm * gamma) @ W.T == x_norm @ (W * gamma).T, so this only rescales
    each Linear's input-feature columns -- valid regardless of whether the
    Linear has a bias, since bias is added after the matmul untouched.
    Requires `norm` to be a plain `nn.RMSNorm` (no bias, no mean-subtraction)
    and every linear in `linears` to take norm's output with no elementwise op
    in between.
    """
    if not isinstance(norm, nn.RMSNorm) or not norm.elementwise_affine:
        return False

    gamma = norm.weight.data

    with torch.no_grad():
        for linear in linears:
            linear.weight.data.mul_(gamma)

    norm.weight = None
    norm.elementwise_affine = False

    return True


def fuse_rms_norms(model: nn.Module, encoder_only: bool = False) -> list:
    """
    Fuse every pre-norm RMSNorm in the ViT (both `ln_mha`/`ln_mlp` inside each
    encoder layer and the classifier head's final `ln`) into the linear
    layer(s) that immediately follow it. Must be called on the fp32 model
    before `_replace_with_brevitas` swaps norms/linears for Brevitas modules.

    Returns the dotted names of the fused norms. For the `QuantRMSNorm` path
    (use_int_norm=False) fusing drops gamma entirely, so there's nothing left
    to retrain. `IntRMSNorm` (use_int_norm=True) always carries its own weight
    regardless of the source norm's affine setting, so these names are still
    passed to `_freeze_norm_weights` post-replacement for that path -- otherwise
    QAT would happily relearn a per-channel scale back into it and silently
    undo the fusion.
    """
    from sparse_vit.model.encoder import ViTEncoderLayer
    from sparse_vit.model.ffn import FeedForwardNetwork, GatedFeedForwardNetwork

    fused_names = []

    for name, module in model.named_modules():
        if isinstance(module, ViTEncoderLayer):

            if _fuse_norm_into_linears(
                module.ln_mha, [module.mha.w_q, module.mha.w_k, module.mha.w_v]
            ):
                fused_names.append(f"{name}.ln_mha" if name else "ln_mha")

            mlp = module.mlp
            if isinstance(mlp, GatedFeedForwardNetwork):
                mlp_linears = [mlp.gate_proj, mlp.up_proj]
            elif isinstance(mlp, FeedForwardNetwork):
                mlp_linears = [mlp.linear_1]
            else:
                mlp_linears = []

            if mlp_linears and _fuse_norm_into_linears(module.ln_mlp, mlp_linears):
                fused_names.append(f"{name}.ln_mlp" if name else "ln_mlp")

    heads = getattr(model, "heads", None)

    if (
        encoder_only is False
        and heads is not None
        and hasattr(heads, "ln")
        and hasattr(heads, "head")
    ):
        if _fuse_norm_into_linears(heads.ln, [heads.head]):
            fused_names.append("heads.ln")

    return fused_names


def _freeze_norm_weights(model: nn.Module, names: list):
    """
    Freeze the gamma of the given (fused) norms so QAT can't relearn a
    per-channel scale back into them. Only `IntRMSNorm` actually has anything
    to freeze here -- it always constructs its own all-ones `.weight`
    regardless of the source norm's affine setting. `QuantRMSNorm` has no
    weight left after fusion (gamma is dropped entirely), so this is a safe
    no-op for that path.
    """
    lookup = dict(model.named_modules())
    for name in names:
        norm = lookup.get(name)
        if norm is None:
            continue
        weight = getattr(norm, "weight", None)
        if weight is None and hasattr(norm, "rms"):
            weight = norm.rms.weight
        if weight is not None:
            weight.requires_grad_(False)


def _fold_layernorm_params(model: nn.Module, bits: int = 8):
    def _fold(param):
        scale = param.data.abs().max() / (2 ** (bits - 1) - 1)
        param.data = (
            param.data.div(scale)
            .round()
            .clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
            .mul(scale)
        )

    for module in model.modules():
        if isinstance(module, IntRMSNorm):
            if module.weight is not None:
                _fold(module.weight)


def _compute_loop_bounds(stats: dict) -> dict:
    bounds = {}
    if "isqrt_x" in stats:
        lo, hi = stats["isqrt_x"]["min"], stats["isqrt_x"]["max"]
        n_halv = (max(0, math.ceil(math.log2(hi)) - 1) + 1) if hi >= 2.0 else 0
        n_doubl = (
            (max(0, math.ceil(-math.log2(lo))) + 1) if (lo > 0 and lo < 1.0) else 0
        )
        bounds["isqrt_n_halv"] = n_halv
        bounds["isqrt_n_doubl"] = n_doubl
    if "inv_x" in stats:
        lo, hi = stats["inv_x"]["min"], stats["inv_x"]["max"]
        n_halv = (max(0, math.ceil(math.log2(hi)) - 1) + 1) if hi >= 2.0 else 0
        n_doubl = (
            (max(0, math.ceil(-math.log2(lo))) + 1) if (lo > 0 and lo < 1.0) else 0
        )
        bounds["inv_n_halv"] = n_halv
        bounds["inv_n_doubl"] = n_doubl
    return bounds


def _apply_loop_bounds(model: nn.Module, stats: dict):
    bounds = _compute_loop_bounds(stats)
    if not bounds:
        return
    n_rms = n_la = 0
    for module in model.modules():
        if isinstance(module, IntRMSNorm):
            if "isqrt_n_halv" in bounds:
                module.isqrt_n_halv = bounds["isqrt_n_halv"]
                module.isqrt_n_doubl = bounds["isqrt_n_doubl"]
            n_rms += 1
        elif isinstance(module, IntLinearAttnNorm):
            if "inv_n_halv" in bounds:
                module.inv_n_halv = bounds["inv_n_halv"]
                module.inv_n_doubl = bounds["inv_n_doubl"]
            n_la += 1
    print(
        f"[loop_bounds] isqrt={bounds.get('isqrt_n_halv','?')}/{bounds.get('isqrt_n_doubl','?')} "
        f"inv={bounds.get('inv_n_halv','?')}/{bounds.get('inv_n_doubl','?')} "
        f"→ updated {n_rms} IntRMSNorm, {n_la} IntLinearAttnNorm"
    )


def recalibrate_loop_bounds(model: nn.Module, loader, device):
    """Re-collect polynomial input ranges and update loop bounds after QAT."""
    model.eval()
    with range_logging_mode() as stats:
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                model(images)
    print("[QAT] Updated range stats after training:")
    print_range_stats()
    _apply_loop_bounds(model, stats)


def finalize_qat(model: nn.Module, loader, device):
    """
    Call once after QAT training is complete, before eval or export.

    If the model uses the integer-domain norm ops (use_int_norm=True at
    quantization time), also re-folds IntRMSNorm weights to the int8 grid
    and re-measures _int_isqrt / _int_inv loop bounds, since the activation
    and weight distributions shift during training.
    """
    model.eval()
    uses_int_norm = any(
        isinstance(m, (IntRMSNorm, IntLinearAttnNorm)) for m in model.modules()
    )
    _setup_dyadic_residuals(model)
    if uses_int_norm:
        _fold_layernorm_params(model)
        recalibrate_loop_bounds(model, loader, device)
    print(
        f"[finalize_qat] done: dyadic residuals frozen{', loop bounds updated' if uses_int_norm else ''}."
    )


def brevitas_ptq_quantize_pruned_model(model: nn.Module, args):
    """
    Replace eligible layers with Brevitas quantized layers and run a calibration pass.

    Parameters
    ----------
    device : torch.device
        Device where quantization will take place.
    wbits : int
        The weight quantization bits.
    abits : int
        The activation quantization bits.
    batches :
        calib_loader
        verbose
        use_gptq: if True, use GPTQ calibration instead of plain PTQ
    use_int_norm : bool
        If True, replace RMSNorm and the linear-attention denominator division
        with the integer-domain approximations (IntRMSNorm / IntLinearAttnNorm,
        built on _int_isqrt / _int_inv) instead of the float rms_norm / float
        division path. No Div/Sqrt ops end up in the exported QONNX graph, at
        the cost of a small approximation error.
    weight_per_tensor : bool
        If True, quantize Linear/Conv2d weights with a single per-tensor scale
        instead of one scale per output channel. Per-channel gives better accuracy
        at low bit-widths.
    weight_scale_po2 : bool
        If True, restrict Linear/Conv2d weight quantizer scales to powers of two.
    weight_weight_po2 : bool
        If True, restrict Linear / Conv2D weights to take values from powers
        of two.
    act_scale_po2 : bool
        If True, restrict every activation quantizer scales to powers of two.
        (QuantLinear / QuantConv2d, QuantRMSNorm, Residual etc.)
    encoder_only : bool
        If True, quantize only the encoder layers.
    """
    model.eval().to(args.device)

    use_int_norm = bool(getattr(args, "use_int_norm", False))
    weight_per_tensor = bool(getattr(args, "weight_per_tensor", False))
    fuse_rms_norm = bool(getattr(args, "fuse_rms_norm", False))
    wpo2_s = bool(getattr(args, "weight_scale_po2", False))
    wpo2_w = bool(getattr(args, "weight_weight_po2", False))
    apo2 = bool(getattr(args, "act_scale_po2", False))
    encoder_only = bool(getattr(args, "encoder_only", False))

    sparsity_before = eval_sparsity(model)
    print(f"[sparsity] before quantization: global={sparsity_before['global']:.4f}")

    fused_norm_names = []
    if fuse_rms_norm:
        fused_norm_names = fuse_rms_norms(model, encoder_only=encoder_only)
        print(
            f"[fuse_rms_norm] fused {len(fused_norm_names)} RMSNorm layer(s) "
            f"into adjacent linears: {fused_norm_names}"
        )

    quantized_layer_names = []

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
                wbits=int(args.wbits),
                abits=int(args.abits),
                use_int_norm=use_int_norm,
                per_tensor=weight_per_tensor,
                wpo2_s=wpo2_s,
                wpo2_w=wpo2_w,
                apo2=apo2,
                device=args.device,
            ),
        )
        quantized_layer_names.append(name)

    if fused_norm_names:
        _freeze_norm_weights(model, fused_norm_names)

    model = model.to(args.device)
    abits = int(args.abits)

    # Path ViT layers
    if encoder_only is False:
        patch_embedding_layer(model, abits=abits, po2=apo2)

    patch_encoder_residuals(model, abits=abits, po2=apo2)
    patch_mha_softmax(model, abits=abits, use_int_norm=use_int_norm, po2=apo2)

    use_gptq = bool(getattr(args, "use_gptq", False))
    verbose = bool(getattr(args, "verbose", True))

    if use_gptq:
        if verbose:
            print("[Brevitas] Using GPTQ calibration.")
        _gptq_calibrate(
            model=model,
            calib_loader=args.calib_loader,
            device=args.device,
            batches=int(args.batches),
            verbose=verbose,
        )
    else:
        if verbose:
            print("[Brevitas] Using plain PTQ calibration.")
        _calibrate(
            model=model,
            calib_loader=args.calib_loader,
            device=args.device,
            batches=int(args.batches),
            verbose=verbose,
        )

    _setup_dyadic_residuals(model)
    if use_int_norm:
        _fold_layernorm_params(model, bits=int(args.wbits))
    _check_quantization(model)
    _check_scales(model)

    sparsity_after = eval_sparsity(model)
    print(f"[sparsity] after  quantization: global={sparsity_after['global']:.4f}")
    print(
        f"[sparsity] delta: {sparsity_after['global'] - sparsity_before['global']:+.4f}"
    )

    acc_metrics = {}
    calib_loader = getattr(args, "calib_loader", None)
    if calib_loader is not None and "val" in calib_loader:
        if use_int_norm:
            # First pass: collect polynomial input ranges to set loop bounds.
            with range_logging_mode():
                evaluate_model(
                    model, calib_loader["val"], args.device, nn.CrossEntropyLoss()
                )
            print_range_stats()
            _apply_loop_bounds(model, _RANGE_STATS)
            # Second pass: evaluate with corrected loop counts.
        acc_metrics = evaluate_model(
            model, calib_loader["val"], args.device, nn.CrossEntropyLoss()
        )
        if acc_metrics:
            print(f"[accuracy] post-quant val acc: {acc_metrics['acc']:.4f}")

    meta = {
        "quantized_layers": len(quantized_layer_names),
        "quant_scope": "auto_linear_conv2d",
        "method": "gptq" if use_gptq else "ptq",
        "norm_mode": "int_approx" if use_int_norm else "float",
        "weight_granularity": "per_tensor" if weight_per_tensor else "per_channel",
        "fused_rms_norms": len(fused_norm_names),
        "weight_scale_po2": wpo2_s,
        "weight_weight_po2": wpo2_w,
        "act_scale_po2": apo2,
        "sparsity_before": sparsity_before["global"],
        "sparsity_after": sparsity_after["global"],
        "val_acc": acc_metrics.get("acc"),
    }

    return model, quantized_layer_names, meta


# EXTRACT TO QONNX
def _is_brevitas_quant_module(mod):
    cls_name = mod.__class__.__name__.lower()
    mod_name = mod.__class__.__module__.lower()
    return ("brevitas" in mod_name) and ("quant" in cls_name or "quant" in mod_name)


def _summarize_quant_modules(model):
    quant_modules = []
    for name, mod in model.named_modules():
        if _is_brevitas_quant_module(mod):
            quant_modules.append((name, mod.__class__.__name__))
    print(f"[debug] brevitas quant-like modules found: {len(quant_modules)}")
    for name, cls_name in quant_modules[:40]:
        print(f"[debug]   {name}: {cls_name}")
    if len(quant_modules) > 40:
        print(f"[debug]   ... ({len(quant_modules) - 40} more)")


def _print_state_load_report(missing, unexpected):
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("[load] missing keys (first 80):")
        for key in missing[:80]:
            print(f"[load]   MISSING {key}")
        if len(missing) > 80:
            print(f"[load]   ... ({len(missing) - 80} more missing keys)")
    if unexpected:
        print("[load] unexpected keys (first 80):")
        for key in unexpected[:80]:
            print(f"[load]   UNEXPECTED {key}")
        if len(unexpected) > 80:
            print(f"[load]   ... ({len(unexpected) - 80} more unexpected keys)")


def _debug_forward_summary(model, x):
    """ """
    with torch.no_grad():
        y = model(x)

    y0 = y[0] if isinstance(y, (tuple, list)) else y

    if hasattr(y0, "value"):
        y0 = y0.value

    if torch.is_tensor(y0):
        print(
            f"[debug] forward ok: output shape={tuple(y0.shape)} dtype={y0.dtype} "
            f"min={float(y0.min()):.6f} max={float(y0.max()):.6f}"
        )
    else:
        print(f"[debug] forward ok: non-tensor output type={type(y0)}")


def _call_with_supported_kwargs(fn, **kwargs):
    """Call fn with only kwargs supported by its signature."""
    sig = inspect.signature(fn)
    supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**supported)


def export_to_qonnx(model: nn.Module, args, out_dir):
    """ """
    model.eval()
    model = model.cpu()
    _summarize_quant_modules(model)

    # Prepare dummy input
    x = torch.randn(1, args.channels, args.img_size, args.img_size, dtype=torch.float32)
    _debug_forward_summary(model, x)

    # Export to ONNX/QONNX
    export_qonnx(model, export_path=out_dir, input_t=x)
    print(f"[export] Model exported to QONNX at: {out_dir}")
