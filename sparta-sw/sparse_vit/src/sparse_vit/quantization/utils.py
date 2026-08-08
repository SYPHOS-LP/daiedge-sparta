import os
import torch
import csv
import torch.nn as nn
import numpy as np
import brevitas.nn as qnn

from collections import OrderedDict
from decimal import Decimal, ROUND_HALF_UP


def _densify_pruning_state_dict(state_dict: OrderedDict):
    """ """
    if not isinstance(state_dict, (dict, OrderedDict)):
        return state_dict, False

    dense = OrderedDict(state_dict)
    transformed = False

    keys = list(dense.keys())
    for key in keys:
        if not key.endswith(".weight_orig"):
            continue

        prefix = key[: -len(".weight_orig")]
        mask_key = prefix + ".weight_mask"
        weight_key = prefix + ".weight"

        if mask_key not in dense:
            continue

        dense[weight_key] = dense[key] * dense[mask_key]
        dense.pop(key, None)
        dense.pop(mask_key, None)
        transformed = True

    return dense, transformed


def _extract_pruned_module_names_from_state_dict(state_dict):
    """ """
    names = set()
    if not isinstance(state_dict, (dict, OrderedDict)):
        return names

    for key in state_dict.keys():
        if key.endswith(".weight_mask"):
            names.add(key[: -len(".weight_mask")])
    return names


def load_checkpoint_state(model: nn.Module, model_path: str, device: torch.device):
    """ """
    raw = torch.load(model_path, map_location=device, weights_only=True)
    state = raw["state"] if isinstance(raw, dict) and "state" in raw else raw
    pruned_module_names = _extract_pruned_module_names_from_state_dict(state)

    state, transformed = _densify_pruning_state_dict(state)

    try:
        missing, unexpected = model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Strict load failed for {model_path}: {e}")

    return transformed, missing, unexpected, pruned_module_names


def evaluate_model(
    model: nn.Module,
    data_loader,
    device: torch.device,
    loss_compute,
    verbose: bool = False,
    log_interval: int = 1,
    stage_name: str = "eval",
):
    """ """
    if data_loader is None:
        return None

    model.eval()
    totals = 0
    corrects = 0
    loss_sum = 0.0
    steps = 0
    total_batches = len(data_loader) if hasattr(data_loader, "__len__") else None

    with torch.no_grad():
        for image, label in data_loader:
            image = image.to(device)
            label = label.to(device)
            logits = model(image)

            # ✅ unwrap QuantTensor if needed
            if hasattr(logits, "value"):
                logits = logits.value

            loss = loss_compute(logits, label)
            preds = torch.argmax(logits, dim=-1)

            totals += label.numel()
            corrects += (preds == label).sum().item()
            loss_sum += float(loss.item())
            steps += 1

            if verbose and (steps % log_interval == 0):
                running_acc = float(corrects) / float(totals) if totals > 0 else 0.0
                running_loss = float(loss_sum) / float(steps)
                if total_batches is None:
                    batch_progress = f"{steps}/UNK"
                else:
                    batch_progress = f"{steps}/{total_batches}"
                print(
                    f"[{stage_name}] batch={batch_progress} "
                    f"running_acc={running_acc:.6f} "
                    f"running_loss={running_loss:.6f}"
                )

    if steps == 0:
        return None

    return {
        "acc": float(corrects) / float(totals),
        "loss": float(loss_sum) / float(steps),
    }


def eval_sparsity(model: nn.Module) -> dict:
    """ """
    total = 0
    zeros = 0
    per_layer = {}
    for name, module in model.named_modules():
        if not isinstance(
            module, (nn.Linear, nn.Conv2d, qnn.QuantLinear, qnn.QuantConv2d)
        ):
            continue
        w = module.weight.data
        layer_total = w.numel()
        layer_zeros = (w == 0).sum().item()
        per_layer[name] = layer_zeros / layer_total if layer_total else 0.0
        total += layer_total
        zeros += layer_zeros
    return {
        "global": zeros / total if total else 0.0,
        "per_layer": per_layer,
    }


def save_quantized_model(
    model: nn.Module,
    out_dir: str,
    in_model_path: str,
    extra_payload: dict | None = None,
    run_id: str | None = None,
    model_name: str | None = None,
):
    """ """
    os.makedirs(out_dir, exist_ok=True)
    if model_name:
        stem = model_name
    else:
        base = os.path.basename(in_model_path)
        stem, _ = os.path.splitext(base)
    run_suffix = f"-{run_id}" if run_id else ""
    out_path = os.path.join(out_dir, f"{stem}{run_suffix}.pth")
    payload = {"state": model.state_dict()}
    if extra_payload:
        payload.update(extra_payload)
    torch.save(payload, out_path)
    return out_path


def save_results_csv(rows, out_dir: str):
    """ """
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "quant_results.csv")
    fieldnames = [
        "input_model",
        "output_model",
        "checkpoint_had_pruning_reparam",
        "quant_scope",
        "pruned_linear_layer_count_in_ckpt",
        "quantized_linear_layer_count",
        "int8_sparsity",
        "int8_acc",
        "int8_loss",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return csv_path


def _dequant(t: torch.Tensor) -> torch.Tensor:
    return t.value if hasattr(t, "value") else t


def _batch_frexp(inputs: torch.Tensor, max_bit: int = 31):
    """
    Decompose a scale ratio into (mantissa, exponent) for dyadic fixed-point arithmetic.
    Ported from I-ViT: ratio ≈ mantissa * 2^(-exponent), where mantissa is a 31-bit integer.
    On hardware this is a multiply followed by an arithmetic right-shift.
    """
    shape = inputs.size()
    vals = inputs.view(-1).cpu().numpy()
    m_raw, e_raw = np.frexp(vals)
    mantissas = np.array(
        [
            int(
                Decimal(float(m) * (2**max_bit)).quantize(
                    Decimal("1"), rounding=ROUND_HALF_UP
                )
            )
            for m in m_raw
        ]
    )
    exponents = float(max_bit) - e_raw
    return (
        torch.from_numpy(mantissas).to(inputs.device).view(shape),
        torch.from_numpy(exponents).to(inputs.device).view(shape),
    )
