import os
import torch.optim as optim

try:
    import wandb
except ImportError:
    print("`wandb` was not found. `wandb` functionalities will be disabled.")

from torch import nn

from .brevitas_quant import (
    brevitas_ptq_quantize_pruned_model,
    export_to_qonnx,
    finalize_qat,
)
from .utils import (
    eval_sparsity,
    evaluate_model,
    load_checkpoint_state,
    save_quantized_model,
)
from sparse_vit.model import build_vit_base_model
from sparse_vit.training import train_epoch_qat, train_distill_epoch_qat
from sparse_vit.training.utils import init_lr_scheduler_cosine_w_warmup

from types import SimpleNamespace


def quantize_vit(
    model_paths,
    quant_device,
    build_vit_kw,
    loss_compute,
    eval_verbose,
    eval_log_interval,
    quant_scope,
    loaders,
    cfg,
    out_dir,
    run_id=None,
    model_name=None,
):
    """
    Quantize ViT.
    """
    results = []

    loaders_train = loaders["train"]
    loaders_valid = loaders["valid"]

    calib_loader = {"train": loaders_train, "val": loaders_valid}

    for idx, path in enumerate(model_paths, start=1):
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(model_paths)}] Quantizing: {path}")

        model = build_vit_base_model(**build_vit_kw).to(quant_device)

        transformed, missing, unexpected, pruned_module_names = load_checkpoint_state(
            model, path, quant_device
        )

        if missing or unexpected:
            print(f"load_state_dict missing={missing} unexpected={unexpected}")

        quant_method = str(cfg["quant"]["method"]).strip().lower()
        extra_payload = None

        if quant_method == "brevitas_ptq":
            if calib_loader is None:
                raise ValueError("Brevitas PTQ requires a calibration loader.")

            brevitas_args = SimpleNamespace(
                device=quant_device,
                wbits=int(cfg["quant"].get("bits", 8)),
                abits=int(cfg["quant"].get("abits", 8)),
                batches=int(cfg["quant"].get("calibrate_steps", 32)),
                calib_loader=calib_loader,
                verbose=bool(cfg["quant"].get("verbose", True)),
                quant_scope=quant_scope,
                use_gptq=bool(cfg["quant"].get("use_gptq", False)),
                use_int_norm=bool(cfg["quant"].get("use_int_norm", False)),
                weight_per_tensor=bool(cfg["quant"].get("weight_per_tensor", False)),
                fuse_rms_norm=bool(cfg["quant"].get("fuse_rms_norm", False)),
                weight_scale_po2=bool(cfg["quant"].get("weight_scale_po2", False)),
                weight_weight_po2=bool(cfg["quant"].get("weight_weight_po2", False)),
                act_scale_po2=bool(cfg["quant"].get("act_scale_po2", False)),
                encoder_only=bool(cfg["quant"].get("encoder_only", False)),
                channels=3,
                img_size=224,
            )

            quant_model, quantized_layer_names, brevitas_meta = (
                brevitas_ptq_quantize_pruned_model(
                    model,
                    brevitas_args,
                )
            )
            quant_eval_device = quant_device
            extra_payload = {
                "quant_meta": {
                    **(brevitas_meta or {}),
                    "method": "brevitas_ptq",
                    "bits": brevitas_args.wbits,
                    "act_bits": brevitas_args.abits,
                    "batches": brevitas_args.batches,
                },
            }

        else:
            raise ValueError("quant.method must be either 'brevitas_ptq'")

        if bool(cfg["quant"].get("qat", False)):
            qat_cfg = cfg.get("train", {}).get("qat", {})
            qat_epochs = int(qat_cfg.get("epochs", 5))
            qat_warmup = int(qat_cfg.get("warmup", 0))
            qat_lr = float(qat_cfg.get("lr", 1e-5))
            qat_wd = float(qat_cfg.get("weight_decay", 0.05))
            qat_total_epochs = qat_warmup + qat_epochs
            qat_distill = bool(qat_cfg.get("distill", False))

            teacher_model = None
            if qat_distill:
                # Fresh fp32 copy of the pre-quantization checkpoint, kept frozen as
                # the distillation teacher (quantization mutated `model` in place above).
                teacher_model = build_vit_base_model(**build_vit_kw).to(quant_device)
                load_checkpoint_state(teacher_model, path, quant_device)
                teacher_model.eval()
                for p in teacher_model.parameters():
                    p.requires_grad_(False)

            # Scale/zero-point params must not get weight decay — they're not weights.
            # 1-D params (biases) are also excluded, matching standard transformer practice.
            decay_params, no_decay_params = [], []
            for name, param in quant_model.named_parameters():
                if param.ndim <= 1 or "scale" in name or "zero_point" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

            optimizer = optim.AdamW(
                [
                    {"params": decay_params, "weight_decay": qat_wd},
                    {"params": no_decay_params, "weight_decay": 0.0},
                ],
                lr=qat_lr,
            )
            scheduler = init_lr_scheduler_cosine_w_warmup(
                optimizer, cosine_epochs=qat_epochs, warmup_epochs=qat_warmup
            )

            # Re-zero pruned weights after every optimizer step
            def _reapply_masks(opt, args, kwargs):
                for m in quant_model.modules():
                    if hasattr(m, "weight_mask"):
                        m.weight.data.mul_(m.weight_mask)

            optimizer.register_step_post_hook(_reapply_masks)

            training_kw = {
                "epochs": qat_total_epochs,
                "batch_size": cfg["dataset"]["dataloader"]["train"]["batch_size"],
            }

            print(
                f"\n[QAT] Starting {qat_total_epochs} epoch(s) (warmup={qat_warmup}, cosine={qat_epochs}) | "
                f"lr={qat_lr} | distill={qat_distill}"
            )
            for epoch in range(1, qat_total_epochs + 1):
                if qat_distill:
                    train_metrics = train_distill_epoch_qat(
                        data_loader=loaders_train,
                        s_model=quant_model,
                        t_model=teacher_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        device=quant_device,
                        loss_compute=loss_compute,
                        training_kw={
                            **training_kw,
                            "dstl_type": qat_cfg.get("dstl_type", "soft"),
                            "model_mode": cfg["train"]["model"]["encoder"].get(
                                "mode", "vit"
                            ),
                            "T": float(qat_cfg.get("temperature", 3.0)),
                            "alpha": float(qat_cfg.get("alpha", 0.5)),
                        },
                    )
                else:
                    train_metrics = train_epoch_qat(
                        data_loader=loaders_train,
                        model=quant_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        device=quant_device,
                        loss_compute=loss_compute,
                        training_kw=training_kw,
                    )
                val_metrics = evaluate_model(
                    quant_model,
                    loaders_valid,
                    quant_device,
                    loss_compute,
                    verbose=False,
                    log_interval=eval_log_interval,
                    stage_name=f"qat_val_epoch{epoch}",
                )
                print(
                    f"[QAT] Epoch {epoch}/{qat_total_epochs} | "
                    f"train_acc={train_metrics['acc']:.4f} | val_acc={val_metrics['acc']:.4f}"
                )

                try:
                    wandb.log(
                        {
                            "qat_val_acc": val_metrics["acc"],
                            "qat_val_loss": val_metrics["loss"],
                        },
                        step=epoch,
                    )
                except Exception as e:
                    print("Logging with `wandb` failed: {}".format(e))

            finalize_qat(quant_model, loaders_train, quant_device)

        q_sparsity = eval_sparsity(quant_model)["global"]
        q_metrics = evaluate_model(
            quant_model,
            loaders_valid,
            quant_eval_device,
            loss_compute,
            verbose=eval_verbose,
            log_interval=eval_log_interval,
            stage_name="eval_int8",
        )

        out_path = save_quantized_model(
            quant_model,
            out_dir,
            path,
            extra_payload=extra_payload,
            run_id=run_id,
            model_name=model_name,
        )

        row = {
            "input_model": path,
            "output_model": out_path,
            "checkpoint_had_pruning_reparam": bool(transformed),
            "quant_scope": quant_scope,
            "pruned_linear_layer_count_in_ckpt": len(pruned_module_names),
            "quantized_linear_layer_count": len(quantized_layer_names),
            "int8_sparsity": float(q_sparsity),
            "int8_acc": None if q_metrics is None else float(q_metrics["acc"]),
            "int8_loss": None if q_metrics is None else float(q_metrics["loss"]),
        }
        results.append(row)

        print(
            f"Saved: {out_path} | int8_sparsity={row['int8_sparsity']:.6f} "
            f"quantized_layers={row['quantized_linear_layer_count']}"
        )
        if q_metrics is not None:
            print(f"Eval int8 acc/loss={row['int8_acc']:.6f}/{row['int8_loss']:.6f}")

        try:
            wandb.log(
                {
                    "quant_" + k: v
                    for k, v in row.items()
                    if k not in ("input_model", "output_model")
                },
                step=idx,
            )
        except Exception as e:
            print("Logging with `wandb` failed: {}".format(e))

        # Export to QONNX (mirrors the saved `.pth` name, including the run id)

        if bool(cfg["quant"].get("export_qonnx", True)):
            os.makedirs(out_dir, exist_ok=True)
            out_path_qonnx = os.path.splitext(out_path)[0] + ".qonnx"
            export_to_qonnx(quant_model, brevitas_args, out_path_qonnx)

    return results
