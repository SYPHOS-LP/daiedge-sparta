from typing import Tuple, Dict, List
from numpy.typing import ArrayLike

import torch
import numpy as np

from torch.nn import functional as F
from torch.amp.grad_scaler import GradScaler

from .utils import (
    log_activation_sparsity,
    log_activation_sparsity_quant,
    log_activation_sparsity__gather_modules,
)
from .utils import _freeze_mlp_weights

try:
    import wandb
except ImportError:
    print("`wandb` was not found. `wandb` functionalities will be disabled.")


def log_multiclass__classification(
    logits: torch.Tensor, label: torch.Tensor | None
) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Prepare predictions for metric calculation.
    """
    # extract probabilities and measure metrics
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)

    if label is None:
        return preds.numpy(force=True), None

    elif label.dim() == 2:
        label = label.argmax(dim=-1)

    return preds.numpy(force=True), label.numpy(force=True)


def log_update_metrics(
    metrics: Dict, preds: ArrayLike, trues: ArrayLike, loss: torch.Tensor
) -> Dict:
    """
    Log classification metrics.
    """
    metrics["totals"] += len(preds)
    metrics["corrects"] += (preds == trues).sum()
    metrics["acc"] = metrics["corrects"] / metrics["totals"]
    metrics["loss"] += loss.item()

    return metrics


def create_plotly_graph_with_error_bars(x, y, error_y):
    """
    Create plotly graph with error bars for wnadb logging.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            error_y=dict(type="data", array=error_y, visible=True),
            mode="markers",
            marker_symbol="square",
            marker_size=15,
            marker_line_color="darkslategrey",
            marker_line_width=2,
            marker_color="cadetblue",
        )
    )

    fig.update_layout(
        title={
            "text": "<b>Sparsity (Mean ± Std) per ViT Module</b>",
            "font": {"size": 32},
            "y": 1.0,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis={
            "title": {"text": "<b>ViT Modules</b>"},
            "title_font": {"family": "Courier New, monospace", "size": 20},
            "tickfont": {"size": 12},
        },
        yaxis={
            "title": {"text": "<b>Sparsity Value [0, 1]</b>"},
            "title_font": {"family": "Courier New, monospace", "size": 20},
            "tickfont": {"size": 12},
        },
    )

    return fig


def log_activation_sparsity_to_wandb(act_sparsity: Dict[str, List]):
    """
    Log activation sparsity to wandb
    """
    # flatten & aggregate data
    data = [
        (mod, float(np.mean(sps)), float(np.std(sps)))
        for mod, sps in act_sparsity.items()
    ]

    mods, means, stds = list(zip(*data))

    return create_plotly_graph_with_error_bars(mods, means, stds)


def train_epoch(
    data_loader, model, optimizer, scheduler, epoch, device, loss_compute, training_kw
):
    """
    Trains model for one epoch and logs training stats.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        A `pytorch` dataloader.
    model : torch.nn.Module,
        A pytorch model.
    optimizer : torch.optim
        A `pytorch` optimizer.
    scheduler : torch.optim.lr_scheduler
        A `pytorch` learning rate scheduler.
    epoch : int
        Current training epoch.
    device : torch.device
        Device to use, i.e. cuda or cpu.
    loss_compute : torch.nn.loss
        A torch loss instance.
    training_kw : Dict
        Model training keyword arguments.

    Returns
    -------
    train_metrics : Dict
        Model training metrics. Logs training metrics to `wandb` logger.
    """
    # switch into training mode and initialize statistics
    model.train()

    epochs = training_kw["epochs"]
    gradient_clip = training_kw.get("gradient_clip")
    label_smoothing = training_kw.get("label_smoothing", 0.0)
    num_classes = training_kw.get("num_classes")

    train_metrics = {"totals": 0, "corrects": 0, "acc": 0.0, "loss": 0.0}

    scaler = GradScaler()

    for idx, (image, label) in enumerate(data_loader, 1):

        optimizer.zero_grad()

        # Apply label smoothing
        label = label * (1 - label_smoothing) + label_smoothing / num_classes

        with torch.autocast(device.type, dtype=torch.bfloat16):
            # mount data to device
            image = image.to(device)
            label = label.to(device)

            # make predictions, compute loss, compute gradients, update weights
            logits = model(image)
            loss = loss_compute(logits, label)

            # loss.backward()
            # optimizer.step()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        scaler.step(optimizer)
        scaler.update()

        # extract probabilities and measure metrics
        preds, trues = log_multiclass__classification(logits, label)

        train_metrics = log_update_metrics(train_metrics, preds, trues, loss)

        print(
            "[TRAIN] [EPOCH:{}/{} ] [IDX: {}/{}] Loss: {:.4f} Accuracy: {:.4f}\t".format(
                epoch, epochs, idx, "UNK", loss.item(), train_metrics["acc"]
            )
        )

    train_metrics["loss"] = train_metrics["loss"] / idx
    train_metrics.pop("totals")

    try:
        _log = {"train_" + k: v for k, v in train_metrics.items()}

        wandb.log(_log, step=epoch)

    except Exception as e:
        print("Logging with `wandb` failed: {}".format(e))

    scheduler.step()

    return train_metrics


def train_epoch_qat(
    data_loader, model, optimizer, scheduler, epoch, device, loss_compute, training_kw
):
    """
    QAT-specific training epoch: float32 only, no GradScaler, handles QuantTensor output.
    Brevitas fake-quantization and STE gradients require full float32 precision.
    """
    model.train()

    epochs = training_kw["epochs"]
    gradient_clip = training_kw.get("gradient_clip")

    train_metrics = {"totals": 0, "corrects": 0, "acc": 0.0, "loss": 0.0}

    for idx, (image, label) in enumerate(data_loader, 1):

        optimizer.zero_grad()

        image = image.to(device)
        label = label.to(device)

        logits = model(image)
        if hasattr(logits, "value"):
            logits = logits.value

        loss = loss_compute(logits, label)
        loss.backward()

        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        preds, trues = log_multiclass__classification(logits, label)
        train_metrics = log_update_metrics(train_metrics, preds, trues, loss)

        print(
            "[QAT-TRAIN] [EPOCH:{}/{} ] [IDX: {}/{}] Loss: {:.4f} Accuracy: {:.4f}\t".format(
                epoch, epochs, idx, len(data_loader), loss.item(), train_metrics["acc"]
            )
        )

    train_metrics["loss"] = train_metrics["loss"] / idx
    train_metrics.pop("totals")

    try:
        _log = {"qat_train_" + k: v for k, v in train_metrics.items()}
        wandb.log(_log, step=epoch)
    except Exception as e:
        print("Logging with `wandb` failed: {}".format(e))

    scheduler.step()

    return train_metrics


def train_distill_epoch_qat(
    data_loader,
    s_model,
    t_model,
    optimizer,
    scheduler,
    epoch,
    device,
    loss_compute,
    training_kw,
):
    """
    QAT distillation epoch: float32 only, no GradScaler/autocast, handles QuantTensor
    output. Brevitas fake-quantization and STE gradients require full float32
    precision on the student; the (unquantized) teacher runs in plain float32 too,
    so no autocast is needed on either side.
    """
    s_model.train()
    t_model.eval()

    epochs = training_kw["epochs"]
    gradient_clip = training_kw.get("gradient_clip")
    dstl_type = training_kw.get("dstl_type", "soft")
    model_mode = training_kw.get("model_mode", "vit")
    T = training_kw.get("T", 3.0)
    alpha = training_kw.get("alpha", 0.5)

    train_metrics = {"totals": 0, "corrects": 0, "acc": 0.0, "loss": 0.0}

    for idx, (image, label) in enumerate(data_loader, 1):

        optimizer.zero_grad()

        image = image.to(device)
        label = label.to(device)

        with torch.no_grad():
            t_logits = t_model(image)
            if hasattr(t_logits, "value"):
                t_logits = t_logits.value

        logits = s_model(image)
        if hasattr(logits, "value"):
            logits = logits.value

        if dstl_type == "hard" and model_mode == "deit":
            cls_logits, dstl_logits = logits
            t_preds = torch.argmax(t_logits, dim=-1)

            loss = (
                loss_compute(cls_logits, label) + loss_compute(dstl_logits, t_preds)
            ) * 0.5

        elif dstl_type == "soft" and model_mode == "deit":
            cls_logits, dstl_logits = logits

            loss = soft_distillation_loss(
                s_logits=cls_logits,
                t_logits=t_logits,
                labels=label,
                dstl_logits=dstl_logits,
                T=T,
                alpha=alpha,
            )

        elif dstl_type == "soft" and model_mode == "vit":
            loss = soft_distillation_loss(
                s_logits=logits, t_logits=t_logits, labels=label, T=T, alpha=alpha
            )

        else:
            raise ValueError("Given `dstl_type` and `encoder.mode` are not valid.")

        loss.backward()

        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(s_model.parameters(), gradient_clip)

        optimizer.step()

        if model_mode == "deit":
            preds, trues = log_multiclass__classification(
                (cls_logits + dstl_logits) / 2, label
            )
        else:
            preds, trues = log_multiclass__classification(logits, label)

        train_metrics = log_update_metrics(train_metrics, preds, trues, loss)

        print(
            "[QAT-DISTILL-TRAIN] [EPOCH:{}/{} ] [IDX: {}/{}] Loss: {:.4f} Accuracy: {:.4f}\t".format(
                epoch, epochs, idx, len(data_loader), loss.item(), train_metrics["acc"]
            )
        )

    train_metrics["loss"] = train_metrics["loss"] / idx
    train_metrics.pop("totals")

    try:
        _log = {"qat_distill_train_" + k: v for k, v in train_metrics.items()}
        wandb.log(_log, step=epoch)
    except Exception as e:
        print("Logging with `wandb` failed: {}".format(e))

    scheduler.step()

    return train_metrics


def soft_distillation_loss(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    labels: torch.Tensor,
    dstl_logits: None | torch.Tensor = None,
    T: float = 3.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Combine soft and hard targets using KL divergence and cross-entropy.

    Parameters
    ----------
    s_logits : torch.Tensor
        Student logits.
    t_logits : torch.Tensor
        Teacher logits.
    labels : torch.Tensor
        The target labels.
    dstl_logits : torch.Tensor | None
        Distillation logits in case they are different than the `s_logits`
        as in the case of Deit models.
    T : float
        Temperature parameter to be applied on softmax.
    alpha : float
        A one-parameter family for weighing hard and soft loss.
    """
    if dstl_logits is None:
        dstl_logits = s_logits

    soft_loss = F.kl_div(
        F.log_softmax(dstl_logits / T, dim=1),
        F.log_softmax(t_logits / T, dim=1),
        reduction="batchmean",
        log_target=True,
    ) * (T * T)

    # hard label loss
    hard_loss = F.cross_entropy(s_logits, labels)

    return alpha * hard_loss + (1 - alpha) * soft_loss


def train_distill_epoch(
    data_loader,
    s_model,
    t_model,
    optimizer,
    scheduler,
    epoch,
    device,
    loss_compute,
    training_kw,
):
    """
    Trains model for one epoch and logs training stats.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        A `pytorch` dataloader.
    s_model : torch.nn.Module,
        A pytorch model to act as student during distillation training.
    t_model : torch.nn.Module,
        A pytorch model to act as teacher during distillation training.
    optimizer : torch.optim
        A `pytorch` optimizer.
    scheduler : torch.optim.lr_scheduler
        A `pytorch` learning rate scheduler.
    epoch : int
        Current training epoch.
    device : torch.device
        Device to use, i.e. cuda or cpu.
    loss_compute : torch.nn.loss
        A torch loss instance.
    training_kw : Dict
        Model training keyword arguments.

    Returns
    -------
    train_metrics : Dict
        Model training metrics. Logs training metrics to `wandb` logger.
    """
    # Put student into training mode, teacher into eval mode
    s_model.train()
    t_model.eval()

    epochs = training_kw["epochs"]
    gradient_clip = training_kw.get("gradient_clip")
    label_smoothing = training_kw.get("label_smoothing", 0.0)
    num_classes = training_kw.get("num_classes")
    dstl_type = training_kw.get("dstl_type", "soft")
    model_mode = training_kw.get("model_mode", "vit")

    # Initialize statistics
    train_metrics = {"totals": 0, "corrects": 0, "acc": 0.0, "loss": 0.0}

    scaler = GradScaler()

    for idx, (image, label) in enumerate(data_loader, 1):

        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Apply label smoothing
        label = label * (1 - label_smoothing) + label_smoothing / num_classes

        # Get teacher labels (no grads)
        with torch.no_grad():

            with torch.autocast(device.type, dtype=torch.bfloat16):
                t_logits = t_model(image)
                t_preds = torch.argmax(t_logits, dim=-1)

        with torch.autocast(device.type, dtype=torch.bfloat16):

            # make predictions, compute loss, compute gradients, update weights
            logits = s_model(image)

            if dstl_type == "hard" and model_mode == "deit":
                cls_logits, dstl_logits = logits

                loss = (
                    loss_compute(cls_logits, label) + loss_compute(dstl_logits, t_preds)
                ) * 0.5

            elif dstl_type == "soft" and model_mode == "deit":
                cls_logits, dstl_logits = logits

                loss = soft_distillation_loss(
                    s_logits=cls_logits,
                    t_logits=t_logits,
                    labels=label,
                    dstl_logits=dstl_logits,
                )

            elif dstl_type == "soft" and model_mode == "vit":
                loss = soft_distillation_loss(
                    s_logits=logits, t_logits=t_logits, labels=label
                )

            else:
                raise ValueError("Given `dstl_type` and `encoder.mode` are not valid.")

            # loss.backward()
            # optimizer.step()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(s_model.parameters(), gradient_clip)

        scaler.step(optimizer)
        scaler.update()

        # extract probabilities and measure metrics
        if model_mode == "deit":
            preds, trues = log_multiclass__classification(
                (cls_logits + dstl_logits) / 2, label
            )
        elif model_mode == "vit":
            preds, trues = log_multiclass__classification(
                logits.detach(), label.detach()
            )

        train_metrics = log_update_metrics(train_metrics, preds, trues, loss.detach())

        print(
            "[TRAIN] [EPOCH:{}/{} ] [IDX: {}/{}] Loss: {:.4f} Accuracy: {:.4f}\t".format(
                epoch, epochs, idx, "UNK", loss.item(), train_metrics["acc"]
            )
        )

    train_metrics["loss"] = train_metrics["loss"] / idx
    train_metrics.pop("totals")

    try:
        _log = {"train_" + k: v for k, v in train_metrics.items()}

        wandb.log(_log, step=epoch)

    except Exception as e:
        print("Logging with `wandb` failed: {}".format(e))

    scheduler.step()

    return train_metrics


def validate_epoch(data_loader, model, epoch, device, loss_compute, valid_kw):
    """
    Validates the model after each epoch and logs the metrics.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        A `pytorch` dataloader.
    model : torch.nn.Models
        A `pytorch` model.
    epoch : int
        Current epoch.
    device : torch.device
        Device to validate on i.e. 'cpu' or 'cuda'.
    loss_compute : torch.nn.loss
        A torch loss instance.
    valid_kw : Dict
        Model validation keyword arguments.

    Returns
    -------
    val_metrics : Dict
        Model validation metrics. Logs validation metrics to `wandb` logger.
    """
    # switch into eval mode
    model.eval()

    val_metrics = {"totals": 0, "corrects": 0, "acc": 0.0, "loss": 0.0}

    epochs = valid_kw["epochs"]

    for idx, (image, label) in enumerate(data_loader, 1):

        # zero gradients
        with torch.no_grad():

            with torch.autocast(device.type, dtype=torch.bfloat16):
                # mount data to device
                image = image.to(device)
                label = label.to(device)

                # make predictions, compute loss, compute gradients
                logits = model(image)
                loss = loss_compute(logits, label)

            preds, trues = log_multiclass__classification(logits, label)

            val_metrics = log_update_metrics(val_metrics, preds, trues, loss)

            print(
                "[VALID] [EPOCH:{}/{} ] [IDX: {}/{}] Loss: {:.4f} Accuracy: {:.4f}\t".format(
                    epoch, epochs, idx, "UNK", loss.item(), val_metrics["acc"]
                )
            )

    val_metrics["loss"] = val_metrics["loss"] / idx
    val_metrics.pop("totals")

    try:
        _log = {"val_" + k: v for k, v in val_metrics.items()}

        wandb.log(_log, step=epoch)

    except Exception as e:
        print("Logging with `wandb` failed: {}".format(e))

    return val_metrics


def infer_epoch(data_loader, model, device, infer_kw):
    """
    Performs inference on a dataset provided by the dataloader.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        A `pytorch` dataloader.
    model : torch.nn.Models
        A `pytorch` model.
    device : torch.device
        Device to validate on i.e. 'cpu' or 'cuda'.
    infer_kw : Dict
        Any additional model inference keyword arguments.

    Returns
    -------
    all_preds: List
        Model predictions for input samples.
    """
    # switch into eval mode
    model.eval()

    all_preds = []

    for idx, (image, _) in enumerate(data_loader):

        # zero gradients
        with torch.no_grad():

            with torch.autocast(device.type, dtype=torch.bfloat16):
                # mount data to device
                image = image.to(device)
                logits = model(image)

            preds, _ = log_multiclass__classification(logits, label=None)

        all_preds.extend(preds.tolist())

    return all_preds


def log_model_stats_epoch(data_loader, model, device, log_kw, epoch=None):
    """
    Estimates and logs activation statistics.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        A `pytorch` dataloader.
    model : torch.nn.Models
        A `pytorch` model.
    epoch : int
        Current epoch.
    device : torch.device
        Device to validate on i.e. 'cpu' or 'cuda'.
    log_kw : Dict
        Log keyword arguments.

    Returns
    -------
    log_metrics : Dict
        Model validation metrics. Logs validation metrics to `wandb` logger.
    """
    if log_kw.get("log_sparsity", False) is False:
        return

    model_type = log_kw.get("model_type", "dense")

    # switch into eval mode
    model.eval()

    log_name = log_kw.get("name", "")
    log_sparsity__modules = log_activation_sparsity__gather_modules(model)

    act_sparsity = {module: [] for module in log_sparsity__modules}

    if model_type == "dense":
        tmp_sparsity = log_activation_sparsity(model, log_sparsity__modules)

    elif model_type == "quant":
        tmp_sparsity = log_activation_sparsity_quant(model, log_sparsity__modules)

    log_metrics = {}

    for idx, (image, label) in enumerate(data_loader, 1):

        # zero gradients
        with torch.no_grad():

            with torch.autocast(device.type, dtype=torch.bfloat16):
                # mount data to device
                image = image.to(device)
                label = label.to(device)

                # make predictions, compute loss, compute gradients
                logits = model(image)

            preds, trues = log_multiclass__classification(logits, label)

        for module, vals in tmp_sparsity.items():
            act_sparsity[module].extend(vals)

    _log_sp = log_name + "_" + "act_sparsity" if log_name else "act_sparsity"

    try:
        _log = {}
        _log[_log_sp] = log_activation_sparsity_to_wandb(act_sparsity)

        wandb.log(_log, step=epoch)

    except Exception as e:
        print("Logging with `wandb` failed: {}".format(e))

    act_sparsity_agg = {
        module: (float(np.mean(sps)), float(np.std(sps)))
        for module, sps in act_sparsity.items()
    }

    log_metrics[_log_sp] = act_sparsity_agg

    return log_metrics
