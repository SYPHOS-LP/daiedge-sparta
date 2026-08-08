from typing import Dict, List

import torch
import torch.optim as optim

from torch import nn


def model_save(model, optimizer, scheduler, epoch, save_path):
    """
    Saves model, optimizer and scheduler history at some specific epoch.

    Parameters
    ----------
    model : torch.nn.Module.
        An instantiation of a pytorch model.
    optimizer : torch.optim
        A `pytorch` optimizer.
    scheduler : torch.optim.lr_scheduler
        A `pytorch` learning rate scheduler.
    epoch : int
        Current epoch in training loop.
    save_path : str
        Save path to store model.

    Returns
    -------
    None
        Saves model state along with optimizer, scheduler history etc.
    """
    save_dict = {
        "state": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
    }

    torch.save(save_dict, save_path)


def model_load(
    model, model_path: str, device: torch.device = "cpu", replace_head: bool = False
) -> nn.Module:
    """
    Load model from state_dict.
    """
    state_dict = torch.load(model_path, map_location=device)["state"]

    if replace_head is True:

        head_layers = [k for k in state_dict.keys() if k.startswith("heads.")]

        for l in head_layers:
            state_dict.pop(l)

    try:
        missing_keys, unexpected_fields = model.load_state_dict(
            state_dict, strict=False
        )

        print(missing_keys, unexpected_fields)
    except Exception as e:
        print("An error has occured when loading the model: {}".format(e))

    return model


def model_load_submodule(
    model, model_path: str, submodule: str, device: torch.device = "cpu"
) -> nn.Module:
    """
    Load model from state_dict.
    """
    state_dict = torch.load(model_path, map_location=device, weights_only=True)["state"]

    # Select submodule layers only
    state_dict = {
        k.removeprefix(submodule + "."): w
        for k, w in state_dict.items()
        if k.startswith(submodule)
    }

    try:
        missing_keys, unexpected_fields = model.load_state_dict(
            state_dict, strict=False
        )

        print(missing_keys, unexpected_fields)
    except Exception as e:
        print("An error has occured when loading the model: {}".format(e))

    return model


def validate_device(device):
    """
    Validate device argument.
    """
    if device.startswith("cuda"):
        return torch.device(device if torch.cuda.is_available() else "cpu")

    elif device == "xpu" and hasattr(torch, "xpu"):
        return torch.device("xpu" if torch.xpu.is_available() else "cpu")

    else:
        return torch.device("cpu")


def freeze_weights_iter(model: nn.Module, layers: str = "heads") -> None:
    """
    Freeze layers weights before training.

    Parameters
    ----------
    model : nn.Module
        Model to freeze its layers.
    layers : str
        Layer to be frozen for training.
    """
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze target layers
    for param in getattr(model, layers).parameters():
        param.requires_grad = True


def freeze_weights(model: nn.Module) -> None:
    """
    Freeze layers weights except `heads` for finetuning.

    Parameters
    ----------
    model : nn.Module
        Model to freeze its layers.
    """
    model.embed.requires_grad_(False)
    model.encoder.requires_grad_(False)


def _freeze_mlp_weights(vit_model: nn.Module, requires_grad: bool = False) -> None:
    """
    Freeze mlp layers weights for finetuning.

    Parameters
    ----------
    model : nn.Module
        Model to freeze its layers.
    """
    names = [
        "encoder.encoder_layer_{}.mlp.linear_1",
        "encoder.encoder_layer_{}.mlp.linear_2",
    ]

    N = len(vit_model._modules["encoder"])

    for idx in range(N):

        for name in names:
            module = vit_model.get_submodule(name.format(idx))
            module.requires_grad_(requires_grad)


def init_lr_scheduler_cosine_w_warmup(
    optimizer, cosine_epochs: int, warmup_epochs: int = 0
) -> optim.lr_scheduler.LRScheduler:
    """
    Initialize a cosine annealing scheduler with linear warmups.
    """
    if warmup_epochs <= 0:
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_epochs, eta_min=1e-6
        )

    warmup_lr = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.05, end_factor=1.0, total_iters=warmup_epochs
    )

    cosine_lr = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=1e-6
    )

    return optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_lr, cosine_lr], milestones=[warmup_epochs]
    )


def log_activation_sparsity(model: nn.Module, modules: List[str]) -> Dict[str, List]:
    """
    Measure and log activation sparsity over input data.
    """
    activation_sparsity = {}

    def activation_sparsity_hook_factory(name):
        """
        Factory function create hooks for measuring activation sparsity.
        """

        def hook_fn(model, input_args, output_tensor: torch.Tensor):
            # flatten all dimensions except the batch dimension
            t = output_tensor.detach().flatten(1)

            # measure sparsity
            activation_sparsity[name] = (t.count_nonzero(1) / t.size(1)).tolist()

        return hook_fn

    for name in modules:

        try:
            layer = model.get_submodule(name)
        except AttributeError:
            print("Layer `{}` was not found.".format(name))
            continue

        layer.register_forward_hook(activation_sparsity_hook_factory(name))

    return activation_sparsity


def log_activation_sparsity_quant(
    model: nn.Module, modules: List[str]
) -> Dict[str, List]:
    """
    Measure and log activation sparsity over input data.
    """
    from brevitas.quant_tensor import IntQuantTensor

    activation_sparsity = {}

    def activation_sparsity_hook_factory(name):
        """
        Factory function create hooks for measuring activation sparsity.
        """

        def hook_fn(model, input_args, output_tensor: IntQuantTensor):
            # flatten all dimensions except the batch dimension
            if isinstance(output_tensor, IntQuantTensor):
                t = output_tensor.int().detach().flatten(1)
            else:
                t = output_tensor.detach().flatten(1)

            # measure sparsity
            activation_sparsity[name] = (t.count_nonzero(1) / t.size(1)).tolist()

        return hook_fn

    for name in modules:

        try:
            layer = model.get_submodule(name)
        except AttributeError:
            print("Layer `{}` was not found.".format(name))
            continue

        layer.register_forward_hook(activation_sparsity_hook_factory(name))

    return activation_sparsity


def log_activation_sparsity__gather_modules(model: nn.Module):
    """
    Gather modules to log activation sparsity for.
    """
    modules = []

    # embedding modules (non changing)
    embed_modules = ["embed.conv_proj", "embed"]

    # encoder modules
    encoder_modules = []
    for idx in range(len(model._modules["encoder"])):

        names = [
            "encoder.encoder_layer_{}.ln_mha",
            "encoder.encoder_layer_{}.mha.activation_q",
            "encoder.encoder_layer_{}.mha.activation_k",
            "encoder.encoder_layer_{}.mha.w_v",
            "encoder.encoder_layer_{}.mha.out_proj",
            "encoder.encoder_layer_{}.ln_mlp",
            "encoder.encoder_layer_{}.mlp.linear_1",
            "encoder.encoder_layer_{}.mlp.activation",
            "encoder.encoder_layer_{}.mlp.linear_2",
        ]

        encoder_modules.extend([name.format(idx) for name in names])

    # output modules ()
    heads_modules = ["heads.head"]

    modules.extend(embed_modules)
    modules.extend(encoder_modules)
    modules.extend(heads_modules)

    return modules


def log_activation_sparsity_quant__gather_modules(model: nn.Module):
    """
    Gather modules to log activation sparsity for.
    """
    modules = []

    # embedding modules (non changing)
    embed_modules = ["embed.conv_proj", "embed"]

    # encoder modules
    encoder_modules = []
    for idx in range(len(model._modules["encoder"])):

        names = [
            "encoder.encoder_layer_{}.ln_mha",
            "encoder.encoder_layer_{}.mha.activation_q",
            "encoder.encoder_layer_{}.mha.activation_k",
            "encoder.encoder_layer_{}.mha.w_v",
            "encoder.encoder_layer_{}.mha.out_proj",
            "encoder.encoder_layer_{}.ln_mlp",
            "encoder.encoder_layer_{}.mlp.linear_1",
            "encoder.encoder_layer_{}.mlp.activation",
            "encoder.encoder_layer_{}.mlp.linear_2",
        ]

        encoder_modules.extend([name.format(idx) for name in names])

    # output modules ()
    heads_modules = ["heads.head"]

    modules.extend(embed_modules)
    modules.extend(encoder_modules)
    modules.extend(heads_modules)

    return modules


# ['encoder.encoder_layer_0.quant_input.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value',
#  'encoder.encoder_layer_0.res1.m_a',
#  'encoder.encoder_layer_0.res1.e_a',
#  'encoder.encoder_layer_0.res1.m_b',
#  'encoder.encoder_layer_0.res1.e_b',
#  'encoder.encoder_layer_0.res1.scale_out',
#  'encoder.encoder_layer_0.res1._ready',
#  'encoder.encoder_layer_0.res1.align_a.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value',
#  'encoder.encoder_layer_0.res1.align_b.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value',
#  'encoder.encoder_layer_0.res1.out_quant.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value',
#  'encoder.encoder_layer_0.res2.m_a',
#  'encoder.encoder_layer_0.res2.e_a',
#  'encoder.encoder_layer_0.res2.m_b',
#  'encoder.encoder_layer_0.res2.e_b',
#  'encoder.encoder_layer_0.res2.scale_out',
#  'encoder.encoder_layer_0.res2._ready',
#  'encoder.encoder_layer_0.res2.align_a.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value',
#  'encoder.encoder_layer_0.res2.align_b.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value',
#  'encoder.encoder_layer_0.res2.out_quant.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value',
#  'encoder.encoder_layer_0.quant_pre_mlp.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value',
#  'encoder.encoder_layer_0.ln_mha.rms.weight',
#  'encoder.encoder_layer_0.ln_mha.output_quant_module.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value',
#  'encoder.encoder_layer_0.ln_mha.weight_quant_module.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value',
#  'encoder.encoder_layer_0.ln_mlp.rms.weight',
#  'encoder.encoder_layer_0.ln_mlp.output_quant_module.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value',
#  'encoder.encoder_layer_0.ln_mlp.weight_quant_module.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value',
#  'encoder.encoder_layer_0.mha.quant_q.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value',
#  'encoder.encoder_layer_0.mha.quant_k.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value',
#  'encoder.encoder_layer_0.mha.quant_v.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value',
#  'encoder.encoder_layer_0.mha.quant_out.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value',
#  'encoder.encoder_layer_0.mha.w_q.weight_mask',
#  'encoder.encoder_layer_0.mha.w_q.weight_orig',
#  'encoder.encoder_layer_0.mha.w_q.input_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value',
#  'encoder.encoder_layer_0.mha.w_k.weight_mask',
#  'encoder.encoder_layer_0.mha.w_k.weight_orig',
#  'encoder.encoder_layer_0.mha.w_k.input_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value',
#  'encoder.encoder_layer_0.mha.w_v.weight_mask',
#  'encoder.encoder_layer_0.mha.w_v.weight_orig',
#  'encoder.encoder_layer_0.mha.w_v.input_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value',
#  'encoder.encoder_layer_0.mha.out_proj.weight_mask',
#  'encoder.encoder_layer_0.mha.out_proj.weight_orig',
#  'encoder.encoder_layer_0.mha.out_proj.input_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value',
#  'encoder.encoder_layer_0.mlp.linear_1.weight_mask',
#  'encoder.encoder_layer_0.mlp.linear_1.weight_orig',
#  'encoder.encoder_layer_0.mlp.linear_1.input_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value',
#  'encoder.encoder_layer_0.mlp.linear_2.weight_mask',
#  'encoder.encoder_layer_0.mlp.linear_2.weight_orig',
#  'encoder.encoder_layer_0.mlp.linear_2.input_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value'
#  ]
