from typing import List, Dict, Tuple, Iterator

import torch
import numpy as np

from torch import nn
from torch.nn.utils import prune


def get_target_modules(
    model: nn.Module, modules: List[str] = ["linear", "conv2d"]
) -> Iterator[nn.Module]:
    """
    Gather target modules.
    """
    MODULES = {
        "linear": nn.Linear,
        "conv2d": nn.Conv2d,
    }

    _modules = tuple([MODULES[module] for module in modules if module in MODULES])

    for module in model.modules():
        if isinstance(module, _modules):
            yield module


def find_layers(
    module: nn.Module, layers: List[str] = ["linear", "conv2d"], name: str = ""
) -> Dict:
    """
    Gather target layers.
    """
    LAYERS = {
        "linear": nn.Linear,
        "conv2d": nn.Conv2d,
    }

    _layers = tuple([LAYERS[layer] for layer in layers if layer in LAYERS])

    if type(module) in _layers:
        return {name: module}

    res = {}

    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )

    return res


def evaluate_model_sparsity(model: nn.Module) -> Dict:
    """
    Evaluate model pruning.
    """
    metrics = {}

    layers = find_layers(module=model)

    for name, layer in layers.items():

        weight = layer.weight.detach()
        zeros = torch.sum(weight == 0).item()
        total = weight.numel()

        metrics[name] = {
            "zeros": zeros,
            "total": total,
            "sparsity": zeros / total if total else 0.0,
        }

    total_zeros = sum(item["zeros"] for _, item in metrics.items())
    total_weights = sum(item["total"] for _, item in metrics.items())
    sparsity = total_zeros / total_weights if total_weights else 0.0

    return {
        "per_layer": metrics,
        "total": {
            "total_zeros": total_zeros,
            "total_weights": total_weights,
            "sparsity": sparsity,
        },
    }


def get_model_total_sparsity(model: nn.Module) -> float:
    """
    Get model total weight sparsity.
    """
    total_zeros = 0
    total_elems = 0

    for module in get_target_modules(model):

        w = module.weight.detach()

        total_zeros += torch.sum(w == 0).item()
        total_elems += w.numel()

    if total_elems == 0:
        return 0.0

    return total_zeros / total_elems


def parse_prune_amounts(amounts: List[float]) -> List[float]:
    """
    Parse prune amounts when given as a list of pruning amounts.
    """
    if any([amount < 0 or amount > 1 for amount in amounts]):
        raise ValueError(
            "Prune amounts must be a list of non-decreasing floats in [0, 1]"
        )

    for i in range(1, len(amounts)):
        if amounts[i] < amounts[i - 1]:
            raise ValueError(
                "For progressive pruning, prune amounts must be non-decreasing."
            )

    return [float(amount) for amount in amounts]


def validate_range_amounts(
    amounts: Tuple[float, float, int],
) -> Tuple[float, float, int]:
    """
    Validate prune amounts when given as 'range' or 'cubic' of the form `(start, end, num)`.
    """
    start, end, num = amounts

    start = float(start)
    end = float(end)
    num = int(num)

    if start < 0.0 or start > 1.0 or end < 0.0 or end > 1.0:
        raise ValueError("Prune schedule `start` and `end` must be in [0, 1].")
    if end < start:
        raise ValueError("Prune schedule `end` should be bigger than `start`")
    if num <= 0:
        raise ValueError("Prune schedule `num` must be an integer > 0.")

    return start, end, num


def parse_range_amounts(start: float, end: float, num: int) -> List[float]:
    """
    Return 'range' prune amounts with input `(start, end, num)`.
    """
    return np.linspace(start, end, num).tolist()


def parse_cubic_amounts(start: float, end: float, num: int) -> List[float]:
    """
    Return 'cubic' prune amounts with input `(start, end, num)` as defined
    in (Zhu & Gupta, 2017), `https://arxiv.org/abs/1710.01878`. Similar to
    our case used in (Srinivas et al., 2022), `https://arxiv.org/abs/2202.01290`

    For given start amount `s_i`, end amount `s_f`, step `j` and number of steps `n`:

                    s(j) = s_f + (s_i - s_f) * (1 - j/n)^3
    """
    cubic_prune = lambda s_i, s_f, i, n: s_f + (s_i - s_f) * (1 - i / n) ** 3

    return [cubic_prune(start, end, i, num) for i in range(num)]


def resolve_prune_amounts(schedule_type: str, schedule_amounts: List) -> List[float]:
    """
    Resolve prune schedule amounts.

    Parameters
    ----------
    schedule_type : str
        Schedule type must be between ('steps', 'range', 'cubic').
    schedule_amounts : List
        If 'steps' then a list of floats, if 'range' or 'cubic', then a three tuple
        of the form (start, end, num).

    Parameters
    ----------
    schedule_amounts : List[float]
        A list of non-decreasing floats in [0,1] pruning amounts
    """
    if schedule_type == "steps":
        schedule_amounts = parse_prune_amounts(schedule_amounts)

    elif schedule_type == "range":
        start, end, num = validate_range_amounts(schedule_amounts)

        schedule_amounts = parse_range_amounts(start, end, num)

    elif schedule_type == "cubic":
        start, end, num = validate_range_amounts(schedule_amounts)

        schedule_amounts = parse_cubic_amounts(start, end, num)

    return schedule_amounts


def parse_training_epochs(cfg: Dict) -> Tuple[List[int], List[int]]:
    """
    Parse warmup and cosine epochs and align them with pruning steps.
    """
    # resolve number of pruning steps
    if cfg["prune"]["schedule"]["schedule_type"] == "steps":
        steps = len(cfg["prune"]["schedule"]["schedule_amounts"])

    elif cfg["prune"]["schedule"]["schedule_type"] in ("range", "cubic"):
        steps = cfg["prune"]["schedule"]["schedule_amounts"][2]

    _cosine = cfg["train"]["lr_scheduler"]["epochs"]
    _warmup = cfg["train"]["lr_scheduler"].get("warmup")

    if isinstance(_cosine, int):
        cosine_epochs_per_step = [_cosine for i in range(steps)]

    if isinstance(_warmup, int):
        warump_epochs_per_step = [_warmup for i in range(steps)]

    if isinstance(_cosine, (list, tuple)):
        cosine_epochs_per_step = _cosine

    if isinstance(_cosine, (list, tuple)):
        warump_epochs_per_step = _warmup

    if len(cosine_epochs_per_step) != len(warump_epochs_per_step):
        raise ValueError("Cosine epochs and warmup epochs must have the same length")

    if len(cosine_epochs_per_step) != steps:
        raise ValueError("Number of cosine/warmup epochs must equal pruning steps.")

    return warump_epochs_per_step, cosine_epochs_per_step


def parse_learning_rates(cfg: Dict) -> List[float]:
    """
    Parse learning rate and align them with pruning steps.
    """
    # resolve number of pruning steps
    if cfg["prune"]["schedule"]["schedule_type"] == "steps":
        steps = len(cfg["prune"]["schedule"]["schedule_amounts"])

    elif cfg["prune"]["schedule"]["schedule_type"] == "steps":
        steps = cfg["prune"]["schedule"]["schedule_amounts"][2]

    _lr = cfg["train"]["optim"]["lr"]

    if isinstance(_lr, (int, float)):
        lr_per_step = [_lr for i in range(steps)]

    if isinstance(_lr, (list, tuple)):
        lr_per_step = _lr

    if len(lr_per_step) != steps:
        raise ValueError("Number of learning rates must equal pruning steps.")

    return lr_per_step


def _build_zero_masks(model: nn.Module) -> tuple[Dict, Dict]:
    """
    Build masks for currently-zero weights.
    """
    layers = find_layers(model)
    masks = {}

    for name, module in layers.items():

        mask = module.weight.detach() != 0
        masks[name] = mask

    return layers, masks


def _apply_zero_masks(model: nn.Module, layers: Dict, masks: Dict) -> nn.Module:
    """
    Apply masks to layers via `torch.nn.utils.prune` module.

    This step introduces two new parameters to the model 'weight_orig' and
    'weight_mask'. The product of these two parameters gives 'weight'.
    """
    for name, layer in layers.items():
        prune.custom_from_mask(layer, name="weight", mask=masks[name])

    return model


def _remove_prune(layers: Dict) -> nn.Module:
    """
    Removes introduced parameters and replaces them with 'weight' parameter
    whose values are the product of 'weight_orig' and 'weight_mask'.
    """
    for name, layer in layers.items():
        prune.remove(layer, name="weight")


def _enforce_zero_masks(layers: Dict, masks: Dict, where: str = "gradients") -> None:
    """
    Keep masked positions exactly zero after optimizer updates.
    """
    if where == "weights":
        with torch.no_grad():
            for name, layer in layers.items():
                layer.weight.mul_(masks[name])

    elif where == "gradients":
        with torch.no_grad():
            for name, layer in layers.items():
                layer.weight.grad.mul_(masks[name])

    else:
        raise ValueError("Choose between ('weights', 'gradients')")


def test_prune_amounts():
    """
    Simple unit-test for the prune amounts implementation.
    """
    steps_inp = [0.0, 0.2, 0.5, 0.8]
    range_inp = [0.0, 0.2, 5]
    cubic_inp = [0.0, 0.2, 5]

    steps_res = [0.0, 0.2, 0.5, 0.8]
    range_res = [0.0, 0.05, 0.1, 0.15, 0.2]
    cubic_res = [0.0, 0.0976, 0.1568, 0.1872, 0.1984]

    assert np.allclose(resolve_prune_amounts("steps", steps_inp), steps_res)
    assert np.allclose(resolve_prune_amounts("range", range_inp), range_res)
    assert np.allclose(resolve_prune_amounts("cubic", cubic_inp), cubic_res)


def test_prune():
    """
    Simple test for pruning and testing a whole network.
    """
    from sparse_vit.pruning.utils import find_layers
    from sparse_vit.pruning.dispatcher import apply_pruning

    model = apply_pruning(model, amount=0.9, prune_method="magnitude")

    layers = find_layers(model)

    def get_tensor_sparsity(x):
        return (x == 0).sum() / x.numel()


def estimate_sparsity_under_random_sparsity_model(s_a, s_b, k):
    """
    Parameters
    ----------
    s_a : float
        Input sparsity of matrix A
    s_b : float
        Input sparsity of matrix B
    k : int
        Their common dimension.
    """
    p_a = 1 - s_a
    p_b = 1 - s_b

    return (1 - p_a * p_b) ** k
