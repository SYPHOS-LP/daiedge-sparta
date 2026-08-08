import torch
import torch.nn as nn

from .utils import get_model_total_sparsity, get_target_modules, find_layers
from .constants import assign_pruning_ratio_levels_across_modules


def prune_tensor_unstructured_w_magnitude_inplace(
    weight: torch.Tensor, amount: float
) -> None:
    """
    Prune weight tensor by a specified amount based on magnitude,

    Parameters
    ----------
    weight : torch.Tensor
        A weight tensor to reach a specific amount of zero entries.
    amount : float
        The percentage of entries to be zero. If float, must be a number between 0 and 1.

    Returns
    -------
    None
        Applies unstructured pruning inplace.
    """
    if amount < 0.0 or amount > 1:
        raise ValueError("Given amount must be a float number between 0 and 1")

    w = weight.detach()

    # current no. of nonzero elements
    cur_mask = w != 0
    cur_count = int(cur_mask.sum().item())

    # target number of nonzero entries
    trg_count = int(w.numel() * (1 - amount))

    # Get the number of elements to remove
    num_prune = cur_count - trg_count

    # if current nonzero entries are more than target nonzero entries, return
    if num_prune <= 0:
        return

    # Get non-zero entries index
    cur_idx = torch.nonzero(cur_mask.view(-1), as_tuple=False).flatten()
    cur_abs = w.abs().view(-1)[cur_idx]

    # Make sure that k will not be larger than number of elements in `cur_abs`
    k = min(num_prune, cur_abs.numel())

    prune_sel = torch.topk(cur_abs, k=k, largest=False).indices
    prune_idx = cur_idx[prune_sel]

    with torch.no_grad():
        weight.view(-1)[prune_idx] = 0


def prune_model_layerwise_unstructured_w_magnitude(
    model: nn.Module, amount: float, allocation: str = "uniform"
) -> nn.Module:
    """
    Apply layerwise unstructured pruning.

    Parameters
    ----------
    model : nn.Module
        The model to perform unstructured prunnig with sparse-gpt on.
    amount : float
        The amount of target sparsity.
    allocation : str
        How to apply pruning ratios are assigned per layer. If 'uniform' use
        same pruning ratio for every layer, if 'finegrained' use different.

    Returns
    -------
    model : nn.Module
        The model with pruned weights.
    """
    subset = find_layers(model, layers=["linear", "conv2d"])

    for name, module in subset.items():

        if allocation == "finegrained":
            _amount = assign_pruning_ratio_levels_across_modules(name, amount)

        elif allocation == "uniform":
            _amount = amount

        prune_tensor_unstructured_w_magnitude_inplace(module.weight, _amount)

    return model


def prune_model_global_unstructured_w_magnitude(
    model: nn.Module, amount: float
) -> nn.Module:
    """
    Prune weight tensor by a specified amount based on magnitude considering all
    model weights simultaneously.

    Parameters
    ----------
    model : nn.Module
        A model to apply unstructured pruning based on magnitude in a global manner.
    amount : float
        The percentage of entries to be zero. If float, must be a number between 0 and 1.

    Returns
    -------
    model : nn.Module
        The model pruned globally.
    """
    if amount < 0.0 or amount > 1:
        raise ValueError("Given amount must be a float number between 0 and 1")

    # Get all weights flattened, get nnz index sets
    weights = []
    nnz_indices = []
    total_nnz_indices = 0
    total_num_weights = 0

    for module in get_target_modules(model):

        # for each module, get weights, detach and flatten
        w = module.weight.detach().view(-1)

        nnz_idx = torch.nonzero(w != 0, as_tuple=False).flatten()

        n_idx = nnz_idx.numel()
        n_all = w.numel()

        if n_idx == 0:
            continue

        weights.append(w)
        nnz_indices.append(nnz_idx)

        total_nnz_indices += n_idx
        total_num_weights += n_all

    # If no index is corresponds to non-zero value
    if total_nnz_indices == 0:
        return model

    # target number of nonzero entries
    trg_count = int(total_num_weights * (1 - amount))

    # Get the number of elements to remove
    num_prune = total_nnz_indices - trg_count

    # if current nonzero entries are more than target nonzero entries, return
    if num_prune <= 0:
        return model

    _abs_nnz_weights = []

    for w, nnz_idx in zip(weights, nnz_indices):
        _abs_nnz_weights.append(w.abs()[nnz_idx])

    abs_nnz_weights = torch.cat(_abs_nnz_weights, dim=0)

    # Make sure that k will not be larger than number of elements in `cur_abs`
    k = min(num_prune, abs_nnz_weights.numel())

    # Select topk weights from global nnz weight vector.
    # The indices correspond to the nnz weights, not all weights
    prune_idx_set = torch.topk(abs_nnz_weights, k=k, largest=False).indices

    # Estimate offsets of flattened weights
    offsets = []
    pointer = 0
    for nnz_idx in nnz_indices:
        nxt = pointer + int(nnz_idx.numel())
        offsets.append((pointer, nxt))
        pointer = nxt

    with torch.no_grad():
        for w, nnz_idx, (start, end) in zip(weights, nnz_indices, offsets):

            # create mask for pruning index set for given weights
            mask = (prune_idx_set >= start) & (prune_idx_set < end)

            # if mask is all zeros continue
            if not bool(mask.any()):
                continue

            # Select pruning index set for given weights, apply offset
            local_sel = prune_idx_set[mask] - start

            # Map pruning index set from nnz weights to all weights
            prune_idx = nnz_idx[local_sel]

            # Apply pruning
            w[prune_idx] = 0

    return model


def prune_unstructured_w_magnitude(
    model: nn.Module,
    amount: float,
    scope: str = "layerwise",
    allocation: str = "uniform",
) -> nn.Module:
    """
    Perform unstructured pruning to an input model.

    Parameters
    ----------
    model : nn.Module
        The model to perform unstructured prunnig with sparse-gpt on.
    amount : float
        The amount of target sparsity.
    scope : str
        Choose between `global` and `layerwise` magnitude pruning.
    allocation : str
        How to apply pruning ratios are assigned per layer. If 'uniform' use
        same pruning ratio for every layer, if "finegrained" use different.

    Returns
    -------
    model : nn.Module
        The model with pruned weights.
    """
    if amount < 0.0 or amount > 1.0:
        raise ValueError("prune_amount must be in [0, 1].")

    if scope == "global":
        model = prune_model_global_unstructured_w_magnitude(model, amount)

    elif scope == "layerwise":
        model = prune_model_layerwise_unstructured_w_magnitude(
            model, amount, allocation
        )

    else:
        raise ValueError(
            "Given scope is not valid. Choose from ('global', 'layerwise')"
        )

    sparsity = get_model_total_sparsity(model)

    print("Total model weight sparsity (linear & conv): {:.2f}%".format(100 * sparsity))

    return model
