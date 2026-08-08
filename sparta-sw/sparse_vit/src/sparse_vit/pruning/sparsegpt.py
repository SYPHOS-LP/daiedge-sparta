from typing import Dict

import math
import time
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from .utils import find_layers

from .constants import assign_pruning_ratio_levels_across_modules

# from ..quant.quant import quantize, Quantizer

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


@torch.no_grad()
def prune_unstructured_w_sparsegpt(
    model: nn.Module,
    amount: float,
    device: str,
    data_loader: DataLoader,
    batches: int = 10,
    allocation: str = "uniform",
) -> nn.Module:
    """
    Perform unstructured pruning with sparse gpt.

    Parameters
    ----------
    model : nn.Module
        The model to perform unstructured prunnig with sparse-gpt on.
    amount : float
        The amount of target sparsity.
    device : torch.device
        Device to use, i.e. cuda or cpu.
    data_loader : torch.utils.data.DataLoader
        A `pytorch` dataloader.
    batches : int
        The number of batches from the dataloader to use as calibration
        dataset.
    allocation : str
        How to apply pruning ratios are assigned per layer. If 'uniform' use
        same pruning ratio for every layer, if "finegrained" use different.

    Returns
    -------
    nn.Module
        A pruned model with the sparseGPT method.
    """
    model.eval()
    model.to(device)

    subset = find_layers(model, layers=["linear", "conv2d"])

    gpts = {}

    for layer_idx, (name, layer) in enumerate(subset.items(), start=1):

        gpts[name] = SparseGPT(layer=layer, name=name)

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handle = layer.register_forward_hook(add_batch(name))

        for batch_idx, (images, _) in enumerate(data_loader):

            if batch_idx >= batches:
                break

            images = images.to(device)
            model(images)

        handle.remove()

        if allocation == "finegrained":
            _amount = assign_pruning_ratio_levels_across_modules(name, amount)

        elif allocation == "uniform":
            _amount = amount

        gpts[name].fasterprune(sparsity=_amount)

        gpts[name].free()

    return model


class SparseGPT:
    """
    Class that represents the application of sparse-GPT method on a single layer
    """

    def __init__(self, layer: nn.Module, name: str = "none"):
        """
        Initializer for sparseGPT.
        """
        self.layer = layer
        self.name = name

        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()

        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)

        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out, blocksize=1024):
        """
        This method is to be registered as a forward hook on a module.
        """
        if DEBUG:
            self.inp1 = inp
            self.out1 = out

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)

        tmp = inp.shape[0]

        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=0.01):
        """
        Prune self.layer weights given sparse gpt.
        """
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)

        W = W.float()

        if hasattr(self, "quantizer"):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1**2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = (
                        W1[:, i : (i + prunem)] ** 2
                        / (torch.diag(Hinv1)[i : (i + prunem)].reshape((1, -1))) ** 2
                    )
                    mask1.scatter_(
                        1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True
                    )

                q = w.clone()
                q[mask1[:, i]] = 0

                # if hasattr(self, "quantizer"):
                #     q = quantize(
                #         q.unsqueeze(1),
                #         self.quantizer.scale,
                #         self.quantizer.zero,
                #         self.quantizer.maxq,
                #     ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = W[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print("time %.2f" % (time.time() - tick))
        print(f"`{self.name}`, error: {str(torch.sum(Losses).item())}")

        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def reset_hessian(self):
        """
        Reset Hessian matrix.
        """
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        torch.cuda.empty_cache()

    def free(self):
        """
        Empty cuda cache.
        """
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
