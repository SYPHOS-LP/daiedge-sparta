from typing import Optional, Dict, List, Tuple

import torch
import numpy as np

from torch import nn


def haar_wavelets_1d(dim: int, n: int, scale: float) -> np.ndarray:
    """
    Generate an overcomplete set of Haar wavelet atom of the 1D Haar
    basis of length `n` at scale `v` and all valid location `ws`.

    Haar wavelet atoms are generated via the function `phi`:

    phi(v(i - w)) = {
        1,   if    0   <= v(i - w) < 1/2
       -1,   if    1/2 <= v(i - w) < 1
        0,   otherwise
    }
    for i in range(0, n), where `v` is the scale and `w` the shift.
    Since, `n` needs to be powers of two, then scale (v) can take the value
    of all fractions whose denominator is `n` and the nominator is a power
    of two up to but not including `n`, i.e.:

    `v = 2^j /n`

    where `j in Z` is selected from `j = {0, 1, ..., log2(n - 1)}`.

    From the above formula we can see that scale can be also represented by
    the scale index `j`, but in this implementation we denote scale as a fraction.

    The shift `w` is also constrained based on the scale index `j`. For `j`
    implied by given scale, `w` can take values in `w = {0, 1, ..., n - n /2^j}`.

    Based on the above we can work out some examples:
    ```
    for n = 2, j = 0:
        v = 1 / 2 and w in {0}

    for n = 4, j = 0:
        v = 1 / 4 and w in {0}

    for n = 4, j = 1:
        v = 2 / 4 and w in {0, 1, 2}

    for n = 8, j = 0:
        v = 1 / 8 and w in {0}

    for n = 8, j = 1:
        v = 2 / 8 and w in {0, 1, ..., 4}

    for n = 8, j = 2:
        v = 4 / 8 and w in {0, 1, ..., 7}
    ...
    ```

    Parameters
    ----------
    dim : int
        The overall dimension. This can be bigger but not smaller than the
        length `n`.
    n : int
        The length of the 1D atoms.
    scale : int
        The dyadic scale (v) defined as `v = 2^j /n`. Scale needs to be a power
        of two and no bigger than `n`. Said differently, `j` needs to be selected
        from `j = 0, 1, ..., log2(n - 1)`.

    Returns
    -------
    H : np.ndarray
        An overcomplete set of 1D Haar wavelet atoms of shape (,dim)
    """
    if not np.log2(n).is_integer():
        raise ValueError("Length of `n` must be a power of 2.")

    if (scale <= 0) or (not np.log2(scale).is_integer()) or scale > 1:
        raise ValueError("`scale` must be a dyadic fraction, i.e. 1/2, 1/4, ...")

    # Enforce stricter bounds on scale?
    j = np.log2(scale * n)
    # if j < 0:
    #     raise ValueError(
    #         (
    #             "`scale` is not valid. For given `n`, it needs"
    #             " to be a dyadic fraction no smaller than 1 / {}"
    #         ).format(n)
    #     )

    # Get shift indices `ws` for given scale
    bound = n - n / 2**j

    ws = np.arange(bound + 1, dtype="int64")

    indices = np.arange(n)

    H = np.zeros((len(ws), dim))

    # Add dimensions to allow an outer subtraction between indices and ws
    vals = scale * (indices[None, :] - ws[:, None])

    mask_p1 = (0 <= vals) & (vals < 1 / 2)
    mask_m1 = (1 / 2 <= vals) & (vals < 1)

    vals[mask_p1] = 1
    vals[mask_m1] = -1
    vals[~mask_p1 & ~mask_m1] = 0

    H[ws, :n] = vals

    # Normalize to get L2 norm equal to 1
    return H * scale**0.5


def haar_wavelets2d(
    d: int = 768,
    dim: int = 16,
    n: int = 16,
    scales: float | List[float] | None = None,
    h: int = 14,
    h_step: int = 1,
    apply_kron: str = "per_scale",
):
    """
    Calculate an overcomplete dictionary of 2D Haar wavelet atoms of size
    (target_length x target_length) as a kronecker product of 1D Haar atoms.

    Parameters
    ----------
    d : int
        Target overally dimensionality. Used to decide whether to truncate excess
        wavelet dimensions.
    dim : int
        The overall dimension. This can be bigger but not smaller than the
        length `n`
    n : int
        The length of the 1D Haar wavelet atoms.
    scales : float | List[float] | None
        A list of scales to calculate haar wavelets for. If None,
        all proper scales will be used.
    h : int
        The length of the 1D Haar wavelet atoms used to generate the 2d wavelets.
    h_step : int
        Sample by step the 1D Haar wavelet atoms.
    apply_kron : str
        How kronecker product is applied. Valid choices include `per_scale`,
        and `all`.

    Returns
    -------
    H : np.ndarray
        An overcomplete dictionary of 2D Haar wavelet atoms
    """
    if not np.log2(n).is_integer():
        raise ValueError("Support `n` must be a power of 2.")

    if apply_kron not in ("per_scale", "all"):
        raise ValueError("Choose between 'per_scale' and 'all'.")

    J = int(np.log2(n))

    # NOTE: We remove the 1 / 2 scale as it is too local
    # Could be helpful in the context learnable modulation
    # Needs experiment
    if scales is None:
        scales = [(2 ** (j) / n) for j in range(J)]

    s = (n - h) // 2
    e = s + h

    Hs = []

    for scale in scales:
        H = haar_wavelets_1d(dim, n, scale)

        H = H[:, s:e:h_step]

        # Keep only non-zero valued rows
        mask = np.any(H != 0, axis=1)

        H = H[mask]

        # Remove excess dimensionality to construct `d` correctly
        H_d, _ = H.shape

        if 2 * (H_d**2) > d:
            _h = int((d / 2) ** 0.5)
            _s = (H_d - _h) // 2
            _e = _s + _h

            H = H[_s:_e]

        Hs.append(H)

    if apply_kron == "all":
        Hs = np.vstack(Hs)

        return np.kron(Hs, Hs)

    elif apply_kron == "per_scale":
        Hs = [np.kron(_H, _H) for _H in Hs]

        return np.vstack(Hs)


def pe_haar_2d(
    d: int = 768,
    h: int = 14,
    add_cls: bool = True,
    add_dstl: bool = False,
    add_regs: int = 0,
    dtype: torch.dtype = torch.float32,
    **kwargs
) -> torch.Tensor:
    """
    Non-learnable positional embeddings on a haar wavelet bases
    """
    haar_kw = {
        "d": d,
        "dim": 32,
        "n": 32,
        "scales": [1 / 8],
        "h": h,
        "apply_kron": "per_scale",
    }
    haar_kw.update(kwargs)

    _H = haar_wavelets2d(**haar_kw)

    # Encode haar atom values as two non-negative quantities
    S = _H != 0

    H_pos = (S + _H) / 2
    H_neg = (S - _H) / 2

    # Stack, reshape and pad (if required)
    H = np.stack([H_pos, H_neg], axis=1)
    H = H.reshape(-1, H.shape[-1])

    H_d, _ = H.shape

    if H_d > d:
        print(
            (
                "Output haar wavelets have dimension {} larger than `{}`."
                "Will be sliced to target dimension but may not bring the desired effect,"
            ).format(H_d, d)
        )
        # slice to target dimensions
        H = H[:d]
    else:
        H = np.pad(H, ((0, d - H.shape[0]), (0, 0)))

    # Add class positional embeddings at the beginning
    if add_cls is True and add_dstl is False:
        H = np.pad(H, ((0, 0), (1, 0)))

    elif add_cls is True and add_dstl is True:
        H = np.pad(H, ((0, 0), (2, 0)))

    # Add register positional embeddings at the end
    if add_regs > 0:
        H = np.pad(H, ((0, 0), (0, add_regs)))

    # transpose -> (n, d), add extra dimension -> (1, n, d), and convert to tensor
    H = H.transpose()[None, :]

    return torch.tensor(H, dtype=dtype)


def pe_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    """
    Non-learnable positional embeddings on a sin-cos basis.
    """
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)

    return pe.type(dtype)


def freq_bands(
    num_bands: int, temperature: float = 10000.0, step: int = 1
) -> torch.Tensor:
    """ """
    exp = (
        torch.arange(0, num_bands, step, dtype=torch.int64).to(torch.float32)
        / num_bands
    )
    bands = 1.0 / (temperature**exp)

    return bands


def ndgrid(*tensors) -> Tuple[torch.Tensor, ...]:
    """
    generate N-D grid in dimension order.

    The ndgrid function is like meshgrid except that the order of the first two input arguments are switched.

    That is, the statement
    [X1,X2,X3] = ndgrid(x1,x2,x3)

    produces the same result as

    [X2,X1,X3] = meshgrid(x2,x1,x3)

    This naming is based on MATLAB, the purpose is to avoid confusion due to torch's change to make
    torch.meshgrid behaviour move from matching ndgrid ('ij') indexing to numpy meshgrid defaults of ('xy').
    """
    try:
        return torch.meshgrid(*tensors, indexing="ij")
    except TypeError:
        # old PyTorch < 1.10 will follow this path as it does not have indexing arg,
        # the old behaviour of meshgrid was 'ij'
        return torch.meshgrid(*tensors)


def pe_sincos_1d(seq_len, d_model):
    """ """
    import math

    pe = torch.zeros(d_model, seq_len)
    position = torch.arange(seq_len)
    dim_index = torch.arange(0, d_model, 2)
    scale = -math.log(10000.0) / d_model
    div_term = torch.exp(dim_index * scale)
    angles = div_term[:, None] * position[None, :]

    # even dimensions ↓ (sine)
    even = torch.sin(angles)
    # odd dimensions ↑ (cosine)
    odd = torch.cos(angles)

    pe[0::2, :] = even
    pe[1::2, :] = odd

    return pe


def pe_sincos_2d(
    feat_shape: List[int],
    dim: int = 64,
    temperature: float = 10000.0,
    reverse_coord: bool = False,
    interleave_sin_cos: bool = False,
    device="cpu",
) -> torch.Tensor:
    """
    Non-learnable positional embeddings on a sin-cos basis.

    Parameters
    ----------
    feat_shape :

    dim : dimension
        Embedding dimension.
    temperature : float
        Temperature variable
    reverse_coord : bool
        stack grid order W, H instead of H, W
    interleave_sin_cos : bool
        Whether to interleave sin-cos values like sin - cos - sin - cos
        stack instead of sin - sin - cos - cos.

    Returns:

    """
    msg = "Embed dimension must be divisible by 4 for sin-cos 2D position embedding"

    assert dim % 4 == 0, msg

    pos_dim = dim // 4
    bands = freq_bands(pos_dim, temperature=temperature, step=1)

    if reverse_coord:
        feat_shape = feat_shape[::-1]  # stack W, H instead of H, W

    grid = (
        torch.stack(
            ndgrid(
                [
                    torch.arange(s, device=device, dtype=torch.int64).to(torch.float32)
                    for s in feat_shape
                ]
            )
        )
        .flatten(1)
        .transpose(0, 1)
    )
    pos2 = grid.unsqueeze(-1) * bands.unsqueeze(0)
    # FIXME add support for unflattened spatial dim?

    stack_dim = (
        2 if interleave_sin_cos else 1
    )  # stack sin, cos, sin, cos  instead of sin sin cos cos
    return torch.stack([torch.sin(pos2), torch.cos(pos2)], dim=stack_dim).flatten(1)


def pe_learnable_2d(
    d, n: int, add_cls: bool = True, add_dstl: bool = False, add_regs: int = 0
) -> torch.Tensor:
    """
    Learnable positional embeddings.
    """
    if add_cls is True:
        n += 1

    if add_dstl is True:
        n += 1

    n += add_regs

    return torch.nn.Parameter(torch.randn(1, n, d))
