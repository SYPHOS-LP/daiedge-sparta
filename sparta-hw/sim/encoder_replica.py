"""Bit-faithful Python replica of the FPGA encoder layer.

Mirrors the HLS kernel in src/ (encoder_layer_top -> MHA block + MLP block),
capturing the exact integer/fixed-point datapath and bitwidths so a model can be
inferenced in seconds instead of a C-simulation run.

    hidden = input  + MHA(RMSNorm(input))     # attention block
    output = hidden + MLP(RMSNorm(hidden))     # feed-forward block

Everything is feature-major int8 (D x N), matching the hardware layout.

What "bit-faithful" means here
------------------------------
The HW is quantized (W4A8): int8 activations, 4-bit signed weights, integer
MAC accumulators, and requantization by a folded fixed-point scale. This replica
reproduces:

  * the exact accumulator widths (int32) and MAC integer arithmetic,
  * ap_fixed<32,16> requant scales (truncation to the fixed grid),
  * round-half-away-from-zero + saturate-to-int8 requantization,
  * RMSNorm summing RAW int8 codes (no dequant) through a log-distributed
    rsqrt LUT, and the reciprocal LUT in the attention denominator,
  * ReLU -> requant -> keep-only-strictly-positive sparsification,
  * per-head band splitting of the softmax-free linear attention.

The POT weight encoding (sign+exponent) and the shift-MAC datapath are NOT
modeled specially: a shift by log2|w| plus sign is bit-identical to a multiply
by the +/-2^e weight int, so the plain integer multiply below matches both the
`POT_SHIFT_MAC` and the multiply build of the kernel.

Config is taken from config/inc/*.h (see the constants below); if those change,
update them here.

Ideal vs HW mode
----------------
The MAC / requant / residual / sparsify path is *exact* — it computes the same
integers the hardware does, so it is identical in both modes.  The only lossy
parts of the design are two approximations: the rsqrt LUT in RMSNorm and the
reciprocal LUT in attention.  `set_mode()` toggles those two:

    "hw"    — use the actual on-chip LUTs (what the silicon computes).
    "ideal" — use exact 1/sqrt and 1/x (what the design INTENDS to compute).

Run both and diff: where "ideal" and "hw" agree, the LUT is not the problem;
where they disagree, you have quantified exactly what that HW approximation
costs.  Diffing HW's real output against "ideal" localizes design problems
without the LUT rounding masking them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Config  (mirrors config/inc/*.h — the single source of truth in hardware)
# ---------------------------------------------------------------------------
D = 768            # ENC_D / MHA_D — model feature width
N_HEADS = 12       # MHA_N_HEADS
D_HEAD = 64        # MHA_D_HEAD  (D / N_HEADS)
D_MLP = 3072       # MLP_MAX_D_H — feed-forward inner width

SCALE_FRAC = 16    # T_Scale = ap_fixed<32,16>  -> 16 fractional bits

RMS_EPS = 1e-6
RMS_LUT_ADDR_BITS = 16                 # RMS_LUT_ADDR_BITS (log-distributed rsqrt LUT)
RMS_MS_HI = 1.10 * 127.0 * 127.0       # RMS_MS_HI
RMS_MS_LO = float(RMS_EPS)             # RMS_MS_LO (log variant)

MHA_DIV_EPS = 1e-6                      # MHA_DIV_EPS (unused: recip LUT keys off d directly)
MHA_RECIP_LUT_ADDR_BITS = 14           # MHA_RECIP_LUT_ADDR_BITS
MHA_RECIP_LO = 1.0
# MHA_RECIP_HI mirrors mha_recip.h:  d_head*127 * (N_TOKENS*127).  This is
# N-DEPENDENT: it MUST track the model's token count (p16 N=197, 32b N=50),
# otherwise the replica builds a different recip LUT than the silicon.
MHA_N_TOKENS = 50                      # MHA_N_TOKENS (32b model; p16 uses 197)
MHA_RECIP_HI = float(D_HEAD) * 127.0 * (float(MHA_N_TOKENS) * 127.0)

# scales[] slot layout — enum EncoderScaleLUTIndex (inc/helpers/encoder_scales.h)
SCALE_Q = 0
SCALE_K = 1
SCALE_V = 2
SCALE_LINEAR_OUT = 3
SCALE_DIV_OUT = 4
SCALE_ATT_RMSNORM_IN = 5          # (unused by the datapath; kept for layout parity)
SCALE_ATT_RMSNORM_OUT_INV = 6
SCALE_ATT_RESIDUAL = 7
SCALE_ATT_BRANCH_RATIO = 8
SCALE_FC1 = 9
SCALE_FC2 = 10
SCALE_FF_RMSNORM_IN = 11          # (unused)
SCALE_FF_RMSNORM_OUT_INV = 12
SCALE_FF_RESIDUAL = 13
SCALE_FF_BRANCH_RATIO = 14
SCALE_IDX_MAX = 15


# ---------------------------------------------------------------------------
# Fixed-point primitives  (model ap_fixed / ap_ufixed truncation + saturation)
# ---------------------------------------------------------------------------
def _fixed(value, total_bits, int_bits, signed=True):
    """Quantize onto an ap_[u]fixed<total_bits, int_bits> grid.

    ap_fixed casts TRUNCATE toward -inf (they drop the low bits), and wrap on
    integer-part overflow. We only need the truncation-to-grid behaviour here;
    the datapath never overflows the wide working types by construction.
    """
    frac = total_bits - int_bits
    scaled = np.floor(np.asarray(value, dtype=np.float64) * (2.0 ** frac))
    if signed:
        lo, hi = -(2 ** (total_bits - 1)), 2 ** (total_bits - 1) - 1
    else:
        lo, hi = 0, 2 ** total_bits - 1
    scaled = np.clip(scaled, lo, hi)
    return scaled * (2.0 ** -frac)


def to_scale(value):
    """Cast onto the T_Scale = ap_fixed<32,16> grid (the folded requant scale)."""
    return _fixed(value, 32, 16, signed=True)


# Rounding rule at exact .5 ties.
#   "away" = round-half-away-from-zero — what the HW quantize.h::saturate_to_int8 does
#            (scaled + copysign(0.5); (int) truncates).  10.5->11, -4.5->-5.
#   "even" = round-half-to-even (banker's) — what SW brevitas RoundSte does.
#            10.5->10, -4.5->-4.  With POT (2^k) folded scales, acc*scale lands on exact
#            half-integers SYSTEMATICALLY (every odd multiple of 128), so the two rules
#            disagree constantly -> a ~1-LSB bias per requant that compounds over layers.
_ROUND_MODE = "away"


def set_round_mode(mode):
    """Tie-break rule for requant: 'away' (current HW) or 'even' (SW/brevitas)."""
    global _ROUND_MODE
    if mode not in ("away", "even"):
        raise ValueError("round mode must be 'away' or 'even'")
    _ROUND_MODE = mode


def get_round_mode():
    return _ROUND_MODE


def _round_ties(scaled):
    """Round to nearest integer using the active tie-break rule (_ROUND_MODE)."""
    scaled = np.asarray(scaled, dtype=np.float64)
    if _ROUND_MODE == "even":
        return np.rint(scaled)                       # numpy rint = round-half-to-even
    # round-half-away-from-zero (HW): + copysign(0.5) then truncate toward zero.
    return np.trunc(scaled + np.where(scaled >= 0.0, 0.5, -0.5))


def saturate_to_int8(scaled):
    """Round (per _ROUND_MODE) then clamp to int8  (quantize.h::saturate_to_int8)."""
    value = _round_ties(scaled).astype(np.int64)
    return np.clip(value, -128, 127).astype(np.int8)


def saturate_to_intN(scaled, bits):
    """Round (per _ROUND_MODE) then clamp to a signed `bits`-wide integer.

    Generalization of saturate_to_int8 for the wider residual-highway datapath
    (Option A): the code values live on a finer grid (bits-8 extra frac bits) but
    the arithmetic is otherwise identical (same round rule, same clamp shape).
    """
    value = _round_ties(scaled).astype(np.int64)
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return np.clip(value, lo, hi).astype(np.int64)


def requant_to_int8(acc, requant_const):
    """Multiply an integer accumulator by the folded scale, round + clamp to int8.

    Mirrors requant_to_int8(): the accumulator is cast to T_Scale, multiplied,
    and the product carries T_Scale's <32,16> truncation before rounding.
    """
    acc_fx = to_scale(acc)                # (T_Scale) acc
    product = to_scale(acc_fx * requant_const)  # T_Scale * requant -> T_Scale grid
    return saturate_to_int8(product)


# ---------------------------------------------------------------------------
# Look-up tables  (log-distributed rsqrt + reciprocal, built once)
# ---------------------------------------------------------------------------
class _LogLut:
    """Log-distributed 1/f(x) LUT with the kernel's geometric-midpoint values.

    key[k] = lo * ratio^k ;  val[k] = f(key[k] * sqrt(ratio)) at the bucket
    geometric midpoint.  Lookup = largest k with key[k] <= x (binary search in
    HW; np.searchsorted here is equivalent).
    """

    def __init__(self, lo, hi, size, value_fn, val_bits, val_int_bits):
        ratio = (hi / lo) ** (1.0 / (size - 1))
        sqrt_ratio = math.sqrt(ratio)
        k = np.arange(size, dtype=np.float64)
        self.key = lo * (ratio ** k)
        val = value_fn(self.key * sqrt_ratio)
        self.val = _fixed(val, val_bits, val_int_bits, signed=False)
        self.lo = lo

    def lookup(self, x):
        x = np.asarray(x, dtype=np.float64)
        idx = np.searchsorted(self.key, x, side="right") - 1
        idx = np.clip(idx, 0, len(self.key) - 1)
        return self.val[idx]


class _ExpMantLut:
    """Exponent-and-mantissa LUT — the SEARCH-FREE addressing (rmsnorm_rsqrt.h
    RMS_RSQRT_EXPMANT / the proposed mha_recip Exp-Mant).

    Models the on-chip bit-manipulation EXACTLY (not math.log):

      x written as m * 2^e, m in [1,2):
        e    = position of the leading 1-bit (a priority encoder in HW)
        addr = the ADDR_BITS mantissa bits just below the leading one
      Table stores only the mantissa function f(m) for m in [1,2); the 2^(power*e)
      factor is applied at runtime as a shift.  See docs/exploration/expmant_lut_explained.md.

    `power` = -0.5 for rsqrt (half-power, needs even/odd parity tables), -1 for
    1/x (plain shift, single table).  `frac_bits` = fractional bits of the
    fixed-point input's raw integer view (so e = lead_pos - frac_bits).
    """

    def __init__(self, addr_bits, value_fn, power, frac_bits,
                 val_bits, val_int_bits, in_bits):
        self.addr_bits = addr_bits
        self.power     = power
        self.frac_bits = frac_bits
        self.in_bits   = in_bits
        self.size      = 1 << addr_bits
        # Table over the mantissa m in [1,2), bucket-center convention (matches
        # rms_rsqrt_populate: m = 1 + (k + 0.5)/SIZE).
        k = np.arange(self.size, dtype=np.float64)
        m = 1.0 + (k + 0.5) / self.size
        even = value_fn(m)                      # f(m)               (even e)
        if abs(power + 0.5) < 1e-9:             # rsqrt: fold 1/sqrt2 for odd e
            self.even = _fixed(even, val_bits, val_int_bits, signed=False)
            self.odd  = _fixed(even * (1.0 / math.sqrt(2.0)),
                               val_bits, val_int_bits, signed=False)
        else:                                   # 1/x: single table, plain 2^-e shift
            self.even = _fixed(even, val_bits, val_int_bits, signed=False)
            self.odd  = None
        # Compat with the zero-guard in mha_recip(): `.val[0]` = value at the LUT
        # floor (denominator == LO == 1 -> m=1, e=0 -> 1/m from bucket 0).
        self.val = self.even

    def _lookup_scalar(self, x):
        # Fixed-point raw integer view: value = raw * 2^-frac_bits.
        raw = int(math.floor(x * (2.0 ** self.frac_bits)))
        if raw <= 0:
            return float(self.even[0])
        raw &= (1 << self.in_bits) - 1
        lead_pos = raw.bit_length() - 1          # position of leading one (== clz)
        e = lead_pos - self.frac_bits
        # Mantissa address = ADDR_BITS bits just below the leading one.
        shift = lead_pos - self.addr_bits
        if shift >= 0:
            addr = (raw >> shift) & (self.size - 1)
        else:
            addr = (raw << (-shift)) & (self.size - 1)
        if self.odd is not None:                 # rsqrt: parity table + 2^(-e/2)
            is_odd = (e & 1) != 0
            half   = (e - 1) // 2 if is_odd else e // 2
            base   = float(self.odd[addr]) if is_odd else float(self.even[addr])
            return base * (2.0 ** (-half))       # -half because power=-1/2 folded as >>half
        else:                                    # 1/x: 2^(-e)
            return float(self.even[addr]) * (2.0 ** (-e))

    def lookup(self, x):
        x = np.asarray(x, dtype=np.float64)
        flat = x.ravel()
        out = np.array([self._lookup_scalar(float(v)) for v in flat], dtype=np.float64)
        return out.reshape(x.shape)


# LUT-variant switch: "log" (current binary-search) or "expmant" (search-free).
# set_lut_variant() rebuilds both LUTs; the accuracy gate compares the two.
_LUT_VARIANT = "log"

# Fixed-point input formats the Exp-Mant addressing keys off (from config/inc):
#   T_RmsSq  = ap_ufixed<56,26> -> 30 frac bits, 56 total   (rsqrt input = mean-square)
#   T_MhaAcc = ap_int<32>       ->  0 frac bits, 32 total   (recip input = denominator)
_RMS_SQ_FRAC = 56 - 26     # 30
_RMS_SQ_BITS = 56
_MHA_ACC_BITS = 32


def _build_luts(variant, rsqrt_bits=None, recip_bits=None):
    """(Re)build the rsqrt + recip LUTs. `rsqrt_bits`/`recip_bits` override the
    config address widths (for the accuracy-vs-bits sweep); None = config default."""
    rb = RMS_LUT_ADDR_BITS if rsqrt_bits is None else rsqrt_bits
    cb = MHA_RECIP_LUT_ADDR_BITS if recip_bits is None else recip_bits
    if variant == "log":
        rsqrt = _LogLut(RMS_MS_LO, RMS_MS_HI, 1 << rb,
                        lambda ms: 1.0 / np.sqrt(ms), 28, 12)
        recip = _LogLut(MHA_RECIP_LO, MHA_RECIP_HI, 1 << cb,
                        lambda d: 1.0 / d, 32, 1)
    elif variant == "expmant":
        # rsqrt: power -1/2 (parity tables), keyed off the 30-frac ap_ufixed<56,26>.
        rsqrt = _ExpMantLut(rb, lambda m: 1.0 / np.sqrt(m),
                            power=-0.5, frac_bits=_RMS_SQ_FRAC,
                            val_bits=28, val_int_bits=12, in_bits=_RMS_SQ_BITS)
        # recip: power -1 (single table, plain 2^-e shift), keyed off int32 T_MhaAcc.
        recip = _ExpMantLut(cb, lambda m: 1.0 / m,
                            power=-1.0, frac_bits=0,
                            val_bits=32, val_int_bits=1, in_bits=_MHA_ACC_BITS)
    else:
        raise ValueError("lut variant must be 'log' or 'expmant'")
    return rsqrt, recip


_RSQRT_LUT, _RECIP_LUT = _build_luts(_LUT_VARIANT)


def set_lut_variant(variant, rsqrt_bits=None, recip_bits=None):
    """Select the LUT addressing: 'log' (binary search) or 'expmant' (search-free).
    Optionally override the rsqrt/recip address-bit widths (accuracy-vs-bits sweep)."""
    global _LUT_VARIANT, _RSQRT_LUT, _RECIP_LUT
    _RSQRT_LUT, _RECIP_LUT = _build_luts(variant, rsqrt_bits, recip_bits)
    _LUT_VARIANT = variant


def get_lut_variant():
    return _LUT_VARIANT


# Mode switch for the two LUT approximations (see module docstring).
# "hw"    -> the on-chip LUTs (what the silicon computes)
# "ideal" -> exact 1/sqrt and 1/x (what the design intends)
_MODE = "hw"

# Option A — residual-highway precision.  The residual "skip" stream (the activation
# carried from layer to layer) and the RMSNorm input use this many bits instead of
# int8.  8 = current HW (int8 highway).  Widening it (e.g. 16) adds RESIDUAL_BITS-8
# extra FRACTIONAL bits: the codes live on a finer grid (s_out / 2^(RESIDUAL_BITS-8)),
# so the per-layer requant rounding that otherwise compounds is largely preserved.
# Q'/K'/V and the MLP hidden (the sparse SpMM operands) stay int8 — only the highway
# widens.  This is a design knob, not HW behaviour, used to measure the recovery.
_RESIDUAL_BITS = 8


def set_residual_bits(bits):
    """Set the residual-highway width (>=8).  8 = current int8 HW; 16 = Option A."""
    global _RESIDUAL_BITS
    if bits < 8:
        raise ValueError("residual bits must be >= 8")
    _RESIDUAL_BITS = int(bits)


def get_residual_bits():
    return _RESIDUAL_BITS


def set_mode(mode):
    """Select the approximation mode: 'hw' (LUTs) or 'ideal' (exact math)."""
    global _MODE
    if mode not in ("hw", "ideal"):
        raise ValueError("mode must be 'hw' or 'ideal'")
    _MODE = mode


def get_mode():
    return _MODE


def rms_rsqrt(mean_square):
    """1/sqrt(mean_square).  Scalar or array (vectorized over tokens).

    hw:    clamp to [LO,HI] then log-LUT lookup (rmsnorm_rsqrt.h, LOG variant).
    ideal: exact 1/sqrt (the value the LUT approximates).
    """
    ms = np.clip(np.asarray(mean_square, dtype=np.float64), RMS_MS_LO, RMS_MS_HI)
    out = 1.0 / np.sqrt(ms) if _MODE == "ideal" else _RSQRT_LUT.lookup(ms)
    return out if out.ndim else float(out)


def mha_recip(denominator):
    """1/denominator for the attention division.  Scalar or array (per token).

    hw:    d<=LO returns entry 0 (forces 0/0 -> 0); else log-LUT (mha_recip.h).
    ideal: exact 1/d, with the same d<=LO -> 0 guard so an all-zero Q'/K' row
           still produces a zero output rather than a divide-by-zero.
    """
    d = np.asarray(denominator, dtype=np.float64)
    small = d <= MHA_RECIP_LO                       # 0/0 guard (numerator is also 0)
    safe = np.where(small, MHA_RECIP_LO + 1.0, d)   # keep the LUT/divide in range
    if _MODE == "ideal":
        val = np.where(small, 0.0, 1.0 / safe)
    else:
        val = np.where(small, float(_RECIP_LUT.val[0]), _RECIP_LUT.lookup(safe))
    return val if val.ndim else float(val)


# ---------------------------------------------------------------------------
# Weights container
# ---------------------------------------------------------------------------
@dataclass
class LayerWeights:
    """One encoder layer's six int4 weight matrices and 15 folded scales.

    Weights are DENSE int matrices (rows = output features, cols = input
    features); the hardware stores them CSR but the arithmetic is identical.
    `scales` is the length-15 folded-scale vector (EncoderScaleLUTIndex order),
    stored on the T_Scale grid.

        w_q, w_k, w_v, w_o : (D, D)      MHA projections
        w1                 : (D_MLP, D)  MLP fc1
        w2                 : (D, D_MLP)  MLP fc2
    """
    w_q: np.ndarray
    w_k: np.ndarray
    w_v: np.ndarray
    w_o: np.ndarray
    w1: np.ndarray
    w2: np.ndarray
    scales: np.ndarray
    # Optional raw attention quant scales, needed ONLY by the SW-faithful attention
    # path (mha with sw=True): the per-tensor int8 scales of Q/K/V/out and the
    # d_k^-0.25 pre-ReLU factor from the SW quant model (conversion.py).  Also the
    # product s_weight*s_z per projection, to turn the replica's integer projection
    # accumulator back into a real value before quantizing at s_q/s_k/s_v.
    sw_attn: dict = None

    def __post_init__(self):
        self.scales = to_scale(np.asarray(self.scales, dtype=np.float64))
        for name in ("w_q", "w_k", "w_v", "w_o", "w1", "w2"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.int32))


# ---------------------------------------------------------------------------
# RMSNorm  (rmsnorm.cpp)
# ---------------------------------------------------------------------------
def rmsnorm(x, inv_out_scale, in_frac=0):
    """RMSNorm over the feature axis, per token, on the raw highway codes.

    Mirrors rmsnorm(): the sum of squares is over the RAW codes (no dequant);
    the input_scale cancels because gamma is fused away and the LUT is
    scale-free, so mean-square uses the codes directly.  Output = round/clamp of
    code * rsqrt(mean_square) * inv_out_scale.

    `in_frac` = extra fractional bits of a widened residual highway (Option A).
    rsqrt is scale-free, so `code * rsqrt(mean(code^2))` is unchanged by a common
    2^in_frac factor; we only divide it out for the LUT lookup so the mean-square
    stays inside the LUT's fixed [eps, ~1.1*127^2] range (a >int8 highway would
    otherwise overflow it and clamp).  The normalize keeps the full-precision codes.

    x : (D, N) int highway codes feature-major.  Returns (D, N) int8 (z).
    """
    x = x.astype(np.int64)
    d, _n = x.shape
    inv_d = 1.0 / float(d)
    scale = 2.0 ** in_frac                              # highway is 2^in_frac finer

    # mean-square of the int8-EQUIVALENT magnitudes (codes / 2^in_frac) -> LUT range.
    xf = x.astype(np.float64) / scale
    acc = np.sum(xf * xf, axis=0)                       # (N,)
    mean_square = acc * inv_d + RMS_EPS
    inv_rms = np.asarray(rms_rsqrt(mean_square))        # (N,) rsqrt on int8-scale ms
    # normalize the int8-equivalent code (xf) so the result is width-independent.
    norm = _fixed(xf * inv_rms[None, :], 32, 16, signed=True)
    return saturate_to_int8(norm * inv_out_scale)


# ---------------------------------------------------------------------------
# Linear projections  (Gustavson SpMM in HW; a dense int matmul here)
# ---------------------------------------------------------------------------
def _project(weight, x):
    """Integer matmul  acc = weight (Dout x Din, int4) @ x (Din x N, int8).

    Returns the int32 accumulators (Dout x N).  This is exactly what the
    Gustavson SpMM computes; the CSR/sparsity only changes the schedule.
    """
    return weight.astype(np.int64) @ x.astype(np.int64)


def _proj_relu(weight, x, scale):
    """Projection + ReLU + requant, with the kernel's sparsification semantics.

    proj_relu_*: keep an output only where the int32 accumulator is > 0, AND the
    requantized int8 is strictly > 0 (so exact-zero and negative codes drop out).
    Returns an int8 (Dout x N) with the pruned entries set to 0 — numerically
    identical to the CSR the hardware emits.
    """
    acc = _project(weight, x)
    q = requant_to_int8(acc, scale).astype(np.int32)
    keep = (acc > 0) & (q > 0)
    return np.where(keep, q, 0).astype(np.int8)


def _proj_dense(weight, x, scale):
    """Dense projection + requant (proj_dense): V and O projections."""
    return requant_to_int8(_project(weight, x), scale)


# ---------------------------------------------------------------------------
# Linear multi-head attention  (mha.cpp: head_core)
# ---------------------------------------------------------------------------
def _mha_hw(z, w, scales):
    """HW-faithful linear attention (mha.cpp head_core), vectorized over tokens.

    Q'/K' are ReLU'd + requantized to int8 BEFORE the attention math (the CSR the
    HW emits); the numerator/denominator are int32 accumulators; the division uses
    the reciprocal LUT (or exact 1/x in 'ideal' mode); O is requantized on the
    T_MhaDiv <32,16> grid.  Bit-identical to the per-token loop, just batched.
    """
    n = z.shape[1]
    q = _proj_relu(w.w_q, z, scales[SCALE_Q])     # (D, N) int8
    k = _proj_relu(w.w_k, z, scales[SCALE_K])     # (D, N) int8
    v = _proj_dense(w.w_v, z, scales[SCALE_V])    # (D, N) int8
    s_div = scales[SCALE_DIV_OUT]

    zhat = np.zeros((D, n), dtype=np.int8)
    for head in range(N_HEADS):
        base = head * D_HEAD
        qh = q[base:base + D_HEAD, :].astype(np.int64)   # (d_head, N)
        kh = k[base:base + D_HEAD, :].astype(np.int64)
        vh = v[base:base + D_HEAD, :].astype(np.int64)

        A = kh @ vh.T                              # (d_head, d_head) int32
        sK = kh.sum(axis=1)                        # (d_head,)

        num = A.T @ qh                             # (d_head, N)  num[:, t] = A^T q_t
        den = sK @ qh                              # (N,)         den[t] = q_t . sK
        recip = mha_recip(den)                     # (N,)  T_MhaRecip
        ratio = _fixed(num.astype(np.float64) * recip[None, :], 32, 16, signed=True)
        zhat[base:base + D_HEAD, :] = saturate_to_int8(ratio * s_div)

    return _proj_dense(w.w_o, zhat, scales[SCALE_LINEAR_OUT])


def _mha_sw(z, w, scales):
    """SW-faithful linear attention (conversion.py patched_forward_linear).

    Reproduces the SW quant model's attention exactly, which differs from the HW:
      * Q/K/V are quantized to int8 (per-tensor s_q/s_k/s_v) from the RAW projection
        (NO ReLU yet);
      * _q = ReLU(Q_int * s_q * scale^0.5), likewise _k  (ReLU AFTER quant, and the
        d_k^-0.25 pre-scale that the HW folds/drops);
      * numerator, denominator and the division are done in FLOAT (num/(den+1e-8));
      * out is quantized to int8 at s_out, then the O projection runs (dense, HW-style
        requant) — the O weights + SCALE_LINEAR_OUT are shared with the HW path.

    Needs w.sw_attn = {s_q, s_k, s_v, s_out, scale_half, swz_q, swz_k, swz_v} where
    swz_* = s_weight * s_z per projection (turns the integer projection accumulator
    into a real value).  Falls back to the HW path if sw_attn is absent.
    """
    if w.sw_attn is None:
        return _mha_hw(z, w, scales)
    a = w.sw_attn
    n = z.shape[1]
    zc = z.astype(np.int64)

    def quant_int8(real, s):
        return np.clip(np.round(real / s), -128, 127)   # brevitas RoundSte + int8 clamp

    # raw projections (integer acc -> real), then quantize to int8 at the SW scale
    q_real = (w.w_q.astype(np.int64) @ zc).astype(np.float64) * a["swz_q"]
    k_real = (w.w_k.astype(np.int64) @ zc).astype(np.float64) * a["swz_k"]
    v_real = (w.w_v.astype(np.int64) @ zc).astype(np.float64) * a["swz_v"]
    q_i = quant_int8(q_real, a["s_q"])              # (D, N)
    k_i = quant_int8(k_real, a["s_k"])
    v_i = quant_int8(v_real, a["s_v"])

    sh = a["scale_half"]
    zhat_real = np.zeros((D, n), dtype=np.float64)
    for head in range(N_HEADS):
        base = head * D_HEAD
        _q = np.maximum(q_i[base:base + D_HEAD, :] * a["s_q"] * sh, 0.0)   # (d_head, N)
        _k = np.maximum(k_i[base:base + D_HEAD, :] * a["s_k"] * sh, 0.0)
        _v = v_i[base:base + D_HEAD, :] * a["s_v"]

        kv = _k @ _v.T                              # (d_head, d_head)  _k V^T over tokens
        num = kv.T @ _q                             # (d_head, N)
        den = _k.sum(axis=1) @ _q                   # (N,)
        zhat_real[base:base + D_HEAD, :] = num / (den[None, :] + 1e-8)

    # quantize attention output to int8 at s_out, then the (HW) dense O projection.
    zhat = quant_int8(zhat_real, a["s_out"]).astype(np.int8)
    return _proj_dense(w.w_o, zhat, scales[SCALE_LINEAR_OUT])


def mha(z, w, scales):
    """Softmax-free linear attention on the normalized input z (D x N int8).

    'ideal' mode uses the SW-faithful attention (matches the SW quant model) when
    w.sw_attn scales are provided; 'hw' mode uses the HW datapath.  Either way the
    Q/K/V/O projections and the RMSNorm/MLP/residual around it are the same code.
    """
    if _MODE == "ideal" and w.sw_attn is not None:
        return _mha_sw(z, w, scales)
    return _mha_hw(z, w, scales)


# ---------------------------------------------------------------------------
# Sparse MLP  (mlp.cpp)
# ---------------------------------------------------------------------------
def mlp(z, w, scales):
    """Two-layer feed-forward:  Y = W2 . ReLU(W1 . z).

    z : (D, N) int8 (RMSNorm output).  Returns (D, N) int8.
    fc1 output is ReLU'd + requantized + sparsified (fused_sdmm_relu); fc2 is a
    dense SpMM over the surviving hidden nonzeros (spmm_dense_row).
    """
    h = _proj_relu(w.w1, z, scales[SCALE_FC1])              # (D_MLP, N) int8, sparse
    return _proj_dense(w.w2, h, scales[SCALE_FC2])          # (D, N) int8


# ---------------------------------------------------------------------------
# Residual add  (residual.cpp)
# ---------------------------------------------------------------------------
def residual_add(residual, branch, residual_scale, branch_scale):
    """y = round( residual*residual_scale + branch*branch_scale )  -> int8.

    Two int8 operands in different quant domains; the folded ratios reconcile
    them to the output scale.

    hw:    mirrors residual.cpp — each branch is cast onto the T_Scale grid
           (truncation) before the sum, matching the HLS ap_fixed datapath.
    ideal: sum the two rescaled branches in full precision and round ONCE, with
           no per-branch truncation.  This is what the SW dyadic residual
           approaches (its per-branch dyadic rounding is exact for the >=1 ratio
           and near-exact for the <1 one), and it avoids the coarse re-grid that
           otherwise dominates the layer-to-layer drift vs the SW model.
    """
    # extra fractional bits carried by the residual highway (Option A).  The SKIP
    # branch arrives already on the finer grid (its codes carry `extra` frac bits, so
    # dividing by 2^extra recovers its int8-scale real value); the OUTPUT is emitted on
    # the same finer grid (scale it back up by 2^extra before the round-to-intN).
    extra = _RESIDUAL_BITS - 8
    scale = 2.0 ** extra
    r = (residual.astype(np.float64) / scale) * float(residual_scale)
    b = branch.astype(np.float64) * float(branch_scale)   # branch is int8 (s_out grid)
    if _MODE == "ideal":
        total = r + b
    else:
        total = to_scale(to_scale(r) + to_scale(b))
    if extra == 0:
        return saturate_to_int8(total)
    return saturate_to_intN(total * scale, _RESIDUAL_BITS)


# ---------------------------------------------------------------------------
# Blocks + top level  (encoder_mha_block / encoder_mlp_block / encoder_layer_top)
# ---------------------------------------------------------------------------
def attention_block(x, w):
    """hidden = x + MHA(RMSNorm(x)).  x/hidden are highway codes (_RESIDUAL_BITS wide)."""
    extra = _RESIDUAL_BITS - 8
    z = rmsnorm(x, w.scales[SCALE_ATT_RMSNORM_OUT_INV], in_frac=extra)  # z is int8
    branch = mha(z, w, w.scales)                                       # branch is int8
    return residual_add(x, branch,
                        w.scales[SCALE_ATT_RESIDUAL], w.scales[SCALE_ATT_BRANCH_RATIO])


def feedforward_block(x, w):
    """output = x + MLP(RMSNorm(x)).  x/output are highway codes (_RESIDUAL_BITS wide)."""
    extra = _RESIDUAL_BITS - 8
    z = rmsnorm(x, w.scales[SCALE_FF_RMSNORM_OUT_INV], in_frac=extra)  # z is int8
    branch = mlp(z, w, w.scales)                                       # branch is int8
    return residual_add(x, branch,
                        w.scales[SCALE_FF_RESIDUAL], w.scales[SCALE_FF_BRANCH_RATIO])


def encoder_layer(x, w):
    """One transformer encoder layer.  x/return are highway codes (_RESIDUAL_BITS wide).

        hidden = attention_block(x)
        output = feedforward_block(hidden)

    With _RESIDUAL_BITS == 8 the highway is plain int8 (unchanged HW).  Wider values
    carry _RESIDUAL_BITS-8 extra fractional bits (Option A) and stay on the highway
    across layers.  Use `run_encoder` to feed an int8 input / read an int8 output; call
    this directly only when x already lives on the highway grid.
    """
    x = np.asarray(x)
    hidden = attention_block(x, w)
    return feedforward_block(hidden, w)


def run_encoder(x_int8, layers, out_frac=None):
    """Run all encoder layers with the current residual-highway width.

    x_int8 : (D, N) int8 encoder input (0 frac).  Promoted onto the highway grid
             (<< extra frac bits) before layer 0 so the skip scale reconciles.
    out_frac: if None, the output is returned on the highway grid (extra frac bits,
             so head-side dequant must use s_out / 2^extra).  Pass 0 to round the
             final output back to int8 (s_out grid) for an int8-in/int8-out contract.

    Returns (codes, frac_bits): the encoder output codes and how many extra
    fractional bits they carry (0 for int8).
    """
    extra = _RESIDUAL_BITS - 8
    x = np.asarray(x_int8, dtype=np.int64) << extra    # promote int8 -> highway grid
    for w in layers:
        x = encoder_layer(x, w)
    if out_frac == 0 and extra > 0:
        x = saturate_to_int8(x.astype(np.float64) / (2.0 ** extra))
        return x, 0
    return x, extra