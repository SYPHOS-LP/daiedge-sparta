#!/usr/bin/env python3
"""Compile the SW team's quant_values.txt into the FPGA runtime payload.

Two stages in one tool:

  (1) LOAD    quant_values.txt -> in-memory QuantModel  (compile_model()).
      The SW team exports the W4A8 ViT into a text file with four sections:

        1. QUANTIZED LINEAR / CONV2D LAYERS  (marker: "LAYER: <name>  (QuantLinear)")
             weight.shape / weight.bit_width(4) / weight.signed / weight.scale (PER-ROW,
             one per output channel) / weight.zero_point / weight.int (dense int4 matrix)
             [optional] bias.* (IGNORED - unused in the network)
             input.bit_width(8) / input.signed / input.scale (per-tensor) / input.zero_point
        2. RMSNORM MODULES                   (marker: "NORM: <name>")
             gamma.* (int8 vector + scalar scale) + output.scale (sZ for ln_mha / sX1 for ln_mlp)
        3. DYADIC  (residual adders)         (marker: "DYADIC: <name>")
             branch_a.scale / branch_b.scale / out.scale / m_a,e_a,m_b,e_b / scale_out
        4. OTHER STANDALONE ACTIVATION QUANTIZERS  (marker: "ACT: <name>")
             act.scale (per-tensor) for quant_q/quant_k/quant_v/quant_out (= sQ'/sK'/sV/
             sZhat), quant_input, quant_pre_mlp, and embed/head quantizers.

  (2) EMIT    QuantModel -> ONE concatenated little-endian model.bin (all layers,
      all blobs) + manifest.json.  The manifest gives every blob an absolute byte
      {offset, bytes, count, dtype} into model.bin, so the runtime host seeks/mmaps
      directly and streams to the encoder_layer_top kernel.  Per encoder layer:

        WEIGHTS (w4 CSR, per weight):
          <w>.values  int8   (w4 value in the byte)   one per nonzero, row-major
          <w>.col     uint16 (input-feature index)    one per nonzero
          <w>.rowptr  int32  (row offsets)            length num_rows+1
        The six weights: w_q,w_k,w_v,out_proj (768x768), linear_1 (3072x768),
        linear_2 (768x3072).  CSR = the nonzeros of weight.int (already int4).

        PER-ROW REQUANT VECTORS (ap_fixed<32,16>, raw int32 of round(s*2^16)):
          s_q_row, s_k_row, s_v_row, s_o_row  (len 768)
          s1_row  (len d_h=3072),  s2_row (len d_out=768)
        Folded per OUTPUT ROW: s[row] = (weight.scale[row] * s_in) / s_out.  The weight
        scale is per-row (per-output-channel), so the requant constant varies per row.

        SCALAR SCALES: scales = the 15 folded SCALE_*_IDX values (ap_fixed<32,16> int32).

        GAMMAS: gamma_mha / gamma_mlp = ap_fixed<16,2> int16 of the RMSNorm gamma
        (real gamma = gamma.int * gamma.scale, re-encoded to the design's ap_fixed grid).

      FOLD FORMULAS (s_in / s_out per projection; weight.scale is the per-row sW):
        Q':  (sWq[row] * sZ)   / sQ'      sZ=ln_mha.output, sQ'=quant_q
        K':  (sWk[row] * sZ)   / sK'                        sK'=quant_k
        V:   (sWv[row] * sZ)   / sV                         sV =quant_v
        O:   (sWo[row] * sZhat)/ sY       sZhat=quant_out,  sY =res1.out
        W1:  (sW1[row] * sX1)  / sH       sX1=ln_mlp.output,sH =linear_2.input
        W2:  (sW2[row] * sH)   / s_M      sH =linear_2.input, s_M=res2.out

      The 15 SCALAR scales (SCALE_*_IDX order) are the non-per-row folds:
        s_div=sV/sZhat; rms in/inv_out; residual res_xr=sX/sOUT, res_mr=sM/sOUT (both blocks).

Weights are DENSE in the file (int4 values incl. the ~80% zeros); the CSR emit drops
the zeros.  Biases are IGNORED (unused in the network).

Usage:
    python scripts/compile_model.py -i <quant_values.txt> [--out out/model]
                                    [--layers 0,1 | all] [--verify]

    # inspection only (no emit):
    python scripts/compile_model.py -i <path> --summary
    python scripts/compile_model.py -i <path> --module encoder.encoder_layer_0.mha.w_q

As a library:
    from compile_model import compile_model
    model = compile_model("quant_values.txt")
    model.linears["encoder.encoder_layer_0.mha.w_q"].weight_int   # np.int16 [768,768]
    model.act_scale("encoder.encoder_layer_0.mha.quant_q")        # 0.01750...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------
# per-block data holders
# ----------------------------------------------------------------------------
@dataclass
class LinearLayer:
    """A QuantLinear block: dense int4 weight + per-row weight scale + input scale."""
    name: str
    shape: tuple = None                     # (out, in)
    weight_bits: int = 0
    weight_signed: bool = True
    weight_scale: np.ndarray = None         # PER-ROW, len = out
    weight_int: np.ndarray = None           # dense int, [out, in] (int4 range, still dense)
    input_scale: float = None               # per-tensor activation-into-this-layer scale
    input_bits: int = 0
    # bias.* are read but intentionally NOT stored (unused in the network).


@dataclass
class NormModule:
    """An RMSNorm block: int8 gamma + gamma scale + output activation scale."""
    name: str
    gamma_int: np.ndarray = None            # int, len D
    gamma_scale: float = None
    gamma_bits: int = 0
    output_scale: float = None              # RMSNorm output act scale (sZ / sX1)
    output_bits: int = 0


@dataclass
class DyadicModule:
    """A residual (dyadic) adder block: branch/out scales + m/e dyadic fold."""
    name: str
    branch_a_scale: float = None
    branch_b_scale: float = None
    out_scale: float = None
    m_a: int = None
    e_a: int = None
    m_b: int = None
    e_b: int = None
    scale_out: float = None


@dataclass
class ActQuant:
    """A standalone activation quantizer block: a single per-tensor scale."""
    name: str
    scale: float = None
    bits: int = 0
    signed: bool = True


@dataclass
class QuantModel:
    """The whole parsed file, keyed by full module name."""
    linears: dict = field(default_factory=dict)   # name -> LinearLayer
    norms: dict = field(default_factory=dict)      # name -> NormModule
    dyadics: dict = field(default_factory=dict)    # name -> DyadicModule
    acts: dict = field(default_factory=dict)       # name -> ActQuant
    meta: dict = field(default_factory=dict)       # header comments

    # -- convenience scale accessors (the "list indexed by known activation") ----
    def act_scale(self, name):
        """Scale of a standalone activation quantizer (quant_q, quant_input, ...)."""
        return self.acts[name].scale

    def norm_out_scale(self, name):
        """RMSNorm output activation scale (ln_mha -> sZ, ln_mlp -> sX1)."""
        return self.norms[name].output_scale

    def input_scale(self, name):
        """A QuantLinear's input activation scale."""
        return self.linears[name].input_scale

    def layers(self):
        """Sorted list of encoder layer indices present in the file."""
        ids = set()
        for n in list(self.linears) + list(self.norms) + list(self.dyadics) + list(self.acts):
            m = re.search(r"encoder\.encoder_layer_(\d+)\.", n)
            if m:
                ids.add(int(m.group(1)))
        return sorted(ids)


# ----------------------------------------------------------------------------
# stage 1: parsing / load  (compile_model)
# ----------------------------------------------------------------------------
_FLOATS = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _floats(s):
    return [float(x) for x in _FLOATS.findall(s)]


def _ints(s):
    # ints only (no decimal point / exponent); the int dumps are whitespace-separated
    return np.fromstring(s, dtype=np.int64, sep=" ")


def compile_model(path):
    """Parse the quant_values.txt into a QuantModel.  Reads line by line; the big
    weight/gamma int dumps are single (very long) lines, so this stays streaming."""
    model = QuantModel()

    # Current block being filled: (kind, name, obj).  kind in {LINEAR,NORM,DYADIC,ACT}.
    kind = name = obj = None
    # When a label announces a following multi-row int matrix (weight.int header),
    # we collect the next `pending_rows` data lines into `pending_target`.
    pending_rows = 0
    pending_target = None       # list to append np rows into
    pending_shape = None

    def flush_matrix():
        nonlocal pending_rows, pending_target, pending_shape
        pending_rows = 0
        pending_target = None
        pending_shape = None

    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            s = line.strip()

            # --- collecting rows of a dense int matrix (weight.int) ---------------
            if pending_rows > 0 and pending_target is not None:
                # a data row is all-integer whitespace-separated tokens
                if s and (s[0].isdigit() or s[0] == "-"):
                    pending_target.append(_ints(s))
                    pending_rows -= 1
                    if pending_rows == 0:
                        arr = np.stack(pending_target).astype(np.int16)
                        obj.weight_int = arr
                        flush_matrix()
                    continue
                # non-data line before we filled all rows -> stop (defensive)
                flush_matrix()

            # --- header comments -------------------------------------------------
            if s.startswith("#"):
                if ":" in s:
                    k, v = s[1:].split(":", 1)
                    model.meta[k.strip()] = v.strip()
                continue

            # --- block markers ---------------------------------------------------
            m = re.match(r"(LAYER|NORM|DYADIC|ACT):\s+(\S+)", s)
            if m:
                mk, nm = m.group(1), m.group(2)
                if mk == "LAYER":
                    kind, name, obj = "LINEAR", nm, LinearLayer(nm)
                    model.linears[nm] = obj
                elif mk == "NORM":
                    kind, name, obj = "NORM", nm, NormModule(nm)
                    model.norms[nm] = obj
                elif mk == "DYADIC":
                    kind, name, obj = "DYADIC", nm, DyadicModule(nm)
                    model.dyadics[nm] = obj
                else:  # ACT
                    kind, name, obj = "ACT", nm, ActQuant(nm)
                    model.acts[nm] = obj
                continue

            if obj is None or ":" not in s:
                continue

            key, val = s.split(":", 1)
            key = key.strip()
            val = val.strip()

            # --- per-kind fields -------------------------------------------------
            if kind == "LINEAR":
                if key == "weight.shape":
                    obj.shape = tuple(int(x) for x in re.findall(r"\d+", val))
                elif key == "weight.bit_width":
                    obj.weight_bits = int(val)
                elif key == "weight.signed":
                    obj.weight_signed = (val == "True")
                elif key == "weight.scale":
                    obj.weight_scale = np.array(_floats(val), dtype=np.float64)
                elif key == "weight.int":
                    # header line "(768 rows x 768 vals)": start collecting rows next.
                    rows = obj.shape[0] if obj.shape else None
                    if rows:
                        pending_rows = rows
                        pending_target = []
                        pending_shape = obj.shape
                elif key == "input.scale":
                    obj.input_scale = float(val)
                elif key == "input.bit_width":
                    obj.input_bits = int(val)
                # bias.* intentionally ignored.

            elif kind == "NORM":
                if key == "gamma.int":
                    obj.gamma_int = _ints(val).astype(np.int16)
                elif key == "gamma.scale":
                    obj.gamma_scale = float(val)
                elif key == "gamma.bit_width":
                    obj.gamma_bits = int(val)
                elif key == "output.scale":
                    obj.output_scale = float(val)
                elif key == "output.bit_width":
                    obj.output_bits = int(val)

            elif kind == "DYADIC":
                dd = {
                    "branch_a.scale": "branch_a_scale", "branch_b.scale": "branch_b_scale",
                    "out.scale": "out_scale", "scale_out": "scale_out",
                }
                if key in dd:
                    setattr(obj, dd[key], float(val))
                elif key in ("m_a", "e_a", "m_b", "e_b"):
                    setattr(obj, key, int(val))

            elif kind == "ACT":
                if key == "act.scale":
                    obj.scale = float(val)
                elif key == "act.bit_width":
                    obj.bits = int(val)
                elif key == "act.signed":
                    obj.signed = (val == "True")

    return model


# ----------------------------------------------------------------------------
# stage 2: emit  (QuantModel -> per-layer FPGA .bin payload)
# ----------------------------------------------------------------------------
# LATENT RISK — PER-TENSOR vs PER-CHANNEL WEIGHT SCALES (read before shipping a new model)
# ---------------------------------------------------------------------------------------
# This tool was audited bit-exact against the POT4 `1ee51p3s` model, whose weight scales are
# PER-TENSOR (weight.scale is a single scalar per QuantLinear).  Two things ride on that and
# would SILENTLY MISCOMPUTE if a future model exports genuinely PER-CHANNEL (per-output-row)
# weight scales:
#
#   1. The shipping kernel consumes ONLY the 15 SCALAR `scales[]` (encoder_layer.cpp reads
#      scales[SCALE_Q_IDX] etc. — one scale per projection, applied to every output row).
#      The per-row `s_q_row`/`s1_row`/... blobs emit_layer writes are therefore DEAD (emitted
#      but never read by the shipping datapath).
#   2. `scalar_scales()` fills the per-projection scalar slots with per_row_fold(...).mean().
#      With per-tensor scales every row is identical, so mean == the exact value.  With
#      per-channel scales the mean would be applied uniformly to all rows -> WRONG requant.
#
# If a new model has per-channel weight scales you must either (a) make the kernel consume the
# per-row `s_*_row` blobs (they are already emitted, just wire them in), or (b) re-fold to a
# per-tensor scale in the SW export.  Do NOT rely on the .mean() shortcut.  (`_is_pot_model`
# and `weight.scale` length in --summary tell you which regime a file is in.)
# ----------------------------------------------------------------------------
# Fixed-point encodings (match the HLS types).
SCALE_FRAC = 16   # ap_fixed<32,16> requant scales (T_MhaQuant/T_MlpQuant/T_Scale)
GAMMA_BITS = 8    # T_RmsGamma = ap_int<8> (int8 code; gamma_scale folded into inv_out)

# The six weights: tag -> module suffix.  Per-row fold s_in/s_out resolved per layer.
WEIGHTS = ["w_q", "w_k", "w_v", "out_proj", "linear_1", "linear_2"]
WMODULE = {
    "w_q": "mha.w_q", "w_k": "mha.w_k", "w_v": "mha.w_v",
    "out_proj": "mha.out_proj", "linear_1": "mlp.linear_1", "linear_2": "mlp.linear_2",
}


class BlobWriter:
    """Appends arrays to ONE open binary file, returning a manifest descriptor
    {offset, bytes, count, dtype} for each blob so the runtime host can seek to it.
    Offsets are absolute byte positions in the single output file."""
    # numpy dtype -> stable string the host/manifest agree on.
    _DT = {"int8": "int8", "uint8": "uint8", "uint16": "uint16",
           "int16": "int16", "int32": "int32"}

    def __init__(self, fh):
        self.fh = fh
        self.offset = 0

    def put(self, arr):
        arr = np.ascontiguousarray(arr)
        raw = arr.tobytes()
        off = self.offset
        self.fh.write(raw)
        self.offset += len(raw)
        # normalize dtype label (strip endianness prefix; the file is little-endian).
        name = np.dtype(arr.dtype).name
        return {"offset": off, "bytes": len(raw), "count": int(arr.size),
                "dtype": self._DT.get(name, name)}


def fx(scale_vec, frac=SCALE_FRAC):
    """Encode a float array to ap_fixed<.,.> raw int32 (round(v * 2^frac))."""
    return np.round(np.asarray(scale_vec, dtype=np.float64) * (1 << frac)).astype("<i4")


# ----------------------------------------------------------------------------
# POT (power-of-two) encodings — for POT4 models (weights + scales are +/- 2^k).
# The kernel then applies weights AND requant as SHIFTS, not multiplies.
# ----------------------------------------------------------------------------
def _log2_exact(x):
    """log2(|x|) as an integer, asserting x is EXACTLY +/- a power of two (x != 0)."""
    a = np.abs(np.asarray(x, dtype=np.float64))
    l = np.log2(a)
    r = np.round(l)
    if not np.all(np.abs(l - r) < 1e-6):
        bad = a[np.abs(l - r) >= 1e-6][:4]
        raise ValueError(f"value(s) not power-of-two: {bad}")
    return r.astype(np.int64)


def pot_shift(scale_vec):
    """Encode a POT scale (2^s) as its SIGNED int8 shift amount s (real = 2^s).
    s>=0 -> left shift (scale>1), s<0 -> right shift (scale<1). Asserts POT."""
    s = _log2_exact(scale_vec)
    if np.any((s < -127) | (s > 127)):
        raise ValueError(f"POT shift out of int8 range: {s.min()}..{s.max()}")
    return s.astype("<i1")


def pot_weight_signexp(vals_int):
    """Encode POT weight ints (+/- 2^e) as one byte per nonzero: (sign<<7)|exponent,
    exponent = log2(|w|) in 0..6, sign 1=negative. Asserts every value is +/- POT, nonzero."""
    v = np.asarray(vals_int, dtype=np.int64)
    if np.any(v == 0):
        raise ValueError("pot_weight_signexp got a zero (CSR should only hold nonzeros)")
    exp = _log2_exact(v)                       # log2(|w|)
    if np.any((exp < 0) | (exp > 7)):
        raise ValueError(f"weight exponent out of 0..7: {exp.min()}..{exp.max()}")
    sign = (v < 0).astype(np.int64)
    return ((sign << 7) | exp).astype("<u1")   # uint8: bit7=sign, bits0..6=exponent


def dense_to_csr(wint):
    """CSR of an int matrix [rows, cols] (nonzeros only, row-major).
    Returns vals as int64 (the raw ints; caller encodes int8 or POT sign+exp)."""
    rows, cols = wint.shape
    vals, colidx = [], []
    rowptr = np.zeros(rows + 1, dtype=np.int64)
    for i in range(rows):
        nz = np.nonzero(wint[i])[0]
        colidx.append(nz.astype(np.uint16))
        vals.append(wint[i, nz].astype(np.int64))
        rowptr[i + 1] = rowptr[i] + nz.size
    vals = np.concatenate(vals) if vals else np.zeros(0, np.int64)
    colidx = np.concatenate(colidx) if colidx else np.zeros(0, np.uint16)
    return vals, colidx.astype("<u2"), rowptr.astype("<i4")


def dense_to_csc(wint):
    """CSC of an int matrix [rows, cols] (nonzeros only, COLUMN-major).

    Returns (vals, rowidx, colptr):
      vals   int64 : the raw nonzero ints, grouped by column (caller encodes int8 / POT).
      rowidx u16   : the ROW index of each nonzero (which output row uses this column).
      colptr i4    : length cols+1; colptr[c]..colptr[c+1] is column c's nonzero span.

    Used for the H-driven W2 stage (Y = W2.H reformulated H-first): the datapath walks a
    hidden column and needs "which output rows use it", i.e. W2 stored column-major.
    Same nonzeros/total as dense_to_csr, just grouped by column instead of row."""
    rows, cols = wint.shape
    vals, rowidx = [], []
    colptr = np.zeros(cols + 1, dtype=np.int64)
    for c in range(cols):
        nz = np.nonzero(wint[:, c])[0]           # ROW indices with a nonzero in column c
        rowidx.append(nz.astype(np.uint16))
        vals.append(wint[nz, c].astype(np.int64))
        colptr[c + 1] = colptr[c] + nz.size
    vals = np.concatenate(vals) if vals else np.zeros(0, np.int64)
    rowidx = np.concatenate(rowidx) if rowidx else np.zeros(0, np.uint16)
    return vals, rowidx.astype("<u2"), colptr.astype("<i4")


def _act_signed_scale(model, name):
    """Activation scale as the HW datapath sees it (SIGNED int8, [-128,127]).

    UNSIGNED quantizers (only mha.quant_out in this model) export a scale for a [0,255]
    range; brevitas computes it as amax/255, i.e. HALF the signed-equivalent step amax/127
    (~amax/128).  The HW attention output is stored/requantised as a SIGNED int8, so its
    effective scale is 2x the exported unsigned scale.  Doubling the unsigned scale here
    reconciles the txt with the SW RUNTIME (which uses quant_out signed) — verified: every
    signed act's txt scale already matches runtime; only the 12 unsigned quant_out were 2x off.

    NOTE: in the current `1ee51p3s` model EVERY act is signed (quant_out was made signed in the
    SW fix), so the 2x branch is inert here.  It is kept for older/unsigned-quant_out exports."""
    a = model.acts[name]
    return a.scale * 2.0 if not a.signed else a.scale


def per_row_fold(model, layer, tag):
    """s[row] = (weight.scale[row] * s_in) / s_out, resolved from the file.

    Returns one fold per output row.  NOTE: for a PER-TENSOR model (this one) weight.scale is
    a single scalar, so this returns a length-1 array (all rows equal).  For a per-channel
    model it returns len==out_rows and scalar_scales' .mean() is NOT valid — see the LATENT
    RISK block at the top of stage 2."""
    P = f"encoder.encoder_layer_{layer}."
    sW = model.linears[P + WMODULE[tag]].weight_scale   # per-row (scalar for a per-tensor model)
    NO = lambda n: model.norms[P + n].output_scale
    A = lambda n: _act_signed_scale(model, P + n)
    LI = lambda n: model.linears[P + n].input_scale
    dy = lambda n: model.dyadics[P + n]
    sZ, sX1 = NO("ln_mha"), NO("ln_mlp")
    if tag == "w_q":      s_in, s_out = sZ,            A("mha.quant_q")
    elif tag == "w_k":    s_in, s_out = sZ,            A("mha.quant_k")
    elif tag == "w_v":    s_in, s_out = sZ,            A("mha.quant_v")
    # out_proj / linear_2 requant to the residual BRANCH grid (branch_b_scale), NOT the
    # coarser residual OUTPUT grid (out_scale).  The residual's branch ratio is already
    # branch_b/out (see SCALE_*_BRANCH_RATIO in scalar_scales), so the branch must arrive
    # at branch_b_scale; folding to out_scale here dropped 3 bits of the attention branch
    # (res1: 8x coarser) and 1 bit of the MLP branch (res2: 2x), which compounded across
    # the 12 layers into wrong classifications (HW twin 1/8 -> 24/24 vs the SW model once
    # fixed).  The finer grid can overflow int8 on <0.01% outlier elements; the projection's
    # saturate handles those (SW likewise clamps only the residual sum).
    elif tag == "out_proj": s_in, s_out = A("mha.quant_out"), dy("res1").branch_b_scale
    elif tag == "linear_1": s_in, s_out = sX1,         LI("mlp.linear_2")
    elif tag == "linear_2": s_in, s_out = LI("mlp.linear_2"), dy("res2").branch_b_scale
    return sW * s_in / s_out


def scalar_scales(model, layer):
    """The 15 SCALE_*_IDX scalar folds (order matches enum EncoderScaleLUTIndex)."""
    P = f"encoder.encoder_layer_{layer}."
    NO = lambda n: model.norms[P + n].output_scale
    A = lambda n: _act_signed_scale(model, P + n)
    LI = lambda n: model.linears[P + n].input_scale
    dy = lambda n: model.dyadics[P + n]
    sZ, sX1 = NO("ln_mha"), NO("ln_mlp")
    # POT4 model: RMSNorm gamma is FUSED into the following linear (gamma_scale/int are None),
    # so there is NO gamma to fold. The RMSNorm output requant is just 1/OUT_SCALE.
    g_mha = model.norms[P + "ln_mha"].gamma_scale
    g_mlp = model.norms[P + "ln_mlp"].gamma_scale
    gamma_fused = (g_mha is None)
    if gamma_fused:
        g_mha = g_mlp = 1.0        # no gamma factor; RMSNORM_OUT_INV = 1/OUT_SCALE
    sV, sZhat = A("mha.quant_v"), A("mha.quant_out")
    sH = LI("mlp.linear_2")
    # Residual-add fold ratios.  The HW residual_add has NO per-branch align step, so each
    # ratio must map the branch's ACTUAL ARRIVING int8 scale straight to out_scale:
    #   - SKIP branch (sX): arrives at the scale the HW actually feeds in.  For res1 that is
    #     the layer's quant_input scale (== previous layer's res2.out, so codes chain); for
    #     res2 it is res1.out (the hidden the MLP block was given).  NOT branch_a_scale, which
    #     is the SW's post-align scale and disagrees with the arriving scale where the SW
    #     align_a rescales (e.g. L2 res1) -> corrupts everything downstream.
    #   - BRANCH (sM): the MHA/MLP output, at the dyadic branch_b scale.
    r1, r2 = dy("res1"), dy("res2")
    sX_mha, sOUT_mha, sM_mha = A("quant_input"), r1.out_scale, r1.branch_b_scale   # MHA block residual
    sX_mlp, sOUT_mlp, sM_mlp = r1.out_scale,      r2.out_scale, r2.branch_b_scale   # MLP block residual
    # SCALE_*_IDX order (see encoder_layer.h):
    return [
        # The shipping kernel uses THESE SCALAR per-projection slots (one scale per projection,
        # applied to every output row) — NOT the per-row s_*_row blobs, which are emitted but
        # dead.  .mean() is EXACT for a per-tensor model (all rows equal); for a per-channel
        # model it is WRONG.  See the LATENT RISK block at the top of stage 2.
        float(per_row_fold(model, layer, "w_q").mean()),      # SCALE_Q_IDX
        float(per_row_fold(model, layer, "w_k").mean()),      # SCALE_K_IDX
        float(per_row_fold(model, layer, "w_v").mean()),      # SCALE_V_IDX
        float(per_row_fold(model, layer, "out_proj").mean()), # SCALE_LINEAR_OUT_IDX
        sV / sZhat,                                           # SCALE_DIV_OUT_IDX
        sZ,                                                   # SCALE_ATT_RMSNORM_IN_IDX
        g_mha / sZ,                                           # SCALE_ATT_RMSNORM_OUT_INV_IDX (gamma_scale folded)
        sX_mha / sOUT_mha,                                    # SCALE_ATT_RESIDUAL_IDX
        sM_mha / sOUT_mha,                                    # SCALE_ATT_BRANCH_RATIO_IDX
        float(per_row_fold(model, layer, "linear_1").mean()), # SCALE_FC1_IDX  (.mean(): see risk note above)
        float(per_row_fold(model, layer, "linear_2").mean()), # SCALE_FC2_IDX  (.mean(): see risk note above)
        sX1,                                                 # SCALE_FF_RMSNORM_IN_IDX
        g_mlp / sX1,                                          # SCALE_FF_RMSNORM_OUT_INV_IDX (gamma_scale folded)
        sX_mlp / sOUT_mlp,                                    # SCALE_FF_RESIDUAL_IDX
        sM_mlp / sOUT_mlp,                                    # SCALE_FF_BRANCH_RATIO_IDX
    ]


def emit_gamma(model, layer, ln, bw, verify):
    # Ship gamma as the model's INT8 code (T_RmsGamma = ap_int<8>).  The per-tensor
    # gamma_scale is NOT applied here — it is folded on the host into that block's
    # RMSNORM_OUT_INV scale (see scalar_scales), so the kernel reconciles it via the
    # existing inv_out multiply.  This halves gamma's DDR footprint (1 B/elem) and
    # keeps the model's true precision (no re-gridding to a wider fixed-point grid).
    n = model.norms[f"encoder.encoder_layer_{layer}.{ln}"]
    q = np.clip(n.gamma_int, -128, 127).astype("<i1")         # int8 code
    entry = bw.put(q)
    entry["abs_max"] = float(np.abs(n.gamma_int.astype(np.float64) * n.gamma_scale).max())
    if verify:
        entry["saturated"] = bool((np.abs(n.gamma_int) > 127).any())   # int8 range guard
    return entry


def emit_layer(model, layer, bw, verify, pot=False, fused=None, pot_scales=None):
    """Append one layer's blobs to the single-file writer `bw`; return its manifest.

    pot=True (POT4 model): WEIGHTS are emitted as one sign+exponent byte per nonzero
    (the shift-MAC kernel shifts instead of multiplying). pot=False keeps int8 weights.

    pot_scales: whether SCALES are emitted as signed int8 shift amounts (True) or as
    ap_fixed<32,16> words (False).  Defaults to `pot`.  The HYBRID config (shift-MAC MAC
    but the shipping MULTIPLY requant) needs pot=True + pot_scales=False: POT sign+exp
    weights that fit int4, and ap_fixed scale VALUES the multiply requant can consume.

    fused: whether gammas are fused into the linears (so none are emitted). Defaults to
    `pot` (POT4 always fuses)."""
    if fused is None:
        fused = pot
    if pot_scales is None:
        pot_scales = pot
    P = f"encoder.encoder_layer_{layer}."
    ld = {"weights": {}, "scale_rows": {}, "gammas": {}}
    fold_names = {"w_q": "s_q_row", "w_k": "s_k_row", "w_v": "s_v_row",
                  "out_proj": "s_o_row", "linear_1": "s1_row", "linear_2": "s2_row"}
    for tag in WEIGHTS:
        wi = model.linears[P + WMODULE[tag]].weight_int.astype(np.int64)
        # linear_2 (W2) is emitted COLUMN-major (CSC) for the H-driven stage-2 datapath
        # (Y = W2.H reformulated to walk H sequentially, scattering through W2 columns).
        # The manifest keeps the same (values/col/rowptr) key names, but for CSC they carry
        # (values, ROW index per nonzero, COLUMN pointer) -- the HW reads the triple as a
        # column-major CSR.  All other weights stay row-major CSR.
        csc = (tag == "linear_2")
        if csc:
            vals, idx, ptr = dense_to_csc(wi)   # idx = row index, ptr = column pointer
        else:
            vals, idx, ptr = dense_to_csr(wi)   # idx = col index, ptr = row pointer
        val_blob = bw.put(pot_weight_signexp(vals)) if pot else bw.put(vals.astype(np.int8))
        ent = {"shape": list(wi.shape), "nnz": int(vals.size),
               "max_row_nnz": int(np.diff(ptr.astype(np.int64)).max()),  # CSC: max COLUMN nnz
               "layout": "csc" if csc else "csr",
               "values": val_blob,
               "col": bw.put(idx),
               "rowptr": bw.put(ptr)}
        if verify:
            recon = np.zeros_like(wi)
            if csc:
                for c in range(wi.shape[1]):
                    s, e = int(ptr[c]), int(ptr[c + 1])
                    recon[idx[s:e].astype(np.int64), c] = vals[s:e]
            else:
                for i in range(wi.shape[0]):
                    s, e = int(ptr[i]), int(ptr[i + 1])
                    recon[i, idx[s:e].astype(np.int64)] = vals[s:e]
            ent["csr_exact"] = bool(np.array_equal(recon, wi))
        ld["weights"][tag] = ent
        # per-row folded scale: POT shift amount (int8) or ap_fixed word
        srow = per_row_fold(model, layer, tag)
        d = bw.put(pot_shift(srow)) if pot_scales else bw.put(fx(srow))
        d["range"] = [float(srow.min()), float(srow.max())]
        if pot_scales:
            d["shift"] = [int(pot_shift(srow).min()), int(pot_shift(srow).max())]
        ld["scale_rows"][fold_names[tag]] = d
    # --- 15 scalar scales: POT shift amounts (int8) or ap_fixed words ---
    sc = scalar_scales(model, layer)
    d = bw.put(pot_shift(sc)) if pot_scales else bw.put(fx(sc))
    d["values_float"] = [float(x) for x in sc]
    if pot_scales:
        d["shift"] = [int(x) for x in pot_shift(sc)]
    ld["scales"] = d
    # --- gammas: only when NOT fused (fused models fold gamma into the linears) ---
    if not fused:
        ld["gammas"]["gamma_mha"] = emit_gamma(model, layer, "ln_mha", bw, verify)
        ld["gammas"]["gamma_mlp"] = emit_gamma(model, layer, "ln_mlp", bw, verify)
    return ld


def _is_pot_model(model):
    """POT4 model = gammas fused away (gamma_int None) AND scale type po2. Auto-detected."""
    scale_ty = model.meta.get("scale type", "")
    any_norm = next(iter(model.norms.values()), None)
    gamma_fused = (any_norm is not None and any_norm.gamma_int is None)
    return gamma_fused and ("weight=po2" in scale_ty)


def emit_model(model, layers, out_dir, verify=False, pot=None, fused=None, pot_scales=None):
    """Emit ONE concatenated model.bin (all layers, all blobs) + manifest.json.

    pot: None -> auto-detect POT4 (fused gamma + po2 scales); True/False to force. POT
    WEIGHTS are one sign+exp byte/nonzero (shift-MAC kernel).
    pot_scales: None -> defaults to `pot`. SCALES as int8 shift amounts (True) or ap_fixed
    words (False). HYBRID (shift-MAC + shipping multiply requant) = pot=True, pot_scales=False.
    fused: None -> defaults to `pot`; gammas emitted only when NOT fused.
    Every blob descriptor carries an absolute byte {offset,bytes,count,dtype}."""
    if pot is None:
        pot = _is_pot_model(model)
    if pot_scales is None:
        pot_scales = pot
    if fused is None:
        fused = pot
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, "model.bin")
    manifest = {
        "source": model.meta.get("source", "quant_values.txt"),
        "bin": "model.bin",
        "endian": "little",
        "format": "pot" if pot else "int",
        "w_bits": 4, "scale_frac": SCALE_FRAC, "gamma_bits": GAMMA_BITS,
        "note_format": ("POT: weight blob = uint8 (bit7=sign, bits0..6=log2|w|); scale "
                        "blobs = int8 shift s (real=2^s); NO gammas (fused into linears)."
                        if pot else
                        "int: weight blob = int8; scale blobs = ap_fixed<32,16> int32; "
                        "int8 gammas per block."),
        "note_offsets": "each blob descriptor {offset,bytes,count,dtype} is an absolute "
                        "position into model.bin.",
        "note_bias": "biases ignored (unused).",
        "layers": {},
    }
    with open(bin_path, "wb") as fh:
        bw = BlobWriter(fh)
        for layer in layers:
            ld = emit_layer(model, layer, bw, verify, pot=pot, fused=fused, pot_scales=pot_scales)
            manifest["layers"][str(layer)] = ld
            w = ld["weights"]
            vtxt = ""
            if verify:
                allok = all(w[t].get("csr_exact") for t in w)
                vtxt = f"  csr_exact={allok}"
            gtxt = "no gammas (fused)" if fused else "2 gammas"
            print(f"  L{layer:02d}: 6 weights (maxrow "
                  f"{max(w[t]['max_row_nnz'] for t in w)}) + 6 scale_rows + scales + {gtxt}{vtxt}")
        total = bw.offset
    manifest["total_bytes"] = total
    manifest["format_detected"] = "pot" if pot else "int"
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  model.bin = {total} bytes ({total/1e6:.1f} MB)")
    return manifest


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _summary(model):
    print(f"linears : {len(model.linears)}")
    print(f"norms   : {len(model.norms)}")
    print(f"dyadics : {len(model.dyadics)}")
    print(f"acts    : {len(model.acts)}")
    print(f"encoder layers present: {model.layers()}")
    print()
    L0 = "encoder.encoder_layer_0."
    print("layer-0 activation scales (the fold ingredients):")
    for nm in sorted(model.acts):
        if nm.startswith(L0):
            print(f"  ACT  {nm:52s} = {model.acts[nm].scale:.9f}")
    for nm in sorted(model.norms):
        if nm.startswith(L0):
            print(f"  NORM {nm:52s} out={model.norms[nm].output_scale:.9f}")
    for nm in sorted(model.linears):
        if nm.startswith(L0):
            lin = model.linears[nm]
            ws = lin.weight_scale
            print(f"  LIN  {nm:52s} in={lin.input_scale:.9f} "
                  f"wscale[{len(ws)}] range[{ws.min():.5f},{ws.max():.5f}] "
                  f"wint{lin.weight_int.shape} nz={int((lin.weight_int != 0).sum())}")


def _module(model, nm):
    if nm in model.linears:
        lin = model.linears[nm]
        print(f"LINEAR {nm}: shape={lin.shape} wbits={lin.weight_bits}")
        print(f"  weight.int {lin.weight_int.shape} dtype={lin.weight_int.dtype} "
              f"range[{lin.weight_int.min()},{lin.weight_int.max()}] "
              f"nz={int((lin.weight_int != 0).sum())}")
        print(f"  weight.scale PER-ROW len={len(lin.weight_scale)} "
              f"range[{lin.weight_scale.min():.6f},{lin.weight_scale.max():.6f}]")
        print(f"  input.scale={lin.input_scale}")
    elif nm in model.norms:
        n = model.norms[nm]
        print(f"NORM {nm}: gamma_int len={len(n.gamma_int)} "
              f"scale={n.gamma_scale} output.scale={n.output_scale}")
    elif nm in model.dyadics:
        print(f"DYADIC {nm}: {model.dyadics[nm]}")
    elif nm in model.acts:
        print(f"ACT {nm}: scale={model.acts[nm].scale}")
    else:
        raise SystemExit(f"module not found: {nm}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", required=True, help="path to quant_values.txt")
    ap.add_argument("--out", default="out/model",
                    help="output directory for model.bin + manifest.json (default out/model)")
    ap.add_argument("--layers", default="all",
                    help="'all' or comma list of encoder layer indices to emit (default all)")
    ap.add_argument("--verify", action="store_true",
                    help="cross-check CSR reconstruction + gamma round-trip during emit")
    ap.add_argument("--summary", action="store_true",
                    help="inspect only: print a structural summary + layer-0 scale table")
    ap.add_argument("--module", default=None,
                    help="inspect only: print one module's parsed contents")
    args = ap.parse_args()

    print(f"Reading {args.input} ...", file=sys.stderr)
    model = compile_model(args.input)
    model.meta["source"] = os.path.basename(args.input)

    # inspection modes short-circuit the emit.
    if args.module:
        _module(model, args.module)
        return
    if args.summary:
        _summary(model)
        return

    all_layers = model.layers()
    layers = (all_layers if args.layers == "all"
              else [int(x) for x in args.layers.split(",") if x.strip()])
    emit_model(model, layers, args.out, args.verify)
    print(f"\nWrote {len(layers)} layer(s) to {args.out}/", file=sys.stderr)


if __name__ == "__main__":
    main()
