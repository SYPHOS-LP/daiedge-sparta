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
                                    [--scales auto|fixed|shift] [--model-index]
                                    [--permute none|random] [--seed S]

    # the shipping kernel's HYBRID payload (POT weights + ap_fixed scales) + model_index.bin:
    python scripts/compile_model.py -i <path> --out sim/e2e --scales fixed --model-index

    # the same model in a randomly permuted feature basis (see apply_permutations):
    python scripts/compile_model.py -i <path> --out sim/e2e/perm --scales fixed --model-index \
                                    --permute random --seed 1

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
import struct
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
# Weights that can be emitted per column block for the tiled SpMM engine (see
# dense_to_tiled_csr): the four MHA projections, selected by --mha-tiles, and the two MLP
# weights, selected by --mlp-tiles.
MHA_TILED_WEIGHTS = ["w_q", "w_k", "w_v", "out_proj"]
MLP_TILED_WEIGHTS = ["linear_1", "linear_2"]
# Weight entries per kernel memory word (MHA_TILE_PACK in mha_cfg.h): 4 entries of
# {8-bit block-local column, 4-bit value} in a 16-bit lane fill a 64-bit word, so one bank is
# one URAM (72-bit words) and each load reads whole words.
TILE_PACK = 4
# Bytes per transfer when the kernel reads the nonzero counts (MHA_COUNT_PACK in mha_cfg.h): the
# counts are one byte each, so one transfer carries this many of them.
COUNT_WORD_BYTES = 8
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


def pot_weight_signexp(vals_int, allow_zero=False):
    """Encode POT weight ints (+/- 2^e) as one byte per nonzero: (sign<<7)|exponent,
    exponent = log2(|w|) in 0..6, sign 1=negative. Asserts every value is +/- POT, nonzero.

    allow_zero=True permits zeros, which a tiled emit uses to pad a block out to a whole
    memory word; they are encoded as byte 0 and never read back (a row's span comes from the
    per-row counts, which exclude padding)."""
    v = np.asarray(vals_int, dtype=np.int64)
    padding = (v == 0)
    if padding.any() and not allow_zero:
        raise ValueError("pot_weight_signexp got a zero (CSR should only hold nonzeros)")
    exp = _log2_exact(np.where(padding, 1, v))   # padding slots stand in as 2^0
    if np.any((exp < 0) | (exp > 7)):
        raise ValueError(f"weight exponent out of 0..7: {exp.min()}..{exp.max()}")
    sign = (v < 0).astype(np.int64)
    encoded = ((sign << 7) | exp).astype("<u1")   # uint8: bit7=sign, bits0..6=exponent
    encoded[padding] = 0
    return encoded


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


def pack_tiled_bank(vals, indices, pot, pack=TILE_PACK, entry_bits=16):
    """Pack tiled weight entries into the words the kernel copies into its banks.

    An entry is `entry_bits` wide: the value in the low bits, then the 8-bit block-local column.
    The value field follows the kernel's datapath, and so does the column's position:

    ==========  ===========  =============  ====================================================
    `pot`       value field  column at bit  datapath
    ==========  ===========  =============  ====================================================
    True        4 bits       4              {sign, exponent}; the MAC shifts the activation
    False       8 bits       8              plain integer; the MAC multiplies on a DSP
    ==========  ===========  =============  ====================================================

    Producing this layout here makes the on-chip load a plain copy instead of passes that unpack,
    re-encode and merge.

    Parameters
    ----------
    vals : ndarray of int
        Nonzero weight values in bank order (output of dense_to_tiled_csr), zeros for padding.
    indices : ndarray of int
        Block-local column of each value, same order and length as `vals`.
    pot : bool
        True for a power-of-two model: the value is encoded as {sign in bit 3, exponent in bits
        0..2}, matching the shift-MAC.  False encodes the plain int4 weight in two's complement.
    pack : int, optional
        Entries per word.
    entry_bits : int, optional
        Bits per entry slot.

    Returns
    -------
    ndarray of uint64, shape (len(vals) // pack,)
        One word per `pack` entries, in the same order as `vals`.

    Raises
    ------
    ValueError
        If `len(vals)` is not a multiple of `pack`, or a column does not fit 6 bits.
    """
    v = np.asarray(vals, dtype=np.int64)
    c = np.asarray(indices, dtype=np.int64)
    if v.size % pack:
        raise ValueError(f"{v.size} entries is not a multiple of pack={pack}")
    if c.size and (c.min() < 0 or c.max() > 0xFF):
        raise ValueError(f"block-local column out of 0..255: {c.min()}..{c.max()}")

    if pot:
        code    = pot_weight_signexp(v, allow_zero=True).astype(np.int64)
        code    = ((code >> 7) << 3) | (code & 0x7)   # {sign bit7, exp bits0..6} -> {sign bit3, exp bits0..2}
        col_lsb = 4
    else:
        # these weights reach +/-64, so the plain-integer field is 8 bits, not 4
        if v.size and (v.min() < -128 or v.max() > 127):
            raise ValueError(f"weight out of int8 range: {v.min()}..{v.max()}")
        code    = v & 0xFF
        col_lsb = 8

    entry = (code | (c << col_lsb)).astype(np.uint64)
    entry = entry.reshape(-1, pack)
    word = np.zeros(entry.shape[0], dtype="<u8")
    for lane in range(pack):
        word |= entry[:, lane] << np.uint64(lane * entry_bits)
    return word


def dense_to_tiled_csr(wint, n_tiles, pad_to=1):
    """Split a weight into `n_tiles` column blocks and return each block as a CSR matrix.

    Column `j` of the weight multiplies row `j` of the activation, so a block of columns only
    ever reads its own block of activation rows.  The kernel keeps each activation block in a
    separate on-chip memory and gives each block of the weight its own bank, so one lane per
    block can read a nonzero in the same cycle.

    Parameters
    ----------
    wint : ndarray, shape (rows, cols)
        Dense weight matrix. Only its nonzeros are kept.
    n_tiles : int
        Number of column blocks. Must divide `cols`.
    pad_to : int, optional
        Pad each block with zero entries so its length is a multiple of this. The kernel packs
        `pad_to` entries per memory word, so padded blocks start on a word boundary and load
        with one read per word. Padding is never read back: a row's span comes from `indptr`.

    Returns
    -------
    data : ndarray of int64, shape (nnz,)
        Nonzero values: all of block 0 first, then block 1, and so on. Within a block the
        values are in row order, as CSR requires, followed by any padding.
    indices : ndarray of uint16, shape (nnz,)
        Column of each value inside its own block, so 0 .. cols // n_tiles - 1.
    indptr : ndarray of int32, shape (n_tiles, rows + 1)
        One CSR row-pointer array per block. Row `r` of block `t` is the slice
        `indptr[t, r] : indptr[t, r + 1]` of that block's values, and block `t` starts at
        `sum(indptr[u, rows] for u < t)` in `data` / `indices`.

    Raises
    ------
    ValueError
        If `n_tiles` does not divide the number of columns.
    """
    rows, cols = wint.shape
    if cols % n_tiles:
        raise ValueError(f"n_tiles={n_tiles} does not divide {cols} columns")
    width = cols // n_tiles

    data, indices, indptr = [], [], []
    for t in range(n_tiles):
        block = wint[:, t * width:(t + 1) * width]
        row, col = np.nonzero(block)            # row-major order == CSR order
        pad = -row.size % pad_to
        data.append(np.concatenate([block[row, col], np.zeros(pad, block.dtype)]))
        indices.append(np.concatenate([col, np.zeros(pad, col.dtype)]))
        indptr.append(np.concatenate(([0], np.cumsum(np.bincount(row, minlength=rows)))))

    return (np.concatenate(data).astype(np.int64),
            np.concatenate(indices).astype("<u2"),
            np.stack(indptr).astype("<i4"))


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


def emit_layer(model, layer, bw, verify, pot=False, fused=None, pot_scales=None, tiles=None,
               pot_weights=None):
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
    if pot_weights is None:
        pot_weights = pot
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
        n_tiles = (tiles or {}).get(tag, 0)
        tiled = n_tiles > 0
        csc = (tag == "linear_2" and not tiled)
        if tiled:
            # One CSR per column block.  The "rowptr" slot carries each block's per-row nonzero
            # counts (block-major), which is what the kernel needs to advance one cursor per block.
            vals, idx, indptr = dense_to_tiled_csr(wi, n_tiles, pad_to=TILE_PACK)
            ptr = np.diff(indptr, axis=1).astype("<i4").ravel()
        elif csc:
            vals, idx, ptr = dense_to_csc(wi)   # idx = row index, ptr = column pointer
        else:
            vals, idx, ptr = dense_to_csr(wi)   # idx = col index, ptr = row pointer
        ent = {"shape": list(wi.shape), "nnz": int(vals.size),
               "max_row_nnz": int(ptr.max() if tiled else np.diff(ptr.astype(np.int64)).max()),
               "layout": "tiled_bank" if tiled else ("csc" if csc else "csr")}
        if tiled:
            # Values and columns in one pre-packed array; "counts" holds the per-(block, row)
            # nonzero counts the kernel walks the rows with.
            # counts blob = [n_tiles first-word offsets, int32, padded to COUNT_WORD_BYTES]
            #               [one byte per (column block, row) nonzero count]
            # The offsets are prefix sums of each block's padded length, which the kernel would
            # otherwise have to recompute by reading every count of the weight.  The counts fit a
            # byte (a row holds at most the block's width), and the kernel reads them
            # COUNT_WORD_BYTES at a time, so a row block's counts cost n_tiles * rows / 16 reads
            # instead of one per count.
            block_words = np.cumsum(
                np.concatenate(([0], (indptr[:, -1] + TILE_PACK - 1) // TILE_PACK)))[:n_tiles]
            if ptr.max() > 255:
                raise ValueError(f"nonzero count {ptr.max()} does not fit a byte")
            header = block_words.astype("<i4").tobytes()
            header += b"\x00" * (-len(header) % COUNT_WORD_BYTES)
            body = ptr.astype("<u1").tobytes()
            body += b"\x00" * (-len(body) % COUNT_WORD_BYTES)
            ent["bank"]   = bw.put(pack_tiled_bank(vals, idx, pot_weights))
            ent["counts"] = bw.put(np.frombuffer(header + body, dtype="<u1"))
            ent["block_words"] = [int(v) for v in block_words]
            ent["counts_header_bytes"] = len(header)
        else:
            ent["values"] = (bw.put(pot_weight_signexp(vals)) if pot_weights
                             else bw.put(vals.astype(np.int8)))
            ent["col"]    = bw.put(idx)
            ent["rowptr"] = bw.put(ptr)
        if tiled:
            ent["n_tiles"] = n_tiles
            ent["tile_nnz"] = [int(n) for n in indptr[:, -1]]   # nonzeros per block
        if verify:
            recon = np.zeros_like(wi)
            if tiled:
                width, pos = wi.shape[1] // n_tiles, 0
                for t in range(n_tiles):
                    for r in range(wi.shape[0]):
                        n = int(ptr[t * wi.shape[0] + r])
                        recon[r, idx[pos:pos + n].astype(np.int64) + t * width] = vals[pos:pos + n]
                        pos += n
                    pos += -pos % TILE_PACK          # skip this block's padding
            elif csc:
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


def emit_model(model, layers, out_dir, verify=False, pot=None, fused=None, pot_scales=None,
               perms=None, tiles=None, pot_weights=None):
    """Emit ONE concatenated model.bin (all layers, all blobs) + manifest.json.

    perms: the permutations already applied to `model` (apply_permutations), if any. They are
    saved next to model.bin and recorded in the manifest so the host can permute the encoder
    input / un-permute its output.

    pot: None -> auto-detect POT4 (fused gamma + po2 scales); True/False to force.  It selects
    gamma fusion and the defaults of the two switches below.
    pot_weights: None -> defaults to `pot`.  True encodes the weight values as {sign, exponent}
    for the kernel's shift datapath, False as plain integers for its DSP-multiply datapath.  The
    model's numbers are the same either way, only their encoding differs.
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
    if pot_weights is None:
        pot_weights = pot
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, "model.bin")
    manifest = {
        "source": model.meta.get("source", "quant_values.txt"),
        "bin": "model.bin",
        "endian": "little",
        "format": "pot" if pot_weights else "int",
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
            ld = emit_layer(model, layer, bw, verify, pot=pot, fused=fused, pot_scales=pot_scales,
                            tiles=tiles, pot_weights=pot_weights)
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
    if perms is not None:
        np.savez(os.path.join(out_dir, PERM_FILE), **perms)
        manifest["permutation"] = {
            "file": PERM_FILE,
            "convention": "perm[old] = new; X_perm = X[inverse(pi)] at the encoder input, "
                          "Y = Y_perm[pi] at the encoder output",
        }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  model.bin = {total} bytes ({total/1e6:.1f} MB)")
    return manifest


MODEL_INDEX_MAGIC = 0x45324D49   # 'IM2E', checked by tb/encoder_e2e_tb.cpp


def emit_model_index(manifest, out_dir, n_heads=12, entry_format=0):
    """Write model_index.bin for tb/encoder_e2e_tb.cpp from an emitted manifest.

    Layout: magic + (n_layers, D, d_h, n_heads, entry_format) + per layer, per weight in WEIGHTS
    order, where entry_format is 0 for power-of-two values (the shift datapath) and 1 for plain
    integers (the multiply datapath); the kernel's testbench checks it against its own build.
    Then
    (nnz, n_ptr) where n_ptr = pointer-array length - 1: rows for CSR, columns for CSC, and
    n_tiles * (rows + 1) for tiled_bank, whose count array holds n_tiles block offsets followed by
    one count per (block, row)."""
    ids = sorted(int(k) for k in manifest["layers"])
    w0 = manifest["layers"][str(ids[0])]["weights"]
    d_model, d_hidden = int(w0["w_q"]["shape"][1]), int(w0["linear_1"]["shape"][0])
    path = os.path.join(out_dir, "model_index.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<I", MODEL_INDEX_MAGIC))
        f.write(struct.pack("<5i", len(ids), d_model, d_hidden, n_heads, entry_format))
        for L in ids:
            for tag in WEIGHTS:
                w = manifest["layers"][str(L)]["weights"][tag]
                layout = w.get("layout")
                if layout == "tiled_bank":
                    # the count array carries n_tiles offsets in front of n_tiles * rows counts
                    n_ptr = int(w["n_tiles"]) * (int(w["shape"][0]) + 1) - 1
                elif layout == "csc":
                    n_ptr = int(w["shape"][1])
                else:
                    n_ptr = int(w["shape"][0])
                f.write(struct.pack("<2i", int(w["nnz"]), n_ptr))
    return path


# ----------------------------------------------------------------------------
# Feature permutations: run the encoder in a reordered feature basis, bit-exactly.
# ----------------------------------------------------------------------------
# RMSNorm (gamma fused) normalizes each token over ALL its features, so it is invariant to their
# order; every other encoder operation is elementwise or a linear map, computed in integer /
# fixed-point with order-independent sums.  Reordering features consistently therefore leaves
# every value unchanged, just stored at a different index.  Four independent orders:
#
#   pi     (D,)                  residual stream, shared by ALL layers: columns of w_q/w_k/w_v/
#                                linear_1, rows of out_proj/linear_2.  The host applies it at the
#                                encoder boundary (input permuted, output un-permuted).
#   sigma  (layers, d_h)         MLP hidden features: rows of linear_1, columns of linear_2.
#   rho    (layers, heads, d_hd) Q'/K' features within each head band: rows of w_q AND w_k.
#   nu     (layers, heads, d_hd) V/Zhat features within each head band: rows of w_v, cols of out_proj.
#
# Convention: perm[old] = new.  Head permutations stay inside their band, so the attention's
# per-head split is untouched.
PERM_FILE = "permutations.npz"


def _inverse(perm):
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


def _band_perm(local):
    """(heads, d_head) per-band permutations -> one (heads*d_head,) permutation."""
    heads, d_head = local.shape
    return (np.arange(heads)[:, None] * d_head + local).reshape(-1)


def balanced_order(load, n_bins):
    """Order items so that `n_bins` equal-sized, contiguous bins carry a similar total load.

    The tiled SpMM gives each column block one lane, and the lanes wait for the busiest block,
    so the emitted column order should spread the nonzeros evenly. Heaviest item first into the
    lightest bin that still has room.

    Parameters
    ----------
    load : ndarray, shape (n_items,)
        Weight of each item, e.g. the nonzeros in each column.
    n_bins : int
        Number of bins. Must divide `n_items`.

    Returns
    -------
    perm : ndarray of int, shape (n_items,)
        New position of each item, i.e. `perm[old] = new`, with bin `b` holding the positions
        `b * n_items // n_bins` onwards.
    """
    load = np.asarray(load, dtype=float)
    n_items = load.size
    capacity = n_items // n_bins
    totals, filled = np.zeros(n_bins), np.zeros(n_bins, dtype=int)
    members = [[] for _ in range(n_bins)]
    for item in np.argsort(-load, kind="stable"):
        open_bins = np.flatnonzero(filled < capacity)
        b = open_bins[np.argmin(totals[open_bins])]
        members[b].append(item)
        filled[b] += 1
        totals[b] += load[item]
    perm = np.empty(n_items, dtype=int)
    for b, items in enumerate(members):
        perm[np.asarray(items)] = b * capacity + np.arange(len(items))
    return perm


def balanced_permutations(model, n_tiles, n_heads=12):
    """Permutations that even out the per-column-block load of the tiled weights.

    Same four orders as `random_permutations` (see `apply_permutations`), but chosen rather than
    random: the residual-stream order balances the columns of w_q/w_k/w_v/linear_1 summed over
    every layer, and the hidden order balances linear_1's rows against linear_2's columns.

    Parameters
    ----------
    model : QuantModel
        Parsed model; every encoder layer is balanced together for the shared stream order.
    n_tiles : int
        Number of column blocks the kernel will use.
    n_heads : int, optional
        Attention heads, for the per-head orders.

    Returns
    -------
    dict
        Same layout as `random_permutations`: layers, pi, sigma, rho, nu.
    """
    layers = model.layers()
    nnz_cols = lambda lin: (lin.weight_int != 0).sum(0) / (lin.weight_int != 0).sum()
    nnz_rows = lambda lin: (lin.weight_int != 0).sum(1) / (lin.weight_int != 0).sum()
    lin = lambda layer, tag: model.linears[f"encoder.encoder_layer_{layer}." + WMODULE[tag]]

    stream = sum(nnz_cols(lin(L, t)) for L in layers for t in ("w_q", "w_k", "w_v", "linear_1"))
    stream += sum(nnz_rows(lin(L, t)) for L in layers for t in ("out_proj", "linear_2"))
    pi = balanced_order(stream, n_tiles)

    d_head = lin(layers[0], "w_q").shape[1] // n_heads
    identity_band = np.tile(np.arange(d_head), (n_heads, 1))
    sigma = [balanced_order(nnz_rows(lin(L, "linear_1")) + nnz_cols(lin(L, "linear_2")), n_tiles)
             for L in layers]
    return {"layers": np.asarray(layers), "pi": pi, "sigma": np.stack(sigma),
            "rho": np.stack([identity_band] * len(layers)),
            "nu": np.stack([identity_band] * len(layers))}


def _fit_group_cap(row_nnz, perm, group, cap, rng):
    """Repair a row permutation so every `group` consecutive rows (in the new order) hold at
    most `cap` nonzeros, by swapping rows between the heaviest and lightest groups.

    The kernel's MLP stage 1 bursts W1 `group` rows at a time into a fixed-size buffer
    (MLP_W1_GROUP_NNZ_MAX / MLP_ROW_PARALLEL in src/layers/mlp.cpp); an arbitrary hidden
    order can overflow it."""
    order = _inverse(perm)                          # order[new] = old
    n_groups = len(order) // group
    for _ in range(100 * len(order)):
        loads = row_nnz[order[:n_groups * group]].reshape(n_groups, group).sum(1)
        heavy = int(loads.argmax())
        if loads[heavy] <= cap:
            return _inverse(order)
        light = int(loads.argmin())
        h = heavy * group + int(row_nnz[order[heavy * group:(heavy + 1) * group]].argmax())
        l = light * group + int(rng.integers(group))
        order[h], order[l] = order[l], order[h]
    raise ValueError(f"could not fit W1 groups of {group} rows under {cap} nonzeros")


def random_permutations(model, seed, n_heads=12, w1_group_cap=None, w1_group=4):
    """Random pi/sigma/rho/nu for every encoder layer in `model`.

    w1_group_cap: if set, each layer's sigma is repaired so no `w1_group` consecutive W1 rows
    exceed it (see _fit_group_cap)."""
    rng = np.random.default_rng(seed)
    layers = model.layers()
    P0 = f"encoder.encoder_layer_{layers[0]}."
    d_model = model.linears[P0 + WMODULE["w_q"]].shape[1]
    d_hidden = model.linears[P0 + WMODULE["linear_1"]].shape[0]
    d_head = d_model // n_heads
    band = lambda: np.stack([rng.permutation(d_head) for _ in range(n_heads)])
    sigma = []
    for layer in layers:
        s = rng.permutation(d_hidden)
        if w1_group_cap is not None:
            w1 = model.linears[f"encoder.encoder_layer_{layer}." + WMODULE["linear_1"]].weight_int
            s = _fit_group_cap((w1 != 0).sum(1), s, w1_group, w1_group_cap, rng)
        sigma.append(s)
    return {
        "layers": np.asarray(layers),
        "pi": rng.permutation(d_model),
        "sigma": np.stack(sigma),
        "rho": np.stack([band() for _ in layers]),
        "nu": np.stack([band() for _ in layers]),
    }


def apply_permutations(model, perms):
    """Reorder every encoder weight (in place) into the basis given by `perms`."""
    pi = perms["pi"]
    for i, layer in enumerate(perms["layers"]):
        P = f"encoder.encoder_layer_{int(layer)}."
        rho, nu, sigma = _band_perm(perms["rho"][i]), _band_perm(perms["nu"][i]), perms["sigma"][i]
        row_col = {"w_q": (rho, pi), "w_k": (rho, pi), "w_v": (nu, pi),
                   "out_proj": (pi, nu), "linear_1": (sigma, pi), "linear_2": (pi, sigma)}
        for tag, (row_perm, col_perm) in row_col.items():
            lin = model.linears[P + WMODULE[tag]]
            inv_row = _inverse(row_perm)
            # W_new[row_perm[r], col_perm[c]] = W[r, c]
            lin.weight_int = lin.weight_int[np.ix_(inv_row, _inverse(col_perm))]
            if lin.weight_scale is not None and lin.weight_scale.size > 1:   # per-row scales
                lin.weight_scale = lin.weight_scale[inv_row]


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
    ap.add_argument("--scales", choices=["auto", "fixed", "shift"], default="auto",
                    help="scale blob format: 'fixed' = ap_fixed<32,16> (the shipping kernel), "
                         "'shift' = int8 POT shifts, 'auto' = shift for POT models, else fixed")
    ap.add_argument("--model-index", action="store_true",
                    help="also write model_index.bin (for tb/encoder_e2e_tb.cpp)")
    ap.add_argument("--permute", choices=["none", "random", "balanced"], default="none",
                    help="emit the model in a permuted feature basis (saves permutations.npz): "
                         "'balanced' evens out the per-column-block load for the tiled engine, "
                         "'random' is the bit-exactness test")
    ap.add_argument("--seed", type=int, default=0, help="seed for --permute random")
    ap.add_argument("--mha-tiles", type=int, default=0,
                    help="emit the MHA projection weights as this many per-column-block CSRs "
                         "(0 = one row-major CSR, the original layout)")
    ap.add_argument("--weights", choices=["auto", "pot", "int"], default="auto",
                    help="weight values in the tiled payload: 'pot' = {sign, exponent} for the "
                         "shift datapath, 'int' = plain integers for the DSP-multiply datapath, "
                         "'auto' = follow POT_SHIFT_MAC in config/build_config.yaml (the same "
                         "switch the kernel is built with)")
    ap.add_argument("--build-config", default="config/build_config.yaml",
                    help="--weights auto reads the kernel's defines from here")
    ap.add_argument("--mlp-tiles", type=int, default=0,
                    help="emit the MLP weights as this many per-column-block CSRs; linear_1 splits "
                         "its D input features, linear_2 its d_h hidden features "
                         "(0 = row-major CSR for linear_1 and column-major CSC for linear_2)")
    ap.add_argument("--w1-group-cap", type=int, default=572,
                    help="--permute: max nonzeros per group of 4 W1 rows; mirrors "
                         "MLP_W1_GROUP_NNZ_MAX in src/layers/mlp.cpp (0 = no limit)")
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
    perms = None
    if args.permute == "random":
        perms = random_permutations(model, args.seed, w1_group_cap=args.w1_group_cap or None)
    elif args.permute == "balanced":
        perms = balanced_permutations(model, args.mha_tiles or 12)
        if args.w1_group_cap:
            rng = np.random.default_rng(args.seed)
            for i, layer in enumerate(perms["layers"]):
                w1 = model.linears[f"encoder.encoder_layer_{int(layer)}." + WMODULE["linear_1"]]
                perms["sigma"][i] = _fit_group_cap((w1.weight_int != 0).sum(1), perms["sigma"][i],
                                                   4, args.w1_group_cap, rng)
    if perms is not None:
        apply_permutations(model, perms)
    if args.weights == "auto":
        try:
            import yaml
            defines = yaml.safe_load(open(args.build_config)).get("defines") or []
        except Exception as exc:
            raise SystemExit(f"--weights auto could not read {args.build_config}: {exc}")
        pot_weights = "POT_SHIFT_MAC" in defines
        print(f"--weights auto: {'pot' if pot_weights else 'int'} "
              f"(POT_SHIFT_MAC {'set' if pot_weights else 'not set'} in {args.build_config})",
              file=sys.stderr)
    else:
        pot_weights = (args.weights == "pot")
    pot_scales = {"auto": None, "fixed": False, "shift": True}[args.scales]
    tiles = {t: args.mha_tiles for t in MHA_TILED_WEIGHTS if args.mha_tiles > 0}
    tiles.update({t: args.mlp_tiles for t in MLP_TILED_WEIGHTS if args.mlp_tiles > 0})
    manifest = emit_model(model, layers, args.out, args.verify, pot_weights=pot_weights,
                          pot_scales=pot_scales, perms=perms, tiles=tiles)
    if args.model_index:
        emit_model_index(manifest, args.out, entry_format=0 if pot_weights else 1)
    print(f"\nWrote {len(layers)} layer(s) to {args.out}/", file=sys.stderr)


if __name__ == "__main__":
    main()
