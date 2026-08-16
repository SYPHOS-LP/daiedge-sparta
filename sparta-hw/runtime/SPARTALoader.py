#!/usr/bin/env python3
"""Load the compiled single-file model payload (model.bin + manifest.json).

compile_model.py emits ONE little-endian model.bin holding every blob for all 12
encoder layers, and a manifest.json giving each blob an absolute byte
{offset, bytes, count, dtype} into that file.  This module turns that pair into
easy-to-use numpy views for the Python host (SPARTA.py).

Nothing here talks to hardware — it is pure host-side data marshalling.

    from SPARTALoader import Loader
    mp = Loader("out/model")          # dir with model.bin + manifest.json
    L  = mp.layer(0)
    L.weights["w_q"].values                 # np.int8  view (CSR nonzeros)
    L.weights["w_q"].col                    # np.uint16
    L.weights["w_q"].rowptr                 # np.int32
    L.scale_rows["s_q_row"]                 # np.int32 (ap_fixed<32,16> raw)
    L.scales                                # np.int32 [15]
    L.gammas["gamma_mha"]                   # np.int16 (ap_fixed<16,2> raw)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

# manifest dtype label -> numpy little-endian dtype.  uint8 carries the POT4
# weight code ({sign, 3-bit exponent}); the payload is copied to DDR verbatim,
# so the dtype only has to give the right element width.
_DT = {"int8": np.dtype("<i1"), "uint8": np.dtype("<u1"),
       "uint16": np.dtype("<u2"),
       "int16": np.dtype("<i2"), "int32": np.dtype("<i4")}


@dataclass
class Weight:
    """One CSR weight (views into model.bin)."""
    shape: tuple
    nnz: int
    max_row_nnz: int
    values: np.ndarray      # int8   [nnz]
    col: np.ndarray         # uint16 [nnz]
    rowptr: np.ndarray      # int32  [rows+1]


@dataclass
class Layer:
    index: int
    weights: dict = field(default_factory=dict)      # tag -> Weight
    scale_rows: dict = field(default_factory=dict)   # name -> int32 view
    scales: np.ndarray = None                        # int32 [15]
    gammas: dict = field(default_factory=dict)       # name -> int16 view


class Loader:
    """mmap of model.bin + parsed manifest, exposing per-layer numpy views."""

    def __init__(self, out_dir):
        self.dir = out_dir
        with open(os.path.join(out_dir, "manifest.json")) as f:
            self.manifest = json.load(f)
        bin_path = os.path.join(out_dir, self.manifest.get("bin", "model.bin"))
        # memory-map read-only; views are zero-copy slices into this.
        self.buf = np.memmap(bin_path, dtype=np.uint8, mode="r")
        assert self.buf.size == self.manifest["total_bytes"], "model.bin size mismatch"
        self.scale_frac = self.manifest["scale_frac"]
        # Optional since the POT4 / folded-gamma payload: the gammas are folded
        # into the datapath, so compile_model.py no longer emits them (the
        # per-layer "gammas" section is empty) and there is no gamma_frac to
        # report.  Older manifests still carry it.
        self.gamma_frac = self.manifest.get("gamma_frac")

    # -- low-level: a manifest descriptor -> numpy view (zero-copy) --------------
    def view(self, desc):
        o, b = desc["offset"], desc["bytes"]
        return self.buf[o:o + b].view(_DT[desc["dtype"]])

    def layer_indices(self):
        return sorted(int(k) for k in self.manifest["layers"])

    def layer(self, i):
        m = self.manifest["layers"][str(i)]
        L = Layer(index=i)
        for tag, w in m["weights"].items():
            L.weights[tag] = Weight(
                shape=tuple(w["shape"]), nnz=w["nnz"], max_row_nnz=w["max_row_nnz"],
                values=self.view(w["values"]), col=self.view(w["col"]),
                rowptr=self.view(w["rowptr"]))
        for name, d in m["scale_rows"].items():
            L.scale_rows[name] = self.view(d)
        L.scales = self.view(m["scales"])
        for name, d in m["gammas"].items():
            L.gammas[name] = self.view(d)
        return L

    # -- helper for the C++/XRT runner: flat offset table -----------------------
    def dump_offsets(self, path):
        """Write a compact CSV the C++ runner reads to locate every blob in
        model.bin: layer,blob,offset,bytes,count,dtype.  Keeps the C++ side free
        of a JSON dependency."""
        rows = []
        for i in self.layer_indices():
            m = self.manifest["layers"][str(i)]
            for tag, w in m["weights"].items():
                for part in ("values", "col", "rowptr"):
                    d = w[part]
                    rows.append((i, f"{tag}.{part}", d["offset"], d["bytes"],
                                 d["count"], d["dtype"]))
            for name, d in m["scale_rows"].items():
                rows.append((i, name, d["offset"], d["bytes"], d["count"], d["dtype"]))
            d = m["scales"]
            rows.append((i, "scales", d["offset"], d["bytes"], d["count"], d["dtype"]))
            for name, d in m["gammas"].items():
                rows.append((i, name, d["offset"], d["bytes"], d["count"], d["dtype"]))
        with open(path, "w") as f:
            f.write("layer,blob,offset,bytes,count,dtype\n")
            for r in rows:
                f.write(",".join(str(x) for x in r) + "\n")
        return len(rows)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Inspect / export the compiled model payload.")
    ap.add_argument("-d", "--dir", required=True, help="dir with model.bin + manifest.json")
    ap.add_argument("--offsets", default=None, help="write the C++ offset table CSV here")
    args = ap.parse_args()
    mp = Loader(args.dir)
    print(f"layers: {mp.layer_indices()}  bytes: {mp.buf.size}  "
          f"scale_frac={mp.scale_frac} gamma_frac={mp.gamma_frac}")
    L0 = mp.layer(0)
    for t, w in L0.weights.items():
        print(f"  L0 {t:9s} nnz={w.nnz:7d} maxrow={w.max_row_nnz:5d} shape={w.shape}")
    if args.offsets:
        n = mp.dump_offsets(args.offsets)
        print(f"wrote {n} offset rows -> {args.offsets}")
