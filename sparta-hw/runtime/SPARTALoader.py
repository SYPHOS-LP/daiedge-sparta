#!/usr/bin/env python3
"""
Load the compiled single-file model payload (model.bin + manifest.json).

This module includes functions and classes for handling model and data
transformations on the host-side.

    from SPARTALoader import Loader
    mp = Loader("out/model")          # dir with model.bin + manifest.json
    L  = mp.layer(0)
    L.weights["w_q"].values                 # np.int8  view (CSR nonzeros)
    L.weights["w_q"].col                    # np.uint16
    L.weights["w_q"].rowptr                 # np.int32
    L.scale_rows["s_q_row"]                 # np.int32 (ap_fixed<32,16> raw)
    L.scales                                # np.int32 [15]
    L.gammas["gamma_mha"]                   # np.int16 (ap_fixed<16,2> raw)

compile_model.py emits ONE little-endian model.bin holding every blob for all 12
encoder layers, and a manifest.json giving each blob an absolute byte
{offset, bytes, count, dtype} into that file.  This module turns that pair into
easy-to-use numpy views for the Python host (SPARTA.py).
"""

from __future__ import annotations

import os
import json
import argparse
import numpy as np

from dataclasses import dataclass, field

# manifest dtype label -> numpy little-endian dtype.  uint8 carries the POT4
# weight code ({sign, 3-bit exponent}); the payload is copied to DDR verbatim,
# so the dtype only has to give the right element width.
_DT = {
    "int8": np.dtype("<i1"),
    "uint8": np.dtype("<u1"),
    "uint16": np.dtype("<u2"),
    "int16": np.dtype("<i2"),
    "int32": np.dtype("<i4"),
    "uint64": np.dtype("<u8"),
}


def parse_args():
    """
    Parse arguments to execute `SPARTALoader.py` as a script.
    """
    parser = argparse.ArgumentParser(description="Inspect the compiled model payload.")
    parser.add_argument(
        "-d",
        "--dir",
        required=True,
        help="Directory that holds the `model.bin` and `manifest.json` files.",
    )
    parser.add_argument(
        "--offsets",
        default=None,
        help="Path rite the C++ offset table CSV here",
    )

    return parser.parse_args()


@dataclass
class Weight:
    """
    A dataclass to represent One weight (views into model.bin).

    layout "csr" / "csc": values + col + rowptr.
    layout "tiled_bank": bank + counts, where bank holds the packed {column, value} entries the
    kernel copies into its banks and counts the nonzeros per (column block, row).
    """

    shape: tuple
    nnz: int
    max_row_nnz: int
    layout: str = "csr"
    values: np.ndarray = None  # int8   [nnz]
    col: np.ndarray = None  # uint16 [nnz]
    rowptr: np.ndarray = None  # int32  [rows+1]
    bank: np.ndarray = None  # uint64 [nnz // entries-per-word]
    counts: np.ndarray = None  # int32  [n_tiles * rows]


@dataclass
class Layer:
    """
    A dataclass to represent the weights of a layer.

    Parameters
    ----------
    index : int
        The layer index.
    weights : dict
        A dictionary [str, Weight] as a map from a string to the layer weights.
    scale_rows : dict

    scales : np.ndarray
        An array including scales.
    gammas : np,ndarray | None
        The RMSNorm weights. These are optional as in the SPARTA models, they are
        folded into the subsequent weights.
    """

    index: int
    weights: dict = field(default_factory=dict)  # tag -> Weight
    scale_rows: dict = field(default_factory=dict)  # name -> int32 view
    scales: np.ndarray = None  # int32 [15]
    gammas: dict = field(default_factory=dict)  # name -> int16 view


class SpartaModelLoader:
    """
    A class representing the loader of a model for deployment on the device.

    A model is fully described by:
    - `model.bin`
        A binary file holding the model's weights
    - `manifest.json`
        A JSON file with information on each layer's index on the binary file.

    The class exposes a memory-map of the model's binary and organizes layer
    weights based on information in `manifest.json`. The weights and layers
    are organized in dataclasses `Weight` and `Layer` respectively.
    """

    def __init__(self, model_dir: str):
        """
        An initializer for the `SpartaModelLoader` class
        """
        self.dir = model_dir

        # Load manifest
        with open(os.path.join(model_dir, "manifest.json")) as f:
            self.manifest = json.load(f)

        # Get binary path
        bin_path = os.path.join(model_dir, self.manifest.get("bin", "model.bin"))

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
        """ """
        o, b = desc["offset"], desc["bytes"]
        return self.buf[o : o + b].view(_DT[desc["dtype"]])

    def layer_indices(self):
        """ """
        return sorted(int(k) for k in self.manifest["layers"])

    def layer(self, i):
        """ """
        m = self.manifest["layers"][str(i)]
        L = Layer(index=i)
        for tag, w in m["weights"].items():
            layout = w.get("layout", "csr")
            if layout == "tiled_bank":
                L.weights[tag] = Weight(
                    shape=tuple(w["shape"]),
                    nnz=w["nnz"],
                    max_row_nnz=w["max_row_nnz"],
                    layout=layout,
                    bank=self.view(w["bank"]),
                    counts=self.view(w["counts"]),
                )
            else:
                L.weights[tag] = Weight(
                    shape=tuple(w["shape"]),
                    nnz=w["nnz"],
                    max_row_nnz=w["max_row_nnz"],
                    layout=layout,
                    values=self.view(w["values"]),
                    col=self.view(w["col"]),
                    rowptr=self.view(w["rowptr"]),
                )
        for name, d in m["scale_rows"].items():
            L.scale_rows[name] = self.view(d)
        L.scales = self.view(m["scales"])
        for name, d in m["gammas"].items():
            L.gammas[name] = self.view(d)
        return L

    # -- helper for the C++/XRT runner: flat offset table -----------------------
    def dump_offsets(self, path):
        """
        Write a compact CSV the C++ runner reads to locate every blob in model.bin:
        layer, blob, offset, bytes, count, dtype. Keeps the C++ side free of a JSON dependency.
        """
        rows = []

        for i in self.layer_indices():
            m = self.manifest["layers"][str(i)]
            for tag, w in m["weights"].items():
                # Same split as layer(): a tiled_bank weight holds {bank, counts}, a csr/csc one
                # holds {values, col, rowptr}.  Every model the current kernel reads is tiled_bank.
                if w.get("layout", "csr") == "tiled_bank":
                    parts = ("bank", "counts")
                else:
                    parts = ("values", "col", "rowptr")
                for part in parts:
                    d = w[part]
                    rows.append(
                        (
                            i,
                            f"{tag}.{part}",
                            d["offset"],
                            d["bytes"],
                            d["count"],
                            d["dtype"],
                        )
                    )
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


def main():
    """
    Main function for executing `SPARTALoader.py` as a script.
    """
    args = parse_args()

    # initialize model loader
    loader = SpartaModelLoader(model_dir=args.dir)

    print("Overview information of the loaded model")
    print(
        "layers: {}, bytes: {}, scale_frac: {}, gamma_frac: {}".format(
            loader.layer_indices(),
            loader.buf.size,
            loader.scale_frac,
            loader.gamma_frac,
        )
    )

    L0 = loader.layer(0)

    for w_tag, w in L0.weights.items():
        print(
            f"  L0 {w_tag:9s} nnz={w.nnz:7d} maxrow={w.max_row_nnz:5d} shape={w.shape}"
        )

    if args.offsets:
        n = loader.dump_offsets(args.offsets)
        print(f"wrote {n} offset rows -> {args.offsets}")


if __name__ == "__main__":
    main()
