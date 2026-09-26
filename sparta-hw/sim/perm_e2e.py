"""Bit-exact check of a permuted-basis model against the original golden outputs.

A model compiled with `compile_model.py --permute ...` runs the encoder in a reordered feature
basis (see compile_model.apply_permutations).  This tool does the host's two boundary steps and
the comparison, so the permutation can be validated with the UNCHANGED kernel:

    prepare : permute the first K images of <src>/inputs.bin into <dir>/inputs.bin
    compare : un-permute <dir>/hw_out.bin and <dir>/hw_layers.bin and byte-compare them with
              <golden>/hw_out.bin (first K images) and <golden>/hw_layers.bin

Usage (numpy only):
    python sim/perm_e2e.py prepare --src sim/e2e --dir sim/e2e/perm --images 2
    E2E_DIR=$PWD/sim/e2e/perm/ <native encoder_e2e_tb>
    python sim/perm_e2e.py compare --dir sim/e2e/perm --golden sim/e2e/golden
"""
import argparse
import os
import struct
import sys

import numpy as np

MAGIC_INPUT = 0x45324E49    # 'IN2E'
MAGIC_OUTPUT = 0x45324F55   # 'UO2E'
N_LAYERS_DUMPED = 12        # tb/encoder_e2e_tb.cpp writes every layer's output for image 0


def read_images(path, magic):
    """-> (n, D, N) int8 array from an inputs.bin / hw_out.bin file."""
    blob = open(path, "rb").read() if os.path.exists(path) else b""
    if len(blob) < 16:
        raise SystemExit(f"{path}: missing or empty (did the testbench run to completion?)")
    m, n, d, tokens = struct.unpack("<I3i", blob[:16])
    if m != magic:
        raise SystemExit(f"{path}: bad magic {m:#x}")
    body = np.frombuffer(blob[16:], dtype=np.int8)
    if body.size != n * d * tokens:
        raise SystemExit(f"{path}: body {body.size} != {n}*{d}*{tokens}")
    return body.reshape(n, d, tokens)


def load_pi(directory):
    """The stream permutation, or None for an unpermuted model (plain golden comparison)."""
    path = os.path.join(directory, "permutations.npz")
    return np.load(path)["pi"] if os.path.exists(path) else None


def prepare(args):
    pi = load_pi(args.dir)
    x = read_images(os.path.join(args.src, "inputs.bin"), MAGIC_INPUT)[:args.images]
    n, d, tokens = x.shape
    inv = np.empty_like(pi)
    inv[pi] = np.arange(len(pi))
    x_perm = np.ascontiguousarray(x[:, inv, :])          # X_perm[pi[f]] = X[f]
    with open(os.path.join(args.dir, "inputs.bin"), "wb") as f:
        f.write(struct.pack("<I3i", MAGIC_INPUT, n, d, tokens))
        f.write(x_perm.tobytes())
    print(f"wrote {args.dir}/inputs.bin: {n} image(s), feature-permuted")


def compare(args):
    pi = load_pi(args.dir)
    ok = True

    y = read_images(os.path.join(args.dir, "hw_out.bin"), MAGIC_OUTPUT)
    if pi is not None:
        y = y[:, pi, :]                                   # Y[f] = Y_perm[pi[f]]
    gold = read_images(os.path.join(args.golden, "hw_out.bin"), MAGIC_OUTPUT)[:len(y)]
    for i in range(len(y)):
        diff = int((y[i] != gold[i]).sum())
        ok &= diff == 0
        print(f"image {i:2d}: {'bit-exact' if diff == 0 else f'{diff} elements differ'}")

    d, tokens = y.shape[1:]
    layers = np.fromfile(os.path.join(args.dir, "hw_layers.bin"), dtype=np.int8)
    gold_layers = np.fromfile(os.path.join(args.golden, "hw_layers.bin"), dtype=np.int8)
    layers = layers.reshape(N_LAYERS_DUMPED, d, tokens)
    if pi is not None:
        layers = layers[:, pi, :]
    gold_layers = gold_layers.reshape(N_LAYERS_DUMPED, d, tokens)
    for layer in range(N_LAYERS_DUMPED):
        diff = int((layers[layer] != gold_layers[layer]).sum())
        ok &= diff == 0
        print(f"image 0 layer {layer:2d}: {'bit-exact' if diff == 0 else f'{diff} elements differ'}")

    kind = "permuted model" if pi is not None else "model"
    print(f"PASS: {kind} is bit-exact vs golden" if ok else "FAIL: outputs differ from golden")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--src", required=True, help="dir with the original inputs.bin")
    p.add_argument("--dir", required=True, help="dir of the permuted model (permutations.npz)")
    p.add_argument("--images", type=int, default=2)
    c = sub.add_parser("compare")
    c.add_argument("--dir", required=True, help="dir of the permuted model and its TB outputs")
    c.add_argument("--golden", required=True, help="dir with the original model's TB outputs")
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args)
        return 0
    return compare(args)


if __name__ == "__main__":
    sys.exit(main())
