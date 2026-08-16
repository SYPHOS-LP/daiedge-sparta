#!/usr/bin/env python3
"""End-to-end ViT inference with the 12 encoder layers running on the FPGA.

The accelerator implements only the encoder. Classifying a real image still
needs the SW embedding (patchify + conv + CLS token + positional embedding) in
front and the SW classifier head behind. Rather than reimplementing those
quantized layers, this borrows them from the sparta-sw brevitas model and
substitutes ONLY the encoder:

    image --[SW embed + quant_input]--> int8 (D x N)
          --[FPGA: 12 x encoder_layer_top]--> int8 (D x N)
          --[* out_scale]--> float --[SW heads.ln + head]--> logits --> argmax

This mirrors sim/full_model_infer.py, which does the same substitution with a
Python replica of the hardware. The boundary handling is deliberately
identical, so a disagreement between this script and that one isolates to the
hardware rather than to the pre/post-processing.

BOUNDARY DETAILS
  * The encoder's int8 input is whatever encoder_layer_0.quant_input emits. It
    is captured with a forward hook during a normal SW forward pass, so the
    exact tensor the SW model would have fed its own encoder is the one sent
    to the FPGA -- no reimplementation, no drift.
  * brevitas works token-major (1, N, D); the kernel works feature-major
    (D, N). Hence the transposes at both boundaries.
  * The encoder's real-valued output = int8_code * out_scale, where out_scale
    is encoder_layer_11.res2.out_quant's scale. heads.ln has its own
    input_quant, so handing it that dequantized float reproduces the head
    exactly.

REQUIREMENTS
  Runs on the board, and needs both:
    - torch + brevitas + sparse_vit  (the sparta-sw venv)
    - pynq                           (the FPGA runtime)
  The PL is expected to be ALREADY PROGRAMMED (xmutil); this script only
  attaches to it.  Load it first, e.g.
    sudo xmutil loadapp <app>

  sudo is required (PYNQ maps /dev/mem) and `sudo -E` specifically: without -E
  the environment PYNQ needs to detect the device is stripped and it fails with
  "No Devices Found".  sudo will not resolve a venv `python` either, so give
  the interpreter's absolute path.

USAGE
  sudo -E /path/to/venv/bin/python runtime/infer_hw.py \
      --cfg    ../sparta-sw/sparse_vit/cfgs/infer_pot4_local.toml \
      --model-dir sim/e2e \
      --images 10
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))

# sparta-sw's package root.  Defaults to a sibling checkout of this repo, which
# is the dev-host layout; SPARTA_SW_SRC overrides it for boards where the two
# trees are not siblings (e.g. runtime/ copied straight into $HOME).
_SW = os.environ.get(
    "SPARTA_SW_SRC",
    os.path.abspath(os.path.join(_REPO, "..", "sparta-sw", "sparse_vit", "src")))

if not os.path.isdir(_SW):
    raise SystemExit(
        f"sparta-sw source not found at {_SW}\n"
        "Set SPARTA_SW_SRC to <sparta-sw>/sparse_vit/src, e.g.\n"
        "  export SPARTA_SW_SRC=$HOME/sparta-sw/sparse_vit/src")

sys.path.insert(0, _HERE)
sys.path.insert(0, _SW)

import SPARTA  # noqa: E402  (the FPGA runtime in this directory)


# ============================================================================
# SW model construction -- borrowed from the sparta-sw inference script so the
# embedding and head are byte-identical to what the SW team runs.
# ============================================================================
def _load_infer_module():
    """Import sparta-sw's infer_quant_vit_simple.py for its builder helpers."""
    path = os.path.join(_SW, "bin", "infer_quant_vit_simple.py")
    spec = importlib.util.spec_from_file_location("iq", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_sw_model(cfg, iq):
    """Build the full SW brevitas quant model and load its checkpoint."""
    import torch  # noqa: F401  (imported for side effects / availability check)
    from sparse_vit.quantization.utils import _densify_pruning_state_dict
    from sparse_vit.quantization.brevitas_quant import _setup_dyadic_residuals

    q = cfg["quant"]
    model = iq.build_quant_model(
        cfg=cfg, wbits=q["bits"], abits=q["abits"],
        weight_per_tensor=q["weight_per_tensor"],
        weight_scale_po2=q["weight_scale_po2"],
        weight_weight_po2=q["weight_weight_po2"],
        act_scale_po2=q["act_scale_po2"],
        fuse_rms_norm=q["fuse_rms_norm"],
    )
    state = iq._resolve_state(iq._load_ckpt(cfg["model"]["path"]))
    state, _ = _densify_pruning_state_dict(state)
    model.load_state_dict(state, strict=False)
    model.eval()
    _setup_dyadic_residuals(model)
    return model


class EncoderBridge:
    """Captures the encoder's int8 input/output boundary from the SW model.

    A forward hook on encoder_layer_0.quant_input records exactly the int8
    tensor the SW encoder would have consumed. Running the SW model forward
    therefore produces both the reference prediction AND the FPGA's input, from
    one pass.
    """

    def __init__(self, model):
        import torch

        self._torch = torch
        self.model = model
        self._enc_in_int8 = None
        self._enc_in_scale = None
        model.encoder.encoder_layer_0.quant_input.register_forward_hook(self._grab_in)
        self.out_scale = float(
            model.encoder.encoder_layer_11.res2.out_quant.act_quant.scale()
        )

    def _grab_in(self, mod, inp, out):
        # `out` is an IntQuantTensor of shape (1, N, D).
        self._enc_in_int8 = out.int().detach().to(self._torch.int32).numpy()[0]
        self._enc_in_scale = float(out.scale.flatten()[0])

    def encoder_input(self):
        """The int8 encoder input as feature-major (D, N), plus its scale."""
        return self._enc_in_int8.T.astype(np.int8), self._enc_in_scale


def run_head(model, enc_out_int8, out_scale):
    """Dequantize the FPGA's int8 encoder output and run the SW head.

    enc_out_int8 : (D, N) feature-major int8 as produced by the kernel.
    """
    import torch
    from brevitas.quant_tensor import IntQuantTensor

    real = enc_out_int8.T.astype(np.float32) * out_scale       # (N, D)
    x = torch.from_numpy(real).unsqueeze(0)                    # (1, N, D)
    with torch.no_grad():
        logits = model.heads(x)
    lv = logits.value if isinstance(logits, IntQuantTensor) else logits
    return lv[0].detach().numpy()


# ============================================================================
# FPGA encoder
# ============================================================================
class HwEncoder:
    """Runs the 12 encoder layers on the FPGA, holding device state open.

    The kernel handle and the activation/scratch buffers are set up once, so
    per-image cost is only the weight staging plus the 12 kernel invocations.
    Weight buffers are freed per layer because all 12 layers' weights do not
    fit in DDR simultaneously.

    By default this ATTACHES to a PL programmed elsewhere (xmutil) rather than
    programming it -- see HwKernel.
    """

    def __init__(self, model_dir, bitstream=None, instance=None, download=False,
                 base_addr=None, debug_dir=None, verbose=False):
        from pynq import allocate
        from SPARTALoader import Loader

        self._allocate = allocate
        self.debug_dir = debug_dir
        self.verbose = verbose
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
        # HwKernel is the hardware seam: attaches to (or programs) the PL and
        # owns the AXI-Lite control registers.
        self.kernel = SPARTA.HwKernel(bitstream, instance=instance,
                                      download=download, base_addr=base_addr)
        self.overlay = self.kernel.overlay      # None on the attach path
        if verbose:
            print(f"  control base {self.kernel._base_addr_str()}, "
                  f"ap_ctrl 0x{self.kernel.ip.read(SPARTA.AP_CTRL):08x}",
                  flush=True)
            print(f"  loading model payload from {model_dir}...",
                  end="", flush=True)
        self.mp = Loader(model_dir)
        self.layers = self.mp.layer_indices()
        if verbose:
            print(f" {len(self.layers)} layers", flush=True)
            print("  allocating DDR buffers...", end="", flush=True)

        n = SPARTA.ACT_ELEMS
        self.a = allocate(shape=(n,), dtype=np.int8)
        self.b = allocate(shape=(n,), dtype=np.int8)
        self.h = allocate(shape=(n,), dtype=np.int8)
        q = SPARTA.MHA_MAX_QK_NNZ
        self.scratch = {
            "q_val": allocate(shape=(q,), dtype=np.int8),
            "q_col": allocate(shape=(q,), dtype=np.uint8),
            "k_val": allocate(shape=(q,), dtype=np.int8),
            "k_col": allocate(shape=(q,), dtype=np.uint8),
        }
        if verbose:
            mb = (3 * n + 4 * q) / (1024 * 1024)
            print(f" {mb:.1f} MB (act 3x{n}, scratch 4x{q})", flush=True)

    def __call__(self, enc_in, n_layers=None, debug_tag=None):
        """enc_in : (D, N) int8 feature-major  ->  (D, N) int8 feature-major.

        n_layers  : run only the first N encoder layers (default: all 12).
                    Stopping early leaves the result partway through the
                    encoder, so the classification is meaningless -- this is
                    for bring-up and bisecting, not for accuracy runs.
        debug_tag : if set (and self.debug_dir is set), write every layer's
                    output to <debug_dir>/<debug_tag>_L<nn>.npy for diffing
                    against the replica.  Off by default: it is a 38KB write
                    per layer per image."""
        flat = np.ascontiguousarray(enc_in).reshape(-1)
        if flat.size != SPARTA.ACT_ELEMS:
            raise ValueError(
                f"encoder input has {flat.size} elements, expected "
                f"{SPARTA.ACT_ELEMS} (ENC_D={SPARTA.ENC_D} x ENC_N={SPARTA.ENC_N}). "
                "The SW model's token count does not match the built kernel.")

        layers = self.layers if n_layers is None else self.layers[:n_layers]

        self.a[:] = flat
        self.a.sync_to_device()
        x_buf, y_buf = self.a, self.b

        for L in layers:
            layer = self.mp.layer(L)
            d_h = layer.weights["linear_1"].shape[0]
            if self.verbose:
                print(f"  L{L:02d}: staging weights...", end="", flush=True)
            t0 = time.monotonic()
            lb = SPARTA._stage_layer(self._allocate, layer)
            t_stage = time.monotonic() - t0
            args = SPARTA._build_args(x_buf, self.h, y_buf, lb, self.scratch,
                                      SPARTA.ENC_N, SPARTA.ENC_D, d_h)
            if self.verbose:
                print(f" {t_stage:.1f}s; invoking kernel...", end="", flush=True)
            t0 = time.monotonic()
            try:
                self.kernel.invoke(args)
            except TimeoutError as e:
                if self.verbose:
                    print(" TIMEOUT")
                raise RuntimeError(f"encoder layer {L} hung: {e}") from e
            t_run = time.monotonic() - t0
            if self.verbose:
                print(f" done in {t_run:.3f}s")
            y_buf.sync_from_device()
            if debug_tag is not None and self.debug_dir is not None:
                self._dump(debug_tag, L, y_buf)
            for buf in lb.values():
                buf.freebuffer()
            x_buf, y_buf = y_buf, x_buf

        x_buf.sync_from_device()
        out = np.array(x_buf[:SPARTA.ACT_ELEMS], dtype=np.int8)
        return out.reshape(SPARTA.ENC_D, SPARTA.ENC_N)

    def _dump(self, tag, layer_idx, buf):
        """Write one layer's output, feature-major (D, N), as .npy."""
        arr = np.array(buf[:SPARTA.ACT_ELEMS], dtype=np.int8).reshape(
            SPARTA.ENC_D, SPARTA.ENC_N)
        path = os.path.join(self.debug_dir, f"{tag}_L{layer_idx:02d}.npy")
        np.save(path, arr)

    def close(self):
        self.kernel.close()
        for b in (self.a, self.b, self.h, *self.scratch.values()):
            b.freebuffer()


# ============================================================================
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True,
                    help="sparta-sw inference .toml (model checkpoint + dataset)")
    ap.add_argument("--model-dir", required=True,
                    help="dir with model.bin + manifest.json (from compile_model.py)")
    ap.add_argument("--bitstream", default=None,
                    help="overlay to program (.bit with its .hwh alongside); "
                         "only needed with --download")
    ap.add_argument("--instance", default=None,
                    help="BD cell name of the kernel (default: discover it)")
    ap.add_argument("--download", action="store_true",
                    help="program the PL from --bitstream instead of attaching "
                         "to one already loaded (e.g. by xmutil). Needs a PYNQ "
                         "whose pynqmetadata can parse the .hwh")
    ap.add_argument("--base-addr", default=None,
                    help=f"s_axi_control base address when attaching "
                         f"(default 0x{SPARTA.HwKernel.CONTROL_BASE:08x})")
    ap.add_argument("--images", type=int, default=10, help="how many images to run")
    ap.add_argument("--layers", type=int, default=None,
                    help="run only the first N encoder layers (default: all). "
                         "Bring-up aid: the prediction is meaningless when the "
                         "encoder is cut short, so this is for bisecting a bad "
                         "layer, not for accuracy")
    ap.add_argument("--debug", action="store_true",
                    help="dump every layer's output to --debug-dir as "
                         ".npy for diffing against the replica (slow: one "
                         "write per layer per image)")
    ap.add_argument("--debug-dir", default="hw_dumps",
                    help="where --debug writes layer dumps (default: hw_dumps)")
    ap.add_argument("--no-reference", action="store_true",
                    help="skip the SW reference prediction (still needs the SW "
                         "forward pass to produce the encoder input, so this "
                         "only suppresses the comparison)")
    args = ap.parse_args()

    if args.download and not args.bitstream:
        ap.error("--download needs --bitstream")

    import toml
    import torch
    from brevitas.quant_tensor import IntQuantTensor

    # Progress logging: the SW model load takes minutes on the board, so
    # without these a slow start is indistinguishable from a hang.
    def step(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    step(f"loading config {args.cfg}")
    cfg = toml.load(args.cfg)
    step("importing sparta-sw inference module")
    iq = _load_infer_module()
    step(f"building SW model + loading checkpoint {cfg['model']['path']}")
    model = build_sw_model(cfg, iq)
    step("SW model ready; hooking encoder boundary")
    bridge = EncoderBridge(model)
    step(f"loading dataset from {cfg['dataset']['path']}")
    data, _ = iq.initialize_map_datasets(cfg)
    step(f"dataset ready ({len(data)} images)")

    step(f"attaching to FPGA (model payload {args.model_dir})")
    hw = HwEncoder(args.model_dir, args.bitstream, args.instance,
                   download=args.download,
                   base_addr=int(args.base_addr, 0) if args.base_addr else None,
                   debug_dir=args.debug_dir if args.debug else None,
                   verbose=True)
    step("FPGA ready")

    if args.layers is not None:
        print(f"NOTE: running only the first {args.layers} of "
              f"{len(hw.layers)} encoder layers -- predictions below are "
              f"NOT meaningful, this is a bring-up/bisect mode.")
    if args.debug:
        print(f"debug: dumping layer outputs to {args.debug_dir}/")

    agree = correct_sw = correct_hw = 0
    print("image | truth | SW pred | HW pred | agree")
    print("------+-------+---------+---------+------")
    try:
        for idx in range(args.images):
            step(f"image {idx}: SW forward pass")
            img, label = data[idx]

            # One SW forward: gives the reference prediction, and fires the hook
            # that captures the exact int8 tensor to hand the FPGA.
            with torch.no_grad():
                sw_logits = model(img.unsqueeze(0))
            sw_lv = (sw_logits.value if isinstance(sw_logits, IntQuantTensor)
                     else sw_logits)
            sw_pred = int(sw_lv.argmax(-1))

            enc_in, _in_scale = bridge.encoder_input()      # (D, N) int8
            if args.debug:
                np.save(os.path.join(args.debug_dir, f"img{idx:03d}_input.npy"),
                        enc_in)
            step(f"image {idx}: running encoder on FPGA")
            enc_out = hw(enc_in, n_layers=args.layers,      # (D, N) int8, on FPGA
                         debug_tag=f"img{idx:03d}")
            step(f"image {idx}: FPGA done, running SW head")
            hw_logits = run_head(model, enc_out, bridge.out_scale)
            hw_pred = int(np.argmax(hw_logits))

            a = sw_pred == hw_pred
            agree += a
            correct_sw += (sw_pred == label)
            correct_hw += (hw_pred == label)
            print(f"{idx:5d} | {label:5d} | {sw_pred:7d} | {hw_pred:7d} | "
                  f"{'yes' if a else 'NO'}")
    finally:
        hw.close()

    n = args.images
    if args.layers is not None:
        print(f"\nran {args.layers}/{len(hw.layers)} layers -- HW columns above "
              f"are a partial encoder, not a prediction.  SW accuracy "
              f"{correct_sw}/{n} is still valid (full SW model).")
    else:
        print(f"\nagreement SW vs HW : {agree}/{n}")
        print(f"accuracy  SW       : {correct_sw}/{n}")
        print(f"accuracy  HW       : {correct_hw}/{n}")


if __name__ == "__main__":
    main()
