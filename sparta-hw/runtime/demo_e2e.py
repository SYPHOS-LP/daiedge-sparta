#!/usr/bin/env python3
"""SPARTA encoder demo — one console run of the whole end-to-end flow ON HARDWARE.

Wraps the end-to-end inference into a single narrated script, so the pipeline can
be demonstrated and recorded in one take, with the 12 encoder layers running on
the Kria FPGA (NOT a simulation):

    image --[SW embed]--> int8 (D x N) --[FPGA: 12 x encoder_layer_top]-->
          int8 --[SW head]--> logits --> prediction

This is the hardware sibling of sim/demo_e2e.py.  That script substituted the
encoder with a Vitis HLS C-simulation, because the Kria deployment was not yet
functional; it replayed artifacts a slow (~4 min/image) csim had produced and
reported throughput from a separate C-synthesis measurement, since csim models no
timing.

The hardware is now functional and invoked through runtime/infer_hw.py, so this
script drives the REAL accelerator instead.  Every stage is live:

  * the SW forward runs per image and its forward hook captures the exact int8
    tensor the SW encoder would have consumed (EncoderBridge) -- that is the
    tensor sent to the FPGA;
  * the 12 encoder layers run on the FPGA (HwEncoder), which returns real int8;
  * the SW head classifies that output.

Latency is MEASURED, not estimated: wall-clock per image always, plus true fabric
cycles from the on-board AXI timer when --timer-base is given (as in infer_hw.py).

REQUIREMENTS  (identical to infer_hw.py -- it runs on the board)
  Needs both the sparta-sw venv (torch + brevitas + sparse_vit) and pynq, and the
  PL must be ALREADY PROGRAMMED (xmutil) unless --download is given.  sudo -E is
  required (PYNQ maps /dev/mem and needs the environment to find the device), and
  sudo will not resolve a venv python, so give the interpreter's absolute path.

USAGE
  sudo -E /path/to/venv/bin/python runtime/demo_e2e.py \
      --cfg       ../sparta-sw/sparse_vit/cfgs/infer_pot4_local.toml \
      --model-dir ../sim/e2e \
      --images 4 --timer-base 0xA0010000

  # attach mode is the default; add --download --bitstream <.bit> to program the PL.
"""
import os
import sys
import time
import argparse

import warnings

# torch/brevitas emit UserWarnings on import and on the first forward; they would
# otherwise interleave into the narration mid-image and spoil a recording.
warnings.filterwarnings("ignore")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import SPARTA
import infer_hw as HW

CIFAR = ["airplane", "automobile", "bird", "cat", "deer",
         "dog", "frog", "horse", "ship", "truck"]

CLOCK_MHZ = 200

# ---------------------------------------------------------------- presentation
BOLD = "\033[1m"; DIM = "\033[2m"; RESET = "\033[0m"
CYAN = "\033[36m"; GREEN = "\033[32m"; YELLOW = "\033[33m"; RED = "\033[31m"


def _init_console():
    """Reconfigure stdout to UTF-8 so the box/bar glyphs survive a cp1253/cp437 console.

    Returns True when wide glyphs are safe to emit; the caller falls back to an
    ASCII-only presentation otherwise (recording must never die on an encode error).
    """
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    if enc.startswith("utf"):
        return True
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        return True
    except Exception:
        return False


_UNICODE = _init_console()


def _supports_colour():
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    if os.name == "nt":
        # Enable ANSI on Windows 10+ consoles; harmless if already on.
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
        except Exception:
            return False
    return True


_COLOUR = _supports_colour()


def c(text, colour):
    return f"{colour}{text}{RESET}" if _COLOUR else text


def say(msg, pause=0.0):
    print(msg, flush=True)
    if pause:
        time.sleep(pause)


# glyph set: box drawing / bars / marks, with ASCII fallbacks
G = dict(
    h="─", hh="═", tl="╔", tr="╗", bl="╚", br="╝", v="║",
    full="█", empty="░", dot="·", tick="✓", cross="✗", arrow="▸",
) if _UNICODE else dict(
    h="-", hh="=", tl="+", tr="+", bl="+", br="+", v="|",
    full="#", empty=".", dot="-", tick="OK", cross="X", arrow=">",
)


def rule(char=None, width=64):
    print(c((char or G["h"]) * width, DIM), flush=True)


def progress(label, seconds, steps=24):
    """A determinate bar for narrating a step whose result is already known."""
    if not _COLOUR:
        say(f"    {label} ...")
        time.sleep(seconds)
        return
    for i in range(steps + 1):
        filled = int(i / steps * 28)
        bar = G["full"] * filled + G["empty"] * (28 - filled)
        pct = int(i / steps * 100)
        sys.stdout.write(f"\r    {label} {c(bar, CYAN)} {pct:3d}%")
        sys.stdout.flush()
        time.sleep(seconds / steps)
    sys.stdout.write("\n")
    sys.stdout.flush()


def progress_around(label, work, expected=None, tick=0.04):
    """Run `work()` on a worker thread and animate the SAME bar as progress()
    while it runs, so the original demo's graphic plays over the REAL FPGA call.

    The bar is paced off `expected` (seconds), the anticipated duration of the
    work: it fills smoothly toward — and holds just under — full over that
    estimate, then snaps to 100% the instant the real call returns.  This mirrors
    the original determinate bar's LOOK (a steady sweep that reaches near-full
    right as the step ends) without ever faking completion: if the work runs long
    the bar simply parks at 95% until it truly finishes.

    `expected` is typically the previous image's measured FPGA time; when it is
    None (the first image, no measurement yet) a conservative default is used so
    the bar still sweeps at a natural pace instead of stalling early.  Returns
    work()'s result (and re-raises anything it threw)."""
    import threading

    box = {}

    def runner():
        try:
            box["result"] = work()
        except BaseException as e:          # re-raised on the main thread below
            box["error"] = e

    if not _COLOUR:
        say(f"    {label} ...")
        t = threading.Thread(target=runner)
        t.start()
        t.join()
    else:
        # Pace to reach ~95% exactly at `expected`; hold there until the work is
        # actually done.  A floor keeps a very fast call from flashing past.
        est = max(float(expected) if expected else 0.0, 0.30)
        t = threading.Thread(target=runner)
        t.start()
        t0 = time.monotonic()
        while t.is_alive():
            frac = min(0.95, (time.monotonic() - t0) / est * 0.95)
            filled = int(frac * 28)
            bar = G["full"] * filled + G["empty"] * (28 - filled)
            sys.stdout.write(f"\r    {label} {c(bar, CYAN)} {int(frac * 100):3d}%")
            sys.stdout.flush()
            time.sleep(tick)
        t.join()
        # Snap to a full, 100% bar now that the work is really done.
        bar = G["full"] * 28
        sys.stdout.write(f"\r    {label} {c(bar, CYAN)} 100%\n")
        sys.stdout.flush()

    if "error" in box:
        raise box["error"]
    return box.get("result")


def banner():
    print()
    title = "  SPARTA " + G["dot"] + " Sparse ViT Encoder Accelerator"
    sub = "  End-to-End inference " + G["dot"] + " Kria KR260 FPGA"
    say(c(G["tl"] + G["hh"] * 45 + G["tr"], CYAN))
    say(c(G["v"], CYAN) + c(title.ljust(45), BOLD) + c(G["v"], CYAN))
    say(c(G["v"], CYAN) + sub.ljust(45) + c(G["v"], CYAN))
    say(c(G["bl"] + G["hh"] * 45 + G["br"], CYAN))
    print()


# ---------------------------------------------------------------- stages
def main():
    ap = argparse.ArgumentParser(
        description="SPARTA encoder end-to-end demo (live FPGA)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True,
                    help="sparta-sw inference .toml (model checkpoint + dataset)")
    ap.add_argument("--model-dir", required=True,
                    help="dir with model.bin + manifest.json (from compile_model.py)")
    ap.add_argument("--images", type=int, default=4,
                    help="how many images to run (default: 4)")
    ap.add_argument("--speed", type=float, default=0.8,
                    help="pacing multiplier for the narration (0 = no pauses)")
    # --- FPGA attach/program plumbing, mirrored from infer_hw.py ---
    ap.add_argument("--bitstream", default=None,
                    help="overlay to program (.bit with its .hwh alongside); "
                         "only needed with --download")
    ap.add_argument("--instance", default=None,
                    help="BD cell name of the kernel (default: discover it)")
    ap.add_argument("--download", action="store_true",
                    help="program the PL from --bitstream instead of attaching "
                         "to one already loaded (e.g. by xmutil)")
    ap.add_argument("--base-addr", default=None,
                    help=f"s_axi_control base address when attaching "
                         f"(default 0x{SPARTA.HwKernel.CONTROL_BASE:08x})")
    ap.add_argument("--timer-base", default=None,
                    help="base address of the diagnostic AXI timer, if present. "
                         "With it, latency is reported in true fabric cycles.")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="pass through infer_hw's per-layer FPGA logging")
    args = ap.parse_args()

    if args.download and not args.bitstream:
        ap.error("--download needs --bitstream")

    sp = args.speed
    banner()

    # ---- setup -----------------------------------------------------------
    # Build the SW model once — it supplies the embedding and the head that sit
    # either side of the accelerator, and (via the boundary hook) the exact int8
    # tensor handed to the FPGA.
    # Reported as steps because they cost very differently: the torch/brevitas
    # import is most of the wait, the checkpoint load less. Attributing all of it
    # to "loading the model" reads like a hang.
    dot0 = c(G["dot"], DIM)

    import toml

    say(c("  Importing deep-learning stack", BOLD) +
        c("   (torch + brevitas - one-off)", DIM))
    t_imp = time.time()
    iq = HW._load_infer_module()
    cfg = toml.load(args.cfg)
    say(f"    {c('ready', GREEN)}  {dot0} {time.time() - t_imp:.1f}s")
    print()

    say(c("  Loading software model", BOLD) +
        c("   (patch embeddings + classifier head + checkpoint)", DIM))
    t_mod = time.time()
    model = HW.build_sw_model(cfg, iq)
    bridge = HW.EncoderBridge(model)
    say(f"    {c('ready', GREEN)}  {dot0} {time.time() - t_mod:.1f}s")
    print()

    say(c("  Loading dataset", BOLD) + c("   (CIFAR-10)", DIM))
    data, _ = iq.initialize_map_datasets(cfg)
    n = min(args.images, len(data))
    D, N = SPARTA.ENC_D, SPARTA.ENC_N
    say(f"    {c('ready', GREEN)}  {dot0} {len(data)} images available")
    print()

    say(c("  Attaching to FPGA", BOLD) +
        c(("   (programming PL)" if args.download else "   (already programmed)"), DIM))
    t_fpga = time.time()
    hw = HW.HwEncoder(
        args.model_dir, args.bitstream, args.instance,
        download=args.download,
        base_addr=int(args.base_addr, 0) if args.base_addr else None,
        verbose=args.verbose,
        timer_base=int(args.timer_base, 0) if args.timer_base else None)
    say(f"    {c('ready', GREEN)}  {dot0} {len(hw.layers)} encoder layers "
        f"{dot0} {time.time() - t_fpga:.1f}s")
    print()

    say(f"    model      {c('12-layer ViT encoder', CYAN)}  ·  FEATURES={D}  TOKENS={N}")
    say(f"    weights    4-bit POT ·  int8 Activations")
    say(f"    kernel     {c('encoder_layer_top', CYAN)}  ·  Kria FPGA "
        f"{c('@ ' + str(CLOCK_MHZ) + ' MHz', DIM)}")
    say(f"    dataset    CIFAR-10  ·  {n} image{'s' if n != 1 else ''}", 0.6 * sp)
    print()
    rule()
    print()

    # ---- per-image -------------------------------------------------------
    import torch
    from brevitas.quant_tensor import IntQuantTensor

    out_scale = bridge.out_scale
    truth, sw_pred, hw_pred = [], [], []
    per_image_wall_ms = []      # wall-clock of the hw() call (host staging + FPGA)
    per_image_hw_ms = []        # true fabric latency (on-board timer), when present

    def fabric_ms_for_image(i):
        """Sum image i's per-layer fabric cycles -> ms, or None if no timer.

        HwEncoder appends one cycle sample per layer per image in order, so
        layer_cycles[L][i] is layer L's cycles on image i; summing over the
        layers gives that image's true encoder latency, free of the host-side
        weight-staging and PYNQ overhead that inflate the wall-clock."""
        if not hw.layer_cycles:
            return None
        try:
            total_cyc = sum(hw.layer_cycles[L][i] for L in hw.layers)
        except (KeyError, IndexError):
            return None
        return total_cyc / (CLOCK_MHZ * 1e6) * 1e3

    try:
        for i in range(n):
            img, label = data[i]
            truth.append(int(label))

            dot = c(G["dot"], DIM)
            arr = "->" if not _UNICODE else "→"
            say(c(f"  IMAGE {i + 1}/{n}", BOLD) +
                c(f"   ground truth: {CIFAR[int(label)]}", DIM))
            say(f"    loading                     {dot} CIFAR-10 sample {i}", 0.10 * sp)

            # One SW forward: the reference prediction AND the hook that captures
            # the exact int8 tensor to hand the FPGA.
            with torch.no_grad():
                sw_logits = model(img.unsqueeze(0))
            sw_lv = (sw_logits.value if isinstance(sw_logits, IntQuantTensor)
                     else sw_logits)
            sw_pred.append(int(sw_lv.argmax(-1)))
            enc_in, _in_scale = bridge.encoder_input()      # (D, N) int8
            say(f"    software embedding          {dot} patchify {arr} int8 "
                f"[{D}x{N}]", 0.10 * sp)

            # --- the encoder, on the FPGA (this is the real work now) ---
            # Pace the bar off the previous image's measured FPGA wall time, so it
            # sweeps smoothly and lands near-full right as the call returns rather
            # than stalling early.  The first image has no estimate yet (None).
            est = (per_image_wall_ms[-1] / 1000.0) if per_image_wall_ms else None
            t_hw = time.time()
            enc_out = progress_around("inference  12 layers on FPGA",
                                      lambda: hw(enc_in),   # (D, N) int8, on FPGA
                                      expected=est)
            dt_hw = time.time() - t_hw
            per_image_wall_ms.append(dt_hw * 1000.0)
            fab_ms = fabric_ms_for_image(i)
            if fab_ms is not None:
                per_image_hw_ms.append(fab_ms)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                logits = HW.run_head(model, enc_out, out_scale)
            pred = int(np.asarray(logits).argmax())
            hw_pred.append(pred)

            ok = pred == truth[i]
            match = pred == sw_pred[i]
            # Headline the TRUE FPGA latency (fabric cycles) when the timer is
            # present; wall-clock includes host staging + PYNQ overhead and is far
            # larger, so it is demoted to a dim parenthetical rather than the
            # number the demo shows off.
            if fab_ms is not None:
                say(f"    FPGA latency                {dot} "
                    f"{c(f'{fab_ms:.2f} ms', BOLD)}")
            else:
                say(f"    FPGA latency                {dot} "
                    f"{c(f'{dt_hw * 1000:.1f} ms', BOLD)} wall"
                    f"   {c('(pass --timer-base for true fabric latency)', DIM)}")
            say(f"    classifier head             {dot} logits {arr} argmax")
            say("    prediction                  " + c(G["arrow"] + " ", CYAN) +
                c(CIFAR[pred].upper(), BOLD if not _COLOUR else GREEN if ok else YELLOW) +
                ("   " + c(G["tick"] + " correct", GREEN) if ok
                 else "   " + c(G["cross"] + " incorrect", RED)))
            say(f"    vs software model           {dot} " +
                (c("match", GREEN) if match else c("DIVERGES", RED)))
            print()
    finally:
        hw.close()

    # ---- results ---------------------------------------------------------
    rule(G["hh"])
    say(c("  RESULTS", BOLD))
    rule(G["hh"])
    print()

    correct = sum(int(p == t) for p, t in zip(hw_pred, truth))
    sw_correct = sum(int(p == t) for p, t in zip(sw_pred, truth))
    agree = sum(int(p == s) for p, s in zip(hw_pred, sw_pred))

    acc = 100.0 * correct / n
    say(f"    Software Accuracy     {100.0 * sw_correct / n:.1f}%   "
        f"({sw_correct}/{n} correct - baseline)")
    say(f"    Hardware Accuracy     {c(f'{acc:.1f}%', BOLD)}   ({correct}/{n} correct)")
    say(f"    Hardware vs Software  {c(f'{100.0 * agree / n:.1f}%', BOLD)}   "
        f"({agree}/{n} predictions identical)")
    print()

    # Latency is MEASURED, not estimated.  When a --timer-base was given, the
    # on-board AXI timer gives the TRUE fabric latency per image -- that is the
    # headline figure.  Wall-clock (host staging + PYNQ overhead + FPGA) is far
    # larger and shown only as a dim reference.  Throughput is quoted from the
    # figure that headlines, so it reflects the accelerator when the timer is on.
    mean_wall_ms = sum(per_image_wall_ms) / len(per_image_wall_ms)

    if per_image_hw_ms:
        mean_hw_ms = sum(per_image_hw_ms) / len(per_image_hw_ms)
        fps = 1000.0 / mean_hw_ms
        say(f"    FPGA latency          {c(f'{mean_hw_ms:.2f} ms', BOLD)} / image   "
            f"{c(f'@ {CLOCK_MHZ} MHz, fabric cycles', DIM)}")
        say(f"    Throughput            {c(f'{fps:.2f} images/s', BOLD)}   "
            f"{c('(fabric only, on-board timer)', DIM)}")
    else:
        fps = 1000.0 / mean_wall_ms
        say(f"    Wall-clock latency    {c(f'{mean_wall_ms:.1f} ms', BOLD)} / image   "
            f"{c('(host + FPGA, measured)', DIM)}")
        say(f"    Throughput            {c(f'{fps:.2f} images/s', BOLD)}   "
            f"{c('(wall-clock; pass --timer-base for true fabric latency)', DIM)}")
    print()

    if agree == n:
        say("    " + c(G["tick"], GREEN) + c(" Hardware datapath reproduces the software "
                                             "model on every image!", GREEN))
    else:
        say("    " + c(G["cross"], RED) +
            c(f" {n - agree} image(s) diverge from the software model.", RED))
    print()


if __name__ == "__main__":
    main()
