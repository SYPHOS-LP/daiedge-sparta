#!/usr/bin/env python3
"""SPARTA encoder demo — one console run of the whole end-to-end flow.

Wraps the three stages that normally run by hand (emit_e2e -> Vitis HLS csim ->
check_e2e) into a single narrated script, so the pipeline can be demonstrated and
recorded in one take:

    image --[SW embed]--> int8 (D x N) --[FPGA csim: 12 x encoder_layer_top]-->
          int8 --[SW head]--> logits --> prediction

Stage 2 is a Vitis HLS C-simulation, which costs minutes per image. Two modes:

  (default)  read the artifacts a previous csim already produced. Runs in seconds.
             This is the mode to record — the predictions are real, they were
             simply computed by an earlier csim rather than during the recording.

  --csim     actually invoke Vitis HLS. Correct but slow (~4 min/image); use it to
             regenerate the artifacts the default mode then reads.

Throughput comes from a separate C-SYNTHESIS measurement, NOT from this run:
C-simulation is functional only and reports no timing.

Usage:
    ../sparta-sw/.venv/Scripts/python.exe sim/demo_e2e.py
    ../sparta-sw/.venv/Scripts/python.exe sim/demo_e2e.py --images 4 --csim
"""
import os
import sys
import json
import time
import argparse
import subprocess

import warnings

# torch/brevitas emit UserWarnings on import and on the first forward; they would
# otherwise interleave into the narration mid-image and spoil a recording.
warnings.filterwarnings("ignore")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

E2E = os.path.join(_HERE, "e2e")
CIFAR = ["airplane", "automobile", "bird", "cat", "deer",
         "dog", "frog", "horse", "ship", "truck"]

# Measured C-synthesis latency: one synthesis run per encoder layer with that layer's
# real activation densities in the loop tripcounts (scripts/tripcount_sweep.py), summed
# over the 12 layers and converted at the signoff clock below. Update if re-measured.
LATENCY_MS = 265.9
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


def banner():
    print()
    title = "  SPARTA " + G["dot"] + " Sparse ViT Encoder Accelerator"
    sub = "  End-to-End inference " + G["dot"] + " Vitis HLS C-simulation"
    say(c(G["tl"] + G["hh"] * 62 + G["tr"], CYAN))
    say(c(G["v"], CYAN) + c(title.ljust(62), BOLD) + c(G["v"], CYAN))
    say(c(G["v"], CYAN) + sub.ljust(62) + c(G["v"], CYAN))
    say(c(G["bl"] + G["hh"] * 62 + G["br"], CYAN))
    print()


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


# ---------------------------------------------------------------- stages
def load_artifacts():
    """Read what the csim produced. Fails loudly if a stage never ran."""
    meta_p = os.path.join(E2E, "meta.json")
    out_p = os.path.join(E2E, "hw_out.bin")
    if not os.path.exists(meta_p):
        sys.exit(f"missing {meta_p}\n  run: python sim/emit_e2e.py N")
    if not os.path.exists(out_p):
        sys.exit(f"missing {out_p}\n  the csim has not been run for these inputs "
                 f"(see sim/README.md stage 2)")
    meta = json.load(open(meta_p))
    blob = open(out_p, "rb").read()
    hdr = np.frombuffer(blob[:16], dtype="<i4")
    if int(hdr[0]) != 0x45324F55:
        sys.exit(f"bad hw_out.bin magic {int(hdr[0]):#x}")
    n, D, N = int(hdr[1]), int(hdr[2]), int(hdr[3])
    if (n, D, N) != (meta["n"], meta["D"], meta["N"]):
        sys.exit(f"hw_out.bin header {(n, D, N)} != meta {(meta['n'], meta['D'], meta['N'])}\n"
                 f"  stale artifacts — re-run emit_e2e.py and the csim")
    hw = np.frombuffer(blob[16:], dtype=np.int8).reshape(n, D, N)
    return meta, hw


def run_csim(images):
    """Stage 1 + 2 for real. Slow; regenerates the artifacts the default mode reads."""
    py = sys.executable
    say(c("  stage 1/2  emitting payload (SW forward + weight compile)", BOLD))
    subprocess.run([py, os.path.join(_HERE, "emit_e2e.py"), str(images)],
                   cwd=_REPO, check=True)
    cfg = os.path.join("workspace", "encoder_e2e", "hls_config.cfg")
    say("")
    say(c("  stage 2/2  Vitis HLS C-simulation  " + DIM + "(minutes per image)", BOLD))
    subprocess.run([py, os.path.join("scripts", "gen_hls_config.py"),
                    os.path.join("config", "build_config.yaml"), cfg],
                   cwd=_REPO, check=True)
    env = dict(os.environ, E2E_DIR=E2E.replace("\\", "/") + "/")
    subprocess.run(["vitis-run", "--mode", "hls", "--csim", "--config", cfg,
                    "--work_dir", os.path.join("workspace", "encoder_e2e")],
                   cwd=_REPO, env=env, check=True)


def main():
    ap = argparse.ArgumentParser(description="SPARTA encoder end-to-end demo")
    ap.add_argument("--images", type=int, default=None,
                    help="how many images to run (default: all in the artifacts)")
    ap.add_argument("--csim", action="store_true",
                    help="really run Vitis HLS (slow) instead of replaying artifacts")
    ap.add_argument("--speed", type=float, default=0.8,
                    help="pacing multiplier for the narration (0 = no pauses)")
    args = ap.parse_args()

    sp = args.speed
    banner()

    if args.csim:
        run_csim(args.images or 4)
        print()

    # ---- setup -----------------------------------------------------------
    say(c("  Initialising", BOLD))
    meta, hw = load_artifacts()
    n = min(args.images, meta["n"]) if args.images else meta["n"]
    D, N = meta["D"], meta["N"]

    say(f"    model      {c('12-layer ViT encoder', CYAN)}  ·  FEATURES={D}  TOKENS={N}")
    say(f"    weights    4-bit POT ·  int8 Activations")
    say(f"    kernel     {c('encoder_layer_top', CYAN)}  ·  Vitis HLS C-simulation")
    say(f"    dataset    CIFAR-10  ·  {n} image{'s' if n != 1 else ''}", 0.6 * sp)
    print()

    # Build the SW model once — it supplies the embedding and the head that sit
    # either side of the accelerator.
    # Reported as two steps because they cost very differently: the torch/brevitas
    # import is ~90% of the wait, the checkpoint load only a few seconds. Attributing
    # all of it to "loading the model" reads like a hang.
    dot0 = c(G["dot"], DIM)
    say(c("  Importing deep-learning stack", BOLD) +
        c("   (torch + brevitas - one-off)", DIM))
    t_imp = time.time()
    import full_model_infer as F
    import toml
    F._CFG = meta.get("cfg") or F._CFG
    iq = F._load_infer_module()
    say(f"    {c('ready', GREEN)}  {dot0} {time.time() - t_imp:.1f}s")
    print()

    say(c("  Loading software model", BOLD) +
        c("   (patch embeddings + classifier head)", DIM))
    t_mod = time.time()
    cfg = toml.load(F._CFG)
    model = F.build_sw_model(cfg, iq)
    say(f"    {c('ready', GREEN)}  {dot0} {time.time() - t_mod:.1f}s")
    print()
    rule()
    print()

    # ---- per-image -------------------------------------------------------
    truth = meta["truth"]
    sw_pred = meta["sw_pred"]
    out_scale = meta["out_scale"]
    hw_pred = []
    t_start = time.time()

    for i in range(n):
        say(c(f"  IMAGE {i + 1}/{n}", BOLD) + c(f"   ground truth: {CIFAR[truth[i]]}", DIM))
        dot = c(G["dot"], DIM)
        arr = "->" if not _UNICODE else "→"
        say(f"    loading                     {dot} CIFAR-10 sample {i}", 0.10 * sp)
        say(f"    software embedding          {dot} patchify {arr} int8 "
            f"[{D}x{N}]", 0.10 * sp)

        progress(f"inference  12 layers on FPGA", 0.5 * sp)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logits = F.run_head(model, hw[i], out_scale)
        pred = int(np.asarray(logits).argmax())
        hw_pred.append(pred)

        ok = pred == truth[i]
        match = pred == sw_pred[i]
        say(f"    classifier head             {dot} logits {arr} argmax")
        say("    prediction                  " + c(G["arrow"] + " ", CYAN) +
            c(CIFAR[pred].upper(), BOLD if not _COLOUR else GREEN if ok else YELLOW) +
            ("   " + c(G["tick"] + " correct", GREEN) if ok
             else "   " + c(G["cross"] + " incorrect", RED)))
        say(f"    vs software model           {dot} " +
            (c("match", GREEN) if match else c("DIVERGES", RED)))
        print()

    elapsed = time.time() - t_start

    # ---- results ---------------------------------------------------------
    rule(G["hh"])
    say(c("  RESULTS", BOLD))
    rule(G["hh"])
    print()

    correct = sum(int(p == t) for p, t in zip(hw_pred, truth))
    sw_correct = sum(int(p == t) for p, t in zip(sw_pred[:n], truth[:n]))
    agree = sum(int(p == s) for p, s in zip(hw_pred, sw_pred[:n]))

    acc = 100.0 * correct / n
    say(f"    Software Accuracy     {100.0 * sw_correct / n:.1f}%   "
        f"({sw_correct}/{n} correct - baseline)")
    say(f"    Hardware Accuracy     {c(f'{acc:.1f}%', BOLD)}   ({correct}/{n} correct)")
    say(f"    Hardware vs Software  {c(f'{100.0 * agree / n:.1f}%', BOLD)}   "
        f"({agree}/{n} predictions identical)")
    print()

    fps = 1000.0 / LATENCY_MS
    say(f"    Latency               {c(f'{LATENCY_MS:.1f} ms', BOLD)} / image   "
        f"{c(f'@ {CLOCK_MHZ} MHz', DIM)}")
    say(f"    Throughput            {c(f'{fps:.2f} images/s', BOLD)}")
    say(c(f"                          from C-synthesis, per-layer at measured activation",
          DIM))
    say(c(f"                          densities — the C-simulation above models no timing",
          DIM))
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
