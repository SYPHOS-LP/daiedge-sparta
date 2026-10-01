#!/usr/bin/env python3
"""Python host: run the 12-layer ViT encoder on the Kria (xck26) accelerator.

Pipeline:
    X (768x50 int8, feature-major)
    for L in 0..11:  X = encoder_layer_top(X, weights[L], scales[L])   (on FPGA)

Run as a script this checks the accelerator against the simulation's golden data,
which needs nothing but numpy and pynq.  Classifying a real image needs the SW
embedding and classifier head as well -- that is infer_hw.py, which imports this
file for the hardware seam and brings the image pipeline with it.

Division of labour (all Python; PYNQ is the only hardware dependency):
  - SPARTALoader.Loader   : parse model.bin / manifest -> numpy views.
  - this file (Python)    : allocate + fill DDR buffers (PYNQ), the 12-layer loop,
                            ping-pong activations, golden comparison.
  - HwKernel (this file)  : the ONE hardware seam — programs the PL from the .bit
                            and invokes encoder_layer_top by writing the PHYSICAL
                            DDR ADDRESSES of those buffers into its s_axi_control
                            registers, then polling ap_done.

FLOW: this is the custom block-design path, NOT the XRT/Vitis path.  The BD in
scripts/encoder_layer_kr260_bd.tcl emits a .bit/.hwh pair (loadable by PYNQ or
xmutil), never an .xclbin, so there is no XRT kernel to bind arguments to --
the arguments ARE control registers, written directly.  Register offsets come
from the HLS-generated xencoder_layer_top_hw.h; see REGMAP below.

Requires: pynq (on-device) and numpy.  pynq is imported lazily so this file can
be linted/inspected on the dev host without it present.

BRING-UP, in this order:
  1. control path only -- no DDR, no ap_start:
       sudo -E python3 SPARTA.py --no-download --selftest --hw-header <x..._hw.h>
     If this fails, no datapath debugging is worth doing yet.
  2. datapath against the simulation's golden:
       sudo -E python3 SPARTA.py --no-download -d <payload> -e <sim/e2e_tiled12> \
            --golden-dir <sim/e2e/golden> --hw-header <x..._hw.h>
  3. real images end to end: infer_hw.py (needs torch + brevitas as well).

PASS --hw-header WHEREVER POSSIBLE.  The built-in REGMAP is a transcription and
goes stale the moment an argument is added or removed; the generated header
cannot.  With the header a mismatch is refused at construction, without it the
arguments are silently written to the wrong registers.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from SPARTALoader import Loader

# Encoder dims (config/inc/encoder_layer_cfg.h).  ENC_N=50 on the 32x32-patch
# baseline: a 7x7 patch grid + 1 CLS token (was 197 on the 16x16-patch model).
ENC_D = 768          # feature width           (ENC_D)
ENC_N = 50           # tokens                  (ENC_MAX_BATCH)
ACT_ELEMS = ENC_D * ENC_N   # the logical activation: D x N int8

# The kernel's activation ports carry PACKED 64-bit words, not bytes: ENC_PACK tokens per word,
# ENC_ROW_WORDS words per feature, a feature's trailing slots zero.  The host packs because on the
# port a byte at a time cost ~8 cycles each -- 310 k cycles to read one activation and 123 k to
# write it, against the 43 k csynth predicted (the port is 64 bits wide, so seven eighths of each
# transfer was empty and the byte requests did not merge into bursts).
ENC_PACK      = 8
ENC_ROW_WORDS = -(-ENC_N // ENC_PACK)          # 7 at N=50
ACT_WORDS     = ENC_D * ENC_ROW_WORDS          # what the DMA buffer holds
ACT_BYTES     = ACT_WORDS * ENC_PACK


def pack_rows(a):
    """(ENC_D, ENC_N) int8  ->  (ACT_WORDS,) uint64, the form the kernel's port expects."""
    padded = np.zeros((ENC_D, ENC_ROW_WORDS * ENC_PACK), dtype=np.uint8)
    padded[:, :ENC_N] = np.ascontiguousarray(a, dtype=np.int8).view(np.uint8)
    return padded.reshape(-1).view(np.uint64)


def unpack_rows(w):
    """(ACT_WORDS,) uint64  ->  (ENC_D, ENC_N) int8.  Drops the per-feature padding."""
    b = np.asarray(w, dtype=np.uint64).view(np.uint8).reshape(ENC_D, ENC_ROW_WORDS * ENC_PACK)
    return np.ascontiguousarray(b[:, :ENC_N]).view(np.int8)

# Fabric clock, for turning measured time into cycles.  Set by the app's device-tree overlay, NOT
# by anything here -- check it on the board with
#   sudo cat /sys/kernel/debug/clk/clk_summary | grep -i pl
# and override with --clock-mhz if it differs.
CLOCK_HZ = 190e6


# ============================================================================
# The kernel's argument set.  Every buffer field is a physical DDR address
# (uint64); the three dimensions are int32.
#
# Retained as a ctypes.Structure purely as a typed carrier: field NAMES are now
# what matters, since HwKernel looks each one up in REGMAP and writes it to its
# own address.  Field ORDER is no longer load-bearing -- the AXI-Lite map is
# addressed, not positional, which removes the silent mis-binding hazard the
# XRT set_arg path had.  Adding a kernel argument means adding it BOTH here and
# in REGMAP.
#
# Not present (removed with the 32x32-patch / on-chip-H baseline): gamma_mha,
# gamma_mlp, s_q_row/s_k_row/s_v_row/s_o_row/s1_row/s2_row (folded into
# `scales`), and the DDR h_val/h_col/h_ptr CSR (H now streams on-chip through
# the MLP's DATAFLOW region).  SPARTALoader still exposes gammas/scale_rows
# from model.bin -- they are simply no longer kernel arguments.
#
# All six weights are packed banks: the MLP's two linears moved onto the shared
# tiled engine, so W1/W2 take {bank, counts} like the MHA's four rather than the
# {values, col_idx, row_ptr} CSR triplets they used before.
# ============================================================================
class SpartaLayerArgs(ctypes.Structure):
    _fields_ = [
        ("input", ctypes.c_uint64),
        ("feature_dim", ctypes.c_int32),
        ("tokens_dim", ctypes.c_int32),
        ("intermediate_dim", ctypes.c_int32),
        ("wq_bank", ctypes.c_uint64), ("wq_counts", ctypes.c_uint64),
        ("wk_bank", ctypes.c_uint64), ("wk_counts", ctypes.c_uint64),
        ("wv_bank", ctypes.c_uint64), ("wv_counts", ctypes.c_uint64),
        ("wo_bank", ctypes.c_uint64), ("wo_counts", ctypes.c_uint64),
        ("w1_bank", ctypes.c_uint64), ("w1_counts", ctypes.c_uint64),
        ("w2_bank", ctypes.c_uint64), ("w2_counts", ctypes.c_uint64),
        ("scales", ctypes.c_uint64),
        ("output", ctypes.c_uint64),
        # The kernel keeps the activation on-chip between layers: only the first layer reads it
        # from DDR and only the last writes it back.
        ("first_layer", ctypes.c_uint32),
        ("last_layer", ctypes.c_uint32),
    ]


# ============================================================================
# s_axi_control register map  (xencoder_layer_top_hw.h, generated by Vitis HLS
# alongside the IP; confirmed against the on-board enc_system_wrapper.hwh).
#
# Two register widths, and they are NOT interchangeable:
#   - the three dimension scalars are 32-bit  -> ONE write
#   - every pointer is 64-bit                 -> TWO writes, low word first
# Writing a pointer as a single 32-bit store leaves the high word holding
# whatever the previous layer put there; on a board with >4GB visible that
# aims the kernel at an arbitrary physical address rather than faulting.
# ============================================================================
AP_CTRL = 0x000          # control/status register
AP_START    = 0x1        # bit0  W/COH -- kernel clears it when it accepts
AP_DONE     = 0x2        # bit1  R     -- LATCHED until ap_continue (see invoke)
AP_IDLE     = 0x4        # bit2  R
AP_READY    = 0x8        # bit3  R/COR
AP_CONTINUE = 0x10       # bit4  W/SC  -- releases the latched done state

# name -> (offset, is_64bit).  Order is documentation only: the AXI-Lite map is
# addressed, not positional, so unlike the XRT set_arg path a misordered entry
# here cannot silently shift the other arguments.
# Transcribed from workspace/encoder_layer/hls/impl/misc/drivers/
# encoder_layer_top_v1_0/src/xencoder_layer_top_hw.h of the CURRENT build.
#
# This table is a convenience for bring-up; prefer --hw-header, which reads the
# generated header directly and therefore cannot drift.  It matters because the
# offsets shift whenever an argument is added or removed: when the MLP's two CSR
# triplets became {bank, counts} pairs the list lost two entries, and every
# offset from w2_bank onward moved down by two slots.  A table left at the old
# offsets does not fail -- it silently aims `output` and `scales` at whatever
# address the neighbouring argument left behind.
REGMAP = {
    "input":            (0x010, True),
    "feature_dim":      (0x01c, False),
    "tokens_dim":       (0x024, False),
    "intermediate_dim": (0x02c, False),
    "wq_bank": (0x034, True), "wq_counts": (0x040, True),
    "wk_bank": (0x04c, True), "wk_counts": (0x058, True),
    "wv_bank": (0x064, True), "wv_counts": (0x070, True),
    "wo_bank": (0x07c, True), "wo_counts": (0x088, True),
    "w1_bank": (0x094, True), "w1_counts": (0x0a0, True),
    "w2_bank": (0x0ac, True), "w2_counts": (0x0b8, True),
    "scales":  (0x0c4, True),
    "output":  (0x0d0, True),
    "first_layer": (0x0dc, False), "last_layer": (0x0e4, False),
}


def load_regmap(hw_header):
    """Register map from the HLS-generated x<top>_hw.h (e.g. xencoder_layer_top_hw.h).

    Reads each '#define X..._CONTROL_ADDR_<NAME>_DATA <offset>' with its matching
    '..._BITS_<NAME>_DATA <width>'.  HLS appends _R to names that are reserved in
    C/RTL (output -> OUTPUT_R), which is stripped to recover the argument name."""
    define = re.compile(r"#define\s+\w+_CONTROL_(ADDR|BITS)_(\w+)_DATA\s+(\w+)")
    addr, bits = {}, {}
    with open(hw_header) as f:
        for line in f:
            m = define.match(line.strip())
            if not m:
                continue
            name = m.group(2).lower()
            if name.endswith("_r"):
                name = name[:-2]
            (addr if m.group(1) == "ADDR" else bits)[name] = int(m.group(3), 0)
    return {name: (off, bits.get(name, 32) > 32) for name, off in addr.items()}


def _cma_total_bytes(default=1 << 30):
    """Size of the CMA pool, for the PS-DDR bank descriptor.

    Only used as the bank's declared size; allocation still fails normally if
    CMA runs out.  /proc/meminfo reports it in kB."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("CmaTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return default


class HwKernel:
    """encoder_layer_top over PYNQ: programs the PL and invokes via AXI-Lite.

    Replaces the libsparta.so/XRT seam.  The custom block design (see
    scripts/encoder_layer_kr260_bd.tcl) produces a .bit/.hwh pair, not an
    .xclbin, so there is no XRT kernel object to bind arguments to -- the
    arguments ARE the control registers, written directly.
    """

    # s_axi_control base address, from the BD's address map (confirmed in
    # enc_system_wrapper.hwh: BASEVALUE 0x80000000, HIGHVALUE 0x8000FFFF, via
    # M_AXI_HPM0_LPD).  Only used when attaching to an already-programmed PL;
    # with an Overlay the address comes from the metadata instead.
    CONTROL_BASE = 0x8000_0000
    CONTROL_SIZE = 0x1_0000

    # AXI Timer register offsets (PG079). Only Timer0 is used.
    TMR_TCSR0 = 0x00   # control/status
    TMR_TLR0  = 0x04   # load register
    TMR_TCR0  = 0x08   # counter value (what we sample)
    TMR_SIZE  = 0x1_0000
    # TCSR0 bits: ENT0 (enable, bit7) | ARHT0 (auto-reload, bit4) | UDT0=0 (up).
    # ENT0|ARHT0 = a free-running up-counter that wraps at 2^32.
    TMR_TCSR0_RUN = (1 << 7) | (1 << 4)

    def __init__(self, bitstream=None, instance=None, download=True,
                 timeout_s=60.0, base_addr=None, timer_base=None, regmap=None):
        self.timeout_s = timeout_s
        # Register map: from the generated header (load_regmap) when given, else REGMAP.
        self.regmap = regmap if regmap is not None else REGMAP
        missing = [name for name, _ in SpartaLayerArgs._fields_ if name not in self.regmap]
        if missing:
            raise ValueError(f"register map has no offsets for kernel arguments {missing}; "
                             "is the header from a different kernel build?")
        self.overlay = None
        # Optional diagnostic AXI timer: a free-running counter on the
        # kernel's ap_clk. When present, invoke() returns fabric cycles too.
        # Set up lazily below only if a base address is provided or the
        # timer IP is discoverable in an overlay.
        self.timer = None
        self._timer_base = timer_base
        self.last_cycles = None   # fabric cycles of the most recent invoke()

        if download:
            # Program the PL and let PYNQ resolve the IP from the metadata.
            from pynq import Overlay
            self.overlay = Overlay(bitstream, download=True)
            self.ip = self._find_ip(instance)
        else:
            # Attach to a PL programmed elsewhere (xmutil).  Deliberately does
            # NOT build an Overlay: that would parse the .hwh, and PYNQ 3.0.1's
            # pynqmetadata (0.1.2) cannot read a Vivado 2025.2 .hwh
            # ("Unknown file format").  Nothing here needs the metadata -- the
            # control base is fixed by the block design and the register
            # offsets come from REGMAP -- so map the register block directly.
            from pynq import MMIO
            self.ip = MMIO(base_addr or self.CONTROL_BASE, self.CONTROL_SIZE)
            self._register_ps_memory()

        self._setup_timer(instance)

    def _setup_timer(self, instance):
        """Map the diagnostic AXI timer, if this bitstream has one.

        On the attach path the timer base is passed explicitly (--timer-base).  
        On the overlay path we try to discover an axi_timer IP from the metadata.  
        Absent either, self.timer stays None and invoke() reports no cycles,
        so a non-debug bitstream just works with the timing feature silently off.

        The host reads the counter itself, around ap_start and ap_done; the kernel
        is not involved and has no timer argument."""
        base = self._timer_base
        if base is None and self.overlay is not None:
            # Best-effort discovery in the overlay's IP dict.
            ipd = getattr(self.overlay, "ip_dict", {}) or {}
            for name, meta in ipd.items():
                if "axi_timer" in name.lower():
                    base = meta.get("phys_addr")
                    break
        if base is None:
            return
        from pynq import MMIO
        self.timer = MMIO(base, self.TMR_SIZE)
        # Arm Timer0 as a free-running up-counter: clear, load 0, then run.
        self.timer.write(self.TMR_TCSR0, 0)
        self.timer.write(self.TMR_TLR0, 0)
        # LOAD0 (bit5) pulses TLR0 -> TCR0, then drop it and enable+auto-reload.
        self.timer.write(self.TMR_TCSR0, (1 << 5))
        self.timer.write(self.TMR_TCSR0, self.TMR_TCSR0_RUN)

    def _timer_cycles(self):
        """Read Timer0's free-running counter (32-bit), or None if no timer."""
        if self.timer is None:
            return None
        return self.timer.read(self.TMR_TCR0) & 0xFFFFFFFF

    @staticmethod
    def _register_ps_memory():
        """Give PYNQ a PS-DDR memory bank so pynq.allocate() works.

        allocate() -> Device.allocate() -> Device.default_memory, and on the
        embedded device default_memory is derived from an overlay's mem_dict
        (or a global-state file written when PYNQ downloads a bitstream).
        Attaching to an xmutil-loaded PL leaves both absent, so allocate()
        dies with "Overlay is not downloaded" even though CMA is perfectly
        usable.

        default_memory picks the mem_dict entry whose base_address is 0, so
        publishing a single PS-DDR descriptor there restores the normal path.
        The buffers still come from PYNQ's CMA allocator -- this only tells
        PYNQ which bank to use, it does not bypass it.  That matters: the
        kernel's m_axi masters DMA to physical addresses with no IOMMU, so the
        buffers must stay contiguous and CMA-backed.
        """
        from pynq.pl_server import Device

        dev = Device.active_device
        if getattr(dev, "mem_dict", None):
            return                              # already populated
        # XrtMemory reads exactly these three keys; idx is the memory-bank
        # index handed to the allocator, and 0 is the PS-DDR bank.
        dev.mem_dict = {
            "PSDDR": {
                "idx": 0,
                "base_address": 0,
                "size": _cma_total_bytes(),
                "type": "PSDDR",
                "streaming": False,
            }
        }

    def _find_ip(self, instance):
        """Resolve the kernel's IP handle, by name or by scanning the overlay.

        The BD names the cell (enc_top_0 in the current design), so the name is
        not guessable from the kernel name -- discovery matches on the HLS VLNV
        instead, which survives a BD rename."""
        if instance is not None:
            return getattr(self.overlay, instance)

        matches = [n for n, d in self.overlay.ip_dict.items()
                   if "encoder_layer_top" in d.get("type", "")]
        if not matches:
            raise RuntimeError(
                "no encoder_layer_top IP in this overlay; found: "
                f"{sorted(self.overlay.ip_dict)}.  Pass instance=... explicitly.")
        if len(matches) > 1:
            raise RuntimeError(f"ambiguous encoder_layer_top instances: {matches}")
        return getattr(self.overlay, matches[0].split("/")[-1])

    def _write_arg(self, name, value):
        off, is64 = self.regmap[name]
        if is64:
            self.ip.write(off,     value        & 0xFFFFFFFF)
            self.ip.write(off + 4, (value >> 32) & 0xFFFFFFFF)
        else:
            self.ip.write(off, value & 0xFFFFFFFF)

    def read_arg(self, name):
        """Read one argument register back (all data registers are Read/Write)."""
        off, is64 = self.regmap[name]
        lo = self.ip.read(off)
        return (self.ip.read(off + 4) << 32 | lo) if is64 else lo

    def selftest(self, verbose=True):
        """Write/read-back every argument register; return list of failures.

        This checks ONE thing: that the PS can address the kernel's AXI-Lite
        control interface correctly.  It does not start the kernel and needs no
        DDR buffers, so it cleanly separates a control-path problem from a
        datapath one -- run it first on any new bitstream.

        The patterns are chosen to catch the failure modes this board has
        actually produced, not just stuck bits:
          - 0xA5A5A5A5 / 0x5A5A5A5A : alternating bits, catches stuck/shorted
            data lines and a byte lane that is not being driven.
          - 0xFFFFFFFF / 0x00000000 : all-ones then all-zeros, catches a lane
            stuck in one direction only.
          - a per-register unique value : catches ADDRESS aliasing -- two
            registers that decode to the same location still pass every
            uniform pattern, because each one reads back the value it just
            wrote.  This is the pattern that catches the GP0 width-converter
            bug (writes to non-16-byte-aligned offsets silently dropped),
            which uniform patterns alone can miss.
        """
        failures = []

        def check(name, off, wrote, got, note):
            if wrote != got:
                failures.append(
                    f"{name}@0x{off:03x} {note}: wrote 0x{wrote:08x}, "
                    f"read 0x{got:08x}")

        # Per-32-bit-word view: a 64-bit arg is two independently addressed
        # words, and only testing the pair together would hide a dead high word.
        words = []
        for name, (off, is64) in self.regmap.items():
            words.append((f"{name}.lo" if is64 else name, off))
            if is64:
                words.append((f"{name}.hi", off + 4))

        for pattern in (0xA5A5A5A5, 0x5A5A5A5A, 0xFFFFFFFF, 0x00000000):
            for name, off in words:
                self.ip.write(off, pattern)
            for name, off in words:
                check(name, off, pattern, self.ip.read(off), f"pattern 0x{pattern:08x}")

        # Address-aliasing pass: every word gets a DIFFERENT value, all written
        # before any is read back, so an aliased pair shows up as one clobbering
        # the other.
        unique = {off: (0xC0DE0000 | (i * 4)) for i, (_, off) in enumerate(words)}
        for _, off in words:
            self.ip.write(off, unique[off])
        for name, off in words:
            check(name, off, unique[off], self.ip.read(off), "unique/aliasing")

        # Leave the control registers zeroed rather than full of test junk: a
        # later ap_start must not inherit a stale pointer from the selftest.
        for _, off in words:
            self.ip.write(off, 0)

        if verbose:
            ctrl = self.ip.read(AP_CTRL)
            print(f"control base : {self._base_addr_str()}")
            print(f"ap_ctrl      : 0x{ctrl:08x} "
                  f"(start={bool(ctrl & AP_START)}, done={bool(ctrl & AP_DONE)}, "
                  f"idle={bool(ctrl & AP_IDLE)}, ready={bool(ctrl & AP_READY)})")
            if ctrl & AP_DONE:
                print("               ^ done is LATCHED from an earlier run "
                      "(ap_ctrl_chain); invoke() clears it before starting.")
            print(f"registers    : {len(words)} 32-bit words "
                  f"({len(self.regmap)} kernel arguments)")
            print(f"patterns     : 5 (4 uniform + 1 unique/aliasing)")
            if failures:
                print(f"\nFAIL -- {len(failures)} register(s) did not read back:")
                for f in failures:
                    print(f"  {f}")
            else:
                print("\nPASS -- every argument register read back correctly.")
        return failures

    def _base_addr_str(self):
        """The IP's control base, for selftest output (best-effort).

        An Overlay-resolved IP keeps its MMIO in .mmio; a directly-mapped one
        (the attach path) IS the MMIO."""
        for obj in (getattr(self.ip, "mmio", None), self.ip):
            addr = getattr(obj, "base_addr", None)
            if addr is not None:
                return f"0x{addr:08x}"
        return "unknown"

    def invoke(self, args):
        """Write every kernel argument, start the kernel, block until it completes.

        `args` is a SpartaLayerArgs -- kept as the argument-carrying type so the
        buffer-address plumbing is shared with the XRT path it replaces.

        This block is ap_ctrl_chain, NOT ap_ctrl_hs (confirmed in the generated
        encoder_layer_top_control_s_axi.v: "bit 4 - ap_continue", and
        int_task_ap_done <= task_ap_done && !int_ap_continue).  So ap_done
        LATCHES after each run and the kernel refuses further starts until
        ap_continue is written.  Handshake per invocation:

            1. drain a latched ap_done from the previous run (ap_continue)
            2. wait for ap_idle -- proof the kernel really is ready
            3. write ap_start
            4. wait for ap_done
            5. write ap_continue to release the done state

        Skipping step 1/5 is not a slow failure but a SILENT one: ap_done reads
        as already-set, the wait falls straight through, and the caller gets an
        untouched output buffer that looks like a successful 1ms run."""
        import time

        # ---- 1+2: make sure the kernel is idle and not parked in done -------
        self._drain_done()

        for name, _ in SpartaLayerArgs._fields_:
            self._write_arg(name, getattr(args, name))

        # ---- 3: start ------------------------------------------------------
        # _drain_done() guarantees ap_done is clear here, so the ap_done seen
        # below can only come from THIS run.
        # Sample the fabric counter as close to ap_start as possible; pair it with
        # a read taken right after ap_done to get true ap_start->ap_done cycles.
        t_start = time.monotonic()
        cyc_start = self._timer_cycles()
        self.ip.write(AP_CTRL, AP_START)

        # ---- 4: wait for completion ---------------------------------------
        deadline = t_start + self.timeout_s
        while True:
            ctrl = self.ip.read(AP_CTRL)
            if ctrl & AP_DONE:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"encoder_layer_top did not assert ap_done within "
                    f"{self.timeout_s}s (ap_ctrl=0x{ctrl:08x}, "
                    f"start={bool(ctrl & AP_START)}, idle={bool(ctrl & AP_IDLE)}). "
                    "A hung kernel usually means a bad buffer address or an "
                    "m_axi bundle that does not reach a PS port.")
        cyc_end = self._timer_cycles()

        # ---- 5: release the done state so the next invoke can start --------
        self.ip.write(AP_CTRL, AP_CONTINUE)

        # Stash fabric cycles for the caller (None if no timer).  The counter is
        # 32-bit and free-running, so subtract modulo 2^32 to survive one wrap.
        if cyc_start is not None and cyc_end is not None:
            self.last_cycles = (cyc_end - cyc_start) & 0xFFFFFFFF
        else:
            self.last_cycles = None
        return time.monotonic() - t_start

    def _drain_done(self):
        """Clear a latched ap_done and wait until the kernel reports idle.

        On ap_ctrl_chain the done state persists across invocations, so this
        runs BEFORE every start rather than only at init -- an aborted run
        (Ctrl-C, an exception between start and continue) would otherwise leave
        the block latched for the next process to trip over."""
        import time

        deadline = time.monotonic() + self.timeout_s
        while True:
            ctrl = self.ip.read(AP_CTRL)
            # Ready to start only when idle AND no completion still latched.
            # Checking idle alone is not enough: the block reports
            # idle|done (0x6) while parked on an unacknowledged completion,
            # and starting from there leaves ap_done set so the next wait
            # falls straight through -- a silent no-op run.
            if (ctrl & AP_IDLE) and not (ctrl & AP_DONE):
                return
            if ctrl & AP_DONE:
                self.ip.write(AP_CTRL, AP_CONTINUE)
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"encoder_layer_top is not startable (ap_ctrl=0x{ctrl:08x}, "
                    f"done={bool(ctrl & AP_DONE)}, idle={bool(ctrl & AP_IDLE)}). "
                    "If done stays latched despite ap_continue, or the block "
                    "never goes idle, reload the PL (xmutil) to reset it.")

    def close(self):
        pass    # the Overlay owns the PL; nothing per-kernel to release.


# ============================================================================
# Reference vectors from the simulation payload
#
# The image pipeline (patch embed, quantizer, classifier head) lives in
# infer_hw.py, which borrows it from the sparta-sw brevitas model.  What is here
# instead is the check that needs no torch: the encoder inputs and expected
# outputs that sim/ already produced, so the board can be held to the same
# bit-exact standard as the native testbench.
# ============================================================================
MAGIC_INPUTS = 0x45324E49    # 'IN2E', sim/emit_e2e.py
MAGIC_OUTPUT = 0x45324F55    # 'OU2E', tb/encoder_e2e_tb.cpp


def read_e2e_blob(path, magic):
    """(n_images, D, N) int8 from an inputs.bin / hw_out.bin, header checked."""
    import struct
    blob = open(path, "rb").read()
    if len(blob) < 16:
        raise SystemExit(f"{path}: too short to hold a header")
    m, n, d, tokens = struct.unpack("<I3i", blob[:16])
    if m != magic:
        raise SystemExit(f"{path}: bad magic {m:#x} (expected {magic:#x})")
    body = np.frombuffer(blob[16:], dtype=np.int8)
    if body.size < n * d * tokens:
        raise SystemExit(f"{path}: holds {body.size} bytes, header claims "
                         f"{n}x{d}x{tokens} = {n * d * tokens}")
    return body[:n * d * tokens].reshape(n, d, tokens)


# ============================================================================
# device buffers / kernel
# ============================================================================
def _allocate(pynq_allocate, nbytes, dtype=np.uint8):
    """Contiguous, cache-coherent DDR buffer (pynq.allocate wrapper)."""
    return pynq_allocate(shape=(nbytes,), dtype=dtype)


# The kernel bursts weight values/columns in wide words (up to 256 bits) and reads the
# final word whole, so every weight buffer must stay readable up to the next word boundary.
WIDE_READ_ALIGN = 64   # bytes


def _stage_layer(alloc, layer):
    """Copy one layer's payload blobs into fresh PYNQ DDR buffers and sync to card.
    Returns a dict of named device buffers used to wire the kernel args."""
    bufs = {}

    def stage(name, arr):
        raw = np.frombuffer(np.ascontiguousarray(arr).tobytes(), dtype=np.uint8)
        padded = -(-raw.size // WIDE_READ_ALIGN) * WIDE_READ_ALIGN
        b = _allocate(alloc, padded, np.uint8)
        b[:raw.size] = raw
        b[raw.size:] = 0
        b.sync_to_device()
        bufs[name] = b

    for tag, w in layer.weights.items():
        if w.layout == "tiled_bank":
            stage(f"{tag}.bank",   w.bank)
            stage(f"{tag}.counts", w.counts)
        else:
            stage(f"{tag}.values", w.values)
            stage(f"{tag}.col",    w.col)
            stage(f"{tag}.rowptr", w.rowptr)
    stage("scales", layer.scales)
    # NOT staged: scale_rows and gammas.  Neither is a kernel argument on this
    # baseline -- the per-row requant folded into `scales` and the gammas into
    # the datapath.  SPARTALoader still exposes them (older manifests carry
    # them), but copying them to DDR would cost a buffer each, per layer, for
    # data the kernel never reads.
    return bufs


def stage_model(alloc, loader):
    """Stage every layer's payload into DDR once: {layer: named device buffers}.

    The whole 12-layer model is ~18 MB, so it stays resident for the lifetime of the
    kernel and per-image work is only activation I/O.  Free with free_staged()."""
    return {L: _stage_layer(alloc, loader.layer(L)) for L in loader.layer_indices()}


def free_staged(staged):
    for bufs in staged.values():
        for buf in bufs.values():
            buf.freebuffer()


def _addr(buf):
    """Physical DDR address of a PYNQ buffer (what the C invoke ABI wants)."""
    return int(buf.device_address)


def _build_args(x_buf, y_buf, lb, n_tokens, d, d_h, first_layer=1, last_layer=1):
    """Fill a SpartaLayerArgs from PYNQ buffer device addresses.

    Field order matches encoder_layer_top()'s signature.  The weight tag names
    (`w_q.bank`, `linear_1.rowptr`, ...) come from compile_model.py's
    manifest via SPARTALoader and are unaffected by the runtime flow."""
    return SpartaLayerArgs(
        input=_addr(x_buf),
        feature_dim=d, tokens_dim=n_tokens, intermediate_dim=d_h,
        wq_bank=_addr(lb["w_q.bank"]), wq_counts=_addr(lb["w_q.counts"]),
        wk_bank=_addr(lb["w_k.bank"]), wk_counts=_addr(lb["w_k.counts"]),
        wv_bank=_addr(lb["w_v.bank"]), wv_counts=_addr(lb["w_v.counts"]),
        wo_bank=_addr(lb["out_proj.bank"]), wo_counts=_addr(lb["out_proj.counts"]),
        w1_bank=_addr(lb["linear_1.bank"]), w1_counts=_addr(lb["linear_1.counts"]),
        w2_bank=_addr(lb["linear_2.bank"]), w2_counts=_addr(lb["linear_2.counts"]),
        scales=_addr(lb["scales"]),
        first_layer=int(first_layer), last_layer=int(last_layer),
        output=_addr(y_buf),
    )


def run_encoder(model_dir, e2e_dir, n_images=1, bitstream=None, instance=None,
                download=True, base_addr=None, hw_header=None, golden_dir=None,
                n_layers=None):
    """Run encoder inputs from the simulation payload and compare against its golden.

    This is the datapath check that needs no torch: `e2e_dir/inputs.bin` holds the
    exact int8 encoder inputs the native testbench uses, and the golden holds what
    it produced, so the board is held to the same bit-exact standard.  A pass means
    the whole chain agrees -- register map, buffer addresses, weight payload and
    arithmetic.  For classifying real images use infer_hw.py, which brings the
    image pipeline with it.

    Returns the number of mismatching images (0 on success).
    """
    from pynq import allocate            # on-device only; import lazily.

    inputs = read_e2e_blob(os.path.join(e2e_dir, "inputs.bin"), MAGIC_INPUTS)
    want = read_e2e_blob(os.path.join(golden_dir or os.path.join(e2e_dir, "golden"),
                                      "hw_out.bin"), MAGIC_OUTPUT)
    if inputs.shape[1:] != (ENC_D, ENC_N):
        raise SystemExit(f"payload is {inputs.shape[1]}x{inputs.shape[2]}, "
                         f"this build is ENC_D={ENC_D} x ENC_N={ENC_N}")
    n_images = min(n_images, inputs.shape[0], want.shape[0])

    mp = Loader(model_dir)
    layers = mp.layer_indices()
    if n_layers is not None:
        # Running a prefix of the encoder leaves the result partway through, so the golden
        # comparison is meaningless -- this is for timing, not accuracy.
        layers = layers[:n_layers]
    kernel = HwKernel(bitstream, instance=instance, download=download,
                      base_addr=base_addr,
                      regmap=load_regmap(hw_header) if hw_header else None)

    # activation ping-pong: X (in) / Y (out), in the packed word form the ports carry.
    a = allocate(shape=(ACT_WORDS,), dtype=np.uint64)
    b = allocate(shape=(ACT_WORDS,), dtype=np.uint64)
    staged = stage_model(allocate, mp)
    d_h = {L: mp.layer(L).weights["linear_1"].shape[0] for L in layers}

    bad = 0
    per_image = []
    per_layer_s = {}       # layer position -> summed kernel time, to see if the twelve differ
    # invoke() returns its own ap_start -> ap_done wall time, so summing those separates the
    # kernel's time from the host's register writes and struct building.  Worth having: the
    # difference is how much of the frame rate Python is responsible for.
    per_image_kernel = []
    try:
        for img in range(n_images):
            t_img = time.monotonic()
            a[:] = pack_rows(inputs[img])
            a.sync_to_device()
            x_buf, y_buf = a, b

            kernel_s = 0.0
            for n, L in enumerate(layers):
                # The activation stays on chip between layers: only the first reads it
                # from DDR and only the last writes it back.
                args = _build_args(x_buf, y_buf, staged[L], ENC_N, ENC_D, d_h[L],
                                   first_layer=(L == layers[0]),
                                   last_layer=(L == layers[-1]))
                before = kernel_s
                try:
                    kernel_s += kernel.invoke(args)
                except TimeoutError as e:
                    raise RuntimeError(f"image {img} layer {L} hung: {e}") from e
                per_layer_s[n] = per_layer_s.get(n, 0.0) + (kernel_s - before)
                x_buf, y_buf = y_buf, x_buf

            x_buf.sync_from_device()
            per_image.append(time.monotonic() - t_img)
            per_image_kernel.append(kernel_s)
            got = unpack_rows(np.array(x_buf[:ACT_WORDS], dtype=np.uint64))
            diff = int(np.count_nonzero(got != want[img]))
            if diff:
                bad += 1
                worst = int(np.abs(got.astype(int) - want[img].astype(int)).max())
                print(f"  image {img}: {diff} of {ACT_ELEMS} bytes differ, "
                      f"max |difference| {worst}")
            else:
                print(f"  image {img}: bit-exact")
    finally:
        free_staged(staged)
        kernel.close()

    print(f"\n{n_images - bad}/{n_images} images bit-exact")
    if per_image:
        # Wall clock per image, which includes the host's per-layer register writes and the
        # ap_start/ap_done handshake -- so it is the end-to-end rate, not the fabric's alone.
        best, mean = min(per_image), sum(per_image) / len(per_image)
        print(f"per image: {1000 * mean:.1f} ms mean, {1000 * best:.1f} ms best"
              f"  ({1.0 / mean:.1f} FPS mean, {1.0 / best:.1f} FPS best)")
        print(f"per layer: {1000 * mean / 12:.2f} ms mean over {len(per_image)} image(s)")
        kmean = sum(per_image_kernel) / len(per_image_kernel)
        print(f"  of which kernel (ap_start -> ap_done): {1000 * kmean:.1f} ms "
              f"({100 * kmean / mean:.1f} %), host {1000 * (mean - kmean):.1f} ms")
        print(f"  kernel per layer: {1000 * kmean / len(layers):.2f} ms"
              f"  =  {kmean / len(layers) * CLOCK_HZ / 1e6:.2f} M cycles at "
              f"{CLOCK_HZ / 1e6:.0f} MHz")
        # Layers 0 and the last are the only ones that convert the activation to/from DDR
        # (first_layer / last_layer), so they should stand out by a known amount.
        print("\n  per-layer kernel time (ms) and cycles:")
        for n in sorted(per_layer_s):
            t = per_layer_s[n] / len(per_image)
            print(f"    layer {layers[n]:2d}: {1000 * t:7.3f} ms  "
                  f"{t * CLOCK_HZ / 1e6:6.3f} M cycles")
    return bad


def main():
    global CLOCK_HZ
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--model-dir", default=None,
                    help="dir with model.bin + manifest.json (from compile_model.py); "
                         "not needed with --selftest")
    ap.add_argument("-b", "--bitstream", default=None,
                    help="overlay to program (.bit, with its .hwh alongside); "
                         "not needed with --no-download")
    ap.add_argument("--instance", default=None,
                    help="BD cell name of the kernel (default: discover it)")
    ap.add_argument("--no-download", action="store_true",
                    help="attach to a PL already programmed (e.g. by xmutil) "
                         "instead of reprogramming it; maps the control "
                         "registers directly and needs no .bit/.hwh")
    ap.add_argument("--base-addr", default=None,
                    help=f"s_axi_control base address when attaching "
                         f"(default 0x{HwKernel.CONTROL_BASE:08x})")
    ap.add_argument("--selftest", action="store_true",
                    help="write/read-back every control register and exit; "
                         "checks the AXI-Lite control path only (no DDR, no "
                         "ap_start). Run this first on a new bitstream.")
    ap.add_argument("-e", "--e2e-dir", default=None,
                    help="sim payload dir with inputs.bin (and golden/hw_out.bin) to "
                         "check against; not needed with --selftest")
    ap.add_argument("--golden-dir", default=None,
                    help="dir holding the reference hw_out.bin (default: <e2e-dir>/golden). "
                         "A payload emitted for a different engine configuration is still "
                         "checked against the ORIGINAL design's golden")
    ap.add_argument("-n", "--images", type=int, default=1,
                    help="how many images from the payload to run (default 1)")
    ap.add_argument("--n-layers", type=int, default=None,
                    help="run only the first N encoder layers; for TIMING only, the result is "
                         "partway through the encoder so the golden comparison will fail")
    ap.add_argument("--clock-mhz", type=float, default=CLOCK_HZ / 1e6,
                    help="fabric clock, for reporting cycles (default %(default)s); the app's "
                         "device-tree overlay sets the real one")
    ap.add_argument("--hw-header", default=None,
                    help="HLS-generated xencoder_layer_top_hw.h to read the register "
                         "offsets from (default: the built-in REGMAP)")
    args = ap.parse_args()
    regmap = load_regmap(args.hw_header) if args.hw_header else None

    if not args.bitstream and not args.no_download:
        ap.error("-b/--bitstream is required unless --no-download is given")
    base = int(args.base_addr, 0) if args.base_addr else None
    CLOCK_HZ = args.clock_mhz * 1e6

    if args.selftest:
        kernel = HwKernel(args.bitstream, instance=args.instance,
                          download=not args.no_download, base_addr=base, regmap=regmap)
        failures = kernel.selftest()
        return 1 if failures else 0

    if not args.model_dir or not args.e2e_dir:
        ap.error("-d/--model-dir and -e/--e2e-dir are required unless --selftest is given")

    bad = run_encoder(args.model_dir, args.e2e_dir, n_images=args.images,
                      bitstream=args.bitstream, instance=args.instance,
                      download=not args.no_download, base_addr=base,
                      hw_header=args.hw_header, golden_dir=args.golden_dir,
                      n_layers=args.n_layers)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
