# Deploying to the KR260, start to finish

How to get from the quantized model export to the accelerator running on the board, and why each
step exists. Written after doing it on 2026-09-30; every step says whether it has actually been
exercised, because two of them have not and that is where to expect friction.

The board is a **Kria KR260** (`xck26-sfvc784-2LV-c`), confirmed by `xbutil examine` reporting
`Model : ZynqMP KR260 revB`. Earlier docs said KV260; they were wrong. Same SOM part either way.

---

## The shape of it

Seven stages. The first five run on the development host, the last two on the board.

|     | stage                     | command                                    | run?          |
| --- | ------------------------- | ------------------------------------------ | ------------- |
| 1   | model export -> payload   | `scripts/compile_model.py`                 | not by me     |
| 2   | reference vectors         | `sim/emit_e2e.py`                          | not by me     |
| 3   | **verify in simulation**  | `make regress`                             | yes           |
| 4   | kernel -> IP-XACT         | `make ip`                                  | yes           |
| 5a  | IP -> block design        | `scripts/encoder_layer_kr260_bd.tcl`       | yes           |
| 5b  | block design -> bitstream | `scripts/build_bitstream.tcl`              | yes           |
| 6   | bitstream -> loadable app | `scripts/package_app.sh`                   | no            |
| 7   | board bring-up            | `runtime/SPARTA.py`, `runtime/infer_hw.py` | selftest only |

Stage 3 is a gate, not a formality: it is the only stage that checks the arithmetic. Everything
after it checks plumbing. Do not skip it, and do not proceed past a failure in it -- a datapath bug
found on the board costs an order of magnitude more to localise than the same bug found here.

---

## Prerequisites

**Development host**

- Vitis / Vivado 2026.1. `source /opt/Xilinx/2026.1/Vitis/settings64.sh` for HLS and the packaging
  tools (`bootgen`, `dtc`); `.../Vivado/settings64.sh` for Vivado. Sourcing both is fine.
- KR260 board files. Check with `get_board_parts *kr260*` in a Vivado shell -- you want
  `xilinx.com:kr260_som:part0:1.1`, which is what the block-design script asks for.
- Python 3 with `pyyaml` and `numpy`. Stage 2 additionally needs torch + brevitas + `sparse_vit`
  (the sparta-sw environment).
- **Memory matters more than cores.** See the OOM trap below.

**Board**

- PYNQ and numpy. On this board PYNQ lives in a virtualenv (`~/sparta-venv`), which creates the
  first trap below.
- Root, for `/dev/mem`.
- CMA pool with ~25 MB free (`grep Cma /proc/meminfo`). The weights are DMA'd to physical addresses
  with no address translation, so the buffers must be contiguous and CMA-backed.

---

## Stage 1 - model export to payload

`compile_model.py` turns the brevitas export into what the kernel reads from DDR: packed weight
entries plus per-row nonzero counts, and the folded requantization scales.

    python3 scripts/compile_model.py \
        -i ../model/vit-base-cifar10-32b-finetuned-pruned-quant-test-1p4g6kml/vit-base-cifar10-32b-finetuned-pruned-quant-test-1p4g6kml.quant_values.txt \
        --out sim/e2e_tiled12/model \
        --mha-tiles 12 --mlp-tiles 12 \
        --weights auto --model-index

Why each flag:

- `--mha-tiles 12 --mlp-tiles 12` must equal `MHA_TILES` in `config/yml/mha.yaml`. The engine splits
  a weight's columns into that many blocks and reads each block from its own memory; the payload has
  to be grouped the same way or the banks are loaded with the wrong entries.
- `--weights auto` reads `config/build_config.yaml` and picks the entry format the kernel was built
  for -- 4-bit power-of-two values with `POT_SHIFT_MAC` set, 8-bit plain integers without. Getting
  this wrong is caught: the format is recorded in `model_index.bin` and the testbench refuses a
  mismatch ("weight entry format mismatch").
- `--model-index` writes that guard file. The board does not read it; the native testbench does.

Produces `model.bin` (~18 MB), `manifest.json`, `model_index.bin`.

## Stage 2 - reference vectors

    python3 sim/emit_e2e.py

Runs the software model, captures the exact int8 tensor its encoder would have consumed, and writes
`inputs.bin`. The golden outputs (`golden/hw_out.bin`) come from running the native testbench on a
known-good kernel.

Keep the golden from the ORIGINAL design. A payload emitted for a different engine configuration is
still checked against it -- that is the whole point, since every change through the engine is
supposed to be bit-exact. `sim/e2e_tiled12` is checked against `sim/e2e/golden`.

## Stage 3 - verify in simulation (the gate)

    make regress IMAGES=32 E2E_DIR=$PWD/sim/e2e_tiled12 GOLDEN_DIR=$PWD/sim/e2e/golden

Three things, in order: `native-units` (edge cases the golden cannot reach -- empty rows, token
counts that are not a multiple of the packing, saturation), `native-e2e` (32 images through the
kernel built with g++), `golden-check` (bit-exact comparison).

`GOLDEN_DIR` defaults to `$(E2E_DIR)/golden`, which is wrong for the tiled payload; pass it.

**A caveat that cost time once:** the native build cannot validate anything inside an HLS pragma.
`g++` ignores unknown pragmas, so an undefined identifier in a `#pragma HLS` compiles cleanly here
and fails only in synthesis. If you change config constants, run `make synth` too.

## Stage 4 - kernel to IP-XACT

    make ip

The board build does not use the Vitis link flow, so the `.xo` that `make synth` produces is not
what is needed: the block design places the kernel as a Vivado IP and wants IP-XACT. This target
runs the same synthesis with `package.output.format=ip_catalog` and copies the result to
`ip/encoder_layer/`.

**`v++ -c --mode hls` only synthesises.** Packaging is a second command,
`vitis-run --mode hls --package`; without it no `component.xml` appears and the block design has
nothing to instantiate. The target runs both.

Two things to take from the output:

- `ip/encoder_layer/component.xml` -- confirm the interfaces match the kernel's `bundle=` pragmas.
  Six of them: `in_mem`, `sc_mem`, `w_mha_mem`, `w1_mem`, `w2_mem`, `y_mem`.
- `ip/encoder_layer/drivers/encoder_layer_top_v1_0/src/xencoder_layer_top_hw.h` -- **the register
  map. Copy this to the board and always pass it as `--hw-header`.** See the trap below.

## Stage 5a - block design

    vivado -mode batch -nojournal -nolog \
        -source scripts/encoder_layer_kr260_bd.tcl -tclargs 200 0

Arguments: fabric clock in MHz, then a debug flag. Builds a project with the PS, the packaged IP and
the AXI wiring; validates it; generates the HDL wrapper; stops before synthesis.

Lands in `workspace/vivado/enc_kr260_200mhz` unless `SPARTA_VIVADO_DIR` says otherwise (the variable
exists so large projects can live off the repo volume; on Windows keep the path short or Vivado
breaks). Both Vivado scripts read it, so they agree on where the project is.

**Vivado exits 0 even when a sourced script errors.** Check for the final line instead:

    CHECKPOINT: block design built, wired, verified, validated, and wrapped OK

Also confirm `PASS -- all 6 bundles reach a physical PS port.` The bundle list in the script must
match the kernel's pragmas exactly; `get_bd_intf_pins` is called without `-quiet`, so a stale name
fails the build rather than being skipped. That is deliberate -- a silently unconnected master would
hang the kernel on the board with no clue why.

And note the assigned control base in the log:

    Slave segment '/enc_top_0/s_axi_control/Reg' ... at <0x8000_0000 [ 64K ]>

That has to match `HwKernel.CONTROL_BASE` in `runtime/SPARTA.py`.

## Stage 5b - synthesis, implementation, bitstream

Run it plainly, one job:

    nice -n 10 vivado -mode batch -nojournal -nolog \
        -source scripts/build_bitstream.tcl -tclargs 200 0 1

Arguments are clock MHz, debug flag, **jobs**, and optionally **threads**. `jobs` bounds how many
runs go in parallel; `threads` bounds the threads inside one run, which `-jobs` does not control.

**Do not wrap this in a systemd scope with `MemoryMax`.** It looks like the responsible thing to do
and it is not -- see the memory trap below.

30-60 minutes, longer at `jobs 1`. Reports worst negative slack and **fails if it is negative**, because a design that
misses timing gives intermittently wrong answers rather than not working -- much harder to diagnose
from the host. Writes the bitstream to `deploy/`.

On the clock: `build_config.yaml` targets 4.0 ns (250 MHz) deliberately, as a "tight synth target to
make the tool try hardest". **200 MHz is the signoff clock** and what to build and quote. The HLS
Fmax estimate (256.82 MHz) is not a signoff number -- it knows nothing about placement or routing.

## Stage 6 - package for xmutil

    scripts/package_app.sh deploy/enc_system_wrapper.bit sparta-tiled-v1 200

`xmutil` does not load a raw `.bit`. It wants a directory of three files:

| file            | what                                                                |
| --------------- | ------------------------------------------------------------------- |
| `<app>.bit.bin` | the bitstream in bootgen's format                                   |
| `<app>.dtbo`    | device-tree overlay: names the `.bit.bin` and sets the fabric clock |
| `shell.json`    | `{"shell_type": "XRT_FLAT", "num_slots": "1"}` -- a fixed constant  |

The overlay in `package_app.sh` reproduces the one already running on the board, recovered with
`dtc -I dtb -O dts` from an existing app, so the reset indices (116-119, the four PL resets) and the
clock index (71, PL0) are copied rather than guessed.

**The clock rate in the overlay must match the build.** The overlay is what sets the fabric
frequency at load time, so a mismatch runs the design at a speed it was never timed for. The script
takes it as a parameter for exactly this reason -- pass the same number as stage 5.

## Stage 7 - board bring-up

Copy, from the host.  `$BOARD` below is whatever root you use on the board -- nothing hardcodes a
path, every location is a command-line argument:

    BOARD=~/sparta-a            # or wherever
    ssh kria "mkdir -p $BOARD/{runtime,payload,golden,hw}"
    scp runtime/*.py                              kria:$BOARD/runtime/
    scp sim/e2e_tiled12/model.bin \
        sim/e2e_tiled12/manifest.json \
        sim/e2e_tiled12/inputs.bin                kria:$BOARD/payload/
    scp sim/e2e/golden/hw_out.bin                 kria:$BOARD/golden/
    scp ip/encoder_layer/drivers/encoder_layer_top_v1_0/src/xencoder_layer_top_hw.h \
                                                  kria:$BOARD/hw/
    scp -r deploy/sparta-tiled-v1                 kria:~/

About 20.6 MB. **Re-copy `runtime/*.py` and the register header whenever either changes on the
host** -- a stale driver whose CLI predates a change fails visibly, but a stale *header* does not:
it writes arguments to the wrong registers silently.  `md5sum` the header against the host's copy
before trusting it.

`model_index.bin` is not needed on the board -- it is the native testbench's guard.
The loader reads only `manifest.json` and `model.bin`, and asserts their sizes agree.

Load the bitstream, on the board:

    sudo cp -r ~/sparta-tiled-v1 /lib/firmware/xilinx/
    sudo xmutil unloadapp
    sudo xmutil loadapp sparta-tiled-v1
    sudo xmutil listapps | grep sparta-tiled-v1      # Active_slot should be 0

Then three checks, in this order. Each isolates one thing, so a failure localises instead of needing
a hunt. `$PY` below is the venv interpreter -- see the first trap.

**1. Control path only.** No memory, does not start the kernel.

    cd $BOARD/runtime
    sudo -E $PY SPARTA.py --no-download --selftest --hw-header $BOARD/hw/xencoder_layer_top_hw.h

Writes and reads back every argument register with five patterns. Four are uniform (stuck and
shorted lines); the fifth gives every register a different value and writes them all before reading
any, which is the one that catches two registers decoding to the same place -- uniform patterns
cannot, because each reads back what it just wrote. If this fails, no datapath debugging is worth
doing.

**2. Datapath, bit-exact against the simulation's golden.** Needs only numpy and pynq.

    sudo -E $PY SPARTA.py --no-download \
        -d $BOARD/payload -e $BOARD/payload \
        --golden-dir $BOARD/golden \
        --hw-header $BOARD/hw/xencoder_layer_top_hw.h -n 1

This holds the board to the same standard as `make golden-check`. A pass means the whole chain
agrees: register map, buffer addresses, weight payload and arithmetic. Raise `-n` once it passes.

**3. Real images end to end.** Additionally needs torch + brevitas + `sparse_vit`.

    sudo -E $PY infer_hw.py --help

---

## Traps

Each of these cost real time.

**`sudo` does not use your virtualenv.** `sudo` replaces `PATH` from its own `secure_path`, so
`sudo -E python3` runs `/usr/bin/python3` and fails with `ModuleNotFoundError: No module named
'pynq'` even though plain `python3` works. `-E` preserves environment *variables*, not which
interpreter is found. Call the venv interpreter by absolute path:

    sudo -E /home/ubuntu/sparta-venv/bin/python3 ...

**Always pass `--hw-header`.** The register offsets shift whenever a kernel argument is added or
removed, and a stale table does not fail -- it silently writes `output` and `scales` to whatever
address the neighbouring argument left behind, and the kernel DMAs wherever it is pointed. With the
header, a mismatch is refused at construction. `SPARTA.REGMAP` is a transcription kept for
convenience; the generated header cannot drift.

This is not hypothetical: when the MLP moved onto the shared engine its two CSR triplets became
`{bank, counts}` pairs, the argument list lost two entries, and every offset from `w2_bank` onward
moved down two slots.

**Stage 5b needs ~5 GB, not the 13 GB I first assumed.** Measured rather than guessed: Vivado's own
log reports `peak = 4748.938` MB at Timing Optimization, and a cgroup kill caught the process at
anon-rss 3.4 GB. What needs care is the **job count**, not the per-process budget.

Three failures were seen on a 15 GB host, and only the first was a genuine shortage:

- `-jobs 12`, no limit -> a genuine system out-of-memory kill. Twelve concurrent Vivado processes
  at ~5 GB each does not fit. The OOM killer took the **desktop**, not Vivado -- an editor usually
  scores higher -- so the run was lost and the session with it. At ~5 GB per run, jobs=2 needs
  ~12 GB free and jobs=1 is right for this host.
- `-jobs 2`, no limit -> the machine **froze** rather than being killed, thrashing against a 2 GB
  swapfile. Worse than an OOM, because an OOM at least ends.
- `MemoryMax=9G` then `12G`, jobs 1 -> **killed both times, by the cgroup, with Vivado at ~4 GB.**
  The limit was the cause, not the cure. `cgroup v2` charges **page cache** to the cgroup as well as
  the process: Vivado's file I/O fills it, `MemorySwapMax=0` leaves dirty pages unreclaimable until
  written back, and the cgroup hits its ceiling while the process is nowhere near it. A
  `MemoryMax` scope does not protect a run that does heavy I/O -- it ends it.

**A killed synthesis reports no error**, which is what made the cgroup cases so confusing.
`runme.log` contains no `ERROR` line -- a process that is killed never gets to write one. The
evidence is instead:

| where | what |
|---|---|
| `runme.log`, near the end | `rdiArgs.sh: line 60: NNNNN Killed "$RDI_PROG" "$@"` |
| the run directory | a pile of `hs_err_pid*.log` worker crash dumps |
| `runme.log` last phase | which stage it reached, e.g. cross-boundary optimization |
| `journalctl --user` | `A process of this unit has been killed by the OOM killer` |
| `journalctl -k` | `constraint=CONSTRAINT_MEMCG` means the **cgroup limit**, not the system |

That last row is the one that settles it. `CONSTRAINT_MEMCG` means a cgroup limit fired; the absence
of any OOM record at all means something else ended the process. And the kill line reports the
victim's actual `anon-rss`, which is how the ~4 GB figure came out -- worth reading before concluding
that a design needs more memory.

If a host genuinely cannot supply ~5 GB plus page cache, the targeted lever is
`-flatten_hierarchy none` on the kernel's IP run: it keeps the module hierarchy rather than
flattening it, which is where that phase's memory goes. It costs quality of result, so it is a last
resort -- at 81 % LUT, closing 200 MHz without cross-boundary optimization is not a given.

So: budget ~3 GB per job plus headroom, run under `systemd-run` with `MemorySwapMax=0` as shown in
stage 5b, drop to `jobs 1`, cap `threads`, and close the editor. If it still will not complete, the
design wants more memory than the host has and should be built elsewhere -- synthesis peak scales
with design size, and this is 81 % LUT on the part.

**A killed run leaves one thing behind, and the script now clears it.** A process killed
mid-synthesis strands a `__synthesis_is_running__` marker in its run directory, and Vivado then
refuses to relaunch that run because it believes it is still going. `build_bitstream.tcl` resets
every run that is not at 100 % before launching, which clears the marker while keeping the
checkpoints of the runs that did finish -- so a retry only redoes what actually failed. After two
interrupted attempts, seven of the eight out-of-context IP runs were already complete and only the
kernel's needed redoing.

No lock files are left unless Vivado died holding the project open for write, which it does not
during a run. Check disk headroom before retrying, though: implementation writes several GB of
intermediate checkpoints.

**A stale bitstream will pass the selftest.** The selftest only proves the processor can address the
control registers; the loaded bitstream may have a completely different argument layout. When we ran
it against an August bitstream, 34 of 35 words passed -- the one failure was `last_layer` at 0x0e4,
which is an unused hole in the old map and a real register in the new one. Useful as a cross-check,
but never mistake it for datapath validation.

**Check `CmaFree`, not `CmaTotal`.** A PYNQ process that exits without freeing its buffers leaves
them allocated. We saw 103 MB free of a 1000 MB pool on an idle board. Enough for the ~21 MB needed,
but reboot before measuring timing so allocation behaviour is clean.
