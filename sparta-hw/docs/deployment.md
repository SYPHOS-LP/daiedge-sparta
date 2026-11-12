# Deploy the encoder to the KR260

This document describes how to put the accelerator on the board. It starts with the quantized model
file and ends with the accelerator that classifies images on the board. The board is a Kria KR260,
`xck26-sfvc784-2LV-c`. 

To confirm the board type, execute on the board the command `xbutil examine`. 
The expected output is `Model : ZynqMP KR260 revB`.


## 1. What the accelerator does

The accelerator holds the 12 encoder layers of a Vision Transformer.

A full classification needs three parts. The accelerator is only the middle part:

| Part              | Where it operates | What it does                                     |
| ----------------- | ----------------- | ------------------------------------------------ |
| Embedding         | Software          | Changes an image into int8 values, 768 by N      |
| 12 encoder layers | **Accelerator**   | Changes those values into 768 by N output values |
| Classifier head   | Software          | Changes the output values into 10 class scores   |

`N` is the token count. Section 4 gives its two values.

A full classification run needs the software model on the board, whereas a test of the
accelerator alone (Section 13.4) does not need the software model.


## 2. The data flow

The accelerator reads two groups of data from DDR memory:

| Group             | How often it changes |
| ----------------- | -------------------- |
| The weights       | One time             |
| Activation values | For each image       |

The accelerator keeps the activation values in its internal memory between layers. Only layer 0
reads the values from DDR memory and only layer 11 writes them back. The arguments `first_layer`
`last_layer` control this behaviour. For each image, the software:

1. It writes the input values to a DDR buffer.
2. It calls the accelerator 12 times, one time for each layer.
3. It reads the output values from a DDR buffer.

[!NOTE]
The weights stay in DDR memory for all images. Thus, timing for each image includes only the
activation transfers, not the 17.2 MB of weights.


## 3. The stages

There are seven stages. Stages 1 to 6 operate on the host computer. Stage 7 operates on the board.

|     | Stage                     | Command                                    | Result                        |
| --- | ------------------------- | ------------------------------------------ | ----------------------------- |
| 1   | Model to payload          | `scripts/compile_model.py`                 | `model.bin`, `manifest.json`  |
| 2   | Reference data            | `sim/emit_e2e.py`                          | `inputs.bin`, `meta.json`     |
| 3   | Test in simulation        | `make regress`                             | Pass or fail                  |
| 4   | Kernel to IP              | `make ip`                                  | `component.xml`, register map |
| 5a  | IP to block design        | `scripts/encoder_layer_kr260_bd.tcl`       | A Vivado project              |
| 5b  | Block design to bitstream | `scripts/build_bitstream.tcl`              | A `.bit` file                 |
| 6   | Bitstream to board app    | `scripts/package_app.sh`                   | A directory for `xmutil`      |
| 7   | Board tests               | `runtime/SPARTA.py`, `runtime/infer_hw.py` | Results from the board        |

Each stage uses the result of the stage before it. Stage 3 tests the calculations. The other stages
test only the connections. 

[!CAUTION]
Do not continue after a failure in stage 3. It is much more difficult to find a
calculation error on the board than in simulation. On the board you cannot see the values inside
the accelerator.


## 4. The token count

The token count is the number of image patches, plus one for the class token. It depends on the 
model's patch size `(16, 32)` and can take values between `(50, 197)`. The accelerator moves data
as in 64-bit words. Therefore 50 tokens need 7 words and 197 tokens need 25 words for each feature.

| Patch size | Calculation         | Tokens | Words for each feature |
| ---------- | ------------------- | ------ | ---------------------- |
| 32 x 32    | (224/32)² + 1 = 50  | 50     | 7                      |
| 16 x 16    | (224/16)² + 1 = 197 | 197    | 25                     |

The token count is in three files:

- `config/yml/mha.yaml`
- `config/yml/encoder_layer.yaml`
- `config/yml/rmsnorm.yaml`

After you change the value, execute `make config`. This makes the C header files again.

[!WARNING] 
The token count of the bitstream and the token count of the payload must agree. The
accelerator sets the size of its internal memories at synthesis time. It cannot accept 
more tokens than this value. The software compares the two values and stops if they do 
not agree.

The token count also controls the software model. The file `sim/infer_p16_local.toml` sets
`p_size = 16`, which gives 197 tokens. A different patch size needs a different file.


## 5. Before you start

### 5.1 The host computer

- Install Vitis and Vivado 2026.1.
- For HLS and the packaging tools, execute `source /opt/Xilinx/2026.1/Vitis/settings64.sh` and for Vivado execute `source /opt/Xilinx/2026.1/Vivado/settings64.sh`. (You can execute both commands).
- Install the KR260 board files. To confirm, execute `get_board_parts *kr260*` in a Vivado shell. The result must include `xilinx.com:kr260_som:part0:1.1`.
- Install Python 3 with `pyyaml` and `numpy`.
- For stage 2, also install `torch`, `brevitas` and `sparse_vit`. A CPU-only version of `torch` is sufficient.
- Make sure that the computer has 5 GB of free memory for each parallel job in stage 5b.

### 5.2 The board

- Install PYNQ and `numpy`. On our deployment in the Kria board, PYNQ is installed in a virtual environment at `~/sparta-venv`.
- Get root permission. The software uses `/dev/mem`.
- Make sure that the CMA pool has 25 MB free. To show the free space, execute
  `grep Cma /proc/meminfo`.
- For test 3 only, also install `torch`, `brevitas` and `sparse_vit`.

[!NOTE]
The accelerator sends data to physical addresses. There is no address translation unit. Thus
the buffers must be continuous and in the CMA pool. This is why the software uses PYNQ buffers and
not usual Python memory.


### 5.3 The model files

You need two files from the same trained model:

| File                | Used by | Function                                      |
| ------------------- | ------- | --------------------------------------------- |
| `.quant_values.txt` | Stage 1 | The weights for the accelerator               |
| `.pth`              | Stage 2 | The software model for the embedding and head |

[!WARNING]
The two files must be from the same trained model. If they are different, the weights and
the input values do not agree. The result will be incorrect, but the software gives no error message.


## 6. Stage 1 - Make the payload
The script `compile_model.py` changes the model export into the data that the accelerator reads
from DDR memory. The data has three parts: the packed weights, the nonzero counts for each row, and
the scales.

Execute this command:

    python3 scripts/compile_model.py \
        -i <model>.quant_values.txt \
        --out sim/e2e_p16/model \
        --mha-tiles 12 --mlp-tiles 12 \
        --weights auto --model-index

The flags have these functions:

- `--mha-tiles` and `--mlp-tiles` set the number of column blocks. The value must agree with
  `MHA_TILES` in `config/yml/mha.yaml`. The accelerator divides the weight columns into 12 blocks.
  It reads each block from a different memory at the same time. This is how it does 12 operations
  in one cycle. If the payload has a different division, the accelerator reads the wrong weights.
- `--weights auto` reads `config/build_config.yaml`. It then selects the weight format of the
  accelerator. With `POT_SHIFT_MAC`, the format is 4-bit power-of-two values, and the accelerator
  shifts instead of multiplies. Without it, the format is 8-bit integers.
- `--model-index` writes the file `model_index.bin`. This file records the weight format. The
  simulation testbench refuses a payload that does not agree. The board does not read this file.

The command writes three files:

| File              | Size   | Content                                           |
| ----------------- | ------ | ------------------------------------------------- |
| `model.bin`       | 18 MB  | All weights for all 12 layers, in one file        |
| `manifest.json`   | 100 kB | The position and size of each item in `model.bin` |
| `model_index.bin` | 600 B  | The weight format and the model dimensions        |

NOTE: The weights do not change with the token count. A 50-token model and a 197-token model have
weight files of the same size.

## 7. Stage 2 - Make the reference data

Execute this command with the `sparta-vit` environment:

    E2E_CFG=sim/infer_p16_local.toml \
    E2E_TXT=<model>.quant_values.txt \
    E2E_OUT=sim/e2e_p16 \
    python sim/emit_e2e.py 32

The last number is the image count.

The script does these operations for each image:

1. It operates the software model on the image.
2. A software hook records the int8 values that the software encoder reads.
3. It writes these values to `inputs.bin`.
4. It records the correct class and the software prediction in `meta.json`.

The recorded values are the exact values that the software encoder uses. The script does not
calculate them a second time. Thus the accelerator and the software model always get the same
input.

The script reads `MHA_TILES` from `config/inc/mha_cfg.h`. Thus the payload division always agrees
with the accelerator.

### 7.1 The golden file

The file `golden/hw_out.bin` is different from the other files. It is the output of the simulation
testbench from an accelerator that is known to be correct. The tests compare new results with this
file, byte for byte.

The golden file shows that the accelerator did not change. It does not show that the accelerator is
correct. A new model has no golden file. Section 13.6 tells you how to make one.

## 8. Stage 3 - Test in simulation

Execute this command:

    make regress IMAGES=32 E2E_DIR=$PWD/sim/e2e_p16 GOLDEN_DIR=$PWD/sim/e2e_p16/golden

The command does three tests in this sequence:

1. `native-units` tests the conditions that the golden data cannot show. These conditions are empty
   rows, token counts that are not a multiple of eight, and saturation. There are 62 tests.
2. `native-e2e` sends 32 images through the accelerator code. The code is the same C++ code, but
   compiled with `g++` for the host computer.
3. `golden-check` compares the result with the golden file.

The expected output is:

    62 checks, 0 failed
    PASS: model is bit-exact vs golden

NOTE: `GOLDEN_DIR` has the default value `$(E2E_DIR)/golden`. If your golden file is in a different
directory, give the `GOLDEN_DIR` value.

CAUTION: The `g++` build cannot test an HLS pragma. The compiler ignores pragmas that it does not
know. Thus an unknown name in a `#pragma HLS` compiles correctly here, but it fails in synthesis.
If you change a configuration value, also execute `make synth`.

## 9. Stage 4 - Make the IP

Execute this command:

    make ip

The board build does not use the Vitis link flow. Thus the `.xo` file from `make synth` is not
correct for this flow. The block design holds the accelerator as a Vivado IP. It needs the IP-XACT
format. This target does the same synthesis with `package.output.format=ip_catalog`. It then copies
the result to `ip/encoder_layer/`.

NOTE: The command `v++ -c --mode hls` only does the synthesis. The packaging is a second command,
`vitis-run --mode hls --package`. Without the second command, there is no `component.xml` file. The
block design then has no IP to use. The `make ip` target does both commands.

Examine two items in the result:

- `ip/encoder_layer/component.xml`. Make sure that the interfaces agree with the `bundle=` pragmas
  of the accelerator. There are six interfaces: `in_mem`, `sc_mem`, `w_mha_mem`, `w1_mem`, `w2_mem`
  and `y_mem`. Each interface is a separate connection to DDR memory.
- `ip/encoder_layer/drivers/encoder_layer_top_v1_0/src/xencoder_layer_top_hw.h`. This is the
  register map file. It gives the address of each accelerator argument. Copy this file to the
  board. Always give it with the `--hw-header` flag. Refer to section 14.2.

To confirm that the IP agrees with your code, compare the report
`workspace/encoder_layer_ip/hls/syn/report/csynth.rpt` with the report from `make synth`. The
resource values must be the same.

## 10. Stage 5a - Make the block design

Execute this command:

    vivado -mode batch -nojournal -nolog \
        -source scripts/encoder_layer_kr260_bd.tcl -tclargs 200 0

The two arguments are the fabric clock in MHz and a debug flag. The script makes a project with the
processor, the IP and the AXI connections. It then tests the design and makes the HDL wrapper. It
stops before the synthesis.

The project goes to `workspace/vivado/enc_kr260_200mhz`. To use a different location, set the
variable `SPARTA_VIVADO_DIR`. Both Vivado scripts read this variable.

CAUTION: The script uses `create_project -force`. This removes a project of the same name. Keep a
copy of an earlier project if you need its reports.

CAUTION: Vivado gives the exit code 0 even when a script has an error. Do not use the exit code. In
the log, find these two lines:

    CHECKPOINT: block design built, wired, verified, validated, and wrapped OK
    PASS -- all 6 bundles reach a physical PS port.

The bundle names in the script must agree with the pragmas of the accelerator. The script calls
`get_bd_intf_pins` without `-quiet`. Thus an incorrect name stops the build. This is intentional. A
master interface with no connection makes the accelerator stop on the board, with no error message.

Find the control base address in the log:

    Slave segment '/enc_top_0/s_axi_control/Reg' ... at <0x8000_0000 [ 64K ]>

This address must agree with `HwKernel.CONTROL_BASE` in `runtime/SPARTA.py`. The board software
uses this address to find the accelerator.

## 11. Stage 5b - Make the bitstream

WARNING: Do not use a `systemd` scope with `MemoryMax` for this stage. The limit stops the build.
Refer to section 14.3.

Execute this command:

    nice -n 10 vivado -mode batch -nojournal -nolog \
        -source scripts/build_bitstream.tcl -tclargs 200 0 1

The arguments are the clock in MHz, a debug flag, the number of jobs, and the number of threads. The
`jobs` value sets how many runs operate at the same time. The `threads` value sets how many threads
operate in one run. The `jobs` value does not control the threads.

The stage takes 30 to 60 minutes.

### 11.1 The timing result

The script shows the worst negative slack. It stops with an error if the slack is negative.

CAUTION: A design that does not meet the timing gives incorrect results some of the time. This is
much more difficult to find than a design that does not operate at all.

These are the measured results of recent builds at 200 MHz:

| Build         | Tokens | Worst negative slack |
| ------------- | ------ | -------------------- |
| 32 x 32 patch | 50     | +0.014 ns            |
| 16 x 16 patch | 197    | +0.006 ns            |

The margin is very small. A small change to the design can make the timing negative.

### 11.2 The clock

NOTE: The file `build_config.yaml` sets a target of 4.0 ns, which is 250 MHz. This is intentional. A
difficult target makes the tool do more work. But 200 MHz is the signoff clock. Build at 200 MHz and
give 200 MHz in your reports.

The HLS estimate of 256.82 MHz is not a signoff value. The HLS tool knows nothing about placement or
routing. Two builds at 250 MHz did not meet the timing. The maximum frequency of this design is
approximately 224 MHz. The board gives only 166.67 MHz, 200 MHz or 250 MHz. Thus 200 MHz is the
correct choice.

### 11.3 The resource use

These are the measured values after implementation:

| Resource        | 50 tokens     | 197 tokens    | Available |
| --------------- | ------------- | ------------- | --------- |
| LUT             | 51,432 (44 %) | 57,964 (50 %) | 117,120   |
| Block RAM tiles | 68 (47 %)     | 77 (53 %)     | 144       |
| URAM            | 20 (31 %)     | 46 (72 %)     | 64        |
| DSP             | 455 (36 %)    | 474 (38 %)    | 1,248     |

URAM has the least free space at 197 tokens.

The script writes the bitstream to `deploy/enc_system_wrapper.bit`.

## 12. Stage 6 - Make the board app

Execute this command:

    scripts/package_app.sh deploy/enc_system_wrapper.bit sparta-p16-v1 200

The three arguments are the bitstream, the app name, and the clock in MHz.

The `xmutil` program cannot load a `.bit` file. It needs a directory with three files:

| File            | Function                                                           |
| --------------- | ------------------------------------------------------------------ |
| `<app>.bit.bin` | The bitstream in the `bootgen` format                              |
| `<app>.dtbo`    | The device-tree overlay. It names the bitstream and sets the clock |
| `shell.json`    | `{"shell_type": "XRT_FLAT", "num_slots": "1"}`. This is a constant |

The overlay in `package_app.sh` is a copy of the overlay that is already on the board. The reset
indices (116 to 119) and the clock index (71) are copies from that overlay. They are not estimates.

WARNING: The clock value in the overlay must be the same as the clock value of the build. The
overlay sets the fabric frequency when the board loads the app. A different value operates the
design at a speed that nobody tested. Give the same value as stage 5b.

---

## 13. Stage 7 - Test on the board

### 13.1 Copy the files

Execute these commands on the host computer. The variable `BOARD` is the directory on the board. No
path is fixed in the software. Each location is a command argument.

    BOARD=~/sparta-a
    ssh kria "mkdir -p $BOARD/{runtime,payload,golden,hw}"
    scp runtime/*.py                       kria:$BOARD/runtime/
    scp sim/e2e_p16/model.bin \
        sim/e2e_p16/manifest.json \
        sim/e2e_p16/inputs.bin             kria:$BOARD/payload/
    scp sim/e2e_p16/golden/hw_out.bin      kria:$BOARD/golden/
    scp ip/encoder_layer/drivers/encoder_layer_top_v1_0/src/xencoder_layer_top_hw.h \
                                           kria:$BOARD/hw/
    scp -r deploy/sparta-p16-v1            kria:~/

The total is 27 MB for a 197-token payload. It is 7 MB for a 50-token payload. The difference is in
`inputs.bin` and the golden file. These two files hold data for each token.

WARNING: Copy `runtime/*.py` and the register map file again after each change on the host computer.
A Python file that is not current fails with a visible error. A register map file that is not
current does not fail. It writes the arguments to the wrong registers. To confirm the file, compare
the `md5sum` values on the host computer and on the board.

NOTE: The board does not need `model_index.bin`. Only the simulation testbench reads it. The board
software reads `manifest.json` and `model.bin`. It compares the size of `model.bin` with the value
in `manifest.json`.

### 13.2 Load the bitstream

Execute these commands on the board:

    sudo cp -r ~/sparta-p16-v1 /lib/firmware/xilinx/
    sudo xmutil unloadapp
    sudo xmutil loadapp sparta-p16-v1
    sudo xmutil listapps | grep sparta-p16-v1

The last command must show `Active_slot` with the value 0.

WARNING: Make sure that `loadapp` is successful before you continue. If `loadapp` fails, and you
then use the flag `--no-download`, the software writes to an address with no device. On this
processor, such an access never completes and has no time limit. The board stops. You must remove
the power and start the board again.

To confirm the clock, execute this command:

    sudo cat /sys/kernel/debug/clk/pl0_ref_clk/clk_rate

The expected output is `200000000` for a 200 MHz build. The overlay sets this value. The value in
the build is only a request.

### 13.3 Test 1 - The control path

This test does not use memory. It does not start the accelerator. It shows only that the processor
can address the registers of the accelerator.

    cd $BOARD/runtime
    sudo -E $PY SPARTA.py --no-download --selftest \
        --hw-header $BOARD/hw/xencoder_layer_top_hw.h

The variable `PY` is the virtual environment interpreter. Refer to section 14.1.

The test writes and reads each argument register with five patterns:

| Pattern                             | What it finds                              |
| ----------------------------------- | ------------------------------------------ |
| `0xA5A5A5A5` and `0x5A5A5A5A`       | Data lines that are stuck or shorted       |
| `0xFFFFFFFF` and `0x00000000`       | A line that is stuck in one direction only |
| A different value for each register | Two registers that have the same address   |

The fifth pattern is the important one. The software writes all values before it reads any value.
The first four patterns cannot find two registers with the same address, because each register
reads back the value that it received.

CAUTION: If this test fails, do not examine the datapath. Correct the control path first.

### 13.4 Test 2 - The datapath

This test needs only `numpy` and `pynq`. It does not need the software model.

    sudo -E $PY SPARTA.py --no-download \
        --tokens 197 \
        -d $BOARD/payload -e $BOARD/payload \
        --golden-dir $BOARD/golden \
        --hw-header $BOARD/hw/xencoder_layer_top_hw.h -n 1

Give the token count of the bitstream with the `--tokens` flag. The default value is 50.

The test sends the recorded int8 values to the accelerator. It compares the output with the golden
file. This is the same test as `make golden-check`, but on the board.

A pass shows that four items agree: the register map, the buffer addresses, the weight payload and
the calculations. After the test passes, increase the value of `-n` to test more images.

The test also shows the time for each image and for each layer. These are the measured results of
the 50-token build:

    32/32 images bit-exact
    per image: 68.5 ms mean  (14.6 FPS)
    kernel per layer: 5.29 ms  =  1.06 M cycles at 200 MHz

A calculation gives approximately 199 ms for each image at 197 tokens. The board has not yet given a
measured value for 197 tokens.

### 13.5 Test 3 - Full images

This test also needs `torch`, `brevitas` and `sparse_vit` on the board. It does the embedding and
the classifier head in software, and the 12 encoder layers on the accelerator.

    sudo -E $PY infer_hw.py \
        --cfg sim/infer_p16_local.toml --model-dir $BOARD/payload \
        --tokens 197 --images 100 --no-reference \
        --hw-header $BOARD/hw/xencoder_layer_top_hw.h

The flag `--no-reference` stops the software forward pass at the embedding. The 12 software encoder
layers and the classifier head do not operate a second time. This makes each image approximately 70
times faster. The test then gives the accelerator accuracy only.

Remove `--no-reference` to also get the software prediction for each image. The software then
operates the full model for each image. Use this for a small number of images only.

NOTE: Use `--no-reference` for a test of the full dataset. Without it, the software model controls
the speed, not the accelerator.

### 13.6 Make a new golden file

A new model has no golden file. Make one in this sequence:

1. Do test 3 with a small number of images and without `--no-reference`.
2. Make sure that the accelerator prediction and the software prediction agree for each image.
3. Copy `hw_out.bin` and `hw_layers.bin` into a `golden/` directory.

CAUTION: Do step 2 before step 3. If you copy an incorrect output, the error stays in the golden
file permanently. All later tests then compare against an incorrect result.

NOTE: The file `hw_layers.bin` holds the output of each of the 12 layers for image 0. Use it to find
the layer where two results start to disagree.

---

## 14. Problems and solutions

### 14.1 The `sudo` command does not find `pynq`

The `sudo` command replaces `PATH` with its own `secure_path`. Thus `sudo -E python3` operates
`/usr/bin/python3`. The result is `ModuleNotFoundError: No module named 'pynq'`. The plain command
`python3` operates correctly. The flag `-E` keeps the environment variables. It does not change
which interpreter the system finds.

Solution: give the full path of the interpreter.

    sudo -E /home/ubuntu/sparta-venv/bin/python3 ...

### 14.2 The register map file is not current

The register addresses move when you add or remove an accelerator argument. A register map that is
not current does not fail. It writes `output` and `scales` to the address of a different argument.
The accelerator then sends data to the wrong location.

Solution: always give the `--hw-header` flag. With the header file, the software refuses a map that
does not agree. The table `SPARTA.REGMAP` is a manual copy of the header. Use it only for
convenience.

This condition is not theoretical. When the MLP moved to the shared engine, its two CSR groups
became `{bank, counts}` pairs. The argument list lost two items. Each address after `w2_bank` moved
down by two positions.

### 14.3 Stage 5b stops with no error

Stage 5b needs approximately 5 GB for each job. The Vivado log shows `peak = 4748.938` MB at Timing
Optimization.

CAUTION: The number of jobs is more important than the memory limit for each process.

These three conditions occurred on a host computer with 15 GB:

| Condition                 | Result                                                           |
| ------------------------- | ---------------------------------------------------------------- |
| `-jobs 12`, no limit      | The system stopped a process. It stopped the editor, not Vivado. |
| `-jobs 2`, no limit       | The computer stopped. It used a 2 GB swap file continuously.     |
| `MemoryMax=9G`, `-jobs 1` | The cgroup stopped Vivado at 4 GB.                               |

The third condition needs an explanation. A cgroup counts the page cache as well as the process
memory. The file operations of Vivado fill the page cache. With `MemorySwapMax=0`, the system cannot
release these pages until it writes them to disk. Thus the cgroup reaches its limit while the
process is much below it.

Solution: use `-jobs 1`. Do not use a memory limit. Close other large programs.

A process that the system stops writes no error message. Find the evidence in these locations:

| Location                  | Evidence                                                   |
| ------------------------- | ---------------------------------------------------------- |
| `runme.log`, near the end | `rdiArgs.sh: line 60: NNNNN Killed "$RDI_PROG" "$@"`       |
| The run directory         | Files with the name `hs_err_pid*.log`                      |
| `journalctl --user`       | `A process of this unit has been killed by the OOM killer` |
| `journalctl -k`           | `constraint=CONSTRAINT_MEMCG` shows a cgroup limit         |

The last row gives the answer. `CONSTRAINT_MEMCG` shows that a cgroup limit stopped the process. If
there is no record at all, a different mechanism stopped it.

If the host computer cannot supply 5 GB, use `-flatten_hierarchy none` on the IP run. This keeps the
module hierarchy and uses less memory. But it gives a worse result. Use it only as a last choice.

### 14.4 A run stopped and will not start again

A process that stops during synthesis leaves a `__synthesis_is_running__` marker in its directory.
Vivado then refuses to start that run again.

Solution: no action is necessary. The script `build_bitstream.tcl` resets each run that is not at
100 %. This removes the marker. It keeps the checkpoints of the runs that are complete. Thus a
second attempt repeats only the work that failed.

NOTE: Examine the free disk space before a second attempt. Implementation writes several GB of
intermediate files.

### 14.5 A bitstream that is not current passes test 1

Test 1 shows only that the processor can address the control registers. The bitstream can have a
different argument layout.

In one example with an old bitstream, 34 of 35 words passed. The one failure was `last_layer` at
address `0x0e4`. This address is empty in the old map and a register in the new map.

Solution: use test 1 as a first check only. Do not use it as a datapath test.

### 14.6 There is not sufficient CMA memory

A PYNQ process that stops without a release of its buffers leaves them allocated. In one example, an
idle board had 103 MB free of a 1000 MB pool.

Solution: examine `CmaFree`, not `CmaTotal`.

    grep Cma /proc/meminfo

NOTE: Start the board again before you measure the time. This gives a clean allocation condition.

### 14.7 The accelerator does not give `ap_done`

The software waits 60 seconds and then shows a time limit error.

These are the common causes:

- A buffer address is incorrect.
- A master interface does not reach a processor port. Section 10 tells you how to confirm this.
- The accelerator is in the done state from an earlier run. The software clears this state before
  each start. A failure here shows that the accelerator cannot start at all.

Solution: load the app again with `xmutil`. This resets the accelerator. If the condition continues,
do test 1 to confirm the control path.
