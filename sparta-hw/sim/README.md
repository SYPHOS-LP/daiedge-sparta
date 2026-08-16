# `sim/` — end-to-end simulation of the encoder

Runs whole images through the real HLS kernel in Vitis HLS C-simulation and checks that
the hardware classifies them identically to the software model it was quantized from.

This is the correctness gate for any change to the kernel: if the hardware still matches
the software predictions, the datapath is intact.

## Files

| file | role |
|------|------|
| `emit_e2e.py` | **Stage 1** — runs the software model, captures the encoder's int8 input, compiles the weights, records ground truth and software predictions. |
| `check_e2e.py` | **Stage 3** — classifies the C-simulation's output and compares hardware vs software vs truth. |
| `demo_e2e.py` | Narrated single-command run of the whole flow, for demonstrating or recording it. |
| `full_model_infer.py` | Shared boundary machinery (software embedding + classifier head). Imported by the three above, not run directly. |
| `encoder_replica.py` | Python replica of the kernel's arithmetic. Imported by `full_model_infer.py`. |
| `sw_accuracy_check.py` | Software-model accuracy over a set of images, independent of the hardware. |
| `infer_pot4_local.toml.template` | Template for the machine-local inference config. |

## Requirements

- **Vitis HLS** with `vitis-run` and `v++` on `PATH` (stage 2 only).
- **The software model**, which lives alongside the accelerator in this repository.
  `full_model_infer.py` resolves it as `../../sparta-sw/sparse_vit/src` relative to
  itself:

  ```
  daiedge-sparta/
    sparta-hw/     <- the accelerator
    sparta-sw/     <- the software model
  ```

- **A Python environment with the software model's dependencies** (PyTorch, Brevitas,
  numpy, toml). By convention it lives at `sparta-sw/.venv`, which is what the commands
  below assume. CPU-only PyTorch is sufficient.

- **A model checkpoint**, not included in this repository. `emit_e2e.py` names the
  checkpoint and config it expects near the top of the file; both must be the same
  trained model.

- **CIFAR-10**, downloaded automatically on first use into the directory named by the
  inference config's `[dataset] path`.

## Running

### 1. Generate the artifacts

```bash
<venv>/python sim/emit_e2e.py 4
```

The argument is the image count. Writes `model.bin`, `model_index.bin`, `inputs.bin`,
`meta.json`, and `manifest.json` into `sim/e2e/`.

`emit_e2e.py` records the config it used in `meta.json`, and `check_e2e.py` reads it back
so the classifier head matches the encoder that produced the output. A mismatch here makes
every image appear to misclassify for reasons unrelated to the hardware.

### 2. Run the C-simulation

```powershell
$env:PATH = '<vitis>/bin;' + $env:PATH
python scripts/gen_hls_config.py config/build_config.yaml workspace/encoder_e2e/hls_config.cfg
$env:E2E_DIR = '<abs-path>/sim/e2e/'
vitis-run --mode hls --csim --config workspace/encoder_e2e/hls_config.cfg --work_dir workspace/encoder_e2e
```

- The **output path's parent directory name** selects which component gets built —
  `workspace/encoder_e2e/` is what picks the end-to-end testbench.
- `config/inc/*.h` must exist first; run `make config` if they do not.
- **`E2E_DIR` must be an absolute path.** Unset, the testbench falls back to a path
  relative to its working directory and exits within a minute reporting that it cannot
  open `model_index.bin`.
- Roughly 3–4 minutes per image. Success ends with `CSim done with 0 errors`.

The testbench writes `hw_out.bin` (final encoder output per image) and `hw_layers.bin`
(per-layer boundaries for image 0).

### 3. Check the result

```bash
<venv>/python sim/check_e2e.py
```

Prints per-image hardware prediction against software prediction and ground truth, then:

```
HW==SW agreement : 32/32
HW accuracy      : 32/32   (SW accuracy: 32/32)
```

**`HW==SW agreement` is the number that matters** — it says the hardware reproduces the
software model. Accuracy is a property of the model, not the hardware; compare it against
the software accuracy in parentheses.

To point `check_e2e.py` at a different config than the one recorded in `meta.json`, set
`E2E_CFG` to an absolute path.

## Demo

```bash
<venv>/python sim/demo_e2e.py
```

Narrates the flow image by image and closes with accuracy, agreement, and throughput.

| flag | effect |
|---|---|
| *(no flag)* | replays the artifacts an earlier C-simulation produced; runs in seconds |
| `--csim` | actually invokes stage 1 and stage 2 to regenerate them (minutes per image) |
| `--images N` | run only the first N images |
| `--speed X` | pacing multiplier for the narration (`0` = no pauses) |

Two things it does not do. Without `--csim` it never re-runs the kernel — the predictions
are real but were computed by the earlier C-simulation, which is why it finishes in
seconds. And the **throughput figure is hardcoded** (`LATENCY_MS` at the top of the
script) from a separate C-synthesis measurement; C-simulation models no timing, so no
timing number comes from this flow.

## Not in git

`sim/models/` (checkpoints), `sim/data/` (CIFAR-10), and `sim/e2e/` (compiled payload and
simulation output) are excluded. A fresh clone has none of them — start at stage 1, and
obtain the checkpoint separately.
