# Testbenches

Two C-simulation drivers for the encoder kernel. Both build the same
`encoder_layer_top()` sources; they differ in where the weights and inputs come from.

```bash
make sim PROJECT=encoder_layer   # synthetic weights, self-checking
make sim PROJECT=encoder_e2e     # real compiled model, checked by sim/
```

## `encoder_layer_tb.cpp` (`PROJECT=encoder_layer`)

Tests `encoder_layer_top()`: `Y = MLP-block(MHA-block(X))`, the full pre-norm encoder
layer chaining both sub-blocks. It requantizes the intermediate `H` to int8 between the
two blocks — the real DDR boundary tensor — exactly as hardware does.

Uses synthetic (mostly identity-projection) weights and the power-of-two test scales in
`tb_pot_scales.h`, so a double-precision reference can match the kernel's requantization
exactly. The testbench recomputes the whole pipeline in double precision, mirroring the
hardware's int8 narrowing at each stage boundary, then compares element-wise against the
kernel's dequantized output. It allows a small per-element tolerance plus a cap on how
many elements may sit exactly one quantization step off, since every stage is a
fixed-point or LUT approximation of the true math.

**Self-checking** — prints `PASS`/`FAIL` per case and a final tally.

## `encoder_e2e_tb.cpp` (`PROJECT=encoder_e2e`)

Runs the real `encoder_layer_top()` through all 12 encoder layers back-to-back on the
real per-layer quantized model, for every image in the payload, in one csim process.
This is the testbench the `sim/` end-to-end flow drives — see
[`sim/README.md`](../sim/README.md).

Reads its data from `sim/e2e/` (override with the `E2E_DIR` environment variable or a
directory argument):

| file | contents |
|---|---|
| `model_index.bin` | per-layer (nnz, n_rows), so the model streams without parsing JSON |
| `model.bin` | per-layer CSR weights (int8 values, u16 column indices, i32 row pointers) + folded scales |
| `inputs.bin` | per-image layer-0 input, feature-major int8 |

Writes `hw_out.bin` (final per-image encoder output) and `hw_layers.bin` (per-layer
hidden/output boundaries for image 0).

**Not self-checking** — it produces output but decides nothing. Correctness comes from
`sim/check_e2e.py`, which classifies `hw_out.bin` and compares the predictions against
the software model.

## `tb_pot_scales.h`

Power-of-two test scale values (`pot_scale`) and a reference requantization helper
(`ref_requant`) mirroring `inc/helpers/quantize.h`. Used by `encoder_layer_tb.cpp`.
