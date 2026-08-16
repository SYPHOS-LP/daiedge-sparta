"""Fresh, self-verifying E2E emit for the HW encoder csim.

Regenerates sim/e2e/{model.bin, model_index.bin, inputs.bin, meta.json} from the 1ee51p3s
POT4 checkpoint, reusing the VALIDATED boundary machinery in full_model_infer.py so the data
ordering is guaranteed correct (not assumed):

  INPUT  : the SW model's own quant_input hook gives the int8 encoder input as (N, D) token-major;
           we transpose to (D, N) FEATURE-MAJOR (what the HW kernel + encoder_e2e_tb.cpp expect).
  OUTPUT : hw_out.bin (written by the TB) is (D, N) feature-major; run_head transposes back to
           (N, D), dequantizes by out_scale, and runs the SW head -> logits -> argmax.
  MODEL  : compile_model.py HYBRID emit (pot=True weights = {sign,exp} bytes for the shift-MAC;
           pot_scales=False = ap_fixed requant scales; fused=True = gammas folded).

meta.json records, per image: truth, sw_pred (full SW float pipeline argmax), and the input/out
scales. So the csim's classification can be checked against sw_pred with the SAME head path.

Run with the sparta-sw venv:
    ../sparta-sw/.venv/Scripts/python.exe sim/emit_e2e.py [N_IMAGES]   (default 4)
"""
import os, sys, json, struct
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))   # compile_model

import full_model_infer as F          # validated SW model + boundary hooks + run_head
import compile_model as C
from brevitas.quant_tensor import IntQuantTensor
import toml

CIFAR = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
# 32x32-patch model (N=50).  TXT (weights) and the SW TOML below MUST be the same checkpoint,
# and the TOML must set p_size=32 so the SW forward produces N=50 encoder inputs matching the
# N=50 hardware config.
TXT = os.path.join(_HERE, "models", "20260814",
                   "vit-base-cifar10-32b-finetuned-pruned-quant-test-1p4g6kml.quant_values.txt")
# Override full_model_infer's default (p16) SW config with the 32b one (p_size=32 -> N=50).
F._CFG = os.path.join(_HERE, "models", "20260814", "infer_32b_local.toml")
OUT = os.path.join(_HERE, "e2e")

# little-endian magics the committed encoder_e2e_tb.cpp checks
MAGIC_INDEX = 0x45324D49   # 'IM2E'
MAGIC_INPUT = 0x45324E49   # 'IN2E'


def emit_inputs(enc_inputs, D, N, path):
    """enc_inputs: list of (D,N) int8 feature-major arrays. Writes inputs.bin the TB reads."""
    with open(path, "wb") as f:
        f.write(struct.pack("<I", MAGIC_INPUT))
        f.write(struct.pack("<3i", len(enc_inputs), D, N))
        for a in enc_inputs:
            assert a.shape == (D, N) and a.dtype == np.int8
            f.write(a.tobytes())            # row-major (D,N) = feature-major, matches TB


def main(n_images=4):
    iq = F._load_infer_module()
    cfg = toml.load(F._CFG)
    model = F.build_sw_model(cfg, iq)
    bridge = F.EncoderBridge(model)
    data, _ = iq.initialize_map_datasets(cfg)

    os.makedirs(OUT, exist_ok=True)

    # --- 1) compile the HYBRID model.bin + model_index.bin (compile_model owns the format) ---
    qm = C.compile_model(TXT)
    qm.meta["source"] = os.path.basename(TXT)
    print("Emitting HYBRID model.bin (pot=True, pot_scales=False, fused=True) ...")
    C.emit_model(qm, qm.layers(), OUT, verify=False, pot=True, pot_scales=False, fused=True)

    # model_index.bin: magic + (n_layers, D, d_h, n_heads) + per-layer 6x(nnz, n_rows).
    # Read the per-weight nnz/n_rows straight from the manifest compile_model just wrote.
    man = json.load(open(os.path.join(OUT, "manifest.json")))
    layer_ids = sorted(int(k) for k in man["layers"].keys())
    WORDER = ["w_q", "w_k", "w_v", "out_proj", "linear_1", "linear_2"]
    D, d_h, n_heads = 768, 3072, 12
    with open(os.path.join(OUT, "model_index.bin"), "wb") as f:
        f.write(struct.pack("<I", MAGIC_INDEX))
        f.write(struct.pack("<4i", len(layer_ids), D, d_h, n_heads))
        for L in layer_ids:
            for tag in WORDER:
                w = man["layers"][str(L)]["weights"][tag]
                # ptr array length-1: CSR -> rows (shape[0]); CSC (W2) -> cols (shape[1]).
                n_ptr = int(w["shape"][1]) if w.get("layout") == "csc" else int(w["shape"][0])
                f.write(struct.pack("<2i", int(w["nnz"]), n_ptr))

    # --- 2) per-image: SW forward -> capture feature-major int8 encoder input + sw_pred + truth ---
    enc_inputs, meta_imgs = [], []
    in_scale = out_scale = None
    print(f"\nSW forward over {n_images} images (capturing encoder-input boundary):")
    for idx in range(n_images):
        img, label = data[idx]
        with torch.no_grad():
            sw_logits = model(img.unsqueeze(0))
        sw_lv = sw_logits.value if isinstance(sw_logits, IntQuantTensor) else sw_logits
        sw_pred = int(sw_lv.argmax(-1))
        enc_in, s_in = bridge.encoder_input()          # (D,N) int8 feature-major, its scale
        enc_in = np.ascontiguousarray(enc_in.astype(np.int8))
        N = enc_in.shape[1]
        assert enc_in.shape == (D, N)
        enc_inputs.append(enc_in)
        in_scale = s_in
        out_scale = bridge.out_scale
        meta_imgs.append(dict(truth=int(label), sw_pred=sw_pred))
        print(f"  img {idx}: truth={CIFAR[label]:>10}  sw_pred={CIFAR[sw_pred]:>10}"
              f"  enc_in {enc_in.shape} range[{int(enc_in.min())},{int(enc_in.max())}]")

    N = enc_inputs[0].shape[1]
    emit_inputs(enc_inputs, D, N, os.path.join(OUT, "inputs.bin"))

    meta = dict(n=n_images, D=D, N=N, d_h=d_h,
                in_scale=float(in_scale), out_scale=float(out_scale),
                # Record the SW config used, so check_e2e builds the MATCHING head
                # (p16 vs 32b differ in N and patch layout -> wrong head = phantom misclassify).
                cfg=os.path.abspath(F._CFG),
                images=meta_imgs,
                truth=[m["truth"] for m in meta_imgs],
                sw_pred=[m["sw_pred"] for m in meta_imgs])
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)

    print(f"\nWrote sim/e2e/{{model.bin, model_index.bin, inputs.bin, meta.json}}")
    print(f"  in_scale={in_scale}  out_scale={out_scale}")
    print(f"  truth   = {[CIFAR[m['truth']] for m in meta_imgs]}")
    print(f"  sw_pred = {[CIFAR[m['sw_pred']] for m in meta_imgs]}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    main(n)
