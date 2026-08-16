"""Pure SW accuracy check: run the brevitas quant model over N images and report top-1.

NO HW, NO compile_model, NO sim/e2e data — this exercises ONLY the SW reference model
(build_sw_model + the dataset), so it validates the checkpoint itself. Used to confirm a
new model (e.g. the 32x32-patch variant) is good before any HW e2e work.

Run with the sparta-sw venv:
    ../sparta-sw/.venv/Scripts/python.exe sim/sw_accuracy_check.py <toml_path> [n_images]
"""
import os, sys
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# Reuse the validated SW-model builder (does NOT touch compile/HW).
import full_model_infer as F
from brevitas.quant_tensor import IntQuantTensor

CIFAR = ["airplane", "automobile", "bird", "cat", "deer", "dog",
         "frog", "horse", "ship", "truck"]


def main(toml_path, n_images=24):
    import toml
    iq = F._load_infer_module()
    cfg = toml.load(toml_path)
    print(f"TOML       : {toml_path}")
    print(f"checkpoint : {cfg['model']['path']}")
    print(f"p_size     : {cfg['infer']['model']['embed']['p_size']}")
    print(f"images     : {n_images}\n")

    model = F.build_sw_model(cfg, iq)   # brevitas quant model + checkpoint
    data, _ = iq.initialize_map_datasets(cfg)

    correct = 0
    print("image | truth      | SW pred    | ok")
    print("------+------------+------------+----")
    for idx in range(n_images):
        img, label = data[idx]
        with torch.no_grad():
            logits = model(img.unsqueeze(0))
        lv = logits.value if isinstance(logits, IntQuantTensor) else logits
        pred = int(lv.argmax(-1))
        ok = (pred == int(label))
        correct += ok
        print(f"  {idx:<3} | {CIFAR[int(label)]:<10} | {CIFAR[pred]:<10} | {'yes' if ok else 'NO'}")

    print(f"\nSW top-1 accuracy: {correct}/{n_images} = {100.0*correct/n_images:.1f}%")


if __name__ == "__main__":
    toml_path = sys.argv[1] if len(sys.argv) > 1 else F._CFG
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    main(toml_path, n)
