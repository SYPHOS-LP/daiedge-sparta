"""Classify the csim's hw_out.bin and compare to SW / truth.

Reads sim/e2e/{hw_out.bin, meta.json}, runs each image's HW encoder output through the SAME
validated SW head path (run_head: (D,N) feature-major int8 -> transpose -> dequant by out_scale
-> SW head -> argmax), and reports per-image HW pred vs sw_pred vs truth.

Run with the sparta-sw venv:
    ../sparta-sw/.venv/Scripts/python.exe sim/check_e2e.py
"""
import os, sys, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import full_model_infer as F
import toml

CIFAR = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
OUT = os.path.join(_HERE, "e2e")


def main():
    meta = json.load(open(os.path.join(OUT, "meta.json")))
    n, D, N = meta["n"], meta["D"], meta["N"]
    out_scale = meta["out_scale"]
    truth = meta["truth"]; sw_pred = meta["sw_pred"]

    # The SW head MUST be built from the SAME checkpoint that generated the artifacts,
    # otherwise a p16 head classifies a 32b encoder output (or vice-versa) and every
    # image "misclassifies" for a reason that has nothing to do with the hardware.
    # emit_e2e records the config it used in meta.json["cfg"]; honour it (env override wins).
    cfg_path = os.environ.get("E2E_CFG") or meta.get("cfg")
    if cfg_path:
        F._CFG = cfg_path if os.path.isabs(cfg_path) else os.path.join(_HERE, cfg_path)

    # hw_out.bin: 16-byte header {magic 0x45324F55, n_images, D, N} then n*(D*N) int8.
    blob = open(os.path.join(OUT, "hw_out.bin"), "rb").read()
    hdr = np.frombuffer(blob[:16], dtype="<i4")
    assert hdr[0] == 0x45324F55, f"bad hw_out magic {hdr[0]:#x}"
    hn, hD, hN = int(hdr[1]), int(hdr[2]), int(hdr[3])
    assert (hn, hD, hN) == (n, D, N), f"hw_out header {(hn,hD,hN)} != meta {(n,D,N)}"
    raw = np.frombuffer(blob[16:], dtype=np.int8)
    assert raw.size == n * D * N, f"hw_out.bin body {raw.size} != n*D*N {n*D*N}"
    hw = raw.reshape(n, D, N)   # (n, D, N) feature-major, matches TB write

    iq = F._load_infer_module()
    cfg = toml.load(F._CFG)
    model = F.build_sw_model(cfg, iq)

    print("image | truth | SW pred | HW pred | HW==SW | HW==truth")
    print("------+-------+---------+---------+--------+----------")
    agree = correct = 0
    for i in range(n):
        logits = F.run_head(model, hw[i], out_scale)   # (D,N) int8 -> logits
        hw_pred = int(np.asarray(logits).argmax())
        a = (hw_pred == sw_pred[i]); c = (hw_pred == truth[i])
        agree += a; correct += c
        print(f"  {i}   | {CIFAR[truth[i]]:>5} | {CIFAR[sw_pred[i]]:>7} | {CIFAR[hw_pred]:>7} |"
              f"  {'yes' if a else 'NO':>3}  |   {'yes' if c else 'NO'}")
    print(f"\nHW==SW agreement : {agree}/{n}")
    print(f"HW accuracy      : {correct}/{n}   (SW accuracy: {sum(int(a==b) for a,b in zip(sw_pred,truth))}/{n})")


if __name__ == "__main__":
    main()
