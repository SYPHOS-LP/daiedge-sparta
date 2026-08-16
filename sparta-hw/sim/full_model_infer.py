"""Shared boundary machinery for the end-to-end flow. Imported, not run directly.

The hardware implements only the 12 encoder layers. Classifying a real image still
needs the SW embedding (patchify + conv + CLS + positional) in front and the SW
classifier head behind, so rather than re-implement those quantized layers this
module borrows them from the sparta-sw brevitas model:

    image --[SW embed + quant_input]--> int8 (D x N) --[encoder]--> int8
          --[* out_scale]--> float --[SW heads.ln + head]--> logits --> argmax

emit_e2e.py uses the front half to capture the encoder's int8 input; check_e2e.py
and demo_e2e.py use run_head() for the back half. Both boundaries are taken from the
SW model's own quantizers, so the tensors match what the SW encoder would have seen.

Boundary facts:
  * the encoder input comes from encoder_layer_0.quant_input (int8, token-major
    (1,N,D)); the kernel works feature-major (D,N), hence the transposes.
  * the encoder output real value = int8_code * the last residual's out_quant scale;
    heads.ln has its own input_quant, so feeding it that dequantized float
    reproduces the head exactly.

Needs the sparta-sw environment (torch + brevitas).
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import torch

# --- make both the replica (this dir) and sparta-sw importable ----------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.abspath(os.path.join(_HERE, "..", "..", "sparta-sw", "sparse_vit", "src"))
sys.path.insert(0, _HERE)
sys.path.insert(0, _SW)

import encoder_replica as R  # noqa: E402

from brevitas.quant_tensor import IntQuantTensor  # noqa: E402
from sparse_vit.quantization.utils import _densify_pruning_state_dict  # noqa: E402
from sparse_vit.quantization.brevitas_quant import _setup_dyadic_residuals  # noqa: E402

_CFG = os.path.join(_SW, "..", "cfgs", "infer_pot4_local.toml")


def _load_infer_module():
    """Import the sparta-sw inference script as a module (for its builder helpers)."""
    path = os.path.join(_SW, "bin", "infer_quant_vit_simple.py")
    spec = importlib.util.spec_from_file_location("iq", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_sw_model(cfg, iq):
    """Build the full SW brevitas quant model and load the 1ee51p3s checkpoint."""
    q = cfg["quant"]
    model = iq.build_quant_model(
        cfg=cfg, wbits=q["bits"], abits=q["abits"],
        weight_per_tensor=q["weight_per_tensor"], weight_scale_po2=q["weight_scale_po2"],
        weight_weight_po2=q["weight_weight_po2"], act_scale_po2=q["act_scale_po2"],
        fuse_rms_norm=q["fuse_rms_norm"],
    )
    state = iq._resolve_state(iq._load_ckpt(cfg["model"]["path"]))
    state, _ = _densify_pruning_state_dict(state)
    model.load_state_dict(state, strict=False)
    model.eval()
    _setup_dyadic_residuals(model)
    return model


class EncoderBridge:
    """Captures the SW encoder's int8 in/out boundary scales via forward hooks."""

    def __init__(self, model):
        self.model = model
        self._enc_in_int8 = None
        self._enc_in_scale = None
        # int8 input to the encoder = layer-0's quant_input output
        model.encoder.encoder_layer_0.quant_input.register_forward_hook(self._grab_in)
        # encoder output real scale = last layer's res2 out_quant scale
        self.out_scale = float(
            model.encoder.encoder_layer_11.res2.out_quant.act_quant.scale()
        )

    def _grab_in(self, mod, inp, out):
        # out is an IntQuantTensor (1, N, D); store int codes + scale.
        self._enc_in_int8 = out.int().detach().to(torch.int32).numpy()[0]  # (N, D)
        self._enc_in_scale = float(out.scale.flatten()[0])

    def encoder_input(self):
        """int8 encoder input as feature-major (D, N), plus its scale."""
        return self._enc_in_int8.T.astype(np.int8), self._enc_in_scale


def attach_sw_attn_scales(model, layers):
    """Populate each LayerWeights.sw_attn with the SW quant model's attention scales.

    Enables encoder_replica's SW-faithful attention path (ReLU-after-quant + float
    division).  swz_* = s_weight * s_z is recovered from the folded scale and the
    raw quant scale:  SCALE_Q = s_wq*s_z / s_q  ->  swz_q = SCALE_Q * s_q.
    """
    for L, w in enumerate(layers):
        mha = getattr(model.encoder, f"encoder_layer_{L}").mha
        s_q = float(mha.quant_q.act_quant.scale())
        s_k = float(mha.quant_k.act_quant.scale())
        s_v = float(mha.quant_v.act_quant.scale())
        s_out = float(mha.quant_out.act_quant.scale())
        scale_half = float(mha.scale) ** 0.5              # d_k^-0.25
        sc = w.scales
        w.sw_attn = {
            "s_q": s_q, "s_k": s_k, "s_v": s_v, "s_out": s_out,
            "scale_half": scale_half,
            "swz_q": float(sc[R.SCALE_Q]) * s_q,          # = s_weight_q * s_z
            "swz_k": float(sc[R.SCALE_K]) * s_k,
            "swz_v": float(sc[R.SCALE_V]) * s_v,
        }


def run_head(model, enc_out_int8, out_scale):
    """Dequantize the replica's int8 encoder output and run the SW head -> logits.

    enc_out_int8 : (D, N) feature-major int8 from the replica.
    """
    real = enc_out_int8.T.astype(np.float32) * out_scale       # (N, D) real values
    x = torch.from_numpy(real).unsqueeze(0)                    # (1, N, D)
    with torch.no_grad():
        logits = model.heads(x)
    lv = logits.value if isinstance(logits, IntQuantTensor) else logits
    return lv[0].detach().numpy()
