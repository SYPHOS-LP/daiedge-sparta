// ============================================================================
// Residual add + requantize — implementation (see residual.h for the contract).
// ============================================================================

#include "residual.h"
#include "encoder_layer_cfg.h" // ENC_FEATURE_W_MAX / ENC_TOKEN_W_MAX (the layer's dimensions)
#include "mha_cfg.h"           // MHA_BANK_PACK / T_MhaBankWord (the shared packed-word form)
#include "quantize.h"          // saturate_to_int8 (shared round+clamp)

// The residual buffers are D x N feature-major and the element count is a runtime argument, so
// this is only the latency-estimation hint: packed words, so one per MHA_BANK_PACK tokens.
#define RESIDUAL_TRIPCOUNT (ENC_FEATURE_W_MAX * ((ENC_TOKEN_W_MAX + MHA_BANK_PACK - 1) / MHA_BANK_PACK))

void residual_add(
    const T_MhaBankWord* residual_input,
    const T_MhaBankWord* branch_input,
    int                  n_words,
    T_Scale              residual_scale,
    T_Scale              branch_scale,
    T_MhaBankWord*       residual_output
) {
    RESIDUAL_LOOP:
    for (int k = 0; k < n_words; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=RESIDUAL_TRIPCOUNT max=RESIDUAL_TRIPCOUNT
        #pragma HLS PIPELINE II=1
        #pragma HLS DEPENDENCE variable=residual_output type=inter false
        #pragma HLS DEPENDENCE variable=branch_input    type=inter false

        const T_MhaBankWord res_word    = residual_input[k];
        const T_MhaBankWord branch_word = branch_input[k];
        T_MhaBankWord       out_word    = 0;

        /* One word carries MHA_BANK_PACK tokens, so the pass costs one cycle per word rather than
         * one per element.  int8 codes in different quantization domains; the folded ratios
         * reconcile them. */
        RESIDUAL_LANE:
        for (int p = 0; p < MHA_BANK_PACK; p++) {
            #pragma HLS UNROLL
            const T_Activation res_v    = (T_Activation) res_word.range(p * 8 + 7, p * 8);
            const T_Activation branch_v = (T_Activation) branch_word.range(p * 8 + 7, p * 8);
            /* Each product is reduced to T_Scale BEFORE the addition, which is how the golden data
             * was produced.  The difference from folding them into one expression is NOT rounding:
             * an int8 operand times a T_Scale has at most T_Scale's 16 fractional bits, so nothing
             * is lost narrowing it.  It is overflow -- T_Scale holds 16 integer bits, so a product
             * past +/-32768 wraps here where the folded form would saturate.  The layer's scales
             * are of order 1, so the two agree in practice; tb/encoder_units_tb.cpp pins both the
             * agreement and the wrap.  See docs/plans/00-open-issues.md. */
            const T_Scale residual_scaled = (T_Scale)(int) res_v * residual_scale;
            const T_Scale branch_scaled   = (T_Scale)(int) branch_v * branch_scale;
            out_word.range(p * 8 + 7, p * 8) =
                (ap_uint<8>) saturate_to_int8(residual_scaled + branch_scaled);
        }
        residual_output[k] = out_word;
    }
}
