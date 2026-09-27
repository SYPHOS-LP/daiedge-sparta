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

        const T_MhaBankWord a = residual_input[k];
        const T_MhaBankWord b = branch_input[k];
        T_MhaBankWord       o = 0;

        /* One word carries MHA_BANK_PACK tokens, so the pass costs one cycle per word rather than
         * one per element.  int8 codes in different quantization domains; the folded ratios
         * reconcile them. */
        RESIDUAL_LANE:
        for (int p = 0; p < MHA_BANK_PACK; p++) {
            #pragma HLS UNROLL
            const T_Activation av = (T_Activation) a.range(p * 8 + 7, p * 8);
            const T_Activation bv = (T_Activation) b.range(p * 8 + 7, p * 8);
            const T_Scale sum = (T_Scale)(int) av * residual_scale
                              + (T_Scale)(int) bv * branch_scale;
            o.range(p * 8 + 7, p * 8) = (ap_uint<8>) saturate_to_int8(sum);
        }
        residual_output[k] = o;
    }
}
