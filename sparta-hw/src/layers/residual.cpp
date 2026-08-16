// ============================================================================
// Residual add + requantize — implementation (see residual.h for the contract).
// ============================================================================

#include "residual.h"
#include "quantize.h"          // saturate_to_int8 (shared round+clamp)

// The encoder residual buffers are D x N feature-major (768 x 197); the flat
// element count is a runtime arg, so this literal is only the LOOP_TRIPCOUNT
// latency-estimation hint (both blocks pass the same 768*197).
#define RESIDUAL_TRIPCOUNT (768 * 197)

void residual_add(
    const T_Activation* residual_input,
    const T_Activation* branch_input,
    int                 n_elem,
    T_Scale             residual_scale,
    T_Scale             branch_scale,
    T_Activation*       residual_output
) {
    RESIDUAL_LOOP:
    for (int k = 0; k < n_elem; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=RESIDUAL_TRIPCOUNT max=RESIDUAL_TRIPCOUNT
        #pragma HLS PIPELINE II=1
        // int8 codes in different quant domains; the folded ratios reconcile them.
        T_Scale residual_scaled = (T_Scale)(int)residual_input[k] * residual_scale;
        T_Scale branch_scaled   = (T_Scale)(int)branch_input[k]   * branch_scale;
        residual_output[k]      = saturate_to_int8(residual_scaled + branch_scaled);
    }
}
