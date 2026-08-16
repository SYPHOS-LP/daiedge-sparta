/**********************************************************************************
 * Root-Mean-Square Normalization Layer - Implementation
 *
 * Please see the respective header file for a high-level description of the layer.
 *
 *********************************************************************************/

#include "rmsnorm.h"
#include "rmsnorm_rsqrt.h"
#include "quantize.h"
#include <cmath>
#ifndef __SYNTHESIS__
#include <cassert>
#endif

void rmsnorm (
    const T_Activation* input,
    int                 tokens_dim,
    int                 feature_dim,
    T_Scale             inverse_output_scale,
    T_Activation*       output,
    T_RmsBankWord*      packed_output
) {
#pragma HLS INLINE off
#ifndef __SYNTHESIS__
    assert(tokens_dim  <= RMS_TOKEN_W_MAX && "tokens_dim exceeds RMS_TOKEN_W_MAX");
    assert(feature_dim <= RMS_FEATURE_W_MAX    && "feature_dim exceeds RMS_FEATURE_W_MAX");
#endif
    /* On-chip scratch for a single token's raw input codes (feature-major row). */
    T_Activation input_row[RMS_FEATURE_W_MAX];
#pragma HLS BIND_STORAGE variable=input_row type=ram_2p impl=bram

    /* Strides used for accessing elements in the input array, which is in feature-major layout */
    const int token_stride    = 1;
    const int feature_stride  = tokens_dim;

    /* Inverse feature dimension and epsilon */
    const T_RmsSq inv_feature_dim = (T_RmsSq) (1.0f / (float)feature_dim);
    const T_RmsSq eps             = (T_RmsSq) RMS_EPS;

    PER_ROW_LOOP:
    for (int t = 0; t < tokens_dim; t++) {
#pragma HLS LOOP_TRIPCOUNT min=RMS_TOKEN_W_MAX max=RMS_TOKEN_W_MAX

        /* Accumulate the sum of squares of the raw int8 codes.  No dequant is needed:
         * the input scale cancels between the sum-of-squares and its rsqrt. */
        T_RmsSq accumulator = (T_RmsSq) 0;
        SUM_SQUARE_LOOP:
        for (int f = 0; f < feature_dim; f++) {
#pragma HLS LOOP_TRIPCOUNT min=RMS_FEATURE_W_MAX max=RMS_FEATURE_W_MAX
#pragma HLS PIPELINE II=1
            T_Activation input_quant = input[t * token_stride + f * feature_stride];

            /* Keep the raw codes in the scratch buffer for the normalize pass. */
            input_row[f] = input_quant;

            /* Sum of squares. */
            T_RmsSq square  = (T_RmsSq) ((int)input_quant * (int)input_quant);
            accumulator    += square;
        }

        /* Divide by feature dimension and add epsilon (mean square of the codes). */
        T_RmsSq mean_square  = (T_RmsSq) (accumulator * inv_feature_dim) + eps;

        /* Look-up reciprocal square root. */
        T_RmsRsqrt inv_rms = rms_rsqrt(mean_square);

        /* Chunk and column indices for packed output. */
        const int word_chunk = t % RMS_BANK_PACK;
        const int word_col   = t / RMS_BANK_PACK;
       
        /* Normalize and quantize, then write to output. */
        NORMALIZE_AND_QUANTIZE_LOOP:
        for (int f = 0; f < feature_dim; f++) {
#pragma HLS LOOP_TRIPCOUNT min=RMS_FEATURE_W_MAX max=RMS_FEATURE_W_MAX
#pragma HLS PIPELINE II=1
/* 
 * Dependency marked as false since consecutive iterations of the loop access different elements of packed_output.
 * The same word is only revisited RMS_BANK_PACK tokens later, so the cross-iteration RMW is far apart.
 */
#pragma HLS DEPENDENCE variable=packed_output type=inter false
            /* Normalize the raw code by rsqrt, then requant by the inverse output scale. */
            T_RmsNorm input_norm   = (T_RmsNorm) input_row[f] * (T_RmsNorm) inv_rms;
            T_Activation quant_out = saturate_to_int8(input_norm * inverse_output_scale);

            if (packed_output) {
                /* Packed word index */
                int word_idx = f * RMS_BANK_ROW_WORDS + word_col;

                /* Set this token's chunk within the URAM packed word. */
                T_RmsBankWord word = packed_output[word_idx];
                word.range(word_chunk * 8 + 7, word_chunk * 8) = (ap_uint<8>)quant_out;
                packed_output[word_idx] = word;
            } 
            else {
                /* Write directly to the output array. */
                output[t * token_stride + f * feature_stride] = quant_out;
            }
        }
    }
}
