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
    const T_RmsBankWord* packed_input,
    int                  tokens_dim,
    int                  feature_dim,
    T_Scale              inverse_output_scale,
    T_RmsBankWord*       packed_output
) {
#pragma HLS INLINE off
#ifndef __SYNTHESIS__
    assert(tokens_dim  <= RMS_TOKEN_W_MAX && "tokens_dim exceeds RMS_TOKEN_W_MAX");
    assert(feature_dim <= RMS_FEATURE_W_MAX && "feature_dim exceeds RMS_FEATURE_W_MAX");
#endif
    /* Input and output hold RMS_BANK_PACK tokens of one feature per word, features in order, so one
     * read covers RMS_BANK_PACK tokens.  Each token is normalized over all its features, which means
     * RMS_BANK_PACK running sums advance per read rather than one. */
    const int row_words = (tokens_dim + RMS_BANK_PACK - 1) / RMS_BANK_PACK;

    T_RmsSq    sum_sq[RMS_BANK_PAD_TOK];
    T_RmsRsqrt inv_rms[RMS_BANK_PAD_TOK];
    #pragma HLS ARRAY_PARTITION variable=sum_sq  cyclic factor=RMS_BANK_PACK dim=1
    #pragma HLS ARRAY_PARTITION variable=inv_rms cyclic factor=RMS_BANK_PACK dim=1

    const T_RmsSq inv_feature_dim = (T_RmsSq) (1.0f / (float) feature_dim);
    const T_RmsSq eps             = (T_RmsSq) RMS_EPS;

    CLEAR_SUMS:
    for (int t = 0; t < RMS_BANK_PAD_TOK; t++) {
        #pragma HLS UNROLL
        sum_sq[t] = (T_RmsSq) 0;
    }

    /* Sum of squares of the raw codes, per token.  No dequantization: the input scale cancels
     * between the sum of squares and its reciprocal square root. */
    SUM_SQUARE_FEATURE:
    for (int f = 0; f < feature_dim; f++) {
        #pragma HLS LOOP_TRIPCOUNT min=RMS_FEATURE_W_MAX max=RMS_FEATURE_W_MAX
        SUM_SQUARE_WORD:
        for (int w = 0; w < row_words; w++) {
            #pragma HLS LOOP_TRIPCOUNT min=RMS_BANK_ROW_WORDS max=RMS_BANK_ROW_WORDS
            #pragma HLS PIPELINE II=1
            const T_RmsBankWord word = packed_input[f * RMS_BANK_ROW_WORDS + w];
            SUM_SQUARE_LANE:
            for (int p = 0; p < RMS_BANK_PACK; p++) {
                #pragma HLS UNROLL
                const T_Activation v = (T_Activation) word.range(p * 8 + 7, p * 8);
                sum_sq[w * RMS_BANK_PACK + p] += (T_RmsSq) ((int) v * (int) v);
            }
        }
    }

    RSQRT_TOKEN:
    for (int t = 0; t < tokens_dim; t++) {
        #pragma HLS LOOP_TRIPCOUNT min=RMS_TOKEN_W_MAX max=RMS_TOKEN_W_MAX
        #pragma HLS PIPELINE II=1
        const T_RmsSq mean_square = (T_RmsSq) (sum_sq[t] * inv_feature_dim) + eps;
        inv_rms[t] = rms_rsqrt(mean_square);
    }

    /* Normalize and requantize, building each output word once.  Slots past tokens_dim are zero. */
    NORMALIZE_FEATURE:
    for (int f = 0; f < feature_dim; f++) {
        #pragma HLS LOOP_TRIPCOUNT min=RMS_FEATURE_W_MAX max=RMS_FEATURE_W_MAX
        NORMALIZE_WORD:
        for (int w = 0; w < row_words; w++) {
            #pragma HLS LOOP_TRIPCOUNT min=RMS_BANK_ROW_WORDS max=RMS_BANK_ROW_WORDS
            #pragma HLS PIPELINE II=1
            const T_RmsBankWord in_word = packed_input[f * RMS_BANK_ROW_WORDS + w];
            T_RmsBankWord       out_word = 0;
            NORMALIZE_LANE:
            for (int p = 0; p < RMS_BANK_PACK; p++) {
                #pragma HLS UNROLL
                const int    t = w * RMS_BANK_PACK + p;
                T_Activation quant_out = (T_Activation) 0;
                if (t < tokens_dim) {
                    const T_Activation v          = (T_Activation) in_word.range(p * 8 + 7, p * 8);
                    const T_RmsNorm    input_norm = (T_RmsNorm) v * (T_RmsNorm) inv_rms[t];
                    quant_out = saturate_to_int8(input_norm * inverse_output_scale);
                }
                out_word.range(p * 8 + 7, p * 8) = (ap_uint<8>) quant_out;
            }
            packed_output[f * RMS_BANK_ROW_WORDS + w] = out_word;
        }
    }
}
