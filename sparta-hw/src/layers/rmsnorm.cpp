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
    /* One flattened walk over (feature, word).  As two loops the outer one is not pipelined, so the
     * inner pipeline's fill and drain were paid once per feature -- 12 cycles per feature where the
     * work is RMS_BANK_ROW_WORDS.  HLS will not flatten them itself because both bounds are runtime
     * values, so the word axis is walked over its compile-time bound and short rows are predicated
     * out; that also makes the recurrence distance below a constant.
     *
     * The cost of that: the pass now takes time proportional to the CONFIGURED maximum token count
     * rather than the runtime tokens_dim, so it is a win when the two agree (the kernel is built per
     * configuration, and this one runs its full RMS_TOKEN_W_MAX) and a loss when a short sequence is
     * run on a build sized for a long one.  The engine's MAC_STEP makes the same trade. */
    const int sum_steps = feature_dim * RMS_BANK_ROW_WORDS;
    int sf = 0, sw = 0;

    SUM_SQUARE_STEP:
    for (int i = 0; i < sum_steps; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=(RMS_FEATURE_W_MAX*RMS_BANK_ROW_WORDS) max=(RMS_FEATURE_W_MAX*RMS_BANK_ROW_WORDS)
        #pragma HLS PIPELINE II=1
        /* Each lane p is its own partition, and within a lane successive iterations move to the next
         * word, so one accumulator recurs only every RMS_BANK_ROW_WORDS iterations. */
        #pragma HLS DEPENDENCE variable=sum_sq type=inter direction=RAW distance=RMS_BANK_ROW_WORDS dependent=true
        if (sw < row_words) {
            const T_RmsBankWord word = packed_input[sf * RMS_BANK_ROW_WORDS + sw];
            SUM_SQUARE_LANE:
            for (int p = 0; p < RMS_BANK_PACK; p++) {
                #pragma HLS UNROLL
                const T_Activation v = (T_Activation) word.range(p * 8 + 7, p * 8);
                sum_sq[sw * RMS_BANK_PACK + p] += (T_RmsSq) ((int) v * (int) v);
            }
        }
        if (++sw == RMS_BANK_ROW_WORDS) { sw = 0; sf++; }
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
