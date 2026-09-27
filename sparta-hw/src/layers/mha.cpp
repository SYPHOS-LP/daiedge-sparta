/**********************************************************************************
 * Linear Multi-Head Attention Layer - Implementation
 *
 * Please see the respective header file for a high-level description of the layer.
 *
 *********************************************************************************/

#include "mha.h"
#include "spmm_tiled.h"
#include "mha_recip.h"
#include "quantize.h"
#ifndef __SYNTHESIS__
#include <cassert>
#endif

#if !defined(MHA_SPARSE_FORMAT_CSR)
#error "mha requires MHA_SPARSE_FORMAT_CSR."
#endif

// On-chip decoded-weight types must cover the values they narrow the DDR CSR to.
static_assert(MHA_FEATURE_W_MAX <= (1 << T_MhaWCol::width), "MHA_FEATURE_W_MAX exceeds T_MhaWCol range (widen T_MhaWCol)");

// Per-row weight-decode buffer must cover the densest emitted row (rows are sparse).
static_assert(MHA_MAX_W_NNZ <= MHA_FEATURE_W_MAX, "MHA_MAX_W_NNZ should not exceed dense row width D");

// Head geometry: D splits evenly into N_HEADS x D_HEAD (code assumes d_head == 64).
static_assert(MHA_FEATURE_W_MAX == MHA_N_HEADS * MHA_FEATURE_HEAD, "MHA_FEATURE_W_MAX != N_HEADS * D_HEAD");
static_assert(MHA_FEATURE_HEAD == 64,                  "code assumes d_head == 64");

// Per-head buffer sizes must match the head dimensions.
static_assert(MHA_HEAD_NNZ_MAX   == MHA_FEATURE_HEAD * MHA_TOKEN_W_MAX, "head CSR capacity != 64*N");
static_assert(MHA_SCR_V_HEAD_CNT == MHA_FEATURE_HEAD * MHA_TOKEN_W_MAX, "V head count != 64*N");
static_assert(MHA_SCR_A_CNT      == MHA_FEATURE_HEAD * MHA_FEATURE_HEAD, "A cnt != d_head^2");

// parallel_out_rows_compute manually unrolls the 8 packed lanes of a bank word.
static_assert(MHA_BANK_PACK >= 8, "MHA_BANK_PACK should cover at least eight tokens per word");


// Wide DDR read words for better bursts on the AXI ports.
 #define MHA_WIDE_BITS   256
#define MHA_VAL_LANES   (MHA_WIDE_BITS / 8)    /* 32 int8 values per beat  */
#define MHA_COL_LANES   (MHA_WIDE_BITS / 16)   /* 16 uint16 columns per beat */
typedef ap_uint<MHA_WIDE_BITS> T_MhaWideWord;

/**
 * @brief Linear (softmax-free) attention for one head's feature band.
 *
 * Computes O = Q'.(K'^T V) / (Q'. sum K'), reassociated so the intermediate is a small
 * d_head x d_head matrix (no N x N score matrix, no softmax):
 *   pass 1 builds key_value_matrix = K'^T V and key_sums = sum K' (walk K' features);
 *   pass 2 builds numerator Q'.key_value_matrix and denominator Q'.key_sums per token,
 *   then divides (one reciprocal per token);
 *   pass 3 packs the head's output into the packed URAM the O projection reads.
 * Heads run serially, so the working buffers are one reused set.
 *
 * @param[in]  head          Head index.
 * @param[in]  tokens_dim    Number of tokens.
 * @param[in]  quant_scale   Folded scale used for quantizing the output.
 * @param[in]  k_val_buf     On-chip K' value slice buffer.
 * @param[in]  k_col_buf     On-chip K' column slice buffer.
 * @param[in]  k_ptr         K' row pointers (absolute; rebased by k_slice_base).
 * @param[in]  k_slice_base  Absolute nnz offset of this head's K' slice.
 * @param[in]  q_val_buf     On-chip Q' value slice buffer.
 * @param[in]  q_col_buf     On-chip Q' column slice buffer.
 * @param[in]  q_ptr         Q' head-major row pointers (absolute; rebased by q_slice_base).
 * @param[in]  q_slice_base  Absolute nnz offset of this head's Q' slice.
 * @param[in]  v_dense       Dense V.
 * @param[out] out_packed    Packed output.
 */


void head_core (
    int             enable,
    int             head,
    int             tokens_dim,
    T_Scale         quant_scale,
    /* This head's K' and Q' CSR and its dense V, all on-chip and indexed from 0. */
    T_Activation   *k_val,
    T_MhaHeadIndex *k_col,
    int            *k_ptr,
    T_Activation   *q_val,
    T_MhaHeadIndex *q_col,
    int            *q_ptr,
    T_Activation   *v_head,
    T_MhaBankWord  *out_packed
) {
    if (!enable) {
        return;
    }
    const int feature_base = head * MHA_FEATURE_HEAD;   /* only for the packed output offset */

    /* Per-head output scratch */
    static T_Activation head_output[MHA_FEATURE_HEAD][MHA_BANK_PAD_TOK];
    #pragma HLS ARRAY_PARTITION variable=head_output cyclic factor=MHA_MAC_UNROLL dim=1
    #pragma HLS BIND_STORAGE variable=head_output type=ram_1p impl=lutram

    /* Per-head intermediate results for KxV and sum(K). */
    T_MhaAcc key_value_matrix[MHA_FEATURE_HEAD * MHA_FEATURE_HEAD];
    T_MhaAcc key_sums[MHA_FEATURE_HEAD];
    #pragma HLS ARRAY_PARTITION variable=key_value_matrix cyclic factor=MHA_MAC_UNROLL dim=1
    #pragma HLS ARRAY_PARTITION variable=key_sums         complete dim=1

    CLEAR_KEY_VALUE_MATRIX:
    for (int i = 0; i < MHA_FEATURE_HEAD * MHA_FEATURE_HEAD; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=MHA_MAC_UNROLL
        key_value_matrix[i] = (T_MhaAcc) 0;
    }

    CLEAR_KEY_SUMS:
    for (int i = 0; i < MHA_FEATURE_HEAD; i++) {
        #pragma HLS PIPELINE II=1
        key_sums[i] = (T_MhaAcc) 0;
    }

    /* Calculate KxV and sum(K) */
    KEY_VALUE_FEATURE:
    for (int feature = 0; feature < MHA_FEATURE_HEAD; feature++) {
        #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_FEATURE_HEAD max=MHA_TC_FEATURE_HEAD
        int nz_begin = k_ptr[feature];
        int nz_end   = k_ptr[feature + 1];

        KEY_VALUE_NONZERO:
        for (int s = nz_begin; s < nz_end; s++) {
            #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_KP_NNZ max=MHA_TC_KP_NNZ
            int          token   = (int) k_col[s];
            T_Activation key_val = k_val[s];
            
            key_sums[feature] += key_val;

            KEY_VALUE_OUT_FEATURE:
            for (int out_feature = 0; out_feature < MHA_FEATURE_HEAD; out_feature++) {
                #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_FEATURE_HEAD max=MHA_TC_FEATURE_HEAD
                #pragma HLS PIPELINE II=1
                #pragma HLS UNROLL factor=MHA_MAC_UNROLL

                /* Multiply-Accumulate K value with MHA_FEATURE_HEAD features of selected token. */
                key_value_matrix[feature * MHA_FEATURE_HEAD + out_feature] +=
                    key_val * v_head[out_feature * tokens_dim + token];
            }
        }
    }

    /* Calculate Qx(KxV) and Qxsum(K), then divide with each other. */
    NUMERATOR_TOKEN:
    for (int token = 0; token <  tokens_dim; token++) {
        #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_TOKEN max=MHA_TC_TOKEN
        T_MhaAcc numerator_row[MHA_FEATURE_HEAD];
        #pragma HLS ARRAY_PARTITION variable=numerator_row complete dim=1

        CLEAR_NUMERATOR_ROW:
        for (int out_feature = 0; out_feature < MHA_FEATURE_HEAD; out_feature++) {
            numerator_row[out_feature] = (T_MhaAcc) 0;
        }

        /* Q' is token-major for this head and q_col holds the head-local feature (0..63). */
        T_MhaAcc denominator = (T_MhaAcc) 0;
        int      nz_begin    = q_ptr[token];
        int      nz_end      = q_ptr[token + 1];
        NUMERATOR_NONZERO:
        for (int s = nz_begin; s < nz_end; s++) {
            #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_QP_NNZ max=MHA_TC_QP_NNZ
            int          band_feature = (int) q_col[s];   // head-local feature (0..63)
            T_Activation query_val    = q_val[s];

            denominator += query_val * key_sums[band_feature];

            NUMERATOR_OUT_FEATURE:
            for (int out_feature = 0; out_feature < MHA_FEATURE_HEAD; out_feature++) {
                #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_FEATURE_HEAD max=MHA_TC_FEATURE_HEAD
                #pragma HLS PIPELINE II=1
                #pragma HLS UNROLL factor=MHA_MAC_UNROLL

                numerator_row[out_feature] +=
                    query_val * key_value_matrix[band_feature * MHA_FEATURE_HEAD + out_feature];
            }
        }

        /* Reciprocal calculated from LUT, see relevant header file. */
        T_MhaRecip reciprocal = mha_recip(denominator);
        NUMERATOR_DIVIDE:
        for (int out_feature = 0; out_feature < MHA_FEATURE_HEAD; out_feature++) {
            #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_FEATURE_HEAD max=MHA_TC_FEATURE_HEAD
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=MHA_MAC_UNROLL
            /* The two wide multiplies (numerator*reciprocal, then *quant_scale) chained in
             * one cycle were the post-route critical path (head_core NUMERATOR_DIVIDE, two
             * DSP muls in series).  Give each mul a registered (latency=2) implementation so
             * the chain is spread across pipeline stages -- keeps II=1, numerically identical. */
            T_MhaDiv ratio = (T_MhaDiv) (numerator_row[out_feature] * reciprocal);
            #pragma HLS BIND_OP variable=ratio op=mul impl=dsp latency=2
            T_MhaDiv scaled = ratio * (T_MhaDiv) quant_scale;
            #pragma HLS BIND_OP variable=scaled op=mul impl=dsp latency=2
            head_output[out_feature][token] = saturate_to_int8(scaled);
        }
    }

    /* Pack feature-major output and write back to URAM. */
    PACK_OUTPUT_FEATURE:
    for (int out_feature = 0; out_feature < MHA_FEATURE_HEAD; out_feature++) {
        #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_FEATURE_HEAD max=MHA_TC_FEATURE_HEAD

        PACK_OUTPUT_WORD:
        for (int w = 0; w < MHA_BANK_ROW_WORDS; w++) {
            #pragma HLS LOOP_TRIPCOUNT min=MHA_BANK_ROW_WORDS max=MHA_BANK_ROW_WORDS
            #pragma HLS PIPELINE II=1

            T_MhaBankWord word = 0;
            for (int p = 0; p < MHA_BANK_PACK; p++) {
                #pragma HLS UNROLL
                int token = w * MHA_BANK_PACK + p;
                T_Activation v = (token <  tokens_dim) ? head_output[out_feature][token] : (T_Activation) 0;
                word.range(p * MHA_ACT_ELEM_BITS + MHA_ACT_ELEM_BITS - 1,
                           p * MHA_ACT_ELEM_BITS) = (ap_uint<MHA_ACT_ELEM_BITS>) v;
            }

            out_packed[(feature_base + out_feature) * MHA_BANK_ROW_WORDS + w] = word;
        }
    }
}

/* Which tail the shared projection runs; see mha_projection(). */
