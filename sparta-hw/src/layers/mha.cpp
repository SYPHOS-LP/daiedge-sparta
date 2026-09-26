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
static_assert((int)T_MhaWVal::width >= 4,       "T_MhaWVal must hold w4 (-7..+7)");

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
static_assert(MHA_BANK_PACK == 8, "code assumes MHA_BANK_PACK == 8");

// Output rows the tiled engine advances per step (parallelism now comes from the blocks).
#define MHA_PROJ_P     MHA_ROWS_PER_STEP
#define MHA_PROJ_P_QK  MHA_ROWS_PER_STEP

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


static void head_core (
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
enum MhaProjMode {
    MHA_PROJ_DENSE     = 0,  /* dense rows to the on-chip scratch (V) */
    MHA_PROJ_FEATMAJOR = 1,  /* ReLU + sparsify, feature-major (K') */
    MHA_PROJ_HEADMAJOR = 2,  /* ReLU into the scratch, then sparsify head-major (Q') */
    MHA_PROJ_DENSE_DDR = 3   /* dense rows straight to the layer output in DDR (out_proj) */
};

/**
 * @brief Compute output rows [row_base, row_base + rows) of one projection.
 *
 * Requires the banks to hold this row block's entries (load_weight_slice).  Per output row the
 * MHA_TILES column blocks accumulate in parallel; the accumulated row is then requantized and,
 * for Q' and K', ReLU'd and emitted as CSR.
 *
 * mha() calls this from a single site.  HLS specialises a function per call site and an
 * ALLOCATION limit does not apply to the functions it outlines loop bodies into, so four call
 * sites gave two copies of this logic at 35k LUT each.
 *
 * @param[in]  mode         MHA_PROJ_DENSE (V), FEATMAJOR (K'), HEADMAJOR (Q'), DENSE_DDR (out_proj).
 * @param[in]  row_base     First output row of this block.
 * @param[in]  rows         Output rows in this block.
 * @param[in]  tokens_dim   Number of tokens.
 * @param[in]  quant_scale  Folded requantization scale.
 * @param[out] out_dense    On-chip dense output (DENSE) or transpose scratch (HEADMAJOR).
 * @param[out] out_ddr      DDR dense output (DENSE_DDR).
 * @param[out] q_val        Q' values / columns / row pointers (HEADMAJOR).
 * @param[out] q_col
 * @param[out] q_ptr
 * @param[out] k_val        K' values / columns / row pointers (FEATMAJOR).
 * @param[out] k_col
 * @param[out] k_ptr
 */
static void project_rows (
    int             mode,
    int             row_base,
    int             rows,
    int             tokens_dim,
    T_Scale         quant_scale,
    T_Activation   *q_scratch,
    T_Activation   *v_head,
    T_Activation   *out_ddr,
    T_Activation   *q_val,
    T_MhaHeadIndex *q_col,
    int            *q_ptr,
    T_Activation   *k_val,
    T_MhaHeadIndex *k_col,
    int            *k_ptr
) {
#pragma HLS INLINE off
    /* The sparse outputs are this head's CSR, so they are written from index 0 on every call. */
    int nnz = 0;
    if (mode == MHA_PROJ_FEATMAJOR) {
        k_ptr[0] = 0;
    } else if (mode == MHA_PROJ_HEADMAJOR) {
        q_ptr[0] = 0;
    }

    PROJECTION_ROW:
    for (int i = 0; i < rows; i += MHA_ROWS_PER_STEP) {
        #pragma HLS LOOP_TRIPCOUNT min=(MHA_FEATURE_HEAD/MHA_ROWS_PER_STEP) max=(MHA_FEATURE_HEAD/MHA_ROWS_PER_STEP)
        const int rows_in_step = (rows - i < MHA_ROWS_PER_STEP) ? (rows - i) : MHA_ROWS_PER_STEP;

        tiled_rows_compute(i, rows_in_step, tokens_dim, 0);

        PROJECTION_TAIL_ROW:
        for (int p = 0; p < MHA_ROWS_PER_STEP; p++) {
            #pragma HLS LOOP_TRIPCOUNT min=MHA_ROWS_PER_STEP max=MHA_ROWS_PER_STEP
            if (p >= rows_in_step) {
                continue;
            }
            /* One loop per mode, each with a branch-free body so that all of them hold II=1. */
            if (mode == MHA_PROJ_DENSE_DDR) {
                TAIL_DENSE_DDR:
                for (int j = 0; j < tokens_dim; j++) {
                    #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_TOKEN max=MHA_TC_TOKEN
                    #pragma HLS PIPELINE II=1
                    out_ddr[(row_base + i + p) * tokens_dim + j] =
                        requant_to_int8(g_proj_rows[p][j], quant_scale);
                }
            } else if (mode == MHA_PROJ_DENSE) {
                TAIL_DENSE:
                for (int j = 0; j < tokens_dim; j++) {
                    #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_TOKEN max=MHA_TC_TOKEN
                    #pragma HLS PIPELINE II=1
                    v_head[(i + p) * tokens_dim + j] = requant_to_int8(g_proj_rows[p][j], quant_scale);
                }
            } else if (mode == MHA_PROJ_HEADMAJOR) {
                TAIL_HEADMAJOR:
                for (int j = 0; j < tokens_dim; j++) {
                    #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_TOKEN max=MHA_TC_TOKEN
                    #pragma HLS PIPELINE II=1
                    const T_MhaAcc acc = g_proj_rows[p][j];
                    q_scratch[(i + p) * tokens_dim + j] =
                        (acc > (T_MhaAcc) 0) ? requant_to_int8(acc, quant_scale) : (T_Activation) 0;
                }
            } else {
                TAIL_FEATMAJOR:
                for (int j = 0; j < tokens_dim; j++) {
                    #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_TOKEN max=MHA_TC_TOKEN
                    #pragma HLS PIPELINE II=1
                    const T_MhaAcc acc = g_proj_rows[p][j];
                    if (acc > (T_MhaAcc) 0) {
                        const T_Activation quant_out = requant_to_int8(acc, quant_scale);
                        if (quant_out > (T_Activation) 0) {
#ifndef __SYNTHESIS__
                            assert(nnz < MHA_HEAD_NNZ_MAX && "K' CSR overflow");
#endif
                            k_val[nnz] = quant_out;
                            k_col[nnz] = (T_MhaHeadIndex) j;
                            nnz++;
                        }
                    }
                }
                k_ptr[i + p + 1] = nnz;
            }
        }
    }

    if (mode == MHA_PROJ_HEADMAJOR) {
    /* Q' is emitted token-major: row j of the CSR holds the nonzero features of token j, with the
     * head-local feature index (0..63) as the column.  Q' . A reads it by token, which is why the
     * dense rows go to a scratch first and are transposed here. */
    RELU_ON_SCRATCH_TOKEN:
    for (int j = 0; j < tokens_dim; j++) {
        #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_TOKEN max=MHA_TC_TOKEN

        PER_BAND_FEATURE_LOOP:
        for (int bf = 0; bf < MHA_FEATURE_HEAD; bf++) {
            #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_FEATURE_HEAD max=MHA_TC_FEATURE_HEAD
            #pragma HLS PIPELINE II=1

            T_Activation out_data = q_scratch[bf * tokens_dim + j];

            if (out_data > (T_Activation) 0) {
#ifndef __SYNTHESIS__
                assert(nnz < MHA_HEAD_NNZ_MAX && "Q' CSR overflow");
#endif
                q_val[nnz] = out_data;
                q_col[nnz] = (T_MhaHeadIndex) bf;
                nnz++;
            }
        }

        q_ptr[j + 1] = nnz;
    }
    }
}

void mha(
    T_MhaBankWord* input_packed,
    int            tokens_dim,
    T_MhaTileWord* wq_bank, int* wq_counts,
    T_MhaTileWord* wk_bank, int* wk_counts,
    T_MhaTileWord* wv_bank, int* wv_counts,
    T_MhaTileWord* wo_bank, int* wo_counts,
    const T_Scale* scales,
    T_Activation*  output
) {
/* Keep mha() a standalone function (do NOT dissolve into the caller).  As one function HLS overlaps
 * the independent projection stages inside it; dissolved into encoder_layer_top they become top-level
 * siblings that schedule sequentially.  Which way HLS goes flips on body size (e.g. the POT shift-MAC
 * enlarges the bodies and triggers the dissolve), inflating latency -- so pin it. */
#pragma HLS INLINE off

    const int feature_dim = MHA_FEATURE_W_MAX;          // all projection weights are D x D
#ifndef __SYNTHESIS__
    assert(tokens_dim <= MHA_TOKEN_W_MAX && "tokens_dim exceeds MHA_TOKEN_W_MAX");
#endif

    /* One head's Q', K' and V, sized for 100 % ReLU survival (MHA_FEATURE_HEAD x N), so overflow is
     * impossible without relying on measured densities.  Full-grid Q'/K' would be 12x this and was
     * the reason they used to be kept in DDR. */
    static T_Activation   v_head[MHA_SCR_V_HEAD_CNT];
    static T_Activation   q_scratch[MHA_SCR_V_HEAD_CNT];
    #pragma HLS BIND_STORAGE variable=v_head    type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=q_scratch type=ram_t2p impl=bram

    static T_Activation   k_val_buf[MHA_HEAD_NNZ_MAX];
    static T_MhaHeadIndex k_col_buf[MHA_HEAD_NNZ_MAX];
    static int            k_ptr_buf[MHA_FEATURE_HEAD + 1];
    static T_Activation   q_val_buf[MHA_HEAD_NNZ_MAX];
    static T_MhaHeadIndex q_col_buf[MHA_HEAD_NNZ_MAX];
    static int            q_ptr_buf[MHA_TOKEN_W_MAX + 1];
    #pragma HLS BIND_STORAGE variable=k_val_buf type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=k_col_buf type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=q_val_buf type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=q_col_buf type=ram_t2p impl=bram

    /* Packed attention output, all heads: the output projection needs every feature. */
    static T_MhaBankWord attn_output[MHA_BANK_WORDS];
    #pragma HLS BIND_STORAGE   variable=attn_output type=ram_t2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=attn_output cyclic factor=MHA_BANK_CYCLIC dim=1

    /* Activation tiles for Q, K and V: one copy of the layer input, split per column block. */
    load_activation_tiles(input_packed, 0, MHA_TILE_WORDS);

    /* Phase 1, stages 0..3*MHA_N_HEADS-1: K, Q and V of one head, then that head's attention, so
     * only one head's projections are live at any time.  Phase 2: the output projection, in
     * MHA_ROW_BLOCKS row blocks.  One call site for project_rows and the weight loaders, so HLS
     * instantiates one of each; the weight pointers are selected in this scope, where the
     * ALLOCATION pragmas apply and HLS multiplexes the AXI ports itself. */
    #pragma HLS ALLOCATION instances=project_rows       limit=1 function
    #pragma HLS ALLOCATION instances=load_weight_slice  limit=1 function
    #pragma HLS ALLOCATION instances=load_weight_bases  limit=1 function

    PROJECTION_STAGE:
    for (int stage = 0; stage < 3 * MHA_N_HEADS + MHA_ROW_BLOCKS; stage++) {
        #pragma HLS LOOP_TRIPCOUNT min=(3*MHA_N_HEADS+MHA_ROW_BLOCKS) max=(3*MHA_N_HEADS+MHA_ROW_BLOCKS)
        #pragma HLS LOOP_FLATTEN off

        const int in_phase1 = (stage < 3 * MHA_N_HEADS);
        const int head      = in_phase1 ? (stage / 3) : 0;
        const int which     = in_phase1 ? (stage % 3) : 3;   // 0 = K, 1 = Q, 2 = V, 3 = out_proj
        const int block     = in_phase1 ? head : (stage - 3 * MHA_N_HEADS);
        const int row_base  = block * MHA_FEATURE_HEAD;
        const int first     = (block == 0);

        /* The output projection multiplies the attention result, not the layer input. */
        if (which == 3 && first) {
            load_activation_tiles(attn_output, 0, MHA_TILE_WORDS);
        }

        /* Weight matrix of this stage; one branch each, all in this scope. */
        if (which == 0) {
            if (first) { load_weight_bases(wk_counts, feature_dim, SPMM_W_K); }
            load_weight_slice(wk_bank, wk_counts, feature_dim, SPMM_W_K, row_base, MHA_FEATURE_HEAD);
        } else if (which == 1) {
            if (first) { load_weight_bases(wq_counts, feature_dim, SPMM_W_Q); }
            load_weight_slice(wq_bank, wq_counts, feature_dim, SPMM_W_Q, row_base, MHA_FEATURE_HEAD);
        } else if (which == 2) {
            if (first) { load_weight_bases(wv_counts, feature_dim, SPMM_W_V); }
            load_weight_slice(wv_bank, wv_counts, feature_dim, SPMM_W_V, row_base, MHA_FEATURE_HEAD);
        } else {
            if (first) { load_weight_bases(wo_counts, feature_dim, SPMM_W_O); }
            load_weight_slice(wo_bank, wo_counts, feature_dim, SPMM_W_O, row_base, MHA_FEATURE_HEAD);
        }

        /* K' is emitted feature-major, Q' dense to a scratch and then token-major, V stays dense,
         * the output projection writes the layer output in DDR. */
        const int     mode  = (which == 0) ? MHA_PROJ_FEATMAJOR
                            : (which == 1) ? MHA_PROJ_HEADMAJOR
                            : (which == 2) ? MHA_PROJ_DENSE
                                           : MHA_PROJ_DENSE_DDR;
        const T_Scale scale = (which == 0) ? scales[SCALE_K_IDX]
                            : (which == 1) ? scales[SCALE_Q_IDX]
                            : (which == 2) ? scales[SCALE_V_IDX]
                                           : scales[SCALE_LINEAR_OUT_IDX];

        project_rows(mode, row_base, MHA_FEATURE_HEAD, tokens_dim, scale,
                     q_scratch, v_head, output,
                     q_val_buf, q_col_buf, q_ptr_buf,
                     k_val_buf, k_col_buf, k_ptr_buf);

        /* V is computed last of the three, so this head's attention runs on the V stage. */
        head_core(which == 2, head, tokens_dim, scales[SCALE_DIV_OUT_IDX],
                  k_val_buf, k_col_buf, k_ptr_buf,
                  q_val_buf, q_col_buf, q_ptr_buf,
                  v_head, attn_output);
    }
}
