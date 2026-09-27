/**********************************************************************************
 * Encoder layer as one sequence of row-block stages - Implementation
 *
 * Please see the respective header file for a description.
 *********************************************************************************/

#include "encoder_stages.h"
#include "mha.h"
#include "rmsnorm.h"
#include "residual.h"
#include "quantize.h"
#ifndef __SYNTHESIS__
#include <cassert>
#endif

// The tails below read one accumulated output row per engine step.
static_assert(MHA_ROWS_PER_STEP == 1, "the tails assume one output row per engine step");
// A W1 row block must fall inside one hidden tile, so a row's packed words go to one memory.
static_assert(MHA_H_TILE_FEATURES % MHA_FEATURE_HEAD == 0,
              "MHA_H_TILE_FEATURES must be a multiple of MHA_FEATURE_HEAD");
static_assert(MLP_HIDDEN_FEATURE_W_MAX == MHA_TILES * MHA_H_TILE_FEATURES,
              "hidden width != MHA_TILES * MHA_H_TILE_FEATURES");

/* What a stage does with its accumulated output rows. */
enum LayerRowsMode {
    ROWS_MHA_K = 0,   /* ReLU + sparsify to CSR, feature-major */
    ROWS_MHA_Q = 1,   /* ReLU to a scratch, then sparsify to CSR, token-major */
    ROWS_MHA_V = 2,   /* dense, this head's V */
    ROWS_MHA_O = 3,   /* dense, the attention half's output */
    ROWS_MLP_H = 4,   /* ReLU + requantize, packed into the tiles' hidden region */
    ROWS_MLP_Y = 5    /* dense, the layer output */
};

/* ---------------------------------------------------------------------------------------
 * Per-head and per-layer working state.
 *
 * Q', K' and V are computed and consumed one head at a time, so they hold MHA_FEATURE_HEAD x N,
 * sized for 100 % ReLU survival (overflow is impossible without relying on measured densities).
 * The attention result needs every feature, so it is kept for the whole layer.
 * ------------------------------------------------------------------------------------- */
static T_Activation   g_v_head[MHA_SCR_V_HEAD_CNT];
static T_Activation   g_q_scratch[MHA_SCR_V_HEAD_CNT];
static T_Activation   g_k_val[MHA_HEAD_NNZ_MAX];
static T_MhaHeadIndex g_k_col[MHA_HEAD_NNZ_MAX];
static int            g_k_ptr[MHA_FEATURE_HEAD + 1];
static T_Activation   g_q_val[MHA_HEAD_NNZ_MAX];
static T_MhaHeadIndex g_q_col[MHA_HEAD_NNZ_MAX];
static int            g_q_ptr[MHA_TOKEN_W_MAX + 1];

/* Packed attention output, all heads: the output projection reads every feature. */
static T_MhaBankWord g_attn_output[MHA_BANK_WORDS];

/* RMSNorm output, packed.  One buffer for both halves of the layer: they run in sequence. */
static T_MhaBankWord g_norm_input[MHA_BANK_WORDS_ALLOC];


/**
 * @brief Compute output rows [row_base, row_base + rows) of one weight matrix.
 *
 * Requires the banks to hold this row block's entries (load_weight_slice).  The engine accumulates
 * the MHA_TILES column blocks of an output row; the tail then requantizes that row into whatever
 * this stage produces.
 *
 * This is the layer's only call site for the engine.
 *
 * @param[in]  mode        Which tail to run (LayerRowsMode).
 * @param[in]  row_base    First output row of this block.
 * @param[in]  rows        Output rows in this block.
 * @param[in]  tokens_dim  Number of tokens.
 * @param[in]  scale       Folded requantization scale of this weight.
 * @param[out] hidden      Attention half's output (ROWS_MHA_O).
 * @param[out] output      Layer output (ROWS_MLP_Y).
 */
static void layer_rows (
    int           mode,
    int           row_base,
    int           rows,
    int           tokens_dim,
    T_Scale       scale,
    T_Activation *hidden,
    T_Activation *output
) {
#pragma HLS INLINE off
    const int row_words = (tokens_dim + MHA_BANK_PACK - 1) / MHA_BANK_PACK;
    /* The MLP's second linear reads H, everything else reads the layer input of its half. */
    const int act_base  = (mode == ROWS_MLP_Y) ? MHA_H_TILE_BASE : 0;
    /* Hidden feature f lives in tile f / MHA_H_TILE_FEATURES at local feature f % that. */
    const int h_tile    = row_base / MHA_H_TILE_FEATURES;
    const int h_local   = row_base % MHA_H_TILE_FEATURES;

    /* The sparse outputs are this head's CSR, so they are written from index 0 on every call. */
    int nnz = 0;
    if (mode == ROWS_MHA_K) {
        g_k_ptr[0] = 0;
    } else if (mode == ROWS_MHA_Q) {
        g_q_ptr[0] = 0;
    }

    STAGE_ROW:
    for (int i = 0; i < rows; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=MHA_FEATURE_HEAD max=MHA_FEATURE_HEAD

        tiled_rows_compute(i, 1, tokens_dim, act_base);

        /* One loop per mode, each with a branch-free body so that all of them hold II=1. */
        if (mode == ROWS_MHA_K) {
            TAIL_K:
            for (int j = 0; j < tokens_dim; j++) {
                #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_TOKEN max=MHA_TC_TOKEN
                #pragma HLS PIPELINE II=1
                const T_MhaAcc acc = g_proj_rows[0][j];
                if (acc > (T_MhaAcc) 0) {
                    const T_Activation quant_out = requant_to_int8(acc, scale);
                    if (quant_out > (T_Activation) 0) {
#ifndef __SYNTHESIS__
                        assert(nnz < MHA_HEAD_NNZ_MAX && "K' CSR overflow");
#endif
                        g_k_val[nnz] = quant_out;
                        g_k_col[nnz] = (T_MhaHeadIndex) j;
                        nnz++;
                    }
                }
            }
            g_k_ptr[i + 1] = nnz;
        } else if (mode == ROWS_MHA_Q) {
            TAIL_Q:
            for (int j = 0; j < tokens_dim; j++) {
                #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_TOKEN max=MHA_TC_TOKEN
                #pragma HLS PIPELINE II=1
                const T_MhaAcc acc = g_proj_rows[0][j];
                g_q_scratch[i * tokens_dim + j] =
                    (acc > (T_MhaAcc) 0) ? requant_to_int8(acc, scale) : (T_Activation) 0;
            }
        } else if (mode == ROWS_MHA_V) {
            TAIL_V:
            for (int j = 0; j < tokens_dim; j++) {
                #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_TOKEN max=MHA_TC_TOKEN
                #pragma HLS PIPELINE II=1
                g_v_head[i * tokens_dim + j] = requant_to_int8(g_proj_rows[0][j], scale);
            }
        } else if (mode == ROWS_MHA_O) {
            TAIL_O:
            for (int j = 0; j < tokens_dim; j++) {
                #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_TOKEN max=MHA_TC_TOKEN
                #pragma HLS PIPELINE II=1
                hidden[(row_base + i) * tokens_dim + j] = requant_to_int8(g_proj_rows[0][j], scale);
            }
        } else if (mode == ROWS_MLP_H) {
            /* ReLU and requantize into int8, packed MHA_BANK_PACK tokens per word, straight into
             * the hidden region of this block's tile: the second linear reads it there, so H is
             * never copied and never leaves the chip. */
            TAIL_H_WORD:
            for (int w = 0; w < row_words; w++) {
                #pragma HLS LOOP_TRIPCOUNT min=MHA_BANK_ROW_WORDS max=MHA_BANK_ROW_WORDS
                #pragma HLS PIPELINE II=1
                T_MhaBankWord word = 0;
                TAIL_H_LANE:
                for (int p = 0; p < MHA_BANK_PACK; p++) {
                    #pragma HLS UNROLL
                    const int    j = w * MHA_BANK_PACK + p;
                    T_Activation v = (T_Activation) 0;
                    if (j < tokens_dim) {
                        const T_MhaAcc acc = g_proj_rows[0][j];
                        if (acc > (T_MhaAcc) 0) {
                            const T_Activation q = requant_to_int8(acc, scale);
                            if (q > (T_Activation) 0) {
                                v = q;
                            }
                        }
                    }
                    word.range(p * MHA_ACT_ELEM_BITS + MHA_ACT_ELEM_BITS - 1,
                               p * MHA_ACT_ELEM_BITS) = (ap_uint<MHA_ACT_ELEM_BITS>) v;
                }
                g_act_tile[h_tile][MHA_H_TILE_BASE + (h_local + i) * MHA_BANK_ROW_WORDS + w] = word;
            }
        } else {
            TAIL_Y:
            for (int j = 0; j < tokens_dim; j++) {
                #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_TOKEN max=MHA_TC_TOKEN
                #pragma HLS PIPELINE II=1
                output[(row_base + i) * tokens_dim + j] = requant_to_int8(g_proj_rows[0][j], scale);
            }
        }
    }

    if (mode == ROWS_MHA_Q) {
        /* Q' is emitted token-major: CSR row j holds the nonzero features of token j, with the
         * head-local feature index (0..63) as the column.  Q' . A reads it by token, which is why
         * the dense rows go to a scratch first and are transposed here. */
        TAIL_Q_TOKEN:
        for (int j = 0; j < tokens_dim; j++) {
            #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_TOKEN max=MHA_TC_TOKEN

            TAIL_Q_FEATURE:
            for (int bf = 0; bf < MHA_FEATURE_HEAD; bf++) {
                #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_FEATURE_HEAD max=MHA_TC_FEATURE_HEAD
                #pragma HLS PIPELINE II=1
                const T_Activation out_data = g_q_scratch[bf * tokens_dim + j];
                if (out_data > (T_Activation) 0) {
#ifndef __SYNTHESIS__
                    assert(nnz < MHA_HEAD_NNZ_MAX && "Q' CSR overflow");
#endif
                    g_q_val[nnz] = out_data;
                    g_q_col[nnz] = (T_MhaHeadIndex) bf;
                    nnz++;
                }
            }
            g_q_ptr[j + 1] = nnz;
        }
    }
}


void encoder_layer_stages(
    const T_Activation *input,
    int                 tokens_dim,
    int                 feature_dim,
    int                 hidden_dim,
    T_MhaTileWord *wq_bank, int *wq_counts,
    T_MhaTileWord *wk_bank, int *wk_counts,
    T_MhaTileWord *wv_bank, int *wv_counts,
    T_MhaTileWord *wo_bank, int *wo_counts,
    T_MhaTileWord *w1_bank, int *w1_counts,
    T_MhaTileWord *w2_bank, int *w2_counts,
    const T_Scale      *scales,
    T_Activation       *hidden,
    T_Activation       *output
) {
#pragma HLS INLINE off
#ifndef __SYNTHESIS__
    assert(tokens_dim <= MHA_TOKEN_W_MAX && "tokens_dim exceeds MHA_TOKEN_W_MAX");
    assert(hidden_dim <= MLP_HIDDEN_FEATURE_W_MAX && "hidden_dim exceeds MLP_HIDDEN_FEATURE_W_MAX");
#endif
    #pragma HLS BIND_STORAGE variable=g_v_head     type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=g_q_scratch  type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=g_k_val      type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=g_k_col      type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=g_q_val      type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=g_q_col      type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE   variable=g_attn_output type=ram_t2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=g_attn_output cyclic factor=MHA_BANK_CYCLIC dim=1
    #pragma HLS BIND_STORAGE   variable=g_norm_input type=ram_t2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=g_norm_input cyclic factor=MHA_BANK_CYCLIC dim=1

    /* Stage layout: 3 per head for K/Q/V, then the output projection's row blocks, then the two
     * MLP linears' row blocks.  All of it in one loop, so layer_rows and the weight loaders have
     * one call site each. */
    const int qkv_stages = 3 * MHA_N_HEADS;
    const int o_stages   = feature_dim / MHA_FEATURE_HEAD;
    const int h_stages   = hidden_dim  / MHA_FEATURE_HEAD;
    const int y_stages   = feature_dim / MHA_FEATURE_HEAD;
    const int o_first    = qkv_stages;
    const int h_first    = o_first + o_stages;
    const int y_first    = h_first + h_stages;

    #pragma HLS ALLOCATION instances=layer_rows         limit=1 function
    #pragma HLS ALLOCATION instances=load_weight_slice  limit=1 function
    #pragma HLS ALLOCATION instances=load_weight_bases  limit=1 function
    #pragma HLS ALLOCATION instances=load_activation_tiles limit=1 function
    #pragma HLS ALLOCATION instances=rmsnorm            limit=1 function
    #pragma HLS ALLOCATION instances=residual_add       limit=1 function

    rmsnorm(input, tokens_dim, feature_dim, scales[SCALE_ATT_RMSNORM_OUT_INV_IDX],
            reinterpret_cast<T_RmsBankWord*>(g_norm_input));
    load_activation_tiles(g_norm_input, 0, MHA_TILE_WORDS);

    LAYER_STAGE:
    for (int stage = 0; stage < y_first + y_stages; stage++) {
        #pragma HLS LOOP_TRIPCOUNT min=108 max=108
        #pragma HLS LOOP_FLATTEN off

        /* Which weight this stage computes, and which of its row blocks. */
        const int mode = (stage <  o_first) ? ((stage % 3 == 0) ? ROWS_MHA_K
                                            : (stage % 3 == 1) ? ROWS_MHA_Q : ROWS_MHA_V)
                       : (stage <  h_first) ? ROWS_MHA_O
                       : (stage <  y_first) ? ROWS_MLP_H
                                            : ROWS_MLP_Y;
        const int head  = (stage < o_first) ? (stage / 3) : 0;
        const int block = (stage <  o_first) ? head
                        : (stage <  h_first) ? (stage - o_first)
                        : (stage <  y_first) ? (stage - h_first)
                                             : (stage - y_first);
        const int row_base = block * MHA_FEATURE_HEAD;
        const int first    = (block == 0);

        /* The output projection reads the attention result, not the layer input. */
        if (mode == ROWS_MHA_O && first) {
            load_activation_tiles(g_attn_output, 0, MHA_TILE_WORDS);
        }
        /* Between the halves: close the attention residual, then normalize for the MLP. */
        if (mode == ROWS_MLP_H && first) {
            residual_add(input, hidden, feature_dim * tokens_dim,
                         scales[SCALE_ATT_RESIDUAL_IDX], scales[SCALE_ATT_BRANCH_RATIO_IDX],
                         hidden);
            rmsnorm(hidden, tokens_dim, feature_dim, scales[SCALE_FF_RMSNORM_OUT_INV_IDX],
                    reinterpret_cast<T_RmsBankWord*>(g_norm_input));
            load_activation_tiles(g_norm_input, 0, MHA_TILE_WORDS);
        }

        /* This stage's weight; one branch each, all in this scope, so HLS multiplexes the AXI
         * ports itself rather than needing a pointer selection in source. */
        if (mode == ROWS_MHA_K) {
            if (first) { load_weight_bases(wk_counts, SPMM_W_K); }
            load_weight_slice(wk_bank, wk_counts, feature_dim, SPMM_W_K, row_base, MHA_FEATURE_HEAD);
        } else if (mode == ROWS_MHA_Q) {
            if (first) { load_weight_bases(wq_counts, SPMM_W_Q); }
            load_weight_slice(wq_bank, wq_counts, feature_dim, SPMM_W_Q, row_base, MHA_FEATURE_HEAD);
        } else if (mode == ROWS_MHA_V) {
            if (first) { load_weight_bases(wv_counts, SPMM_W_V); }
            load_weight_slice(wv_bank, wv_counts, feature_dim, SPMM_W_V, row_base, MHA_FEATURE_HEAD);
        } else if (mode == ROWS_MHA_O) {
            if (first) { load_weight_bases(wo_counts, SPMM_W_O); }
            load_weight_slice(wo_bank, wo_counts, feature_dim, SPMM_W_O, row_base, MHA_FEATURE_HEAD);
        } else if (mode == ROWS_MLP_H) {
            if (first) { load_weight_bases(w1_counts, SPMM_W_MLP1); }
            load_weight_slice(w1_bank, w1_counts, hidden_dim, SPMM_W_MLP1, row_base, MHA_FEATURE_HEAD);
        } else {
            if (first) { load_weight_bases(w2_counts, SPMM_W_MLP2); }
            load_weight_slice(w2_bank, w2_counts, feature_dim, SPMM_W_MLP2, row_base, MHA_FEATURE_HEAD);
        }

        const T_Scale scale = (mode == ROWS_MHA_K) ? scales[SCALE_K_IDX]
                            : (mode == ROWS_MHA_Q) ? scales[SCALE_Q_IDX]
                            : (mode == ROWS_MHA_V) ? scales[SCALE_V_IDX]
                            : (mode == ROWS_MHA_O) ? scales[SCALE_LINEAR_OUT_IDX]
                            : (mode == ROWS_MLP_H) ? scales[SCALE_FC1_IDX]
                                                   : scales[SCALE_FC2_IDX];

        layer_rows(mode, row_base, MHA_FEATURE_HEAD, tokens_dim, scale, hidden, output);

        /* V is the last of a head's three projections, so the head's attention runs here.  Called
         * unconditionally with an enable: a call inside an `if` makes HLS specialise the loop body
         * for that case, which duplicates layer_rows. */
        head_core(mode == ROWS_MHA_V, head, tokens_dim, scales[SCALE_DIV_OUT_IDX],
                  g_k_val, g_k_col, g_k_ptr,
                  g_q_val, g_q_col, g_q_ptr,
                  g_v_head, g_attn_output);
    }

    residual_add(hidden, output, feature_dim * tokens_dim,
                 scales[SCALE_FF_RESIDUAL_IDX], scales[SCALE_FF_BRANCH_RATIO_IDX],
                 output);
}
