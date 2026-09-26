/**********************************************************************************
 * Sparse MLP Layer - Implementation
 *
 * Please see the respective header file for a high-level description of the layer.
 *********************************************************************************/

#include "mlp.h"
#include "spmm_tiled.h"
#include "quantize.h"
#ifndef __SYNTHESIS__
#include <cassert>
#endif

// Both stages run one output row at a time on the shared engine.
static_assert(MHA_ROWS_PER_STEP == 1, "the MLP tails assume one output row per engine step");
// A W1 row block must fall inside one hidden tile, so that the row's packed words go to one memory.
static_assert(MHA_H_TILE_FEATURES % MHA_FEATURE_HEAD == 0,
              "MHA_H_TILE_FEATURES must be a multiple of MHA_FEATURE_HEAD");
static_assert(MLP_HIDDEN_FEATURE_W_MAX == MHA_TILES * MHA_H_TILE_FEATURES,
              "hidden width != MHA_TILES * MHA_H_TILE_FEATURES");

/**
 * @brief Compute output rows [row_base, row_base + MHA_FEATURE_HEAD) of one of the two linears.
 *
 * Requires the banks to hold this row block's entries (load_weight_slice).  The engine accumulates
 * the MHA_TILES column blocks of one output row; this then either requantizes the row with ReLU and
 * writes it packed into the hidden region of the activation tiles (W1), or requantizes it into the
 * layer output (W2).
 *
 * Both stages call it from here, so the engine has one call site in the MLP.
 *
 * @param[in]  to_hidden   Non-zero for the first linear (write H), zero for the second (write Y).
 * @param[in]  row_base    First output row of this block.
 * @param[in]  tokens_dim  Number of tokens.
 * @param[in]  scale       Folded requantization scale of this linear.
 * @param[out] output      Layer output in DDR (second linear only).
 */
static void mlp_rows (
    int           to_hidden,
    int           row_base,
    int           tokens_dim,
    T_Scale       scale,
    T_Activation *output
) {
#pragma HLS INLINE off
    const int row_words = (tokens_dim + MHA_BANK_PACK - 1) / MHA_BANK_PACK;
    /* W1 reads the layer input in the first region of each tile, W2 reads H in the second. */
    const int act_base = to_hidden ? 0 : MHA_H_TILE_BASE;
    /* Hidden feature f lives in tile f / MHA_H_TILE_FEATURES at local feature f % that. */
    const int h_tile  = row_base / MHA_H_TILE_FEATURES;
    const int h_local = row_base % MHA_H_TILE_FEATURES;

    MLP_ROW:
    for (int i = 0; i < MHA_FEATURE_HEAD; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=MHA_FEATURE_HEAD max=MHA_FEATURE_HEAD

        tiled_rows_compute(i, 1, tokens_dim, act_base);

        if (to_hidden) {
            /* ReLU and requantize into int8, packed MHA_BANK_PACK tokens per word, straight into
             * the hidden region of this block's tile: the second stage reads it there, so H is
             * never copied and never leaves the chip. */
            H_PACK_WORD:
            for (int w = 0; w < row_words; w++) {
                #pragma HLS LOOP_TRIPCOUNT min=MHA_BANK_ROW_WORDS max=MHA_BANK_ROW_WORDS
                #pragma HLS PIPELINE II=1
                T_MhaBankWord word = 0;
                H_PACK_LANE:
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
            Y_TOKEN:
            for (int j = 0; j < tokens_dim; j++) {
                #pragma HLS LOOP_TRIPCOUNT min=MLP_TC_BATCH max=MLP_TC_BATCH
                #pragma HLS PIPELINE II=1
                output[(row_base + i) * tokens_dim + j] = requant_to_int8(g_proj_rows[0][j], scale);
            }
        }
    }
}


void mlp(
    T_MlpBankWord *input_packed,
    int            feature_dim,
    int            hidden_dim,
    int            batch,
    T_MhaTileWord *w1_bank,
    int           *w1_counts,
    T_MhaTileWord *w2_bank,
    int           *w2_counts,
    const T_Scale *scales,
    T_Activation  *output
) {
#pragma HLS INLINE off
#ifndef __SYNTHESIS__
    assert(hidden_dim <= MLP_HIDDEN_FEATURE_W_MAX && "hidden_dim exceeds MLP_HIDDEN_FEATURE_W_MAX");
    assert(batch <= MLP_TOKEN_W_MAX && "batch exceeds MLP_TOKEN_W_MAX");
#endif

    /* The layer input, split per column block, read by every row block of the first linear. */
    load_activation_tiles(input_packed, 0, MHA_TILE_WORDS);

    const int w1_blocks = hidden_dim / MHA_FEATURE_HEAD;    /* 48 blocks of 64 hidden rows */
    const int w2_blocks = feature_dim / MHA_FEATURE_HEAD;   /* 12 blocks of 64 output rows */

    /* One loop over both linears, so mlp_rows and the weight loaders have one call site each and
     * HLS instantiates one of each.  H is complete once the first linear's blocks are done, which
     * is why the second linear's blocks follow rather than interleave. */
    #pragma HLS ALLOCATION instances=mlp_rows           limit=1 function
    #pragma HLS ALLOCATION instances=load_weight_slice  limit=1 function
    #pragma HLS ALLOCATION instances=load_weight_bases  limit=1 function

    MLP_STAGE:
    for (int stage = 0; stage < w1_blocks + w2_blocks; stage++) {
        #pragma HLS LOOP_TRIPCOUNT min=(MLP_HIDDEN_FEATURE_W_MAX/MHA_FEATURE_HEAD + MLP_FEATURE_W_MAX/MHA_FEATURE_HEAD) max=(MLP_HIDDEN_FEATURE_W_MAX/MHA_FEATURE_HEAD + MLP_FEATURE_W_MAX/MHA_FEATURE_HEAD)
        #pragma HLS LOOP_FLATTEN off

        const int to_hidden = (stage < w1_blocks);
        const int block     = to_hidden ? stage : (stage - w1_blocks);
        const int row_base  = block * MHA_FEATURE_HEAD;
        const int first     = (block == 0);

        if (to_hidden) {
            if (first) { load_weight_bases(w1_counts, hidden_dim, SPMM_W_MLP1); }
            load_weight_slice(w1_bank, w1_counts, hidden_dim, SPMM_W_MLP1, row_base, MHA_FEATURE_HEAD);
        } else {
            if (first) { load_weight_bases(w2_counts, feature_dim, SPMM_W_MLP2); }
            load_weight_slice(w2_bank, w2_counts, feature_dim, SPMM_W_MLP2, row_base, MHA_FEATURE_HEAD);
        }

        mlp_rows(to_hidden, row_base, batch,
                 to_hidden ? scales[SCALE_FC1_IDX] : scales[SCALE_FC2_IDX],
                 output);
    }
}
