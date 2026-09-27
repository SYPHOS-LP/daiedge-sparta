/**********************************************************************************
 * Tiled block-sparse matrix multiply - shared engine
 *
 * One sparse operand (a weight matrix) times one dense operand (an activation matrix), used by
 * every projection of the encoder layer.
 *
 * The feature axis of the weight is split into MHA_TILES column blocks.  Block t owns one
 * activation tile (the activation rows its columns read) and one weight bank (the nonzeros whose
 * column falls in that block, rows ascending), each in its own on-chip memory, so the MHA_TILES
 * lanes each read one nonzero and one activation word per cycle without copying the activation.
 * The output rows are computed in blocks: a caller loads one (row block, column block) slice per
 * block and then walks the rows.
 *
 * compile_model.py emits the weights to match (see pack_tiled_bank there): one array of packed
 * entries in column block order, each entry a value and its block-local column, each block padded
 * to a whole MHA_TILE_PACK word, plus the nonzeros per (column block, row).
 *
 * All callers must go through ONE call site per function: HLS instantiates a function per call
 * site, and a second engine costs ~26k LUT.
 *********************************************************************************/

#ifndef SPMM_TILED_H
#define SPMM_TILED_H

#include "mha_cfg.h"

/* Weight matrices whose DDR position is tracked independently, because the stages of the layer
 * interleave: each slot keeps its own per-column-block cursor into its DDR array. */
enum SpmmWeightSlot {
    SPMM_W_K = 0, SPMM_W_Q = 1, SPMM_W_V = 2, SPMM_W_O = 3,   /* MHA projections */
    SPMM_W_MLP1 = 4, SPMM_W_MLP2 = 5,                          /* MLP linears */
    SPMM_N_WEIGHTS = 6
};

/* The accumulated output rows of the last tiled_rows_compute() call. */
extern T_MhaAcc g_proj_rows[MHA_ROWS_PER_STEP][MHA_TOKEN_W_MAX];

/* The activation tiles, one memory per column block, in two regions per tile: words
 * [0, MHA_TILE_WORDS) for a 64-feature block, [MHA_H_TILE_BASE, MHA_TILE_WORDS_MAX) for a
 * 256-feature hidden block.  Exposed because the MLP's first stage writes its output straight
 * into the hidden region, which is what lets its second stage read it without a copy. */
extern T_MhaBankWord g_act_tile[MHA_TILES][MHA_TILE_WORDS_MAX];

void load_weight_bases (int* w_counts, int which);

void load_weight_slice (T_MhaTileWord* w_bank, int* w_counts,
                        int feature_dim, int which, int row_base, int rows);

void load_activation_tiles (T_MhaBankWord* packed_input, int base, int words);

void tiled_rows_compute (int row_local, int rows_in_step, int tokens_dim, int act_base);

#endif // SPMM_TILED_H
