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

/* Packed weight entry layout, which follows the datapath the kernel is built with.  An entry is
 * MHA_TILE_ENTRY_BITS wide: the value in the low bits, then the 8-bit block-local column.
 *
 *   POT_SHIFT_MAC: the value is a power of two, {sign(bit3), exponent(bits0..2)} -> 4 bits, and
 *                  the MAC shifts the activation by the exponent.
 *   otherwise:     the value is the plain integer weight -> 8 bits (these weights reach +/-64, so
 *                  int4 does not hold them), and the MAC multiplies on a DSP.
 *
 * compile_model.py --weights writes the matching layout and records it in model_index.bin, which
 * the testbench checks against this build.
 */
#ifdef POT_SHIFT_MAC
#define MHA_ENTRY_VAL_BITS 4
#define MHA_ENTRY_FORMAT   0
#else
#define MHA_ENTRY_VAL_BITS 8
#define MHA_ENTRY_FORMAT   1
#endif
#define MHA_ENTRY_COL_LSB  MHA_ENTRY_VAL_BITS

/* Weight matrices whose DDR position is tracked independently, because the stages of the layer
 * interleave: each slot keeps its own per-column-block cursor into its DDR array. */
enum SpmmWeightSlot {
    SPMM_W_K = 0, SPMM_W_Q = 1, SPMM_W_V = 2, SPMM_W_O = 3,   /* MHA projections */
    SPMM_W_MLP1 = 4, SPMM_W_MLP2 = 5,                          /* MLP linears */
    SPMM_N_WEIGHTS = 6
};

/* Per-lane partial sums of the output row the last tiled_rows_compute() call produced.  They are
 * the engine's accumulators, so they are kept unmerged: the consumer already sweeps the row once to
 * requantize and pack it, and summing the lanes in that same sweep costs nothing, where a merge
 * pass of its own cost a second sweep, 12 cycles per output row.
 *
 * Read them only through tiled_lane_sum(), which enforces the other half of that arrangement. */
extern T_MhaAcc g_lane_partial[MHA_TILES][MHA_TOKEN_W_MAX];

/**
 * @brief Take one token's accumulator out of the engine: sum the lanes and clear them.
 *
 * The engine accumulates into g_lane_partial across a row and never clears it, so every token the
 * engine may have written has to be drained before the next call to tiled_rows_compute() or the
 * rows silently add together.  Draining is therefore the consumer's obligation, and doing it here
 * keeps that rule next to the array instead of in a comment in the consumer's file.
 *
 * Inline on purpose: the consumer calls this inside its own pipelined loop, and the adder tree has
 * to fold into that loop's datapath rather than become a function of its own.
 *
 * The caller must cover every token < tokens_dim, since that is exactly the range the engine's MAC
 * writes.  Tokens past it are never written and must not be drained.
 */
static inline T_MhaAcc tiled_lane_sum (int token) {
#pragma HLS INLINE
    T_MhaAcc sum = (T_MhaAcc) 0;
    for (int t = 0; t < MHA_TILES; t++) {
        #pragma HLS UNROLL
        sum += g_lane_partial[t][token];
        g_lane_partial[t][token] = (T_MhaAcc) 0;
    }
    return sum;
}

/* The activation tiles, one memory per column block, in two regions per tile: words
 * [0, MHA_TILE_WORDS) for a 64-feature block, [MHA_H_TILE_BASE, MHA_TILE_WORDS_MAX) for a
 * 256-feature hidden block.  Exposed because the MLP's first stage writes its output straight
 * into the hidden region, which is what lets its second stage read it without a copy. */
extern T_MhaBankWord g_act_tile[MHA_TILES][MHA_TILE_WORDS_MAX];

void load_weight_bases (T_MhaCountWord* w_counts, int which);

void load_weight_slice (T_MhaTileWord* w_bank, T_MhaCountWord* w_counts,
                        int feature_dim, int which, int row_base, int rows);

void load_activation_tiles (T_MhaBankWord* packed_input, int base, int words);

void tiled_rows_compute (int row_local, int tokens_dim, int act_base);

#endif // SPMM_TILED_H
