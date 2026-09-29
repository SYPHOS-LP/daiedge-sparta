/**********************************************************************************
 * Tiled block-sparse matrix multiply - shared engine implementation
 *
 * Please see the respective header file for a description.
 *
 *********************************************************************************/

#include "spmm_tiled.h"
#include "quantize.h"
#ifndef __SYNTHESIS__
#include <cassert>
#endif

// The MAC loop does one multiply-accumulate per token packed in an activation word.
static_assert(MHA_BANK_PACK == 8, "the MAC loop writes out MHA_BANK_PACK token operations by hand");
// A packed entry holds the value field and an 8-bit block-local column.
static_assert(MHA_ENTRY_COL_LSB + 8 <= MHA_TILE_ENTRY_BITS,
              "value + column do not fit MHA_TILE_ENTRY_BITS");
static_assert((int)T_MhaWVal::width >= MHA_ENTRY_VAL_BITS,
              "T_MhaWVal must hold the entry's value field");

/* ---------------------------------------------------------------------------------------
 * Tiled SpMM state.
 *
 * The feature axis is cut into MHA_TILES column blocks.  Block t owns ONE activation tile
 * (its own memory, MHA_TILE_FEATURES rows of packed tokens) and ONE weight bank (its own
 * memory, holding every nonzero whose column falls in that block, rows ascending).  The
 * MHA_TILES lanes therefore each read one nonzero and one activation word per cycle without
 * any copy of the activation: the four replicas this replaces cost 4x the whole D x N bank.
 *
 * compile_model.py --mha-tiles emits the weights to match: one array of packed entries in column
 * block order, each entry a value and its block-local column, each block padded to a whole
 * MHA_TILE_PACK word, plus the per-(column block, row) nonzero counts.
 * ------------------------------------------------------------------------------------ */
T_MhaBankWord  g_act_tile[MHA_TILES][MHA_TILE_WORDS_MAX];
static T_MhaTileWord  g_bank[MHA_TILES][MHA_TILE_BANK_WORDS];
static T_MhaTileCount g_bank_count[MHA_TILES][MHA_FEATURE_HEAD];

/* Engine state kept at file scope on purpose: with no caller-specific pointer arguments every
 * call to tiled_rows_compute() has identical bindings, which is what lets HLS build ONE engine
 * instead of specialising one per call site. */
static int      g_bank_cursor[MHA_TILES];

/* DDR position of the next row block, per weight matrix slot (SpmmWeightSlot) and column block.
 * g_block_word0 is the column block's first packed word in DDR, g_slice_entry the entry index
 * within the block where the next row block starts, and g_slice_cursor0 that entry's slot within
 * its packed word.  Reads start at the word boundary below the first entry (up to
 * MHA_TILE_PACK-1 unused entries precede the slice in the bank) so that no data realignment is
 * needed.  The weight matrices are interleaved across stages, so each keeps its own position. */
static int g_block_word0[SPMM_N_WEIGHTS][MHA_TILES];
static int g_slice_entry[SPMM_N_WEIGHTS][MHA_TILES];
static int g_slice_cursor0[MHA_TILES];

/* The engine's accumulators; declared in spmm_tiled.h, which also carries the rule for draining
 * them (tiled_lane_sum). */
T_MhaAcc g_lane_partial[MHA_TILES][MHA_TOKEN_W_MAX];

/* Wide DDR views used to fill one packed bank word per read. */


/**
 * @brief Unpack one weight entry from a bank word: {block-local column, POT value}.
 */
static inline void bank_entry(T_MhaTileWord word, int slot, T_MhaWVal &val, T_MhaTileCol &col) {
#pragma HLS INLINE
    const int lo = slot * MHA_TILE_ENTRY_BITS;
    val = (T_MhaWVal)   word.range(lo + MHA_ENTRY_VAL_BITS - 1, lo);
    col = (T_MhaTileCol) word.range(lo + MHA_ENTRY_COL_LSB + 7, lo + MHA_ENTRY_COL_LSB);
}

/**
 * @brief Read the first DDR word of every column block of one weight matrix.
 *
 * Called once per weight matrix.  compile_model.py emits the MHA_TILES offsets in front of the
 * counts, so this is MHA_TILES reads.  Recomputing them here meant first reading every count of
 * the weight, which cost 83k cycles per layer across the six matrices.
 *
 * @param[in] w_counts  The counts blob: MHA_TILES int32 block offsets, padded to a whole
 *                  transfer, then one byte per (column block, row) count.
 * @param[in] which     Weight matrix slot (SpmmWeightSlot).
 */
void load_weight_bases (T_MhaCountWord* w_counts, int which) {
#pragma HLS INLINE off
    /* The offsets are int32 in the first MHA_COUNT_HDR_WORDS transfers of the blob. */
    BASE_WORD:
    for (int w = 0; w < MHA_COUNT_HDR_WORDS; w++) {
        #pragma HLS LOOP_TRIPCOUNT min=MHA_COUNT_HDR_WORDS max=MHA_COUNT_HDR_WORDS
        #pragma HLS PIPELINE II=1
        const T_MhaCountWord word = w_counts[w];
        BASE_LANE:
        for (int p = 0; p < MHA_COUNT_PACK / 4; p++) {
            #pragma HLS UNROLL
            const int t = w * (MHA_COUNT_PACK / 4) + p;
            if (t < MHA_TILES) {
                g_block_word0[which][t] = (int) word.range(p * 32 + 31, p * 32);
                g_slice_entry[which][t] = 0;
            }
        }
    }
}

/**
 * @brief Read the entries of output rows [row_base, row_base + rows) into the per-column-block banks.
 *
 * Within a column block the entries are ordered by row, so the rows of one row block are
 * contiguous and one burst per column block is sufficient: 977 entries worst case against 9704
 * for a whole column block.  One read per word: the entries arrive already packed as the banks
 * hold them, so no unpacking, re-encoding or merging is needed here.
 *
 * @param[in] w_bank       Packed weight entries in DDR, {column, value} per slot, block order.
 * @param[in] w_counts     The counts blob (see load_weight_bases).
 * @param[in] feature_dim  Output rows of the weight matrix (row stride of w_counts).
 * @param[in] which        Weight matrix slot (SpmmWeightSlot).
 * @param[in] row_base     First output row of this stage.
 * @param[in] rows         Output rows in this stage.
 */
void load_weight_slice (T_MhaTileWord* w_bank, T_MhaCountWord* w_counts,
                        int feature_dim, int which, int row_base, int rows) {
#pragma HLS INLINE off
    /* A bank is 64-bit words, MHA_TILE_BANK_WORDS deep: one URAM each (72 b x 4096), where BRAM
     * would need ~12 BRAM18 per bank for the same data (18 b per port, 1024 deep). */
    #pragma HLS BIND_STORAGE variable=g_bank type=ram_t2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=g_bank_count complete dim=1
    int slice_words[MHA_TILES], slice_word0[MHA_TILES];
    #pragma HLS ARRAY_PARTITION variable=slice_words complete dim=1
    #pragma HLS ARRAY_PARTITION variable=slice_word0 complete dim=1

    /* Nonzero counts of this row block, which the MAC loop uses to index the banks, and their sum
     * per column block, which is the length of the DDR read.  One column block at a time: the
     * counts of one block are contiguous in DDR, and reading all MHA_TILES of a row in the same
     * cycle would need 12 concurrent reads on one AXI port. */
    int slice_len[MHA_TILES];
    #pragma HLS ARRAY_PARTITION variable=slice_len complete dim=1
    /* Counts arrive MHA_COUNT_PACK per transfer.  A block's counts start on a transfer boundary
     * (the header is a whole number of words, and feature_dim and row_base are multiples of
     * MHA_COUNT_PACK), so a row block is exactly rows / MHA_COUNT_PACK transfers. */
    /* One flattened walk over (column block, count word) rather than an inner loop per block.  As
     * two loops the outer one is not pipelined, so every block paid the DDR read latency again --
     * ~74 cycles of it, 12 times, for 96 reads of actual work.  Flattened, the reads issue back to
     * back on the one port and the latencies overlap. */
    const int count_words = rows / MHA_COUNT_PACK;
    const int count_steps = MHA_TILES * count_words;

    SLICE_LEN_INIT:
    for (int t = 0; t < MHA_TILES; t++) {
        #pragma HLS UNROLL
        slice_len[t] = 0;
    }

    int ct = 0, cw = 0;
    SLICE_COUNT:
    for (int i = 0; i < count_steps; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=(MHA_TILES*MHA_FEATURE_HEAD/MHA_COUNT_PACK) max=(MHA_TILES*MHA_FEATURE_HEAD/MHA_COUNT_PACK)
        #pragma HLS PIPELINE II=1
        const int base = MHA_COUNT_HDR_WORDS + (ct * feature_dim + row_base) / MHA_COUNT_PACK;
        const T_MhaCountWord word = w_counts[base + cw];
        int len = 0;
        SLICE_COUNT_LANE:
        for (int p = 0; p < MHA_COUNT_PACK; p++) {
            #pragma HLS UNROLL
            const T_MhaTileCount n = (T_MhaTileCount) word.range(p * 8 + 7, p * 8);
            g_bank_count[ct][cw * MHA_COUNT_PACK + p] = n;
            len += (int) n;
        }
        slice_len[ct] += len;
        if (++cw == count_words) { cw = 0; ct++; }
    }

    SLICE_PLAN:
    for (int t = 0; t < MHA_TILES; t++) {
        #pragma HLS UNROLL
        const int start = g_slice_entry[which][t];
        const int lane  = start % MHA_TILE_PACK;   /* lane of the first entry in its word */
#ifndef __SYNTHESIS__
        assert(lane + slice_len[t] <= MHA_TILE_NNZ_MAX &&
               "weight row block exceeds MHA_TILE_NNZ_MAX (raise the cap)");
#endif
        g_slice_cursor0[t]      = lane;
        g_slice_entry[which][t] = start + slice_len[t];
        slice_word0[t]          = g_block_word0[which][t] + start / MHA_TILE_PACK;
        slice_words[t]     = (lane + slice_len[t] + MHA_TILE_PACK - 1) / MHA_TILE_PACK;
    }

    /* One read per bank word: compile_model.py emits the entries already packed. */
    BANK_TILE:
    for (int t = 0; t < MHA_TILES; t++) {
        #pragma HLS LOOP_TRIPCOUNT min=MHA_TILES max=MHA_TILES
        const int words = slice_words[t];
        BANK_WORD:
        for (int w = 0; w < words; w++) {
            #pragma HLS LOOP_TRIPCOUNT min=64 avg=104 max=MHA_TILE_BANK_WORDS
            #pragma HLS PIPELINE II=1
            g_bank[t][w] = w_bank[slice_word0[t] + w];
        }
    }
}

/**
 * @brief Split the packed activation into the per-block tiles the lanes read.
 *
 * Feature f lives in block f / MHA_TILE_FEATURES, and the bank is feature-major, so block t is
 * simply the contiguous word range [t * words, (t+1) * words) of the input.  One copy is made, not
 * one per lane.  `base` selects the region inside the tile: 0 for a 64-feature column block of the
 * MHA or W1, MHA_H_TILE_BASE for a 256-feature hidden block of the MLP's W2.
 *
 * @param[in] packed_input  Packed activation bank (feature-major, MHA_BANK_PACK tokens/word).
 */
void load_activation_tiles (T_MhaBankWord* packed_input, int base, int words) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=g_act_tile complete dim=1
    /* A tile is MHA_TILE_WORDS_MAX x 64 b = one URAM, which also covers the 1792-word hidden
     * region the MLP's second stage reads. */
    #pragma HLS BIND_STORAGE variable=g_act_tile type=ram_t2p impl=uram

    ACTIVATION_TILE:
    for (int t = 0; t < MHA_TILES; t++) {
        #pragma HLS LOOP_TRIPCOUNT min=MHA_TILES max=MHA_TILES
        ACTIVATION_TILE_WORD:
        for (int w = 0; w < words; w++) {
            #pragma HLS LOOP_TRIPCOUNT min=MHA_TILE_WORDS max=MHA_H_TILE_WORDS
            #pragma HLS PIPELINE II=1
            g_act_tile[t][base + w] = packed_input[t * words + w];
        }
    }
}

/**
 * @brief Unpack one int8 activation from a packed word, multiply with given weight
 * and accumulate it into an output row.
 *
 * @param[in]     input       Packed input
 * @param[in]     weight      Weight used for multiplication
 * @param[in]     token_idx   Token index this slot maps to
 * @param[in]     slot        Which of the MHA_BANK_PACK activations in the word to use
 * @param[in]     tokens_dim  Number of tokens, used to skip padding
 * @param[in,out] out_row     Output row
 */
static inline void mac_pe (
    T_MhaBankWord input,
    T_MhaWVal     weight,
    int           token_idx,
    int           slot,
    int           tokens_dim,
    T_MhaAcc      out_row[MHA_TOKEN_W_MAX]
) {
#pragma HLS INLINE
    if (token_idx < tokens_dim) {
        /* Unpack */
        T_Activation activation = (T_Activation) input.range(slot * 8 + 7, slot * 8);
#ifdef POT_SHIFT_MAC
        /* The weight is a power of two, {sign(bit3), exponent(bits0..2)}, so the product is the
         * activation shifted left by the exponent: a barrel shifter instead of a multiplier, and
         * a 16-bit addend instead of a 32-bit one.  int8 shifted by up to 7 spans
         * -16384..16256, which fits 16 bits. */
        ap_uint<3> e   = weight.range(2, 0);
        bool       neg = weight[3];
        ap_int<16> mag = ((ap_int<16>) activation) << e;
        out_row[token_idx] += neg ? (T_MhaAcc)(-mag) : (T_MhaAcc) mag;
#else
        /* Plain integer weight: a real multiply, on a DSP.  These weights reach +/-64, so the
         * value field is 8 bits wide (see MHA_ENTRY_VAL_BITS). */
        ap_int<8> w    = (ap_int<8>) (ap_uint<8>) weight;
        T_MhaAcc  prod = (T_MhaAcc) activation * (T_MhaAcc) w;
        #pragma HLS BIND_OP variable=prod op=mul impl=dsp
        out_row[token_idx] += prod;
#endif
    }
}

/**
 * @brief Compute one output row of the tiled SpMM across MHA_TILES lanes.
 *
 * Lane t reads only block t's weight bank and only block t's activation tile, so the lanes never
 * contend for a memory and no copy of the activation is needed.  Each lane accumulates into its own
 * partial row.  The lanes advance together, so the row costs the busiest lane; removing that cost
 * is plan B in docs/plans/05-lane-balance.md.
 *
 * Reads g_bank / g_act_tile / g_bank_count, advances g_bank_cursor, and leaves the row's per-lane
 * partial sums in g_lane_partial for the caller to drain through tiled_lane_sum().
 *
 * @param[in] row_local   Output row within the weight slice the banks currently hold (0..rows-1).
 * @param[in] tokens_dim  Number of tokens.
 * @param[in] act_base    Which activation region inside each tile (see load_activation_tiles).
 */
void tiled_rows_compute (
    int row_local,
    int tokens_dim,
    int act_base
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=g_act_tile   complete dim=1
#pragma HLS ARRAY_PARTITION variable=g_bank       complete dim=1
#pragma HLS ARRAY_PARTITION variable=g_bank_count complete dim=1
#pragma HLS ARRAY_PARTITION variable=g_bank_cursor complete dim=1
    /* Lane t only ever touches g_lane_partial[t], so the lanes' updates are independent. */
#pragma HLS ARRAY_PARTITION variable=g_lane_partial complete dim=1
#pragma HLS ARRAY_PARTITION variable=g_lane_partial cyclic factor=MHA_BANK_PACK dim=2
    /* The walk below reads, adds and writes one accumulator per token lane every cycle.  Successive
     * iterations move to the next activation word, so within a lane they touch a different address;
     * the same address returns only MHA_BANK_ROW_WORDS iterations later.  Saying so lets the walk
     * run one iteration per cycle -- without it HLS assumes consecutive iterations may collide and
     * halves the rate, and the alternatives (all accumulators in registers, or two-port block
     * memories) cost ~80k registers or 96 memory blocks respectively. */
    #pragma HLS DEPENDENCE variable=g_lane_partial type=inter direction=RAW distance=MHA_BANK_ROW_WORDS dependent=true

    // One output row per call.  Several rows in flight was measured slower and far larger (see
    // docs/plans/05-lane-balance.md), so the row index and its bookkeeping are not carried here.
    static_assert(MHA_ROWS_PER_STEP == 1, "the engine computes one output row per call");

    /* A slice sits in the banks from the word that contains its first entry, so the walk starts at
     * that entry's lane.  Done here, on the first row, to keep the cursors private to this unit. */
    if (row_local == 0) {
        CURSOR_RESET:
        for (int t = 0; t < MHA_TILES; t++) {
            #pragma HLS UNROLL
            g_bank_cursor[t] = g_slice_cursor0[t];
        }
    }

    /* How many nonzeros this row holds in each lane, and the longest of them, which sets the walk. */
    int lane_nnz[MHA_TILES];
    #pragma HLS ARRAY_PARTITION variable=lane_nnz complete dim=1
    int max_nnz = 0;
    LANE_INIT:
    for (int t = 0; t < MHA_TILES; t++) {
        #pragma HLS UNROLL
        const int nnz = (int) g_bank_count[t][row_local];
        lane_nnz[t] = nnz;
        if (nnz > max_nnz) {
            max_nnz = nnz;
        }
    }

    /* Walk the step's nonzeros; every lane covers one nonzero's worth of tokens per pass. */
    /* One pipelined loop over (weight, activation word) pairs.  As two nested loops the inner one's
     * fill and drain were paid once per weight -- 15 cycles per weight where the work is
     * MHA_BANK_ROW_WORDS -- and HLS refuses to flatten them because both bounds are runtime values.
     * Flattening by hand over a compile-time word count lets successive weights flow through the
     * same pipeline at no hardware cost.  A given token's accumulator recurs every
     * MHA_BANK_ROW_WORDS iterations, which is distance enough for one iteration per cycle. */
    const int mac_steps = max_nnz * MHA_BANK_ROW_WORDS;

    MAC_STEP:
    for (int idx = 0; idx < mac_steps; idx++) {
        #pragma HLS LOOP_TRIPCOUNT min=(MHA_TC_W_NNZ*MHA_ROWS_PER_STEP/MHA_TILES*MHA_BANK_ROW_WORDS) max=(MHA_TC_W_NNZ*MHA_ROWS_PER_STEP/MHA_TILES*MHA_BANK_ROW_WORDS)
        #pragma HLS PIPELINE II=1
        const int nz = idx / MHA_BANK_ROW_WORDS;   /* which nonzero of this row */
        const int w  = idx - nz * MHA_BANK_ROW_WORDS;   /* which activation word */

        MAC_LANE:
        for (int t = 0; t < MHA_TILES; t++) {
            #pragma HLS UNROLL
            if (nz < lane_nnz[t]) {
                const int    entry = g_bank_cursor[t] + nz;
                T_MhaWVal    val;
                T_MhaTileCol col;
                bank_entry(g_bank[t][entry / MHA_TILE_PACK], entry % MHA_TILE_PACK, val, col);

                const T_MhaBankWord act = g_act_tile[t][act_base + (int) col * MHA_BANK_ROW_WORDS + w];
                const int           tok = w * MHA_BANK_PACK;
                /* One multiply-accumulate per token in the word: the activation word is the vector,
                 * the weight the scalar applied to all of it.  Written out rather than looped: as a
                 * loop with UNROLL the same eight operations cost ~4.8k LUT more. */
                mac_pe(act, val, tok + 0, 0, tokens_dim, g_lane_partial[t]);
                mac_pe(act, val, tok + 1, 1, tokens_dim, g_lane_partial[t]);
                mac_pe(act, val, tok + 2, 2, tokens_dim, g_lane_partial[t]);
                mac_pe(act, val, tok + 3, 3, tokens_dim, g_lane_partial[t]);
                mac_pe(act, val, tok + 4, 4, tokens_dim, g_lane_partial[t]);
                mac_pe(act, val, tok + 5, 5, tokens_dim, g_lane_partial[t]);
                mac_pe(act, val, tok + 6, 6, tokens_dim, g_lane_partial[t]);
                mac_pe(act, val, tok + 7, 7, tokens_dim, g_lane_partial[t]);
            }
        }
    }

    CURSOR_ADVANCE:
    for (int t = 0; t < MHA_TILES; t++) {
        #pragma HLS UNROLL
        g_bank_cursor[t] += lane_nnz[t];
    }
}
