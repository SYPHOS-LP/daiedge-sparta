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

// The MAC loop unrolls the packed lanes of an activation word by hand.
static_assert(MHA_BANK_PACK == 8, "code assumes MHA_BANK_PACK == 8");

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
T_MhaAcc g_proj_rows[MHA_ROWS_PER_STEP][MHA_TOKEN_W_MAX];

/* DDR position of the next row block, per weight matrix slot (SpmmWeightSlot) and column block.  g_block_word0 is the column block's first packed word in DDR, g_slice_entry the
 * entry index within the block where the next row block starts, g_slice_cursor0 that entry's lane
 * within its packed word.  Reads start at the word boundary below the first entry (up to
 * MHA_TILE_PACK-1 unused entries precede the slice in the bank) so that no data realignment is
 * needed.  The weight matrices are interleaved across stages, so each keeps its own position. */
#define MHA_N_WEIGHTS 4
static int g_block_word0[SPMM_N_WEIGHTS][MHA_TILES];
static int g_slice_entry[SPMM_N_WEIGHTS][MHA_TILES];
static int g_slice_cursor0[MHA_TILES];

/* Wide DDR views used to fill one packed bank word per read. */


/**
 * @brief Unpack one weight entry from a bank word: {block-local column, POT value}.
 */
static inline void bank_entry(T_MhaTileWord word, int lane, T_MhaWVal &val, T_MhaTileCol &col) {
#pragma HLS INLINE
    const int lo = lane * MHA_TILE_ENTRY_BITS;
    val = (T_MhaWVal)   word.range(lo + 3, lo);
    col = (T_MhaTileCol) word.range(lo + 11, lo + 4);
}

/**
 * @brief Compute the first DDR word of every column block of one weight matrix.
 *
 * Called once per weight matrix.  Each column block is padded to a whole packed word, so block t
 * starts after the padded lengths of blocks 0..t-1, which requires the counts of all their rows.
 *
 * @param[in] w_counts    Nonzeros per (column block, row), block-major: w_counts[t * rows + r].
 * @param[in] feature_dim Output rows of the projection.
 * @param[in] which       Weight matrix slot (SpmmWeightSlot).
 */
void load_weight_bases (int* w_counts, int feature_dim, int which) {
#pragma HLS INLINE off
    int next_word = 0;
    BASE_TILE:
    for (int t = 0; t < MHA_TILES; t++) {
        #pragma HLS LOOP_TRIPCOUNT min=MHA_TILES max=MHA_TILES
        int len = 0;
        BASE_ROW:
        for (int r = 0; r < feature_dim; r++) {
            #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_FEATURE max=MHA_TC_FEATURE
            #pragma HLS PIPELINE II=1
            len += w_counts[t * feature_dim + r];
        }
        g_block_word0[which][t] = next_word;
        g_slice_entry[which][t] = 0;
        next_word              += (len + MHA_TILE_PACK - 1) / MHA_TILE_PACK;
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
 * @param[in] w_counts     Nonzeros per (column block, row), block-major.
 * @param[in] feature_dim  Output rows of the weight matrix (row stride of w_counts).
 * @param[in] which        Weight matrix slot (SpmmWeightSlot).
 * @param[in] row_base     First output row of this stage.
 * @param[in] rows         Output rows in this stage.
 */
void load_weight_slice (T_MhaTileWord* w_bank, int* w_counts,
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
    SLICE_COUNT_TILE:
    for (int t = 0; t < MHA_TILES; t++) {
        #pragma HLS LOOP_TRIPCOUNT min=MHA_TILES max=MHA_TILES
        int len = 0;
        SLICE_COUNT_ROW:
        for (int r = 0; r < rows; r++) {
            #pragma HLS LOOP_TRIPCOUNT min=MHA_FEATURE_HEAD max=MHA_FEATURE_HEAD
            #pragma HLS PIPELINE II=1
            const int n = w_counts[t * feature_dim + row_base + r];
            g_bank_count[t][r] = (T_MhaTileCount) n;
            len += n;
        }
        slice_len[t] = len;
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
 * @param[in]     token_idx   Token index this chunk maps to
 * @param[in]     chunk       Chunk of packed input to be used
 * @param[in]     tokens_dim  Number of tokens, used to skip padding
 * @param[in,out] out_row     Output row
 */
static inline void mac_pe (
    T_MhaBankWord input,
    T_MhaWVal     weight,
    int           token_idx,
    int           chunk,
    int           tokens_dim,
    T_MhaAcc      out_row[MHA_TOKEN_W_MAX]
) {
#pragma HLS INLINE
    if (token_idx < tokens_dim) {
        /* Unpack */
        T_Activation input_word = (T_Activation) input.range(chunk * 8 + 7, chunk * 8);
#ifdef POT_SHIFT_MAC
        /* POT weight is {sign(bit3), exponent(bits0..2)}. LUTs are
         * a scarce resource at this point, so we bind this operation
         * to a DSP, as these are underutilized.
         */
        ap_uint<3> e   = weight.range(2, 0);
        bool       neg = weight[3];
        T_MhaAcc   pot = neg ? (T_MhaAcc)(-(1 << e)) : (T_MhaAcc)(1 << e);
        T_MhaAcc   prod = (T_MhaAcc) input_word * pot;
        #pragma HLS BIND_OP variable=prod op=mul impl=dsp
        out_row[token_idx] += prod;
#else
        /* Multiply - Accumulate */
        out_row[token_idx] += weight * input_word;
#endif
    }
}

/**
 * @brief One step of the tiled SpMM: MHA_ROWS_PER_STEP output rows across MHA_TILES lanes.
 *
 * Lane t reads only block t's weight bank and only block t's activation tile, so the lanes
 * never contend for a memory and no copy of the activation is needed.  Each lane accumulates
 * into its own partial row; the partials are summed once at the end of the step.  The lanes
 * advance together, so a step costs the busiest lane -- grouping several output rows lets a
 * light row in one block offset a heavy one in another.
 *
 * @param[in]     row_base     First output row of this step.
 * @param[in]     rows_in_step Output rows in this step (<= MHA_ROWS_PER_STEP).
 * @param[in]     tokens_dim   Number of tokens.
 *
 * Reads g_bank / g_act_tile / g_bank_count, advances g_bank_cursor and writes g_proj_rows.
 * act_base selects the activation region inside each tile (see load_activation_tiles).
 */
void tiled_rows_compute (
    int row_local,
    int rows_in_step,
    int tokens_dim,
    int act_base
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=g_act_tile   complete dim=1
#pragma HLS ARRAY_PARTITION variable=g_bank       complete dim=1
#pragma HLS ARRAY_PARTITION variable=g_bank_count complete dim=1
#pragma HLS ARRAY_PARTITION variable=g_bank_cursor complete dim=1
#pragma HLS ARRAY_PARTITION variable=g_proj_rows   complete dim=1
#pragma HLS ARRAY_PARTITION variable=g_proj_rows   cyclic factor=MHA_BANK_PACK dim=2

    /* One partial row per lane: lane t only ever touches partial[t], so the lanes' updates
     * are independent. */
    static T_MhaAcc partial[MHA_TILES][MHA_ROWS_PER_STEP][MHA_TOKEN_W_MAX];
    #pragma HLS ARRAY_PARTITION variable=partial complete dim=1
    #pragma HLS ARRAY_PARTITION variable=partial complete dim=2
    #pragma HLS ARRAY_PARTITION variable=partial cyclic factor=MHA_BANK_PACK dim=3

    const int row_words = (tokens_dim + MHA_BANK_PACK - 1) / MHA_BANK_PACK;

    /* A slice sits in the banks from the word that contains its first entry, so the walk starts at
     * that entry's lane.  Done here, on the first row, to keep the cursors private to this unit. */
    if (row_local == 0) {
        CURSOR_RESET:
        for (int t = 0; t < MHA_TILES; t++) {
            #pragma HLS UNROLL
            g_bank_cursor[t] = g_slice_cursor0[t];
        }
    }

    CLEAR_PARTIAL_WORD:
    for (int w = 0; w < row_words; w++) {
        #pragma HLS LOOP_TRIPCOUNT min=MHA_BANK_ROW_WORDS max=MHA_BANK_ROW_WORDS
        #pragma HLS PIPELINE II=1
        for (int t = 0; t < MHA_TILES; t++) {
            #pragma HLS UNROLL
            for (int r = 0; r < MHA_ROWS_PER_STEP; r++) {
                #pragma HLS UNROLL
                for (int p = 0; p < MHA_BANK_PACK; p++) {
                    #pragma HLS UNROLL
                    const int token = w * MHA_BANK_PACK + p;
                    if (token < tokens_dim) {
                        partial[t][r][token] = (T_MhaAcc) 0;
                    }
                }
            }
        }
    }

    /* Per-lane walk: how many nonzeros this step holds, and where each row starts within them
     * (prefix[r] is compared against the walk index instead of re-reading the counts). */
    int lane_len[MHA_TILES];
    int prefix[MHA_TILES][MHA_ROWS_PER_STEP];
    #pragma HLS ARRAY_PARTITION variable=lane_len complete dim=1
    #pragma HLS ARRAY_PARTITION variable=prefix   complete dim=0
    int max_len = 0;
    LANE_INIT:
    for (int t = 0; t < MHA_TILES; t++) {
        #pragma HLS UNROLL
        int len = 0;
        for (int r = 0; r < MHA_ROWS_PER_STEP; r++) {
            #pragma HLS UNROLL
            prefix[t][r] = len;
            if (r < rows_in_step) {
                len += (int) g_bank_count[t][row_local + r];
            }
        }
        lane_len[t] = len;
        if (len > max_len) {
            max_len = len;
        }
    }

    /* Walk the step's nonzeros; every lane covers one nonzero's worth of tokens per pass. */
    MAC_ENTRY:
    for (int k = 0; k < max_len; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=(MHA_TC_W_NNZ*MHA_ROWS_PER_STEP/MHA_TILES) max=(MHA_TC_W_NNZ*MHA_ROWS_PER_STEP/MHA_TILES)
        MAC_WORD:
        for (int w = 0; w < row_words; w++) {
            #pragma HLS LOOP_TRIPCOUNT min=MHA_BANK_ROW_WORDS max=MHA_BANK_ROW_WORDS
            #pragma HLS PIPELINE II=1
            MAC_LANE:
            for (int t = 0; t < MHA_TILES; t++) {
                #pragma HLS UNROLL
                if (k < lane_len[t]) {
                    const int    entry = g_bank_cursor[t] + k;
                    T_MhaWVal    val;
                    T_MhaTileCol col;
                    bank_entry(g_bank[t][entry / MHA_TILE_PACK], entry % MHA_TILE_PACK, val, col);

                    /* Which of the step's rows this nonzero belongs to. */
                    int row_sel = 0;
                    for (int r = 1; r < MHA_ROWS_PER_STEP; r++) {
                        #pragma HLS UNROLL
                        if (k >= prefix[t][r]) {
                            row_sel = r;
                        }
                    }

                    const T_MhaBankWord act = g_act_tile[t][act_base + (int) col * MHA_BANK_ROW_WORDS + w];
                    const int           tok = w * MHA_BANK_PACK;
                    mac_pe(act, val, tok + 0, 0, tokens_dim, partial[t][row_sel]);
                    mac_pe(act, val, tok + 1, 1, tokens_dim, partial[t][row_sel]);
                    mac_pe(act, val, tok + 2, 2, tokens_dim, partial[t][row_sel]);
                    mac_pe(act, val, tok + 3, 3, tokens_dim, partial[t][row_sel]);
                    mac_pe(act, val, tok + 4, 4, tokens_dim, partial[t][row_sel]);
                    mac_pe(act, val, tok + 5, 5, tokens_dim, partial[t][row_sel]);
                    mac_pe(act, val, tok + 6, 6, tokens_dim, partial[t][row_sel]);
                    mac_pe(act, val, tok + 7, 7, tokens_dim, partial[t][row_sel]);
                }
            }
        }
    }

    /* Sum the lanes into the output rows, MHA_BANK_PACK tokens per cycle. */
    MERGE_ROW:
    for (int r = 0; r < rows_in_step; r++) {
        #pragma HLS LOOP_TRIPCOUNT min=MHA_ROWS_PER_STEP max=MHA_ROWS_PER_STEP
        MERGE_WORD:
        for (int w = 0; w < row_words; w++) {
            #pragma HLS LOOP_TRIPCOUNT min=MHA_BANK_ROW_WORDS max=MHA_BANK_ROW_WORDS
            #pragma HLS PIPELINE II=1
            for (int p = 0; p < MHA_BANK_PACK; p++) {
                #pragma HLS UNROLL
                const int token = w * MHA_BANK_PACK + p;
                if (token < tokens_dim) {
                    T_MhaAcc sum = (T_MhaAcc) 0;
                    for (int t = 0; t < MHA_TILES; t++) {
                        #pragma HLS UNROLL
                        sum += partial[t][r][token];
                    }
                    g_proj_rows[r][token] = sum;
                }
            }
        }
    }

    CURSOR_ADVANCE:
    for (int t = 0; t < MHA_TILES; t++) {
        #pragma HLS UNROLL
        g_bank_cursor[t] += lane_len[t];
    }
}
