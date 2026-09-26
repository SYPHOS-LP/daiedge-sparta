/**********************************************************************************
 * Linear Multi-Head Attention Layer - Implementation
 *
 * Please see the respective header file for a high-level description of the layer.
 *
 *********************************************************************************/

#include "mha.h"
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

/* ---------------------------------------------------------------------------------------
 * Tiled SpMM state.
 *
 * The feature axis is cut into MHA_TILES column blocks.  Block t owns ONE activation tile
 * (its own memory, MHA_TILE_FEATURES rows of packed tokens) and ONE weight bank (its own
 * memory, holding every nonzero whose column falls in that block, rows ascending).  The
 * MHA_TILES lanes therefore each read one nonzero and one activation word per cycle without
 * any copy of the activation: the four replicas this replaces cost 4x the whole D x N bank.
 *
 * compile_model.py --mha-tiles emits the weights to match: values and block-local columns in
 * block order (each block padded to a whole MHA_TILE_PACK word) plus, in the row-pointer slot,
 * the per-(block, row) nonzero counts that let each lane walk row by row.
 * ------------------------------------------------------------------------------------ */
static T_MhaBankWord  g_act_tile[MHA_TILES][MHA_TILE_WORDS];
static T_MhaTileWord  g_bank[MHA_TILES][MHA_TILE_BANK_WORDS];
static T_MhaTileCount g_bank_count[MHA_TILES][MHA_FEATURE_HEAD];

/* Engine state kept at file scope on purpose: with no caller-specific pointer arguments every
 * call to tiled_rows_compute() has identical bindings, which is what lets HLS build ONE engine
 * instead of specialising one per call site. */
static int      g_bank_cursor[MHA_TILES];
static T_MhaAcc g_proj_rows[MHA_ROWS_PER_STEP][MHA_TOKEN_W_MAX];

/* DDR position of the next row block, per weight matrix (0 = W_K, 1 = W_Q, 2 = W_V, 3 = W_O) and
 * column block.  g_block_word0 is the column block's first packed word in DDR, g_slice_entry the
 * entry index within the block where the next row block starts, g_slice_cursor0 that entry's lane
 * within its packed word.  Reads start at the word boundary below the first entry (up to
 * MHA_TILE_PACK-1 unused entries precede the slice in the bank) so that no data realignment is
 * needed.  The weight matrices are interleaved across stages, so each keeps its own position. */
#define MHA_N_WEIGHTS 4
static int g_block_word0[MHA_N_WEIGHTS][MHA_TILES];
static int g_slice_entry[MHA_N_WEIGHTS][MHA_TILES];
static int g_slice_cursor0[MHA_TILES];

/* Wide DDR views used to fill one packed bank word per read. */
typedef ap_uint<MHA_TILE_PACK * 8>  T_MhaBankValWord;   /* MHA_TILE_PACK POT bytes   */
typedef ap_uint<MHA_TILE_PACK * 16> T_MhaBankColWord;   /* MHA_TILE_PACK uint16 cols */

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

/**
 * @brief Unpack one weight entry from a bank word: {block-local column, POT value}.
 */
static inline void bank_entry(T_MhaTileWord word, int lane, T_MhaWVal &val, T_MhaTileCol &col) {
#pragma HLS INLINE
    const int lo = lane * MHA_TILE_ENTRY_BITS;
    val = (T_MhaWVal)   word.range(lo + 3, lo);
    col = (T_MhaTileCol) word.range(lo + 9, lo + 4);
}

/**
 * @brief Compute the first DDR word of every column block of one weight matrix.
 *
 * Called once per weight matrix.  Each column block is padded to a whole packed word, so block t
 * starts after the padded lengths of blocks 0..t-1, which requires the counts of all their rows.
 *
 * @param[in] w_counts    Nonzeros per (column block, row), block-major: w_counts[t * rows + r].
 * @param[in] feature_dim Output rows of the projection.
 * @param[in] which       Weight matrix index (0..MHA_N_WEIGHTS-1).
 */
static void load_weight_bases (int* w_counts, int feature_dim, int which) {
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
 * for a whole column block.  Two passes at one read per cycle: values, then column indices merged
 * into the same packed word, so the second pass reads back the word it updates.
 *
 * @param[in] w_val       Weight values in DDR (POT bytes, block order).
 * @param[in] w_col        Block-local column indices in DDR (block order).
 * @param[in] w_counts     Nonzeros per (column block, row), block-major.
 * @param[in] feature_dim  Output rows of the weight matrix (row stride of w_counts).
 * @param[in] which        Weight matrix index (0..MHA_N_WEIGHTS-1).
 * @param[in] row_base     First output row of this stage.
 * @param[in] rows         Output rows in this stage.
 */
static void load_weight_slice (T_Activation* w_val, T_MhaIndex* w_col, int* w_counts,
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

    /* Values: one read holds exactly one bank word's worth of entries. */
    T_MhaBankValWord* val_words = reinterpret_cast<T_MhaBankValWord*>(w_val);
    BANK_VALUES_TILE:
    for (int t = 0; t < MHA_TILES; t++) {
        #pragma HLS LOOP_TRIPCOUNT min=MHA_TILES max=MHA_TILES
        const int words = slice_words[t];
        BANK_VALUES_WORD:
        for (int w = 0; w < words; w++) {
            #pragma HLS LOOP_TRIPCOUNT min=64 avg=104 max=MHA_TILE_BANK_WORDS
            #pragma HLS PIPELINE II=1
            T_MhaBankValWord raw = val_words[slice_word0[t] + w];
            T_MhaTileWord    packed = 0;
            BANK_VALUE_LANE:
            for (int l = 0; l < MHA_TILE_PACK; l++) {
                #pragma HLS UNROLL
                const ap_uint<8> b  = (ap_uint<8>) raw.range(l * 8 + 7, l * 8);
                const int        lo = l * MHA_TILE_ENTRY_BITS;
#ifdef POT_SHIFT_MAC
                /* POT weight: DDR byte is {sign(bit7), exponent(bits0..6)}; repack to the
                 * on-chip {sign(bit3), exponent(bits0..2)} the shift-MAC reads. */
                packed.range(lo + 3, lo) = (ap_uint<4>)((b[7] << 3) | (b & 0x7));
#else
                packed.range(lo + 3, lo) = (ap_uint<4>) b;
#endif
            }
            g_bank[t][w] = packed;
        }
    }

    /* Columns: same word layout, merged into the values already stored. */
    T_MhaBankColWord* col_words = reinterpret_cast<T_MhaBankColWord*>(w_col);
    BANK_COLUMNS_TILE:
    for (int t = 0; t < MHA_TILES; t++) {
        #pragma HLS LOOP_TRIPCOUNT min=MHA_TILES max=MHA_TILES
        const int words = slice_words[t];
        BANK_COLUMNS_WORD:
        for (int w = 0; w < words; w++) {
            #pragma HLS LOOP_TRIPCOUNT min=64 avg=104 max=MHA_TILE_BANK_WORDS
            #pragma HLS PIPELINE II=1
            T_MhaBankColWord raw    = col_words[slice_word0[t] + w];
            T_MhaTileWord    packed = g_bank[t][w];
            BANK_COLUMN_LANE:
            for (int l = 0; l < MHA_TILE_PACK; l++) {
                #pragma HLS UNROLL
                const int lo = l * MHA_TILE_ENTRY_BITS;
                packed.range(lo + 9, lo + 4) = (ap_uint<6>) raw.range(l * 16 + 5, l * 16);
            }
            g_bank[t][w] = packed;
        }
    }
}

/**
 * @brief Split the packed activation into the per-block tiles the lanes read.
 *
 * Feature f lives in block f / MHA_TILE_FEATURES, and the bank is feature-major, so block t is
 * simply the contiguous word range [t * MHA_TILE_WORDS, (t+1) * MHA_TILE_WORDS).  One copy is
 * made, not one per lane.
 *
 * @param[in] packed_input  Packed activation bank (feature-major, MHA_BANK_PACK tokens/word).
 */
static void load_activation_tiles (T_MhaBankWord* packed_input) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=g_act_tile complete dim=1
    /* One tile is 448 words x 64 b, which fits a single BRAM36; keep them out of URAM, which the
     * weight banks need. */
    #pragma HLS BIND_STORAGE variable=g_act_tile type=ram_t2p impl=bram

    ACTIVATION_TILE:
    for (int t = 0; t < MHA_TILES; t++) {
        #pragma HLS LOOP_TRIPCOUNT min=MHA_TILES max=MHA_TILES
        ACTIVATION_TILE_WORD:
        for (int w = 0; w < MHA_TILE_WORDS; w++) {
            #pragma HLS LOOP_TRIPCOUNT min=MHA_TILE_WORDS max=MHA_TILE_WORDS
            #pragma HLS PIPELINE II=1
            g_act_tile[t][w] = packed_input[t * MHA_TILE_WORDS + w];
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
 */
static void tiled_rows_compute (
    int row_local,
    int rows_in_step,
    int tokens_dim
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

                    const T_MhaBankWord act = g_act_tile[t][(int) col * MHA_BANK_ROW_WORDS + w];
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
    /* Reads start at a packed-word boundary, so the first entry of the row block is at lane
     * g_slice_cursor0[t], not at 0. */
    CURSOR_INIT:
    for (int t = 0; t < MHA_TILES; t++) {
        #pragma HLS UNROLL
        g_bank_cursor[t] = g_slice_cursor0[t];
    }

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

        tiled_rows_compute(i, rows_in_step, tokens_dim);

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
    T_Activation* wq_values, T_MhaIndex* wq_col_idx, int* wq_row_ptr,
    T_Activation* wk_values, T_MhaIndex* wk_col_idx, int* wk_row_ptr,
    T_Activation* wv_values, T_MhaIndex* wv_col_idx, int* wv_row_ptr,
    T_Activation* wo_values, T_MhaIndex* wo_col_idx, int* wo_row_ptr,
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
    load_activation_tiles(input_packed);

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
            load_activation_tiles(attn_output);
        }

        /* Weight matrix of this stage; one branch each, all in this scope. */
        if (which == 0) {
            if (first) { load_weight_bases(wk_row_ptr, feature_dim, 0); }
            load_weight_slice(wk_values, wk_col_idx, wk_row_ptr, feature_dim, 0, row_base, MHA_FEATURE_HEAD);
        } else if (which == 1) {
            if (first) { load_weight_bases(wq_row_ptr, feature_dim, 1); }
            load_weight_slice(wq_values, wq_col_idx, wq_row_ptr, feature_dim, 1, row_base, MHA_FEATURE_HEAD);
        } else if (which == 2) {
            if (first) { load_weight_bases(wv_row_ptr, feature_dim, 2); }
            load_weight_slice(wv_values, wv_col_idx, wv_row_ptr, feature_dim, 2, row_base, MHA_FEATURE_HEAD);
        } else {
            if (first) { load_weight_bases(wo_row_ptr, feature_dim, 3); }
            load_weight_slice(wo_values, wo_col_idx, wo_row_ptr, feature_dim, 3, row_base, MHA_FEATURE_HEAD);
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
