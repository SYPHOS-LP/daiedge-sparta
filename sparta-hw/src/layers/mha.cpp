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

/* One lane's share of a K'.V row: the MHA_BANK_PACK output features that lane owns, packed into a
 * single word.  At 32 bits an entry that is MHA_BANK_PACK * 32 = 256 bits wide, which block memory
 * cannot supply in one cycle but a logic-cell memory can -- there is no width limit on those.  That
 * is what lets all MHA_V_GROUPS lanes fetch their part of a row in the same cycle. */
typedef ap_uint<MHA_BANK_PACK * T_MhaAcc::width> T_MhaKvWord;
static_assert(MHA_FEATURE_HEAD % MHA_BANK_PACK == 0,
              "the K'.V table is packed MHA_BANK_PACK output features per word");
/* MHA_SCR_V_HEAD_CNT and MHA_V_ROW_STRIDE now size only the Q' reorder scratch, not V.
 *
 * They were V's: V used to be stored one byte per entry at out_feature * MHA_V_ROW_STRIDE + token,
 * spread across MHA_BANK_PACK memories, and the stride had to be odd so that the MHA_MAC_UNROLL
 * reads of consecutive output features fell in different memories.  V is now ordered by token and
 * packed (MHA_V_GROUPS in mha.h), read as whole words, so neither the stride nor its oddness has
 * any effect on V.  The Q' scratch is indexed with tokens_dim, so it needs only the capacity. */
static_assert(MHA_SCR_V_HEAD_CNT >= MHA_FEATURE_HEAD * MHA_TOKEN_W_MAX,
              "Q' reorder scratch is smaller than 64 x N");
static_assert(MHA_V_PACKED_WORDS == MHA_TOKEN_W_MAX * MHA_V_GROUPS,
              "packed V word count != N x (64 / MHA_BANK_PACK)");
static_assert(MHA_FEATURE_HEAD % MHA_BANK_PACK == 0,
              "packed V needs MHA_FEATURE_HEAD to divide into whole words");
static_assert(MHA_SCR_A_CNT      == MHA_FEATURE_HEAD * MHA_FEATURE_HEAD, "A cnt != d_head^2");

// The engine's MAC walk manually unrolls the packed lanes of a bank word.
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
 * @param[in]  v_packed      Dense V, token-major and packed (see MHA_V_GROUPS in mha.h).
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
    T_MhaBankWord  *v_packed,
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

    /* Per-head intermediate results for KxV and sum(K).
     *
     * The table is MHA_FEATURE_HEAD rows of MHA_FEATURE_HEAD accumulators, held as rows of
     * MHA_V_GROUPS packed words: entry [row][lane] is at row * MHA_V_GROUPS + lane.  A cyclic split
     * by MHA_V_GROUPS therefore gives each lane its own piece, 64 words of 256 bits, and the
     * lutram binding keeps all of it out of block memory -- the dense 32-bit table cost 16 blocks,
     * because two block ports carry about 18 bits each and this reads 2 048 bits per cycle. */
    T_MhaKvWord key_value_matrix[MHA_FEATURE_HEAD * MHA_V_GROUPS];
    T_MhaAcc    key_sums[MHA_FEATURE_HEAD];
    #pragma HLS ARRAY_PARTITION variable=key_value_matrix cyclic factor=MHA_V_GROUPS dim=1
    #pragma HLS BIND_STORAGE    variable=key_value_matrix type=ram_1p impl=lutram
    #pragma HLS ARRAY_PARTITION variable=key_sums         complete dim=1

    /* No clearing pass for key_value_matrix: every entry is written below, from a register
     * accumulator, rather than accumulated into. */
    CLEAR_KEY_SUMS:
    for (int i = 0; i < MHA_FEATURE_HEAD; i++) {
        #pragma HLS PIPELINE II=1
        key_sums[i] = (T_MhaAcc) 0;
    }

    /* Calculate KxV and sum(K) */
    /* Row f of K' occupies k_ptr[f] .. k_ptr[f+1], so one row's end is the next row's start.
     * Carrying it across iterations costs one read per row instead of two, which lets the pointer
     * array live in single-port logic-cell memory: with two reads per row a one-port memory had to
     * serialise them and the loop paid an extra cycle per row. */
    int nz_begin = k_ptr[0];
    KEY_VALUE_FEATURE:
    for (int feature = 0; feature < MHA_FEATURE_HEAD; feature++) {
        #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_FEATURE_HEAD max=MHA_TC_FEATURE_HEAD
        const int nz_end = k_ptr[feature + 1];

        /* One feature's row of the K.V matrix, held in registers while it is accumulated.  In the
         * matrix itself the accumulation was a read-modify-write of the same memory address on
         * successive nonzeros, so the loop could not run faster than a BRAM read plus an add plus a
         * write -- II 3.  Here `complete` makes each out_feature its own scalar in a flip-flop, so
         * all MHA_FEATURE_HEAD of them are updated in the same cycle with no memory port to share
         * and nothing to combine at the end. */
        T_MhaAcc row_acc[MHA_FEATURE_HEAD];
        #pragma HLS ARRAY_PARTITION variable=row_acc complete dim=1
        KEY_VALUE_ROW_CLEAR:
        for (int out_feature = 0; out_feature < MHA_FEATURE_HEAD; out_feature++) {
            #pragma HLS UNROLL
            row_acc[out_feature] = (T_MhaAcc) 0;
        }

        KEY_VALUE_NONZERO:
        for (int s = nz_begin; s < nz_end; s++) {
            #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_KP_NNZ max=MHA_TC_KP_NNZ
            int          token   = (int) k_col[s];
            T_Activation key_val = k_val[s];

            key_sums[feature] += key_val;

            KEY_VALUE_OUT_FEATURE:
            for (int out_feature = 0; out_feature < MHA_FEATURE_HEAD; out_feature++) {
                #pragma HLS UNROLL

                /* Multiply-Accumulate K value with MHA_FEATURE_HEAD features of selected token.
                 *
                 * Unrolled completely, which makes this the attention's equivalent of the engine's
                 * lanes: the MHA_FEATURE_HEAD copies of this body fall into MHA_V_GROUPS groups of
                 * MHA_BANK_PACK, each group reading ONE word from its own slice of V.  So the whole
                 * row of V for this token arrives in one cycle, MHA_FEATURE_HEAD multiplies happen,
                 * and one K' value is retired per cycle.
                 *
                 * The lanes are over OUTPUT features, not over tokens.  Splitting the sparse list
                 * by token instead would make each output row cost the busiest lane -- measured at
                 * 1.64x imbalance, the same effect as docs/plans/05-lane-balance.md records for the
                 * engine.  Here every lane sees the same K' value, so there is no imbalance and no
                 * partial sums to combine: row_acc stays MHA_FEATURE_HEAD registers.
                 *
                 * V is read here rather than fetched into registers beforehand.  Fetching the words
                 * in a loop of their own first cost 2 720 cycles per head: a short pipelined loop
                 * nested inside this one pays its fill and drain on every K' value, and it also
                 * stopped the reads overlapping the multiplies. */
                const int lo = (out_feature % MHA_BANK_PACK) * MHA_ACT_ELEM_BITS;
                const T_MhaBankWord v_word = v_packed[token * MHA_V_GROUPS
                                                      + out_feature / MHA_BANK_PACK];
                const T_Activation v_val = (T_Activation)
                    (ap_int<MHA_ACT_ELEM_BITS>) v_word.range(lo + MHA_ACT_ELEM_BITS - 1, lo);
                row_acc[out_feature] += key_val * v_val;
            }
        }

        /* Pack the finished row into one word per lane.  Fully unrolled, so the whole row lands in
         * one go and no lane waits for another. */
        KEY_VALUE_ROW_STORE:
        for (int lane = 0; lane < MHA_V_GROUPS; lane++) {
            #pragma HLS UNROLL
            T_MhaKvWord word = 0;
            KEY_VALUE_ROW_PACK:
            for (int b = 0; b < MHA_BANK_PACK; b++) {
                #pragma HLS UNROLL
                const int lo = b * T_MhaAcc::width;
                word.range(lo + T_MhaAcc::width - 1, lo) =
                    (ap_uint<T_MhaAcc::width>) row_acc[lane * MHA_BANK_PACK + b];
            }
            key_value_matrix[feature * MHA_V_GROUPS + lane] = word;
        }

        nz_begin = nz_end;
    }

    /* Calculate Qx(KxV) and Qxsum(K), then divide with each other. */
    /* Same carry as KEY_VALUE_FEATURE above, for the same reason. */
    int q_nz_begin = q_ptr[0];
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
        T_MhaAcc  denominator = (T_MhaAcc) 0;
        const int nz_end      = q_ptr[token + 1];
        NUMERATOR_NONZERO:
        for (int s = q_nz_begin; s < nz_end; s++) {
            #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_QP_NNZ max=MHA_TC_QP_NNZ
            int          band_feature = (int) q_col[s];   // head-local feature (0..63)
            T_Activation query_val    = q_val[s];

            denominator += query_val * key_sums[band_feature];

            NUMERATOR_OUT_FEATURE:
            for (int out_feature = 0; out_feature < MHA_FEATURE_HEAD; out_feature++) {
                #pragma HLS UNROLL

                /* Unrolled completely, the same arrangement as KEY_VALUE_OUT_FEATURE above: the
                 * MHA_FEATURE_HEAD copies fall into MHA_V_GROUPS lanes of MHA_BANK_PACK, each lane
                 * reading ONE word from its own piece of the table.  So a whole table row arrives
                 * in one cycle and one Q' value is retired per cycle.
                 *
                 * The lanes are over output features, not over the summed axis, so every lane sees
                 * the same Q' value and keeps its own MHA_BANK_PACK totals in registers.  Nothing
                 * has to be combined at the end and no lane can be the slow one. */
                const int lo = (out_feature % MHA_BANK_PACK) * T_MhaAcc::width;
                const T_MhaKvWord kv_word = key_value_matrix[band_feature * MHA_V_GROUPS
                                                             + out_feature / MHA_BANK_PACK];
                const T_MhaAcc kv_val = (T_MhaAcc)
                    (ap_int<T_MhaAcc::width>) kv_word.range(lo + T_MhaAcc::width - 1, lo);
                numerator_row[out_feature] += query_val * kv_val;
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

        q_nz_begin = nz_end;
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
