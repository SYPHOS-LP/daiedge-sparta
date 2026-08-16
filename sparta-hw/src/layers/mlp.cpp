/**********************************************************************************
 * Sparse MLP Layer - Implementation
 *
 * Please see the respective header file for a high-level description of the layer.
 *********************************************************************************/

#include "mlp.h"
#include "quantize.h"
#include <cassert>
#include <hls_stream.h>

/* H streamed producer->consumer in the DATAFLOW MLP over ONE channel using a count-first
 * per-row protocol: for each hidden row the producer emits a HEADER token (is_header=1,
 * count = that row's surviving-nonzero count), then `count` VALUE tokens {val, token}.
 * The header lets the consumer walk each row with a BOUNDED for-loop (for s in 0..count),
 * which HLS pipelines/flattens with the inner W2 scatter — a data-dependent while(!last)
 * cannot be, and measured ~1.5x slower.  A single stream (vs two) keeps the DATAFLOW
 * producer/consumer coupling to one channel, which HLS's "produced/consumed exactly once"
 * dataflow check requires.  This replaces the DDR H CSR entirely, so H never touches DDR. */
struct T_MlpHTok {
    T_Activation val;        /* int8 hidden activation (value token) */
    T_MlpHCol    token;      /* token index (value token) */
    int          count;      /* row nnz count (header token) */
    bool         is_header;  /* 1 => header (count valid); 0 => value token */
};

#if !defined(MLP_SPARSE_FORMAT_CSR)
#error "mlp requires MLP_SPARSE_FORMAT_CSR."
#endif

/* The fused MAC has no unroll-1 fallback. */
#if MLP_MAC_UNROLL < 2
#error "MLP_MAC_UNROLL must be >= 2 (fused_sdmm_relu_streaming has no unroll-1 path)."
#endif

/* On-chip narrowed types must cover the values they hold, else the store wraps. */
static_assert(MLP_TOKEN_W_MAX <= (1 << T_MlpHCol::width),  "MLP_TOKEN_W_MAX exceeds T_MlpHCol range");
static_assert(MLP_FEATURE_W_MAX  <= (1 << T_MlpW1Col::width), "MLP_FEATURE_W_MAX exceeds T_MlpW1Col range");
static_assert(MLP_HIDDEN_FEATURE_W_MAX   <= (1 << T_MlpW2Col::width), "MLP_HIDDEN_FEATURE_W_MAX exceeds T_MlpW2Col range");
static_assert(MLP_FEATURE_OUT_W_MAX <= (1 << T_MlpW2Row::width), "MLP_FEATURE_OUT_W_MAX exceeds T_MlpW2Row range (W2 CSC row idx)");
static_assert((int)T_MlpWVal::width >= 4,                "T_MlpWVal must hold w4 (-7..+7)");

/* Per-row weight-decode buffers must cover the densest emitted row (rows are sparse). */
static_assert(MLP_MAX_W1_NNZ <= MLP_FEATURE_W_MAX, "W1 decode depth should not exceed dense row width D");
static_assert(MLP_MAX_W2_NNZ <= MLP_HIDDEN_FEATURE_W_MAX,  "W2 decode depth should not exceed dense row width d_h");

/* Second linear layer output-row parallelism. */
static_assert(MLP_OUT_ROW_PARALLEL >= 1, "MLP_OUT_ROW_PARALLEL must be >= 1");

/* fused_sdmm_relu_streaming manually unrolls the 8 packed lanes of a bank word. */
static_assert(MLP_BANK_PACK == 8, "code assumes MLP_BANK_PACK == 8");

/**
 * @brief Burst-read one CSR weight row from DDR into on-chip buffers, narrowing the
 *        value/column to their on-chip widths.
 *
 * @tparam T_OCVal  On-chip value type.
 * @tparam T_OCCol  On-chip column type.
 *
 * @param[in]  in_val   DDR CSR values.
 * @param[in]  in_col   DDR CSR column indices.
 * @param[in]  in_ptr   DDR CSR row pointers.
 * @param[in]  row_id   Row to decode.
 * @param[out] out_val  On-chip row values.
 * @param[out] out_col  On-chip row column indices.
 * @param[out] out_len  Number of nonzeros written to the row.
 */
template <typename T_OCVal, typename T_OCCol>
static inline void mlp_decode(
    T_Activation *in_val,
    T_MlpIndex   *in_col,
    int          *in_ptr,
    int           row_id,
    T_OCVal      *out_val,
    T_OCCol      *out_col,
    int          &out_len
) {
#pragma HLS INLINE
    int row_start = in_ptr[row_id];
    int row_end   = in_ptr[row_id + 1];
    out_len       = row_end - row_start;

    /* Burst the CSR slice from DDR; the stores narrow the value and the 16-bit column
     * to the on-chip widths. */
    CSR_ROW_LOAD_LOOP:
    for (int i = 0; i < out_len; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=MLP_TC_W1_NNZ max=MLP_TC_W2_NNZ
        #pragma HLS PIPELINE II=1
#ifdef POT_SHIFT_MAC
        /* POT weight: DDR byte is {sign(bit7), exponent(bits0..6)}; repack to the
         * on-chip {sign(bit3), exponent(bits0..2)} the shift-MAC reads. */
        ap_uint<8> b = (ap_uint<8>) in_val[row_start + i];
        out_val[i]   = (T_OCVal)((ap_uint<4>)((b[7] << 3) | (b & 0x7)));
#else
        out_val[i] = (T_OCVal) in_val[row_start + i];   /* narrow int8 -> w4 */
#endif
        out_col[i] = (T_OCCol) in_col[row_start + i];
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
static inline void mac_pe(
    T_MlpBankWord input,
    T_MlpWVal     weight,
    int           token_idx,
    int           chunk,
    int           tokens_dim,
    T_MlpAcc      out_row[MLP_TOKEN_W_MAX]
) {
#pragma HLS INLINE
    if (token_idx < tokens_dim) {
        /* Unpack */
        T_Activation input_word = (T_Activation) input.range(chunk * 8 + 7, chunk * 8);
#ifdef POT_SHIFT_MAC
        /* POT weight is {sign(bit3), exponent(bits0..2)}; the multiply-accumulate is a
         * shift by the stored exponent plus a sign (the compiler emits log2|w| directly). */
        ap_uint<3> e   = weight.range(2, 0);
        bool       neg = weight[3];
        T_MlpAcc shifted = (T_MlpAcc) input_word << e;
        out_row[token_idx] += neg ? (T_MlpAcc)(-shifted) : shifted;
#else
        /* Multiply - Accumulate */
        out_row[token_idx] += weight * input_word;
#endif
    }
}

/**
 * @brief First linear layer fused with ReLU on its output, so that the output is sparse.
 *
 * Each hidden row's surviving nonzeros are pushed to `h_stream` using a count-first
 * per-row protocol (see T_MlpHTok).  Used as the DATAFLOW producer so stage-2 consumes
 * H as it is produced (H never round-trips through DDR).
 *
 * @param[in]  input_packed  Packed input activation banks.
 * @param[in]  w1_values     W1 weight values.
 * @param[in]  w1_col_idx    W1 column indices.
 * @param[in]  w1_row_ptr    W1 row pointers.
 * @param[in]  hidden_dim    Hidden dimension.
 * @param[in]  feature_dim   Input feature dimension.
 * @param[in]  batch         Token count.
 * @param[in]  scale         Requantization scale.
 * @param[out] h_stream      Hidden-activation token stream to the consumer.
 */
static void fused_sdmm_relu_streaming (
    T_MlpBankWord *input_packed,
    T_Activation  *w1_values,
    T_MlpIndex    *w1_col_idx,
    int           *w1_row_ptr,
    int            hidden_dim,
    int            feature_dim,
    int            batch,
    T_Scale        scale,
    hls::stream<T_MlpHTok> &h_stream
) {
    #pragma HLS ARRAY_PARTITION variable=input_packed cyclic factor=MLP_BANK_CYCLIC dim=1
#ifndef __SYNTHESIS__
    assert(batch       <= MLP_TOKEN_W_MAX && "batch exceeds MLP_TOKEN_W_MAX");
    assert(feature_dim <= MLP_FEATURE_W_MAX  && "feature_dim exceeds MLP_FEATURE_W_MAX");
    assert(feature_dim <= MAX_ROWS      && "feature_dim exceeds MAX_ROWS");
#endif

    /* Scratch buffers used per processing engine. */
    T_MlpWVal  w_val[MLP_ROW_PARALLEL][MLP_MAX_W1_NNZ];
    T_MlpW1Col w_col[MLP_ROW_PARALLEL][MLP_MAX_W1_NNZ];
    int        w_len[MLP_ROW_PARALLEL];
    T_MlpAcc   h_row[MLP_ROW_PARALLEL][MLP_TOKEN_W_MAX];
    #pragma HLS ARRAY_PARTITION variable=w_val complete dim=1
    #pragma HLS ARRAY_PARTITION variable=w_col complete dim=1
    #pragma HLS ARRAY_PARTITION variable=w_len complete dim=1
    #pragma HLS ARRAY_PARTITION variable=h_row complete dim=1
    #pragma HLS ARRAY_PARTITION variable=h_row cyclic factor=MLP_MAC_UNROLL dim=2

    const T_MlpAcc ZERO = (T_MlpAcc) 0;

    /* Bulk-load the whole W1 row-pointer array on-chip once, up front, via a tight
     * pipelined burst -- mlp_decode() below then reads per-row spans from this on-chip
     * copy instead of taking a second, isolated trip to DDR for every one of the
     * hidden_dim rows.  Same "fine in C-sim, hangs on real hardware" bare-scalar-m_axi-
     * read pattern already fixed for W2 by stage_w2_csc(); ported from
     * sparta-hw-armagedon-kr260-v3 (9df179a). */
    static int w1_rptr_buffer[MLP_HIDDEN_FEATURE_W_MAX + 1];
    #pragma HLS BIND_STORAGE variable=w1_rptr_buffer type=ram_t2p impl=bram
    W1_ROW_POINTER_LOAD:
    for (int h = 0; h <= hidden_dim; h++) {
        #pragma HLS LOOP_TRIPCOUNT min=(MLP_TC_HIDDEN_FEATURE+1) max=(MLP_TC_HIDDEN_FEATURE+1)
        #pragma HLS PIPELINE II=1
        w1_rptr_buffer[h] = w1_row_ptr[h];
    }

    HIDDEN_ROW_GROUP:
    for (int i = 0; i < hidden_dim; i += MLP_ROW_PARALLEL) {
        #pragma HLS LOOP_TRIPCOUNT min=(MLP_TC_HIDDEN_FEATURE/MLP_ROW_PARALLEL) max=(MLP_TC_HIDDEN_FEATURE/MLP_ROW_PARALLEL)

        /* Burst read a single weight row for all parallel engines and clear the hidden buffer. */
        DECODE_AND_CLEAR:
        for (int pe = 0; pe < MLP_ROW_PARALLEL; pe++) {
            #pragma HLS UNROLL
            int row = i + pe;

            if (row >= hidden_dim) {
                w_len[pe] = 0;
                continue;
            }

            mlp_decode(w1_values, w1_col_idx, w1_rptr_buffer, row,
                       w_val[pe], w_col[pe], w_len[pe]);

            CLEAR_ACCUMULATOR:
            for (int j = 0; j < batch; j++) {
                #pragma HLS LOOP_TRIPCOUNT min=MLP_TC_BATCH max=MLP_TC_BATCH
                #pragma HLS PIPELINE II=1
                #pragma HLS UNROLL factor=MLP_MAC_UNROLL
                h_row[pe][j] = ZERO;
            }
        }

        int max_nnz = 0;
        LONGEST_ROW_SPAN:
        for (int pe = 0; pe < MLP_ROW_PARALLEL; pe++) {
            #pragma HLS UNROLL
            if (w_len[pe] > max_nnz) {
                max_nnz = w_len[pe];
            }
        }

        NONZERO_WALK:
        for (int k = 0; k < max_nnz; k++) {
            #pragma HLS LOOP_TRIPCOUNT min=MLP_TC_W1_NNZ max=MLP_TC_W1_NNZ

            TOKEN_TILES:
            for (int token_base = 0; token_base < batch; token_base += MLP_MAC_UNROLL) {
                #pragma HLS LOOP_TRIPCOUNT min=(MLP_TC_BATCH/MLP_MAC_UNROLL+1) max=(MLP_TC_BATCH/MLP_MAC_UNROLL+1)
                #pragma HLS PIPELINE II=1

                PARALLEL_HIDDEN_ROWS:
                for (int pe = 0; pe < MLP_ROW_PARALLEL; pe++) {
                    #pragma HLS UNROLL
                    if (k < w_len[pe]) {
                        T_MlpWVal weight         = w_val[pe][k];
                        int       xrow_word_base = (int)w_col[pe][k] * MLP_BANK_ROW_WORDS;
                        int       tile_word_base = xrow_word_base + (token_base / MLP_BANK_PACK);

                        TILE_WORDS:
                        for (int wo = 0; wo < MLP_BANK_CYCLIC; wo++) {
                            #pragma HLS UNROLL
                            T_MlpBankWord packed_x = input_packed[tile_word_base + wo];
                            int token0 = token_base + wo * MLP_BANK_PACK;
                            mac_pe(packed_x, weight, token0 + 0, 0, batch, h_row[pe]);
                            mac_pe(packed_x, weight, token0 + 1, 1, batch, h_row[pe]);
                            mac_pe(packed_x, weight, token0 + 2, 2, batch, h_row[pe]);
                            mac_pe(packed_x, weight, token0 + 3, 3, batch, h_row[pe]);
                            mac_pe(packed_x, weight, token0 + 4, 4, batch, h_row[pe]);
                            mac_pe(packed_x, weight, token0 + 5, 5, batch, h_row[pe]);
                            mac_pe(packed_x, weight, token0 + 6, 6, batch, h_row[pe]);
                            mac_pe(packed_x, weight, token0 + 7, 7, batch, h_row[pe]);
                        }
                    }
                }
            }
        }

        /* ReLU + emit each hidden row's nonzeros to the stream (row-sequential). */
        DRAIN_PE:
        for (int pe = 0; pe < MLP_ROW_PARALLEL; pe++) {
            int row = i + pe;
            if (row >= hidden_dim) {
                continue;
            }

            /* Buffer this row's surviving nonzeros so the count is known before the
             * header token is written (count-first protocol). */
            T_Activation row_val[MLP_TOKEN_W_MAX];
            T_MlpHCol    row_tok[MLP_TOKEN_W_MAX];
            int          row_nnz = 0;

            SPARSIFY:
            for (int j = 0; j < batch; j++) {
                #pragma HLS LOOP_TRIPCOUNT min=MLP_TC_BATCH max=MLP_TC_BATCH
                #pragma HLS PIPELINE II=1
                T_MlpAcc acc_val = h_row[pe][j];
                if (acc_val > ZERO) {
                    T_Activation hv = requant_to_int8(acc_val, scale);
                    if (hv > (T_Activation) 0) {
                        row_val[row_nnz] = hv;
                        row_tok[row_nnz] = (T_MlpHCol) j;
                        row_nnz++;
                    }
                }
            }

            /* Count-first protocol: emit a HEADER token with this row's nnz count, then its
             * value tokens.  An empty row emits just the header (count 0). */
            T_MlpHTok hdr;
            hdr.val = (T_Activation) 0;
            hdr.token = (T_MlpHCol) 0;
            hdr.count = row_nnz;
            hdr.is_header = true;
            h_stream.write(hdr);

            EMIT_ROW:
            for (int s = 0; s < row_nnz; s++) {
                #pragma HLS LOOP_TRIPCOUNT min=MLP_TC_HIDDEN_NNZ max=MLP_TC_HIDDEN_NNZ
                #pragma HLS PIPELINE II=1
                T_MlpHTok t;
                t.val       = row_val[s];
                t.token     = row_tok[s];
                t.count     = 0;
                t.is_header = false;
                h_stream.write(t);
            }
        }
    }
}

/* Global W2 CSC staging (whole W2, column-major, staged on-chip ONCE per layer -> URAM).
 * W2 is a STATIC weight so its nnz is exact/compile-time known.  To use URAM efficiently
 * (HLS stores 1 elem/URAM-row = 72b), each nonzero {val 4b, row 10b}=14b is PACKED
 * MLP_W2_PACK per word: g_w2_csc[word] lane k = {row(10b) : val(4b)} at bit k*14.
 *   g_w2_csc_ptr : per-hidden-column span [ptr[h], ptr[h+1]) in NONZERO units.
 * The H-driven stage-2 walks H sequentially and, per hidden column h, scatters that
 * column's nonzeros into the output accumulators. */
/* Per-slice W2 CSC: PE p holds ONLY the nonzeros whose out_row is in its contiguous slice
 * [p*MLP_OUT_ROW_SLICE, (p+1)*MLP_OUT_ROW_SLICE).  g_w2_csc_ptr[p][h] is the per-column span
 * (nonzero units) within slice p's own packed value array g_w2_csc[p].  Packing is contiguous
 * across columns per slice (no per-column flush).  Splitting the nonzero WORK per PE (not just
 * predicating one shared walk) is what makes the P scatter lanes overlap. */
static T_MlpW2Word g_w2_csc[MLP_OUT_ROW_PARALLEL][MLP_W2_WORDS_PE];
static int         g_w2_csc_ptr[MLP_OUT_ROW_PARALLEL][MLP_HIDDEN_FEATURE_W_MAX + 1];

/* Unpack a W2 nonzero from an ALREADY-FETCHED packed word (word = g_w2_csc[s/PACK]).
 * The caller fetches the word once per PACK consecutive nonzeros (cuts URAM reads +
 * the per-nonzero shift LUT). */
static inline void w2_lane_from_word(T_MlpW2Word word, int lane_idx,
                                     T_MlpWVal &val, T_MlpW2Row &row) {
#pragma HLS INLINE
    const int lane = lane_idx * MLP_W2_LANE_BITS;
    val = (T_MlpWVal)  word.range(lane + 3,  lane);
    row = (T_MlpW2Row) word.range(lane + 13, lane + 4);
}

/* Packed acc lane read/modify-write: acc word holds MLP_ACC_PACK int20 lanes along token.
 * lane = token % PACK (runtime -> a small PACK-way barrel insert on the write). */
static inline T_MlpAccLane acc_get(T_MlpAccWord word, int lane_idx) {
#pragma HLS INLINE
    const int lo = lane_idx * MLP_ACC_BITS;
    return (T_MlpAccLane) word.range(lo + MLP_ACC_BITS - 1, lo);
}
static inline void acc_set(T_MlpAccWord &word, int lane_idx, T_MlpAccLane v) {
#pragma HLS INLINE
    const int lo = lane_idx * MLP_ACC_BITS;
    word.range(lo + MLP_ACC_BITS - 1, lo) = (ap_uint<MLP_ACC_BITS>) v;
}

/**
 * @brief Burst-stage the whole W2 (column-major CSC) from DDR into on-chip PACKED URAM once.
 *
 * @param[in]  w2_values   W2 CSC values in DDR
 * @param[in]  w2_row_idx  W2 CSC row indices
 * @param[in]  w2_col_ptr  W2 CSC column pointers
 * @param[in]  hidden_dim  Number of hidden columns
 */
static void stage_w2_csc(
    T_Activation *w2_values,
    T_MlpIndex   *w2_row_idx,
    int          *w2_col_ptr,
    int           hidden_dim
) {
#pragma HLS INLINE off
    #pragma HLS BIND_STORAGE variable=g_w2_csc type=ram_t2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=g_w2_csc     complete dim=1
    #pragma HLS ARRAY_PARTITION variable=g_w2_csc_ptr complete dim=1

    /* Per-slice packing cursors: partial word, lane-in-word, and word index for each PE. */
    T_MlpW2Word word[MLP_OUT_ROW_PARALLEL];
    int         lane[MLP_OUT_ROW_PARALLEL];
    int         w[MLP_OUT_ROW_PARALLEL];
    #pragma HLS ARRAY_PARTITION variable=word complete dim=1
    #pragma HLS ARRAY_PARTITION variable=lane complete dim=1
    #pragma HLS ARRAY_PARTITION variable=w    complete dim=1
    W2_SLICE_INIT:
    for (int p = 0; p < MLP_OUT_ROW_PARALLEL; p++) {
        #pragma HLS UNROLL
        word[p] = 0; lane[p] = 0; w[p] = 0;
        g_w2_csc_ptr[p][0] = 0;
    }

    /* Walk W2 column-by-column; route each nonzero to its out_row slice, appending to that
     * slice's packed array and advancing that slice's per-column pointer. */
    W2_CSC_COL:
    for (int h = 0; h < hidden_dim; h++) {
        #pragma HLS LOOP_TRIPCOUNT min=MLP_HIDDEN_FEATURE_W_MAX max=MLP_HIDDEN_FEATURE_W_MAX
        const int c_begin = w2_col_ptr[h];
        const int c_end   = w2_col_ptr[h + 1];

        W2_CSC_NZ:
        for (int i = c_begin; i < c_end; i++) {
            #pragma HLS LOOP_TRIPCOUNT min=0 max=MLP_TC_W2COL_NNZ
            #pragma HLS PIPELINE II=1
#ifdef POT_SHIFT_MAC
            ap_uint<8> b  = (ap_uint<8>) w2_values[i];
            T_MlpWVal  v  = (T_MlpWVal)((ap_uint<4>)((b[7] << 3) | (b & 0x7)));
#else
            T_MlpWVal  v  = (T_MlpWVal) w2_values[i];
#endif
            const int  r  = (int) w2_row_idx[i];
            const int  p  = r / MLP_OUT_ROW_SLICE;   /* which contiguous out_row slice owns it */
            const int  lo = lane[p] * MLP_W2_LANE_BITS;
            word[p].range(lo + 3,  lo)     = (ap_uint<4>)  v;
            word[p].range(lo + 13, lo + 4) = (ap_uint<10>) (T_MlpW2Row) r;
            lane[p]++;
            if (lane[p] == MLP_W2_PACK) { g_w2_csc[p][w[p]++] = word[p]; word[p] = 0; lane[p] = 0; }
        }

        /* Per-column span in NONZERO units, per slice (contiguous packing across columns). */
        W2_COL_PTR:
        for (int p = 0; p < MLP_OUT_ROW_PARALLEL; p++) {
            #pragma HLS UNROLL
            g_w2_csc_ptr[p][h + 1] = w[p] * MLP_W2_PACK + lane[p];
        }
    }

    /* Flush each slice's final partial word. */
    W2_SLICE_FLUSH:
    for (int p = 0; p < MLP_OUT_ROW_PARALLEL; p++) {
        #pragma HLS UNROLL
        if (lane[p] != 0) g_w2_csc[p][w[p]] = word[p];
    }
}

/**
 * @brief H-driven second linear layer: Y = W2 . H, reformulated so H is read ONCE,
 *        SEQUENTIALLY, and scattered through W2's COLUMNS (CSC).
 *
 * Y[out,tok] = sum_h W2[out,h] * H[h,tok].  Walk hidden row h (H storage order, sequential)
 * and, per H[h] nonzero (token,val), scatter W2 column h's (out_row, weight) into
 * acc[out_row][token].  H arrives from `h_stream` rather than DDR, so it is consumed as the
 * producer emits it; acc holds the full Y (feature-major).  Used as the DATAFLOW consumer.
 * g_w2_csc / g_w2_csc_ptr are staged (read-only here) before the DATAFLOW region.
 *
 * The scatter's acc update has a data-dependent address but consecutive iterations (walking
 * a W2 column) target DISTINCT out_row, so the inter-iteration acc dependence is false
 * (asserted) — restoring II~1.
 *
 * @param[in]  h_stream     Hidden-activation token stream from the producer
 * @param[in]  feature_dim  Output feature dimension
 * @param[in]  hidden_dim   Hidden dimension
 * @param[in]  batch        Token count
 * @param[in]  scale        Requantization scale
 * @param[out] output       Y = W2 . H, feature-major int8
 */
static void spmm_dense_streaming(
    hls::stream<T_MlpHTok> &h_stream,
    int           feature_dim,
    int           hidden_dim,
    int           batch,
    T_Scale       scale,
    T_Activation* output
) {
#pragma HLS INLINE off
    /* Full Y accumulator (feature-major), PACKED int20 lanes per URAM word (token axis).
     * Split into P disjoint banks by out_row slice: acc[p] owns rows [p*SLICE, (p+1)*SLICE),
     * indexed LOCALLY (out_row - p*SLICE).  Distinct URAMs -> the P scatter lanes write in the
     * same II=1 iteration without a shared port. */
    static T_MlpAccWord acc[MLP_OUT_ROW_PARALLEL][MLP_OUT_ROW_SLICE][MLP_ACC_ROW_WORDS];
    #pragma HLS BIND_STORAGE variable=acc type=ram_t2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=1

    CLEAR_ACC_ROW:
    for (int r = 0; r < MLP_OUT_ROW_SLICE; r++) {
        #pragma HLS LOOP_TRIPCOUNT min=MLP_OUT_ROW_SLICE max=MLP_OUT_ROW_SLICE
        CLEAR_ACC_WORD:
        for (int wd = 0; wd < MLP_ACC_ROW_WORDS; wd++) {
            #pragma HLS LOOP_TRIPCOUNT min=MLP_ACC_ROW_WORDS max=MLP_ACC_ROW_WORDS
            #pragma HLS PIPELINE II=1
            CLEAR_ACC_PE:
            for (int p = 0; p < MLP_OUT_ROW_PARALLEL; p++) {
                #pragma HLS UNROLL
                acc[p][r][wd] = (T_MlpAccWord) 0;
            }
        }
    }

    /* Walk hidden rows in storage order; H nonzeros arrive from the stream. */
    HIDDEN_ROW:
    for (int h = 0; h < hidden_dim; h++) {
        #pragma HLS LOOP_TRIPCOUNT min=MLP_TC_HIDDEN_FEATURE max=MLP_TC_HIDDEN_FEATURE

        /* Per-slice W2 column span (nonzero units) and the longest across slices. */
        int  w_begin[MLP_OUT_ROW_PARALLEL], w_span[MLP_OUT_ROW_PARALLEL];
        #pragma HLS ARRAY_PARTITION variable=w_begin complete dim=1
        #pragma HLS ARRAY_PARTITION variable=w_span  complete dim=1
        int  max_span = 0;
        SPAN_PE:
        for (int p = 0; p < MLP_OUT_ROW_PARALLEL; p++) {
            #pragma HLS UNROLL
            w_begin[p] = g_w2_csc_ptr[p][h];
            w_span[p]  = g_w2_csc_ptr[p][h + 1] - w_begin[p];
            if (w_span[p] > max_span) max_span = w_span[p];
        }

        /* Count-first: read this column's HEADER (nnz count), then buffer exactly that many H
         * value tokens.  Buffering decouples the single H stream from the P per-slice scatter
         * walks (which have different lengths), so H is read ONCE and broadcast to all lanes. */
        const T_MlpHTok hdr = h_stream.read();
        const int row_nnz = hdr.count;

        T_Activation h_val_buf[MLP_TOKEN_W_MAX];
        int          h_wd_buf[MLP_TOKEN_W_MAX];   /* packed-acc word index (token / PACK) */
        int          h_ln_buf[MLP_TOKEN_W_MAX];   /* packed-acc lane (token % PACK) */
        #pragma HLS ARRAY_PARTITION variable=h_val_buf cyclic factor=2 dim=1
        #pragma HLS ARRAY_PARTITION variable=h_wd_buf  cyclic factor=2 dim=1
        #pragma HLS ARRAY_PARTITION variable=h_ln_buf  cyclic factor=2 dim=1
        H_TOKEN_LOAD:
        for (int s = 0; s < row_nnz; s++) {
            #pragma HLS LOOP_TRIPCOUNT min=MLP_TC_HIDDEN_NNZ max=MLP_TC_HIDDEN_NNZ
            #pragma HLS PIPELINE II=1
            T_MlpHTok t = h_stream.read();
            const int token = (int) t.token;
            h_val_buf[s] = t.val;
            h_wd_buf[s]  = token / MLP_ACC_PACK;
            h_ln_buf[s]  = token % MLP_ACC_PACK;
        }

        /* FUSED scatter: one II=1 pipeline over (token, wpos) pairs, P lanes UNROLLED in the
         * body.  Each lane p walks its OWN slice's span w_span[p] into its OWN acc bank acc[p]
         * -> the P read-modify-writes are on distinct URAMs and co-schedule in the same
         * iteration (the proven PARALLEL_HIDDEN_ROWS shape).  wpos runs to max_span; a lane
         * predicates on wpos < w_span[p] so shorter slices idle their tail.  Wall-clock per
         * column = row_nnz * max_span ~= row_nnz * (w_span / P) when the slices are balanced. */
        int          token_i = 0, wpos = 0;
        T_Activation hval = (T_Activation) 0;
        int          acc_wd = 0, acc_ln = 0;

        /* Per-lane W2-word cache + accumulator-forwarding register (same-word RMW bypass). */
        T_MlpW2Word  w2word[MLP_OUT_ROW_PARALLEL];
        int          cur_wword[MLP_OUT_ROW_PARALLEL];
        int          prev_lr[MLP_OUT_ROW_PARALLEL], prev_wd[MLP_OUT_ROW_PARALLEL];
        T_MlpAccWord prev_aw[MLP_OUT_ROW_PARALLEL];
        #pragma HLS ARRAY_PARTITION variable=w2word    complete dim=1
        #pragma HLS ARRAY_PARTITION variable=cur_wword complete dim=1
        #pragma HLS ARRAY_PARTITION variable=prev_lr   complete dim=1
        #pragma HLS ARRAY_PARTITION variable=prev_wd   complete dim=1
        #pragma HLS ARRAY_PARTITION variable=prev_aw   complete dim=1
        SCATTER_INIT:
        for (int p = 0; p < MLP_OUT_ROW_PARALLEL; p++) {
            #pragma HLS UNROLL
            cur_wword[p] = -1; prev_lr[p] = -1; prev_wd[p] = -1; prev_aw[p] = 0;
        }

        const int total = row_nnz * max_span;
        H_SCATTER:
        for (int idx = 0; idx < total; idx++) {
            #pragma HLS LOOP_TRIPCOUNT min=(MLP_TC_HIDDEN_NNZ*MLP_TC_W2COL_NNZ) max=(MLP_TC_HIDDEN_NNZ*MLP_TC_W2COL_NNZ)
            #pragma HLS PIPELINE II=1
            #pragma HLS DEPENDENCE variable=acc type=inter direction=RAW false
            #pragma HLS DEPENDENCE variable=acc type=inter direction=WAR false

            if (wpos == 0) {
                /* New token: fetch its buffered value + packed-acc address, reset W2 caches. */
                hval   = h_val_buf[token_i];
                acc_wd = h_wd_buf[token_i];
                acc_ln = h_ln_buf[token_i];
                SCATTER_WWORD_RESET:
                for (int p = 0; p < MLP_OUT_ROW_PARALLEL; p++) {
                    #pragma HLS UNROLL
                    cur_wword[p] = -1;   /* force W2 word re-fetch at the start of each token */
                }
            }

            PARALLEL_OUT_ROWS:
            for (int p = 0; p < MLP_OUT_ROW_PARALLEL; p++) {
                #pragma HLS UNROLL
                if (wpos < w_span[p]) {
                    const int w     = w_begin[p] + wpos;
                    const int wword = w / MLP_W2_PACK;
                    if (wword != cur_wword[p]) { w2word[p] = g_w2_csc[p][wword]; cur_wword[p] = wword; }

                    T_MlpWVal  weight;
                    T_MlpW2Row w2row;
                    w2_lane_from_word(w2word[p], w % MLP_W2_PACK, weight, w2row);
                    const int  local_row = (int) w2row - p * MLP_OUT_ROW_SLICE;  /* index into acc[p] */

                    /* Stage A: URAM read (or forward the previous write if same word). */
                    const bool   same_word = (local_row == prev_lr[p]) && (acc_wd == prev_wd[p]);
                    T_MlpAccWord aw = same_word ? prev_aw[p] : acc[p][local_row][acc_wd];

                    /* Stage B: extract lane, shift-add, insert lane, write back. */
                    T_MlpAcc a = (T_MlpAcc) acc_get(aw, acc_ln);
#ifdef POT_SHIFT_MAC
                    ap_uint<3> e   = weight.range(2, 0);
                    bool       neg = weight[3];
                    T_MlpAcc shifted = (T_MlpAcc) hval << e;
                    a += neg ? (T_MlpAcc)(-shifted) : shifted;
#else
                    a += weight * hval;
#endif
                    acc_set(aw, acc_ln, (T_MlpAccLane) a);
                    acc[p][local_row][acc_wd] = aw;

                    prev_lr[p] = local_row;
                    prev_wd[p] = acc_wd;
                    prev_aw[p] = aw;
                }
            }

            if (wpos + 1 == max_span) { wpos = 0; token_i++; }
            else                      { wpos++; }
        }
    }

    /* Requantize + write Y feature-major (unpack the acc lanes).  Global output row r maps to
     * slice p = r / SLICE, local row r % SLICE within that slice's acc bank. */
    WRITE_Y_ROW:
    for (int r = 0; r < feature_dim; r++) {
        #pragma HLS LOOP_TRIPCOUNT min=MLP_TC_FEATURE_OUT max=MLP_TC_FEATURE_OUT
        const int p         = r / MLP_OUT_ROW_SLICE;
        const int local_row = r % MLP_OUT_ROW_SLICE;
        WRITE_Y_TOK:
        for (int j = 0; j < batch; j++) {
            #pragma HLS LOOP_TRIPCOUNT min=MLP_TC_BATCH max=MLP_TC_BATCH
            #pragma HLS PIPELINE II=1
            T_MlpAcc a = (T_MlpAcc) acc_get(acc[p][local_row][j / MLP_ACC_PACK], j % MLP_ACC_PACK);
            output[r * batch + j] = requant_to_int8(a, scale);
        }
    }
}

/**
 * @brief DATAFLOW region: overlap the streaming stage-1 producer and stage-2 consumer.
 *
 * Isolated in its own function so #pragma HLS DATAFLOW scopes ONLY the two streaming
 * stages (W2 staging must complete before this, so it stays in the caller).
 */
static void mlp_stage12_dataflow(
    T_MlpBankWord *input_packed,
    T_Activation  *w1_values,
    T_MlpIndex    *w1_col_idx,
    int           *w1_row_ptr,
    int            feature_dim,
    int            hidden_dim,
    int            batch,
    T_Scale        scale_fc1,
    T_Scale        scale_fc2,
    T_Activation  *output
) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    /* Single H channel (header + value tokens).  Deep FIFO so the producer can run ahead
     * while the consumer is busy scattering (the consumer is the slower stage). */
    hls::stream<T_MlpHTok> h_stream;
    #pragma HLS STREAM variable=h_stream depth=512

    fused_sdmm_relu_streaming(
        input_packed,
        w1_values, w1_col_idx, w1_row_ptr,
        hidden_dim, feature_dim, batch,
        scale_fc1,
        h_stream
    );
    spmm_dense_streaming(
        h_stream,
        feature_dim, hidden_dim, batch,
        scale_fc2,
        output
    );
}


void mlp(
    T_MlpBankWord *input_packed,
    int            feature_dim,
    int            hidden_dim,
    int            batch,
    T_Activation*  w1_values,
    T_MlpIndex*    w1_col_idx,
    int*           w1_row_ptr,
    T_Activation*  w2_values,
    T_MlpIndex*    w2_col_idx,
    int*           w2_row_ptr,
    const T_Scale* scales,
    T_Activation*  output
) {
#ifndef __SYNTHESIS__
    assert(hidden_dim <= MLP_HIDDEN_FEATURE_W_MAX && "hidden_dim exceeds MLP_HIDDEN_FEATURE_W_MAX");
#endif

    /* Stage the whole W2 (CSC) into on-chip URAM once, BEFORE the DATAFLOW region: the
     * consumer reads g_w2_csc read-only inside the region (read-only shared access is
     * allowed; a producer/consumer on a shared static RAM would not be). */
    stage_w2_csc(w2_values, w2_col_idx, w2_row_ptr, hidden_dim);

    /* Overlap stage-1 (W1 MAC, produces H) and stage-2 (H-driven scatter, consumes H) as a
     * producer/consumer pair.  H streams on-chip so stage-2 scatters row h as soon as
     * stage-1 emits it -> the two ~22.5ms / ~38.7ms stages overlap instead of summing. */
    mlp_stage12_dataflow(
        input_packed,
        w1_values, w1_col_idx, w1_row_ptr,
        feature_dim, hidden_dim, batch,
        scales[SCALE_FC1_IDX], scales[SCALE_FC2_IDX],
        output
    );
}
