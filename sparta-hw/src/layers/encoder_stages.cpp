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
static_assert(ENC_HIDDEN_W_MAX == MHA_TILES * MHA_H_TILE_FEATURES,
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

/* Packed per-head attention result, all heads, BEFORE the output projection: that projection reads
 * every feature, so the heads are gathered here first. */
static T_MhaBankWord g_attn_heads[MHA_BANK_WORDS];

/* RMSNorm output, packed.  One buffer for both halves of the layer: they run in sequence. */
static T_MhaBankWord g_norm_input[MHA_BANK_WORDS_ALLOC];

/* The residual stream, in the same packed form as everything else on chip.  Two buffers, because
 * each residual add needs its two operands separately, and they alternate roles:
 *
 *   g_stream      layer input  ->  (MLP branch output)  ->  layer output
 *   g_stream_mid  (attention branch output)  ->  stream after the attention residual
 *
 * so each holds a branch result until the add that consumes it, then the stream itself.  Keeping
 * them packed is what lets RMSNorm, the residual adds and the tails move MHA_BANK_PACK tokens per
 * cycle; external memory is read once at the start and written once at the end. */
static T_MhaBankWord g_stream[MHA_BANK_WORDS];
static T_MhaBankWord g_stream_mid[MHA_BANK_WORDS];


/**
 * @brief Move the layer activation between external memory and the on-chip buffer.
 *
 * The external buffer carries the SAME packed form everything on chip uses: MHA_BANK_ROW_WORDS
 * words per feature, MHA_BANK_PACK tokens in each, with the trailing slots of a feature's last word
 * zero.  So this is a flat copy, one word per cycle.
 *
 * It used to cross the port as individual int8 bytes, and that was expensive in a way csynth did not
 * show: the port is 64 bits wide, so seven eighths of every transfer was empty, and the byte
 * requests did not merge into bursts.  Measured on the board at ~8 cycles per byte against the one
 * the scheduler assumed -- 310 k cycles to read 38 400 bytes and 123 k to write them, where csynth
 * predicted 43 k for each.  As words it is 5 376 transfers instead of 38 400.
 *
 * The host now does the packing, which is where it belongs: it is a numpy reshape there, and the
 * padding it has to add is the same padding the kernel was generating anyway.
 */
static void copy_activation_in (const T_MhaBankWord *src, T_MhaBankWord *dst, int words) {
#pragma HLS INLINE off
    ACT_COPY_IN:
    for (int w = 0; w < words; w++) {
        #pragma HLS LOOP_TRIPCOUNT min=MHA_BANK_WORDS max=MHA_BANK_WORDS
        #pragma HLS PIPELINE II=1
        dst[w] = src[w];
    }
}

static void copy_activation_out (const T_MhaBankWord *src, T_MhaBankWord *dst, int words) {
#pragma HLS INLINE off
    ACT_COPY_OUT:
    for (int w = 0; w < words; w++) {
        #pragma HLS LOOP_TRIPCOUNT min=MHA_BANK_WORDS max=MHA_BANK_WORDS
        #pragma HLS PIPELINE II=1
        dst[w] = src[w];
    }
}


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
 * @param[in]  bank        This row block's packed entries, from load_weight_slice.
 * @param[in]  counts      Its nonzeros per (column block, row).
 * @param[in]  cursor0     Its first entry's slot within its packed word, per column block.
 */
static void layer_rows (
    int           mode,
    int           row_base,
    int           rows,
    int           tokens_dim,
    T_Scale       scale,
    const T_MhaSliceBank   bank,
    const T_MhaSliceCount  counts,
    const T_MhaSliceCursor cursor0
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

        tiled_rows_compute(i, tokens_dim, act_base, bank, counts, cursor0);

        /* Requantize the accumulated row once, MHA_BANK_PACK values per cycle, into row_q; the
         * tails then only move bytes.  The six tails are mutually exclusive, so giving each its own
         * requantize units built 27 multipliers where 8 suffice.  The MLP's hidden output is packed
         * here as well, since its destination takes MHA_BANK_PACK values per word anyway.
         *
         * The ReLU has to test the ACCUMULATOR, not the requantized value.  requant_to_int8 starts
         * by casting to T_Scale, which has 16 integer bits, while the accumulators are int32 sums
         * over feature_dim products and routinely exceed that range, so the cast wraps and a large
         * negative accumulator can come back positive.  The tails that apply a ReLU therefore keep
         * the decision on the accumulator, as they did before this pass was shared. */
        T_Activation row_q[MHA_TOKEN_W_MAX];
        T_Activation row_relu[MHA_TOKEN_W_MAX];
        #pragma HLS ARRAY_PARTITION variable=row_q    cyclic factor=MHA_BANK_PACK dim=1
        #pragma HLS ARRAY_PARTITION variable=row_relu cyclic factor=MHA_BANK_PACK dim=1

        REQUANT_ROW:
        for (int w = 0; w < row_words; w++) {
            #pragma HLS LOOP_TRIPCOUNT min=MHA_BANK_ROW_WORDS max=MHA_BANK_ROW_WORDS
            #pragma HLS PIPELINE II=1
            T_MhaBankWord relu_word  = 0;   /* ReLU'd, for the MLP's hidden output */
            T_MhaBankWord quant_word = 0;   /* as requantized, for the two dense outputs */
            REQUANT_LANE:
            for (int p = 0; p < MHA_BANK_PACK; p++) {
                #pragma HLS UNROLL
                const int    j = w * MHA_BANK_PACK + p;
                T_Activation q    = (T_Activation) 0;   /* as requantized: V, out_proj, MLP output */
                T_Activation relu = (T_Activation) 0;   /* ReLU'd: K', Q', MLP hidden */
                if (j < tokens_dim) {
                    /* This sweep is also where the engine's lanes are drained, which is why there
                     * is no merge pass: merging on its own re-read the row this loop reads anyway,
                     * 12 cycles per output row.  The j < tokens_dim guard makes this cover exactly
                     * the tokens the engine writes, which is what tiled_lane_sum requires. */
                    const T_MhaAcc acc = tiled_lane_sum(j);
                    q = requant_to_int8(acc, scale);
                    relu = (acc > (T_MhaAcc) 0) ? q : (T_Activation) 0;
                    row_q[j]    = q;
                    row_relu[j] = relu;
                }
                relu_word.range(p * MHA_ACT_ELEM_BITS + MHA_ACT_ELEM_BITS - 1,
                                p * MHA_ACT_ELEM_BITS) = (ap_uint<MHA_ACT_ELEM_BITS>) relu;
                quant_word.range(p * MHA_ACT_ELEM_BITS + MHA_ACT_ELEM_BITS - 1,
                                 p * MHA_ACT_ELEM_BITS) = (ap_uint<MHA_ACT_ELEM_BITS>) q;
            }
            /* The three packed destinations are written here, so those tails cost nothing more. */
            if (mode == ROWS_MLP_H) {
                g_act_tile[h_tile][MHA_H_TILE_BASE + (h_local + i) * MHA_BANK_ROW_WORDS + w] = relu_word;
            } else if (mode == ROWS_MHA_O) {
                g_stream_mid[(row_base + i) * MHA_BANK_ROW_WORDS + w] = quant_word;
            } else if (mode == ROWS_MLP_Y) {
                g_stream[(row_base + i) * MHA_BANK_ROW_WORDS + w] = quant_word;
            }
        }

        if (mode == ROWS_MHA_K) {
            /* Sparse emission appends, so it stays one token per cycle. */
            TAIL_K:
            for (int j = 0; j < tokens_dim; j++) {
                #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_TOKEN max=MHA_TC_TOKEN
                #pragma HLS PIPELINE II=1
                const T_Activation k_value = row_relu[j];
                if (k_value > (T_Activation) 0) {
#ifndef __SYNTHESIS__
                    assert(nnz < MHA_HEAD_NNZ_MAX && "K' CSR overflow");
#endif
                    g_k_val[nnz] = k_value;
                    g_k_col[nnz] = (T_MhaHeadIndex) j;
                    nnz++;
                }
            }
            g_k_ptr[i + 1] = nnz;
        } else if (mode == ROWS_MHA_Q || mode == ROWS_MHA_V) {
            /* Q' goes to the transpose scratch (ReLU'd) and V to this head's dense rows.  Both are
             * read at arbitrary token positions later, so they stay byte-addressed and are split
             * across MHA_BANK_PACK memories instead of packed. */
            TAIL_BYTES:
            for (int w = 0; w < row_words; w++) {
                #pragma HLS LOOP_TRIPCOUNT min=MHA_BANK_ROW_WORDS max=MHA_BANK_ROW_WORDS
                #pragma HLS PIPELINE II=1
                TAIL_BYTES_LANE:
                for (int p = 0; p < MHA_BANK_PACK; p++) {
                    #pragma HLS UNROLL
                    const int j = w * MHA_BANK_PACK + p;
                    if (j < tokens_dim) {
                        if (mode == ROWS_MHA_Q) {
                            g_q_scratch[i * tokens_dim + j] = row_relu[j];
                        } else {
                            g_v_head[i * MHA_V_ROW_STRIDE + j] = row_q[j];
                        }
                    }
                }
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
            for (int head_feature = 0; head_feature < MHA_FEATURE_HEAD; head_feature++) {
                #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_FEATURE_HEAD max=MHA_TC_FEATURE_HEAD
                #pragma HLS PIPELINE II=1
                const T_Activation q_value = g_q_scratch[head_feature * tokens_dim + j];
                if (q_value > (T_Activation) 0) {
#ifndef __SYNTHESIS__
                    assert(nnz < MHA_HEAD_NNZ_MAX && "Q' CSR overflow");
#endif
                    g_q_val[nnz] = q_value;
                    g_q_col[nnz] = (T_MhaHeadIndex) head_feature;
                    nnz++;
                }
            }
            g_q_ptr[j + 1] = nnz;
        }
    }
}


/* ------------------------------------------------------------------------------------
 * The stage loop, as a loader half and a compute half
 *
 * The two halves are concurrent processes, so stage N+1's weight slice is fetched while stage N is
 * still being computed (docs/plans/06-load-compute-overlap.md).  Three constraints follow, and all
 * three are why this is one loop of two calls rather than the readable single loop it replaces:
 *
 *  - ONE loop over every stage.  A DATAFLOW region cannot share hardware with anything outside it,
 *    so an engine inside a per-weight region is a SECOND engine; marking two MLP loops was measured
 *    at 3 copies of layer_rows and 2x the LUT.  One region means one loop.
 *  - A body of nothing but calls.  Every per-stage choice -- which weight, which tail, which scale,
 *    which boundary work -- therefore moves inside the two halves, which derive it from `stage`.
 *  - The slice reaches the engine as arguments, not globals, so HLS can place the second copy that
 *    lets one half write while the other reads.
 * ------------------------------------------------------------------------------------ */

/**
 * @brief What stage `stage` computes: which weight, which row block of it, and whether that block
 *        is the weight's first (the point at which its DDR bases must be re-read).
 *
 * Derived from the stage index alone, so the two halves agree on the schedule without exchanging
 * anything -- the loader is a stage ahead and cannot be told where it is.
 */
static void stage_plan (int stage, int o_first, int h_first, int y_first,
                        int &mode, int &head, int &row_base, int &first) {
#pragma HLS INLINE
    int block;
    if (stage < o_first) {
        const int phase = stage % 3;
        mode  = (phase == 0) ? ROWS_MHA_K : (phase == 1) ? ROWS_MHA_Q : ROWS_MHA_V;
        head  = stage / 3;
        block = head;
    } else if (stage < h_first) {
        mode = ROWS_MHA_O; head = 0; block = stage - o_first;
    } else if (stage < y_first) {
        mode = ROWS_MLP_H; head = 0; block = stage - h_first;
    } else {
        mode = ROWS_MLP_Y; head = 0; block = stage - y_first;
    }
    row_base = block * MHA_FEATURE_HEAD;
    first    = (block == 0);
}

/**
 * @brief Loader half: fetch this stage's weight slice into the banks.
 *
 * Reads DDR and the loader's own position state and nothing else, so it depends on nothing the
 * compute half produces.  That independence is what lets it run a stage ahead.
 *
 * `enable` is clear on the loop's last round, where the stage one ahead does not exist.  An enable
 * rather than a call inside an `if`, because a conditional call makes HLS specialise the loop body
 * for that case and duplicate what it contains.
 */
static void stage_load (
    int enable,
    int stage, int o_first, int h_first, int y_first,
    int feature_dim, int hidden_dim,
    T_MhaTileWord *wq_bank, T_MhaCountWord *wq_counts,
    T_MhaTileWord *wk_bank, T_MhaCountWord *wk_counts,
    T_MhaTileWord *wv_bank, T_MhaCountWord *wv_counts,
    T_MhaTileWord *wo_bank, T_MhaCountWord *wo_counts,
    T_MhaTileWord *w1_bank, T_MhaCountWord *w1_counts,
    T_MhaTileWord *w2_bank, T_MhaCountWord *w2_counts,
    T_MhaSliceBank bank, T_MhaSliceCount counts, T_MhaSliceCursor cursor0
) {
#pragma HLS INLINE off
    #pragma HLS ALLOCATION instances=load_weight_slice limit=1 function
    #pragma HLS ALLOCATION instances=load_weight_bases limit=1 function

    if (!enable) { return; }

    int mode, head, row_base, first;
    stage_plan(stage, o_first, h_first, y_first, mode, head, row_base, first);

    /* The counts' row stride is the weight's output row count: the feature width for the four
     * projections and the MLP's second linear, the hidden width for its first. */
    if (mode == ROWS_MHA_K) {
        if (first) { load_weight_bases(wk_counts, SPMM_W_K); }
        load_weight_slice(wk_bank, wk_counts, feature_dim, SPMM_W_K, row_base, MHA_FEATURE_HEAD,
                          bank, counts, cursor0);
    } else if (mode == ROWS_MHA_Q) {
        if (first) { load_weight_bases(wq_counts, SPMM_W_Q); }
        load_weight_slice(wq_bank, wq_counts, feature_dim, SPMM_W_Q, row_base, MHA_FEATURE_HEAD,
                          bank, counts, cursor0);
    } else if (mode == ROWS_MHA_V) {
        if (first) { load_weight_bases(wv_counts, SPMM_W_V); }
        load_weight_slice(wv_bank, wv_counts, feature_dim, SPMM_W_V, row_base, MHA_FEATURE_HEAD,
                          bank, counts, cursor0);
    } else if (mode == ROWS_MHA_O) {
        if (first) { load_weight_bases(wo_counts, SPMM_W_O); }
        load_weight_slice(wo_bank, wo_counts, feature_dim, SPMM_W_O, row_base, MHA_FEATURE_HEAD,
                          bank, counts, cursor0);
    } else if (mode == ROWS_MLP_H) {
        if (first) { load_weight_bases(w1_counts, SPMM_W_MLP1); }
        load_weight_slice(w1_bank, w1_counts, hidden_dim, SPMM_W_MLP1, row_base, MHA_FEATURE_HEAD,
                          bank, counts, cursor0);
    } else {
        if (first) { load_weight_bases(w2_counts, SPMM_W_MLP2); }
        load_weight_slice(w2_bank, w2_counts, feature_dim, SPMM_W_MLP2, row_base, MHA_FEATURE_HEAD,
                          bank, counts, cursor0);
    }
}

/**
 * @brief Compute half: the boundary work this stage opens with, then the engine over its row block.
 *
 * RMSNorm, both residual adds and the activation tiling live here rather than between loops because
 * only one process may touch g_stream / g_stream_mid / g_act_tile if those are to stay ordinary
 * memories instead of becoming channels.  Each of the three phases reads a different activation, so
 * each opens with the work that produces it.
 */
static void stage_compute (
    int stage, int n_stages, int o_first, int h_first, int y_first,
    int tokens_dim, int feature_dim,
    const T_Scale *scales,
    const T_MhaSliceBank bank, const T_MhaSliceCount counts, const T_MhaSliceCursor cursor0
) {
#pragma HLS INLINE off
    #pragma HLS ALLOCATION instances=layer_rows            limit=1 function
    #pragma HLS ALLOCATION instances=load_activation_tiles limit=1 function
    #pragma HLS ALLOCATION instances=rmsnorm               limit=1 function
    #pragma HLS ALLOCATION instances=residual_add          limit=1 function

    #pragma HLS BIND_STORAGE    variable=g_v_head     type=ram_t2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=g_v_head cyclic factor=MHA_BANK_PACK dim=1
    #pragma HLS BIND_STORAGE    variable=g_q_scratch  type=ram_t2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=g_q_scratch cyclic factor=MHA_BANK_PACK dim=1
    #pragma HLS BIND_STORAGE    variable=g_k_val      type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE    variable=g_k_col      type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE    variable=g_q_val      type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE    variable=g_q_col      type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE    variable=g_attn_heads type=ram_t2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=g_attn_heads cyclic factor=MHA_BANK_CYCLIC dim=1
    #pragma HLS BIND_STORAGE    variable=g_stream_mid type=ram_t2p impl=uram
    #pragma HLS BIND_STORAGE    variable=g_norm_input type=ram_t2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=g_norm_input cyclic factor=MHA_BANK_CYCLIC dim=1

    int mode, head, row_base, first;
    stage_plan(stage, o_first, h_first, y_first, mode, head, row_base, first);

    const int stream_words = feature_dim * MHA_BANK_ROW_WORDS;

    if (stage == 0) {
        /* The attention input: the layer input, normalized. */
        rmsnorm(reinterpret_cast<const T_RmsBankWord*>(g_stream), tokens_dim, feature_dim,
                scales[SCALE_ATT_RMSNORM_OUT_INV_IDX],
                reinterpret_cast<T_RmsBankWord*>(g_norm_input));
        load_activation_tiles(g_norm_input, 0, MHA_TILE_WORDS);
    } else if (stage == o_first) {
        /* The output projection reads the attention result, not the layer input. */
        load_activation_tiles(g_attn_heads, 0, MHA_TILE_WORDS);
    } else if (stage == h_first) {
        /* Between the halves: close the attention residual, then normalize for the MLP. */
        residual_add(g_stream, g_stream_mid, stream_words,
                     scales[SCALE_ATT_RESIDUAL_IDX], scales[SCALE_ATT_BRANCH_RATIO_IDX],
                     g_stream_mid);
        rmsnorm(reinterpret_cast<const T_RmsBankWord*>(g_stream_mid), tokens_dim, feature_dim,
                scales[SCALE_FF_RMSNORM_OUT_INV_IDX],
                reinterpret_cast<T_RmsBankWord*>(g_norm_input));
        load_activation_tiles(g_norm_input, 0, MHA_TILE_WORDS);
    }

    const T_Scale scale = (mode == ROWS_MHA_K) ? scales[SCALE_K_IDX]
                        : (mode == ROWS_MHA_Q) ? scales[SCALE_Q_IDX]
                        : (mode == ROWS_MHA_V) ? scales[SCALE_V_IDX]
                        : (mode == ROWS_MHA_O) ? scales[SCALE_LINEAR_OUT_IDX]
                        : (mode == ROWS_MLP_H) ? scales[SCALE_FC1_IDX]
                                               : scales[SCALE_FC2_IDX];

    layer_rows(mode, row_base, MHA_FEATURE_HEAD, tokens_dim, scale, bank, counts, cursor0);

    /* V is the last of a head's three projections, so the head's attention runs here.  Called
     * unconditionally with an enable: a call inside an `if` makes HLS specialise the body for that
     * case, which duplicates layer_rows. */
    head_core(mode == ROWS_MHA_V, head, tokens_dim, scales[SCALE_DIV_OUT_IDX],
              g_k_val, g_k_col, g_k_ptr,
              g_q_val, g_q_col, g_q_ptr,
              g_v_head, g_attn_heads);

    /* The feed-forward residual closes the layer. */
    if (stage == n_stages - 1) {
        residual_add(g_stream_mid, g_stream, stream_words,
                     scales[SCALE_FF_RESIDUAL_IDX], scales[SCALE_FF_BRANCH_RATIO_IDX],
                     g_stream);
    }
}

/**
 * @brief Every stage of the layer, the loader running one ahead of the engine.
 *
 * `n_stages` is an argument rather than an expression because a DATAFLOW loop's bound has to be a
 * constant or a function argument.
 */
static void stage_loop (
    int n_stages, int o_first, int h_first, int y_first,
    int tokens_dim, int feature_dim, int hidden_dim,
    T_MhaTileWord *wq_bank, T_MhaCountWord *wq_counts,
    T_MhaTileWord *wk_bank, T_MhaCountWord *wk_counts,
    T_MhaTileWord *wv_bank, T_MhaCountWord *wv_counts,
    T_MhaTileWord *wo_bank, T_MhaCountWord *wo_counts,
    T_MhaTileWord *w1_bank, T_MhaCountWord *w1_counts,
    T_MhaTileWord *w2_bank, T_MhaCountWord *w2_counts,
    const T_Scale *scales
) {
#pragma HLS INLINE off
#ifndef __SYNTHESIS__
    assert((n_stages % 2) == 0 && "the skewed stage loop steps by two");
#endif
    /* TWO slices, used alternately, and declared here rather than inside the loop body.
     *
     * The point of the skew is that a slice fetched in one round is computed in the NEXT, so the
     * two halves of a round touch DIFFERENT slices and are independent -- which is the only way
     * they can run at the same time.  Letting HLS manage a single slice as a channel cannot express
     * that: a channel hands data from producer to consumer WITHIN a round, so the consumer always
     * waits for the producer and the round costs load + compute.  That is what the first attempt
     * did, and the board measured it: no gain and a small loss (plan 06).
     *
     * The loop therefore steps by two, so which slice each half uses is fixed at compile time.  A
     * run-time choice would index a partitioned array with a run-time value, which silently undoes
     * the partitioning.
     *
     * A bank is 64-bit words, MHA_TILE_BANK_WORDS deep: one URAM each (72 b x 4096), where BRAM
     * would need ~12 BRAM18 per bank for the same data (18 b per port, 1024 deep). */
    T_MhaSliceBank   bank_a,    bank_b;
    T_MhaSliceCount  counts_a,  counts_b;
    T_MhaSliceCursor cursor0_a, cursor0_b;
    #pragma HLS BIND_STORAGE    variable=bank_a type=ram_t2p impl=uram
    #pragma HLS BIND_STORAGE    variable=bank_b type=ram_t2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=bank_a    complete dim=1
    #pragma HLS ARRAY_PARTITION variable=bank_b    complete dim=1
    #pragma HLS ARRAY_PARTITION variable=counts_a  complete dim=1
    #pragma HLS ARRAY_PARTITION variable=counts_b  complete dim=1
    #pragma HLS ARRAY_PARTITION variable=cursor0_a complete dim=1
    #pragma HLS ARRAY_PARTITION variable=cursor0_b complete dim=1

    /* Prologue: the first slice, with nothing to compute alongside it. */
    stage_load(1, 0, o_first, h_first, y_first, feature_dim, hidden_dim,
               wq_bank, wq_counts, wk_bank, wk_counts, wv_bank, wv_counts,
               wo_bank, wo_counts, w1_bank, w1_counts, w2_bank, w2_counts,
               bank_a, counts_a, cursor0_a);

    STAGE:
    for (int stage = 0; stage < n_stages; stage += 2) {
        #pragma HLS LOOP_TRIPCOUNT min=54 max=54
        #pragma HLS LOOP_FLATTEN off

        /* Fetch stage+1 into B while stage runs out of A, then the same with the slices swapped.
         * Nothing in either pair depends on the other, which is what allows them to overlap. */
        stage_load(1, stage + 1, o_first, h_first, y_first, feature_dim, hidden_dim,
                   wq_bank, wq_counts, wk_bank, wk_counts, wv_bank, wv_counts,
                   wo_bank, wo_counts, w1_bank, w1_counts, w2_bank, w2_counts,
                   bank_b, counts_b, cursor0_b);
        stage_compute(stage, n_stages, o_first, h_first, y_first, tokens_dim, feature_dim,
                      scales, bank_a, counts_a, cursor0_a);

        stage_load(stage + 2 < n_stages, stage + 2, o_first, h_first, y_first,
                   feature_dim, hidden_dim,
                   wq_bank, wq_counts, wk_bank, wk_counts, wv_bank, wv_counts,
                   wo_bank, wo_counts, w1_bank, w1_counts, w2_bank, w2_counts,
                   bank_a, counts_a, cursor0_a);
        stage_compute(stage + 1, n_stages, o_first, h_first, y_first, tokens_dim, feature_dim,
                      scales, bank_b, counts_b, cursor0_b);
    }
}


void encoder_layer_stages(
    const T_MhaBankWord *input,
    int                 tokens_dim,
    int                 feature_dim,
    int                 hidden_dim,
    T_MhaTileWord *wq_bank, T_MhaCountWord *wq_counts,
    T_MhaTileWord *wk_bank, T_MhaCountWord *wk_counts,
    T_MhaTileWord *wv_bank, T_MhaCountWord *wv_counts,
    T_MhaTileWord *wo_bank, T_MhaCountWord *wo_counts,
    T_MhaTileWord *w1_bank, T_MhaCountWord *w1_counts,
    T_MhaTileWord *w2_bank, T_MhaCountWord *w2_counts,
    const T_Scale      *scales,
    T_MhaBankWord     *output,
    int                 first_layer,
    int                 last_layer
) {
#pragma HLS INLINE off
#ifndef __SYNTHESIS__
    assert(tokens_dim <= MHA_TOKEN_W_MAX && "tokens_dim exceeds MHA_TOKEN_W_MAX");
    assert(hidden_dim <= ENC_HIDDEN_W_MAX && "hidden_dim exceeds ENC_HIDDEN_W_MAX");
#endif
    #pragma HLS BIND_STORAGE   variable=g_stream    type=ram_t2p impl=uram

    /* Stage layout: 3 per head for K/Q/V, then the output projection's row blocks, then the two
     * MLP linears' row blocks. */
    const int qkv_stages = 3 * MHA_N_HEADS;
    const int o_stages   = feature_dim / MHA_FEATURE_HEAD;
    const int h_stages   = hidden_dim  / MHA_FEATURE_HEAD;
    const int y_stages   = feature_dim / MHA_FEATURE_HEAD;
    const int o_first    = qkv_stages;
    const int h_first    = o_first + o_stages;
    const int y_first    = h_first + h_stages;
    const int n_stages   = y_first + y_stages;

    /* g_stream keeps its contents between calls, and the previous layer left its output there, so
     * only the first layer reads the activation from external memory and only the last writes back.
     * Both conversions cost one transfer per byte, the ports carrying bytes. */
    if (first_layer) {
        copy_activation_in(input, g_stream, feature_dim * MHA_BANK_ROW_WORDS);
    }

    stage_loop(n_stages, o_first, h_first, y_first, tokens_dim, feature_dim, hidden_dim,
               wq_bank, wq_counts, wk_bank, wk_counts, wv_bank, wv_counts,
               wo_bank, wo_counts, w1_bank, w1_counts, w2_bank, w2_counts,
               scales);

    if (last_layer) {
        copy_activation_out(g_stream, output, feature_dim * MHA_BANK_ROW_WORDS);
    }
}
