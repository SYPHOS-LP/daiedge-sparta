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

// Scratch buffer sizes must match the model dimensions.
static_assert(MHA_SCR_QP_VAL_CNT == MHA_MAX_QK_NNZ,   "Qp val cnt != MHA_MAX_QK_NNZ");
static_assert(MHA_SCR_KP_VAL_CNT == MHA_MAX_QK_NNZ,   "Kp val cnt != MHA_MAX_QK_NNZ");
static_assert(MHA_SCR_V_CNT      == MHA_FEATURE_W_MAX * MHA_TOKEN_W_MAX,   "V cnt != D*N");
static_assert(MHA_SCR_A_CNT      == MHA_FEATURE_HEAD * MHA_FEATURE_HEAD, "A cnt != d_head^2");

// parallel_out_rows_compute manually unrolls the 8 packed lanes of a bank word.
static_assert(MHA_BANK_PACK == 8, "code assumes MHA_BANK_PACK == 8");

// Per-projection output-row parallelism.  Q'/K' were P=4 (a 66->60ms proj win) then dropped
// to P=2 when the PESSIMISTIC LUT estimate (88-95%) said the H-driven MLP no longer fit.
// RESTORED to 4: real Vivado LUT is only ~52% (the estimate was wrong), so the row-parallel
// budget is back.  Parallelizes the 768 output rows (768/4 divides evenly) -- the RIGHT
// dimension for the small-N model (token-unroll failed, rows are plentiful).
#define MHA_PROJ_P     4
#define MHA_PROJ_P_QK  4

/* Global weight buffers shared between all projections, burst loaded from DDR. */
static T_MhaWVal g_weight_val_buffer[MHA_MAX_W_TOTAL_NNZ];
static T_MhaWCol g_weight_col_buffer[MHA_MAX_W_TOTAL_NNZ];
static int       g_weight_rptr_buffer[MHA_FEATURE_W_MAX + 1];

/* Global buffer for input activation replicas used by the projections. */
static T_MhaBankWord g_activation_replicas[MHA_PROJ_P_QK][MHA_BANK_WORDS];

/**
 * @brief Burst load projection weights from DDR into the shared on-chip buffers.
 * 
 * This component also typecasts to the appropriate bitlength, for better 
 * resource efficiency.
 *
 * @param[in] w_val       Weight values in DDR.
 * @param[in] w_col       Weight column indices in DDR.
 * @param[in] w_ptr       Weight row pointers in DDR.
 * @param[in] feature_dim Feature dimension.
 */
static void load_weight_from_ddr (T_Activation* w_val, T_MhaIndex* w_col, int* w_ptr, int feature_dim) {
    LOAD_ROW_POINTER:
    for (int i = 0; i <= feature_dim; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=(MHA_TC_FEATURE+1) max=(MHA_TC_FEATURE+1)
        #pragma HLS PIPELINE II=1
        g_weight_rptr_buffer[i] = w_ptr[i];
    }
    // total_nz == w_ptr[feature_dim], already loaded above as the loop's last
    // iteration -- read the on-chip copy instead of a second, isolated DDR access.
    // A bare one-off scalar m_axi read outside any pipelined loop is a documented
    // "fine in C-sim, hangs on real hardware" HLS pattern (see the KR260 bring-up
    // changelog); ported from sparta-hw-armagedon-kr260-v3 (9df179a).
    int total_nz = g_weight_rptr_buffer[feature_dim];

    LOAD_VALUES_AND_COLUMNS:
    for (int i = 0; i < total_nz; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_FEATURE*MHA_TC_W_NNZ max=MHA_TC_FEATURE*MHA_TC_W_NNZ
        #pragma HLS PIPELINE II=1
        /* Typecast to 4 and 10 bits. */
#ifdef POT_SHIFT_MAC
        /* POT weight: DDR byte is {sign(bit7), exponent(bits0..6)}; repack to the
         * on-chip {sign(bit3), exponent(bits0..2)} the shift-MAC reads. */
        ap_uint<8> b = (ap_uint<8>) w_val[i];
        g_weight_val_buffer[i] = (T_MhaWVal)((ap_uint<4>)((b[7] << 3) | (b & 0x7)));
#else
        g_weight_val_buffer[i] = (T_MhaWVal) w_val[i];
#endif
        g_weight_col_buffer[i] = (T_MhaWCol) w_col[i];
    }
}

/**
 * @brief Broadcast input activation into the first ACTIVE_P of the shared URAM replicas.
 *
 * @tparam P  Number of replicas created.
 *
 * @param[in]  input           Input activations.
 * @param[out] input_replicas  The shared replica buffer.
 */
template <int P>
static void replicate_input_activations (T_MhaBankWord* input, T_MhaBankWord input_replicas[MHA_PROJ_P_QK][MHA_BANK_WORDS]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=input_replicas complete dim=1
#pragma HLS ARRAY_PARTITION variable=input_replicas cyclic factor=MHA_BANK_CYCLIC dim=2
#pragma HLS BIND_STORAGE    variable=input_replicas type=ram_t2p impl=uram

    REPLICATE_INPUT:
    for (int word_idx = 0; word_idx < MHA_BANK_WORDS; word_idx++) {
        #pragma HLS LOOP_TRIPCOUNT min=MHA_BANK_WORDS max=MHA_BANK_WORDS
        #pragma HLS PIPELINE II=1
        T_MhaBankWord input_val = input[word_idx];

        for (int rep_idx = 0; rep_idx < P; rep_idx++) {
            #pragma HLS UNROLL
            input_replicas[rep_idx][word_idx] = input_val;
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
        /* POT weight is {sign(bit3), exponent(bits0..2)}; the multiply-accumulate is a
         * shift by the stored exponent plus a sign (the compiler emits log2|w| directly). */
        ap_uint<3> e   = weight.range(2, 0);
        bool       neg = weight[3];
        T_MhaAcc shifted = (T_MhaAcc) input_word << e;
        out_row[token_idx] += neg ? (T_MhaAcc)(-shifted) : shifted;
#else
        /* Multiply - Accumulate */
        out_row[token_idx] += weight * input_word;
#endif
    }
}

/**
 * @brief Processing engine for P output rows in parallel, performing the Gustavson
 * algorithm for Sparse-Dense Matrix Multiplication, between the sparse weights and
 * the dense activation. Output buffer is properly partitioned to allow for P parallel
 * writes, while the input activation is replicated P times to allow for P parallel reads.
 * 
 * The input buffer resides in URAM and the 8-bit values it consists of are packed, so
 * that each URAM element holds 8 values for better utilization of the URAM.
 *
 * @tparam P            number of output-row lanes computed concurrently
 * 
 * @param[in]  input        Input activations, packed in URAM
 * @param[in]  out_row_base Base output row, output will contain rows [out_row_base, out_row_base+P]
 * @param[in]  tokens_dim   Number of tokens
 * @param[in]  weight_val   Weight values (CSR data format)
 * @param[in]  weight_col   Weight columns (CSR data format)
 * @param[in]  weight_ptr   Weight row pointers (CSR data format)
 * @param[out] out_rows     Output rows
 */
template <int P>
static void parallel_out_rows_compute (
    T_MhaBankWord input[MHA_PROJ_P_QK][MHA_BANK_WORDS],
    int           out_row_base,
    int           tokens_dim,
    T_MhaWVal    *weight_val,
    T_MhaWCol    *weight_col,
    int          *weight_ptr,
    T_MhaAcc      out_rows[P][MHA_TOKEN_W_MAX]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=input    cyclic factor=MHA_BANK_CYCLIC dim=2
#pragma HLS ARRAY_PARTITION variable=input    complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_rows cyclic factor=MHA_MAC_UNROLL dim=2
#pragma HLS ARRAY_PARTITION variable=out_rows complete dim=1
    
    /* Clear all accumulators to be used. */
    ACC_CLEAR:
    for (int i = 0; i <  tokens_dim; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_TOKEN max=MHA_TC_TOKEN
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=MHA_MAC_UNROLL
        for (int lane = 0; lane < P; lane++) {
            #pragma HLS UNROLL
            out_rows[lane][i] = (T_MhaAcc) 0;
        }
    }

    /* Since the parallel lanes iterate simultaneously,
     * we measure the largest non-zero span across all
     * the parallel lanes.
     */
    int max_nz_span = 0;
    int nz_begin[P];
    int nz_end[P];
    #pragma HLS ARRAY_PARTITION variable=nz_begin complete dim=1
    #pragma HLS ARRAY_PARTITION variable=nz_end   complete dim=1
    IDENTIFY_NZ_SPAN:
    for (int lane = 0; lane < P; lane++) {
        #pragma HLS UNROLL
        nz_begin[lane] = weight_ptr[out_row_base + lane];
        nz_end[lane]   = weight_ptr[out_row_base + lane + 1];

        int nz_count = nz_end[lane] - nz_begin[lane];

        if (nz_count > max_nz_span) {
            max_nz_span = nz_count;
        }
    }

    /* All parallel rows are traversed simultaneously through their non-zero elements. */
    NONZERO_ELEMENTS:
    for (int nz = 0; nz < max_nz_span; nz++) {
        #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_W_NNZ max=MHA_TC_W_NNZ
        
        /* Each iteration processes a number of token tiles equal to the unroll factor. */
        TOKEN_TILES:
        for (int token_idx = 0; token_idx < tokens_dim; token_idx += MHA_MAC_UNROLL) {
            #pragma HLS LOOP_TRIPCOUNT min=(MHA_TC_TOKEN/MHA_MAC_UNROLL+1) max=(MHA_TC_TOKEN/MHA_MAC_UNROLL+1)
            #pragma HLS PIPELINE II=1
            PARALLEL_OUTPUT_ROWS:
            for (int lane = 0; lane < P; lane++) {
                #pragma HLS UNROLL
                int nz_idx = nz_begin[lane] + nz;

                /* If no more nonzeros in this lane, skip computations. */
                if (nz_idx < nz_end[lane]) {
                    /* Identify the feature row being processed and compute the base tile word index */
                    int feature_row_base = (int) weight_col[nz_idx] * MHA_BANK_ROW_WORDS;
                    int tile_word_base   = feature_row_base + (token_idx / 8);

                    /* Handle as many parallel words as possible based on cyclic factor of the input bank */
                    PARALLEL_WORDS:
                    for (int offset = 0; offset < MHA_BANK_CYCLIC; offset++) {
                        #pragma HLS UNROLL
                        /* Read from dedicated bank within input replica. */
                        T_MhaBankWord input_packed = input[lane][tile_word_base + offset];
                        int           token_idx_0  = token_idx + offset * 8;
                        mac_pe(input_packed, weight_val[nz_idx], token_idx_0 + 0, 0, tokens_dim, out_rows[lane]);
                        mac_pe(input_packed, weight_val[nz_idx], token_idx_0 + 1, 1, tokens_dim, out_rows[lane]);
                        mac_pe(input_packed, weight_val[nz_idx], token_idx_0 + 2, 2, tokens_dim, out_rows[lane]);
                        mac_pe(input_packed, weight_val[nz_idx], token_idx_0 + 3, 3, tokens_dim, out_rows[lane]);
                        mac_pe(input_packed, weight_val[nz_idx], token_idx_0 + 4, 4, tokens_dim, out_rows[lane]);
                        mac_pe(input_packed, weight_val[nz_idx], token_idx_0 + 5, 5, tokens_dim, out_rows[lane]);
                        mac_pe(input_packed, weight_val[nz_idx], token_idx_0 + 6, 6, tokens_dim, out_rows[lane]);
                        mac_pe(input_packed, weight_val[nz_idx], token_idx_0 + 7, 7, tokens_dim, out_rows[lane]);
                    }
                }
            }
        }
    }
}

/**
 * @brief Dense projection
 *
 * @param[in]  w_val       Weight values in DDR.
 * @param[in]  w_col       Weight column indices in DDR.
 * @param[in]  w_ptr       Weight row pointers in DDR.
 * @param[in]  feature_dim Number of features.
 * @param[in]  tokens_dim  Number of tokens.
 * @param[in]  quant_scale Folded scale used for quantizing the output.
 * @param[out] out_dense   Dense output array.
 */
static void proj_dense (
    T_Activation *w_val,
    T_MhaIndex   *w_col,
    int          *w_ptr,
    int           feature_dim,
    int           tokens_dim,
    T_Scale       quant_scale,
    T_Activation *out_dense
) {
    load_weight_from_ddr(w_val, w_col, w_ptr, feature_dim);

    T_MhaAcc rows[MHA_PROJ_P][MHA_TOKEN_W_MAX];
    #pragma HLS ARRAY_PARTITION variable=rows cyclic factor=MHA_MAC_UNROLL dim=2
    #pragma HLS ARRAY_PARTITION variable=rows complete dim=1
    
    DENSE_PROJECTION_PER_ROW:
    for (int i = 0; i < feature_dim; i += MHA_PROJ_P) {
        #pragma HLS LOOP_TRIPCOUNT min=(MHA_TC_FEATURE/MHA_PROJ_P) max=(MHA_TC_FEATURE/MHA_PROJ_P)
        
        parallel_out_rows_compute<MHA_PROJ_P>(g_activation_replicas, 
                                              i, 
                                              tokens_dim, 
                                              g_weight_val_buffer, 
                                              g_weight_col_buffer, 
                                              g_weight_rptr_buffer, 
                                              rows);
        DENSE_PROJECTION_WRITE:
        for (int j = 0; j <  tokens_dim; j++) {
            #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_TOKEN max=MHA_TC_TOKEN
            #pragma HLS PIPELINE II=1
           
            for (int l = 0; l < MHA_PROJ_P; l++) {
                #pragma HLS UNROLL
                out_dense[(i + l) *  tokens_dim + j] = requant_to_int8(rows[l][j], quant_scale);
            }
        }
    }
}

/**
 * @brief Sparse projection (projection followed by ReLU).
 * 
 * Output activations are produced in feature-major order.
 *
 * @param[in]  w_val       Weight values in DDR.
 * @param[in]  w_col       Weight column indices in DDR.
 * @param[in]  w_ptr       Weight row pointers in DDR.
 * @param[in]  feature_dim Number of features.
 * @param[in]  tokens_dim  Number of tokens.
 * @param[in]  quant_scale Folded scale used for quantizing the output.
 * @param[out] out_val     Output activation values.
 * @param[out] out_col     Output activation column indices.
 * @param[out] out_ptr     Output activation row pointers.
 */
static void proj_relu_featmajor (
    T_Activation   *w_val, 
    T_MhaIndex     *w_col,
    int            *w_ptr,
    int             feature_dim,
    int             tokens_dim, 
    T_Scale         quant_scale,
    T_Activation   *out_val, 
    T_MhaHeadIndex *out_col, 
    int            *out_ptr
) {
    load_weight_from_ddr(w_val, w_col, w_ptr, feature_dim);   
    
    T_MhaAcc rows[MHA_PROJ_P_QK][MHA_TOKEN_W_MAX];
    #pragma HLS ARRAY_PARTITION variable=rows cyclic factor=MHA_MAC_UNROLL dim=2
    #pragma HLS ARRAY_PARTITION variable=rows complete dim=1
    int nnz    = 0;
    out_ptr[0] = 0;

    SPARSE_PROJECTION_ROW:
    for (int i = 0; i < feature_dim; i += MHA_PROJ_P_QK) {
        #pragma HLS LOOP_TRIPCOUNT min=(MHA_TC_FEATURE/MHA_PROJ_P_QK) max=(MHA_TC_FEATURE/MHA_PROJ_P_QK)

        parallel_out_rows_compute<MHA_PROJ_P_QK>(g_activation_replicas, 
                                                 i, 
                                                 tokens_dim, 
                                                 g_weight_val_buffer, 
                                                 g_weight_col_buffer, 
                                                 g_weight_rptr_buffer, 
                                                 rows);
        PARALLEL_LANES:
        for (int p = 0; p < MHA_PROJ_P_QK; p++) {
            
            RELU_QUANTIZE_SPARSIFY:
            for (int j = 0; j <  tokens_dim; j++) {
                #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_TOKEN max=MHA_TC_TOKEN
                #pragma HLS PIPELINE II=1

                if (rows[p][j] > (T_MhaAcc) 0) {
                    T_Activation quant_out = requant_to_int8(rows[p][j], quant_scale);

                    if (quant_out > (T_Activation) 0) {
#ifndef __SYNTHESIS__
                        assert(nnz < MHA_MAX_QK_NNZ && "K CSR overflow (raise MHA_MAX_QK_NNZ)");
#endif
                        out_val[nnz] = quant_out;
                        out_col[nnz] = (T_MhaHeadIndex) j;
                        nnz++;
                    }
                }
            }

            out_ptr[i + p + 1] = nnz;
        }
    }
}

/**
 * @brief Sparse Q projection (projection followed by ReLU), emitted HEAD-MAJOR.
 *
 * Output activations are bucketed by head: 12 head blocks, each a token-major CSR
 * (row_ptr of n+1 at out_ptr[h*(n+1)+..], columns are band-local features 0..63).
 * This lets each attention head read a CONTIGUOUS slice with no band filter.
 * A scratch buffer keeps the intermediate dense result, as we transpose before ReLU.
 *
 * @param[in]  w_val          Weight values in DDR.
 * @param[in]  w_col          Weight column indices in DDR.
 * @param[in]  w_ptr          Weight row pointers in DDR.
 * @param[in]  feature_dim    Number of features.
 * @param[in]  tokens_dim     Number of tokens.
 * @param[in]  quant_scale    Folded scale used for quantizing the output.
 * @param[in]  scratch_buffer Scratch buffer for intermediate transpose.
 * @param[out] out_val        Output activation values.
 * @param[out] out_col        Output activation column indices.
 * @param[out] out_ptr        Output activation row pointers.
 */
static void proj_relu_headmajor (
    T_Activation   *w_val, 
    T_MhaIndex     *w_col,
    int            *w_ptr,
    int             feature_dim,
    int             tokens_dim, 
    T_Scale         quant_scale,
    T_Activation   *scratch_buffer,
    T_Activation   *out_val, 
    T_MhaHeadIndex *out_col, 
    int            *out_ptr
) {
    load_weight_from_ddr(w_val, w_col, w_ptr, feature_dim);

    T_MhaAcc rows[MHA_PROJ_P_QK][MHA_TOKEN_W_MAX];
    #pragma HLS ARRAY_PARTITION variable=rows cyclic factor=MHA_MAC_UNROLL dim=2
    #pragma HLS ARRAY_PARTITION variable=rows complete dim=1
    
    DENSE_PROJECTION_ROW:
    for (int i = 0; i < feature_dim; i += MHA_PROJ_P_QK) {
        #pragma HLS LOOP_TRIPCOUNT min=(MHA_TC_FEATURE/MHA_PROJ_P_QK) max=(MHA_TC_FEATURE/MHA_PROJ_P_QK)
        
        parallel_out_rows_compute<MHA_PROJ_P_QK>(g_activation_replicas, 
                                                 i, 
                                                 tokens_dim, 
                                                 g_weight_val_buffer, 
                                                 g_weight_col_buffer, 
                                                 g_weight_rptr_buffer, 
                                                 rows);

        QUANTIZE_AND_TRANSPOSE:
        for (int j = 0; j <  tokens_dim; j++) {
            #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_TOKEN max=MHA_TC_TOKEN
            #pragma HLS PIPELINE II=1
            /* Each (i+p, j) is a distinct scratch address across iterations (j varies, p
             * distinct) -> no real inter-iteration dependency; assert it to hold II=1. */
            #pragma HLS DEPENDENCE variable=scratch_buffer type=inter false

            for (int p = 0; p < MHA_PROJ_P_QK; p++) {
                #pragma HLS UNROLL
                T_Activation quant_out = (rows[p][j] > (T_MhaAcc) 0) ? requant_to_int8(rows[p][j], quant_scale) : (T_Activation) 0;
                scratch_buffer[(i + p) *  tokens_dim + j] = quant_out;
            }
        }
    }

    /* HEAD-MAJOR CSR emit: bucket Q' by head so each head reads a CONTIGUOUS slice
     * (like K'), eliminating the per-head full scan + band filter.  Layout: 12 head
     * blocks, each a token-major row_ptr of (n+1) entries at out_ptr[h*(n+1) + ...].
     * Within a head block the column is the BAND-LOCAL feature (0..63).  Features are
     * emitted ascending within each (head,token), matching the old within-band order. */
    int nnz = 0;
    const int n_heads = feature_dim / MHA_FEATURE_HEAD;
    RELU_ON_SCRATCH_HEAD:
    for (int h = 0; h < n_heads; h++) {
        #pragma HLS LOOP_TRIPCOUNT min=MHA_N_HEADS max=MHA_N_HEADS
        const int feature_base = h * MHA_FEATURE_HEAD;
        const int ptr_base     = h * (tokens_dim + 1);
        out_ptr[ptr_base] = nnz;

        RELU_ON_SCRATCH_TOKEN:
        for (int j = 0; j < tokens_dim; j++) {
            #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_TOKEN max=MHA_TC_TOKEN

            PER_BAND_FEATURE_LOOP:
            for (int bf = 0; bf < MHA_FEATURE_HEAD; bf++) {
                #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_FEATURE_HEAD max=MHA_TC_FEATURE_HEAD
                #pragma HLS PIPELINE II=1

                T_Activation out_data = scratch_buffer[(feature_base + bf) * tokens_dim + j];

                if (out_data > (T_Activation) 0) {
#ifndef __SYNTHESIS__
                    assert(nnz < MHA_MAX_QK_NNZ && "Q' CSR overflow (raise MHA_MAX_QK_NNZ)");
#endif
                    out_val[nnz] = out_data;
                    out_col[nnz] = (T_MhaHeadIndex) bf;   // band-local feature (0..63)
                    nnz++;
                }
            }

            out_ptr[ptr_base + j + 1] = nnz;
        }
    }
}

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
 * @brief Burst one head's contiguous K'/Q' slice from DDR into on-chip buffers.
 *
 * K'/Q' are the sparse FIRST operand and are read as a per-head contiguous slice
 * (K' feature-major band [h*64,(h+1)*64); Q' head-major block h).  Streaming the
 * slice on-chip removes the DDR reads from the head compute loops.  The slice fits
 * the fully-dense bound (MHA_*_STREAM_NNZ = 64*197), so it never overflows.
 * Buffers are indexed slice-locally: absolute nnz index s maps to s - slice_base.
 *
 * @param[in]  k_val_ddr/k_col_ddr  DDR K' values / columns.
 * @param[in]  k_slice_base/k_slice_len  This head's K' slice [base, base+len) in DDR.
 * @param[in]  q_val_ddr/q_col_ddr  DDR Q' values / columns.
 * @param[in]  q_slice_base/q_slice_len  This head's Q' slice [base, base+len) in DDR.
 * @param[out] k_val_buf/k_col_buf/q_val_buf/q_col_buf  On-chip slice buffers.
 */
static void load_head_slice (
    T_Activation   *k_val_ddr, T_MhaHeadIndex *k_col_ddr, int k_slice_base, int k_slice_len,
    T_Activation   *q_val_ddr, T_MhaHeadIndex *q_col_ddr, int q_slice_base, int q_slice_len,
    T_Activation   *k_val_buf, T_MhaHeadIndex *k_col_buf,
    T_Activation   *q_val_buf, T_MhaHeadIndex *q_col_buf
) {
#pragma HLS INLINE off
    LOAD_K_SLICE:
    for (int i = 0; i < k_slice_len; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=0 max=MHA_KP_STREAM_NNZ
        #pragma HLS PIPELINE II=1
        k_val_buf[i] = k_val_ddr[k_slice_base + i];
        k_col_buf[i] = k_col_ddr[k_slice_base + i];
    }
    LOAD_Q_SLICE:
    for (int i = 0; i < q_slice_len; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=0 max=MHA_QP_STREAM_NNZ
        #pragma HLS PIPELINE II=1
        q_val_buf[i] = q_val_ddr[q_slice_base + i];
        q_col_buf[i] = q_col_ddr[q_slice_base + i];
    }
}

static void head_core (
    int             head,
    int             tokens_dim,
    T_Scale         quant_scale,
    /* On-chip K'/Q' slice buffers (this head's slice), + the DDR row_ptr (span bounds)
     * and the slice base offset to rebase absolute nnz indices to buffer-local. */
    T_Activation   *k_val_buf,
    T_MhaHeadIndex *k_col_buf,
    int            *k_ptr,
    int             k_slice_base,
    T_Activation   *q_val_buf,
    T_MhaHeadIndex *q_col_buf,
    int            *q_ptr,
    int             q_slice_base,
    T_Activation   *v_dense,
    T_MhaBankWord  *out_packed
) {
    const int feature_base = head * MHA_FEATURE_HEAD;

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
        int feature_row = feature_base + feature;
        int nz_begin    = k_ptr[feature_row];
        int nz_end      = k_ptr[feature_row + 1];

        KEY_VALUE_NONZERO:
        for (int s = nz_begin; s < nz_end; s++) {
            #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_KP_NNZ max=MHA_TC_KP_NNZ
            int          bi      = s - k_slice_base;   // slice-local buffer index
            int          token   = (int) k_col_buf[bi];
            T_Activation key_val = k_val_buf[bi];
            
            key_sums[feature] += key_val;

            KEY_VALUE_OUT_FEATURE:
            for (int out_feature = 0; out_feature < MHA_FEATURE_HEAD; out_feature++) {
                #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_FEATURE_HEAD max=MHA_TC_FEATURE_HEAD
                #pragma HLS PIPELINE II=1
                #pragma HLS UNROLL factor=MHA_MAC_UNROLL

                /* Multiply-Accumulate K value with MHA_FEATURE_HEAD features of selected token. */
                key_value_matrix[feature * MHA_FEATURE_HEAD + out_feature] +=
                    key_val * v_dense[(feature_base + out_feature) *  tokens_dim + token];
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

        /* Q' is HEAD-MAJOR: this head's token-major row_ptr block starts at head*(n+1),
         * and q_col holds the BAND-LOCAL feature (0..63) directly -- no scan/filter. */
        T_MhaAcc denominator = (T_MhaAcc) 0;
        const int q_ptr_base = head * (tokens_dim + 1);
        int      nz_begin    = q_ptr[q_ptr_base + token];
        int      nz_end      = q_ptr[q_ptr_base + token + 1];
        NUMERATOR_NONZERO:
        for (int s = nz_begin; s < nz_end; s++) {
            #pragma HLS LOOP_TRIPCOUNT min=MHA_TC_QP_NNZ max=MHA_TC_QP_NNZ
            int          bi           = s - q_slice_base;   // slice-local buffer index
            int          band_feature = (int) q_col_buf[bi]; // already band-local (0..63)
            T_Activation query_val    = q_val_buf[bi];

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

void mha(
    T_MhaBankWord* input_packed,
    int            tokens_dim,
    T_Activation* wq_values, T_MhaIndex* wq_col_idx, int* wq_row_ptr,
    T_Activation* wk_values, T_MhaIndex* wk_col_idx, int* wk_row_ptr,
    T_Activation* wv_values, T_MhaIndex* wv_col_idx, int* wv_row_ptr,
    T_Activation* wo_values, T_MhaIndex* wo_col_idx, int* wo_row_ptr,
    const T_Scale* scales,
    /* Q'/K' val/col CSR are DDR-resident scratch (full grid D*N); each head streams its
     * slice on-chip.  Row pointers are produced+consumed within mha() so they stay on-chip
     * (k_ptr_local/q_ptr_local) and are NOT interface args. */
    T_Activation* q_val, T_MhaHeadIndex* q_col,
    T_Activation* k_val, T_MhaHeadIndex* k_col,
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

    /* Dense V array. */
    static T_Activation v_dense[MHA_SCR_V_CNT];
    #pragma HLS BIND_STORAGE variable=v_dense type=ram_t2p impl=bram

    /* Packed attention output. */
    static T_MhaBankWord attn_output[MHA_BANK_WORDS];
    #pragma HLS BIND_STORAGE   variable=attn_output type=ram_t2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=attn_output cyclic factor=MHA_BANK_CYCLIC dim=1

    /* On-chip per-head K'/Q' slice buffers (streamed from DDR before each head).
     * Sized to the fully-dense per-head slice -> never overflows.  head_core reads
     * these instead of DDR, removing DDR from the head compute loops. */
    static T_Activation   k_val_buf[MHA_KP_STREAM_NNZ];
    static T_MhaHeadIndex k_col_buf[MHA_KP_STREAM_NNZ];
    static T_Activation   q_val_buf[MHA_QP_STREAM_NNZ];
    static T_MhaHeadIndex q_col_buf[MHA_QP_STREAM_NNZ];
    #pragma HLS BIND_STORAGE variable=k_val_buf type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=k_col_buf type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=q_val_buf type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=q_col_buf type=ram_t2p impl=bram

    /* K'/Q' row pointers are PRODUCED and CONSUMED entirely within this mha() call, so
     * they live on-chip -- never round-tripped through DDR.  (val/col are still DDR: the
     * full grid doesn't fit on-chip, so each head streams its slice.) */
    static int k_ptr_local[MHA_SCR_KP_PTR_CNT];
    static int q_ptr_local[MHA_SCR_QP_PTR_CNT];
    #pragma HLS BIND_STORAGE variable=k_ptr_local type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=q_ptr_local type=ram_t2p impl=bram

    /* Replicate input activation. Reused for all Q, K and V computations. */
    replicate_input_activations<MHA_PROJ_P_QK>(input_packed, g_activation_replicas);

    /* Compute Q in token-major order. V dense array temporarily used as scratch. */
    proj_relu_headmajor(wq_values,
                         wq_col_idx,
                         wq_row_ptr,
                         feature_dim,
                         tokens_dim,
                         scales[SCALE_Q_IDX],
                         v_dense,
                         q_val,
                         q_col,
                         q_ptr_local);

    /* Compute K in feature-major order. */
    proj_relu_featmajor(wk_values,
                        wk_col_idx,
                        wk_row_ptr,
                        feature_dim,
                        tokens_dim,
                        scales[SCALE_K_IDX],
                        k_val,
                        k_col,
                        k_ptr_local);
    
    /* Compute the dense V. */
    proj_dense(wv_values, 
               wv_col_idx, 
               wv_row_ptr, 
               feature_dim, 
               tokens_dim,
               scales[SCALE_V_IDX], 
               v_dense);

    /* Per-head attention: stream this head's K'/Q' slice on-chip, then compute. */
    HEAD_LOOP:
    for (int head = 0; head < MHA_N_HEADS; head++) {
        #pragma HLS LOOP_TRIPCOUNT min=MHA_N_HEADS max=MHA_N_HEADS

        /* K' feature-major: head owns feature rows [head*64, (head+1)*64). */
        const int k_slice_base = k_ptr_local[head * MHA_FEATURE_HEAD];
        const int k_slice_len  = k_ptr_local[(head + 1) * MHA_FEATURE_HEAD] - k_slice_base;
        /* Q' head-major: head's token-major block starts at head*(n+1). */
        const int q_ptr_base   = head * (tokens_dim + 1);
        const int q_slice_base = q_ptr_local[q_ptr_base];
        const int q_slice_len  = q_ptr_local[q_ptr_base + tokens_dim] - q_slice_base;

        load_head_slice(k_val, k_col, k_slice_base, k_slice_len,
                        q_val, q_col, q_slice_base, q_slice_len,
                        k_val_buf, k_col_buf, q_val_buf, q_col_buf);

        head_core(head,
                  tokens_dim,
                  scales[SCALE_DIV_OUT_IDX],
                  k_val_buf, k_col_buf, k_ptr_local, k_slice_base,
                  q_val_buf, q_col_buf, q_ptr_local, q_slice_base,
                  v_dense,
                  attn_output);
    }

    /* Replicate attention output. */
    replicate_input_activations<MHA_PROJ_P>(attn_output, g_activation_replicas);

    /* Dense output projection. */
    proj_dense(wo_values, 
               wo_col_idx, 
               wo_row_ptr, 
               feature_dim, 
               tokens_dim,
               scales[SCALE_LINEAR_OUT_IDX],
               output);
}
