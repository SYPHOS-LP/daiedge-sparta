/**********************************************************************************
 * Top-Level Encoder Layer for Vision Transformers - Implementation
 *
 * Please see the respective header file for a high-level description of the layer.
 *
 *********************************************************************************/

#include "encoder_layer.h"
#ifndef __SYNTHESIS__
#include <cassert>
#endif

// The layer dims are the single source of truth for both sub-blocks; assert the
// per-block config literals agree so they cannot drift across the config headers.
static_assert(ENC_FEATURE_W_MAX == ENC_MHA_FEATURE_W_MAX && ENC_FEATURE_W_MAX == ENC_MLP_FEATURE_W_MAX,
              "ENC_FEATURE_W_MAX must match ENC_MHA_FEATURE_W_MAX / ENC_MLP_FEATURE_W_MAX");
static_assert(ENC_TOKEN_W_MAX == ENC_MHA_TOKEN_W_MAX && ENC_TOKEN_W_MAX == ENC_MLP_TOKEN_W_MAX,
              "ENC_TOKEN_W_MAX must match ENC_MHA_TOKEN_W_MAX / ENC_MLP_TOKEN_W_MAX");

void encoder_layer_top(
    const T_Activation*   input,
    int               feature_dim,
    int               tokens_dim,
    int               intermediate_dim,
    T_Activation* wq_values, T_MhaIndex* wq_col_idx, int* wq_row_ptr,
    T_Activation* wk_values, T_MhaIndex* wk_col_idx, int* wk_row_ptr,
    T_Activation* wv_values, T_MhaIndex* wv_col_idx, int* wv_row_ptr,
    T_Activation* wo_values, T_MhaIndex* wo_col_idx, int* wo_row_ptr,
    T_Activation* w1_values, T_MlpIndex* w1_col_idx, int* w1_row_ptr,
    T_Activation* w2_values, T_MlpIndex* w2_col_idx, int* w2_row_ptr,
    const T_Scale* scales,
    /* DDR-resident activation scratch: Q'/K' val/col (full grid D*N).  Q'/K' row pointers
     * are on-chip in mha() (produced+consumed there).  The MLP's hidden H no longer needs a
     * DDR bundle: it streams producer->consumer on-chip in the MLP's DATAFLOW region. */
    T_Activation* q_val, T_MhaHeadIndex* q_col,
    T_Activation* k_val, T_MhaHeadIndex* k_col,
    T_Activation* hidden,
    T_Activation* output
) {
    #pragma HLS INTERFACE s_axilite port=return     bundle=control
    #pragma HLS INTERFACE s_axilite port=tokens_dim bundle=control
    #pragma HLS INTERFACE m_axi port=scales bundle=sc_mem depth=SCALE_IDX_MAX num_read_outstanding=2 num_write_outstanding=2
    #pragma HLS INTERFACE s_axilite port=intermediate_dim bundle=control
    #pragma HLS INTERFACE s_axilite port=feature_dim      bundle=control
    #pragma HLS INTERFACE m_axi port=input     bundle=in_mem depth=(ENC_FEATURE_W_MAX*ENC_TOKEN_W_MAX) num_read_outstanding=2 num_write_outstanding=2
    #pragma HLS INTERFACE m_axi port=wq_values  bundle=w_mha_mem depth=MAX_NNZ num_read_outstanding=2 num_write_outstanding=2
    #pragma HLS INTERFACE m_axi port=wq_col_idx bundle=w_mha_mem depth=MAX_NNZ num_read_outstanding=2 num_write_outstanding=2
    #pragma HLS INTERFACE m_axi port=wq_row_ptr bundle=w_mha_mem depth=MAX_ROWS+1 num_read_outstanding=2 num_write_outstanding=2
    #pragma HLS INTERFACE m_axi port=wk_values  bundle=w_mha_mem depth=MAX_NNZ num_read_outstanding=2 num_write_outstanding=2
    #pragma HLS INTERFACE m_axi port=wk_col_idx bundle=w_mha_mem depth=MAX_NNZ num_read_outstanding=2 num_write_outstanding=2
    #pragma HLS INTERFACE m_axi port=wk_row_ptr bundle=w_mha_mem depth=MAX_ROWS+1 num_read_outstanding=2 num_write_outstanding=2
    #pragma HLS INTERFACE m_axi port=wv_values  bundle=w_mha_mem depth=MAX_NNZ num_read_outstanding=2 num_write_outstanding=2
    #pragma HLS INTERFACE m_axi port=wv_col_idx bundle=w_mha_mem depth=MAX_NNZ num_read_outstanding=2 num_write_outstanding=2
    #pragma HLS INTERFACE m_axi port=wv_row_ptr bundle=w_mha_mem depth=MAX_ROWS+1 num_read_outstanding=2 num_write_outstanding=2
    #pragma HLS INTERFACE m_axi port=wo_values  bundle=w_mha_mem depth=MAX_NNZ num_read_outstanding=2 num_write_outstanding=2
    #pragma HLS INTERFACE m_axi port=wo_col_idx bundle=w_mha_mem depth=MAX_NNZ num_read_outstanding=2 num_write_outstanding=2
    #pragma HLS INTERFACE m_axi port=wo_row_ptr bundle=w_mha_mem depth=MAX_ROWS+1 num_read_outstanding=2 num_write_outstanding=2
    #pragma HLS INTERFACE m_axi port=w1_values  bundle=w1_mem depth=MAX_NNZ num_read_outstanding=2 num_write_outstanding=2
    #pragma HLS INTERFACE m_axi port=w1_col_idx bundle=w1_mem depth=MAX_NNZ num_read_outstanding=2 num_write_outstanding=2
    #pragma HLS INTERFACE m_axi port=w1_row_ptr bundle=w1_mem depth=MAX_ROWS+1 num_read_outstanding=2 num_write_outstanding=2
    #pragma HLS INTERFACE m_axi port=w2_values  bundle=w2_mem depth=MAX_NNZ num_read_outstanding=2 num_write_outstanding=2
    #pragma HLS INTERFACE m_axi port=w2_col_idx bundle=w2_mem depth=MAX_NNZ num_read_outstanding=2 num_write_outstanding=2
    #pragma HLS INTERFACE m_axi port=w2_row_ptr bundle=w2_mem depth=MAX_ROWS+1 num_read_outstanding=2 num_write_outstanding=2
    #pragma HLS INTERFACE m_axi port=hidden bundle=h_mem depth=(ENC_FEATURE_W_MAX*ENC_TOKEN_W_MAX) num_read_outstanding=2 num_write_outstanding=2
    #pragma HLS INTERFACE m_axi port=output bundle=y_mem depth=(ENC_FEATURE_W_MAX*ENC_TOKEN_W_MAX) num_read_outstanding=2 num_write_outstanding=2
    /* DDR-resident activation scratch: Q'/K' + H (single region each). */
    /* Each CSR array on its OWN bundle: sharing a bundle across val/col/ptr blocks HLS bursting
     * ("multiple potential writes to the same bundle in the same region"). One array per bundle
     * lets each be bursted independently. */
    #pragma HLS INTERFACE m_axi port=q_val bundle=qv_mem depth=MHA_MAX_QK_NNZ num_read_outstanding=2 num_write_outstanding=2
    #pragma HLS INTERFACE m_axi port=q_col bundle=qc_mem depth=MHA_MAX_QK_NNZ num_read_outstanding=2 num_write_outstanding=2
    #pragma HLS INTERFACE m_axi port=k_val bundle=kv_mem depth=MHA_MAX_QK_NNZ num_read_outstanding=2 num_write_outstanding=2
    #pragma HLS INTERFACE m_axi port=k_col bundle=kc_mem depth=MHA_MAX_QK_NNZ num_read_outstanding=2 num_write_outstanding=2

#ifndef __SYNTHESIS__
    assert(tokens_dim <= ENC_TOKEN_W_MAX && "tokens_dim exceeds ENC_TOKEN_W_MAX");
    assert(feature_dim == ENC_FEATURE_W_MAX && "feature_dim != ENC_FEATURE_W_MAX");
#endif

    /* Burst read the folded quantization scales from DDR once. */
    T_Scale layer_scales[SCALE_IDX_MAX];
    #pragma HLS ARRAY_PARTITION variable=layer_scales complete dim=1
    SCALE_LOAD:
    for (int i = 0; i < SCALE_IDX_MAX; i++) {
        #pragma HLS PIPELINE II=1
        layer_scales[i] = scales[i];
    }

    /* hidden = input + MHA(RMSNorm(input)) */
    encoder_mha_block(input,
                      tokens_dim,
                      feature_dim,
                      wq_values, wq_col_idx, wq_row_ptr,
                      wk_values, wk_col_idx, wk_row_ptr,
                      wv_values, wv_col_idx, wv_row_ptr,
                      wo_values, wo_col_idx, wo_row_ptr,
                      layer_scales,
                      q_val, q_col,
                      k_val, k_col,
                      hidden);

    /* output = hidden + MLP(RMSNorm(hidden)) */
    encoder_mlp_block(hidden,
                      tokens_dim,
                      feature_dim,
                      intermediate_dim,
                      w1_values, w1_col_idx, w1_row_ptr,
                      w2_values, w2_col_idx, w2_row_ptr,
                      layer_scales,
                      output);
}
