/**********************************************************************************
 * Encoder Multi-Head Attention Block - Implementation
 *
 * Please see the respective header file for a high-level description of the layer.
 *
 *********************************************************************************/

#include "encoder_mha_block.h"
#include "residual.h"
#include <cmath>
#ifndef __SYNTHESIS__
#include <cassert>
#endif

// Block dims mirror the callee bounds; assert they agree so the literals can't drift.
static_assert(ENC_MHA_FEATURE_W_MAX == RMS_FEATURE_W_MAX && ENC_MHA_FEATURE_W_MAX == MHA_FEATURE_W_MAX,
              "ENC_MHA_FEATURE_W_MAX must match RMS_FEATURE_W_MAX / MHA_FEATURE_W_MAX");
static_assert(ENC_MHA_TOKEN_W_MAX == RMS_TOKEN_W_MAX && ENC_MHA_TOKEN_W_MAX == MHA_TOKEN_W_MAX,
              "ENC_MHA_TOKEN_W_MAX must match RMS_TOKEN_W_MAX / MHA_TOKEN_W_MAX");

void encoder_mha_block(
    const T_Activation* input,
    int                 tokens_dim,
    int                 feature_dim,
    T_Activation* wq_values, T_MhaIndex* wq_col_idx, int* wq_row_ptr,
    T_Activation* wk_values, T_MhaIndex* wk_col_idx, int* wk_row_ptr,
    T_Activation* wv_values, T_MhaIndex* wv_col_idx, int* wv_row_ptr,
    T_Activation* wo_values, T_MhaIndex* wo_col_idx, int* wo_row_ptr,
    const T_Scale*      scales,
    T_Activation* q_val, T_MhaHeadIndex* q_col,
    T_Activation* k_val, T_MhaHeadIndex* k_col,
    T_Activation*       hidden
) {
#ifndef __SYNTHESIS__
    assert(feature_dim <= ENC_MHA_FEATURE_W_MAX && "feature_dim exceeds ENC_MHA_FEATURE_W_MAX");
    assert(tokens_dim  <= ENC_MHA_TOKEN_W_MAX && "tokens_dim exceeds ENC_MHA_TOKEN_W_MAX");
#endif

    /* RMSNorm output written in URAM in packed format. */
    static T_MhaBankWord norm_input[MHA_BANK_WORDS];
    #pragma HLS BIND_STORAGE    variable=norm_input type=ram_t2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=norm_input cyclic factor=MHA_BANK_CYCLIC dim=1
    rmsnorm(input,
            tokens_dim,
            feature_dim,
            scales[SCALE_ATT_RMSNORM_OUT_INV_IDX],
            (T_Activation*) nullptr,
            reinterpret_cast<T_RmsBankWord*>(norm_input));

    mha(norm_input,
        tokens_dim,
        wq_values, wq_col_idx, wq_row_ptr,
        wk_values, wk_col_idx, wk_row_ptr,
        wv_values, wv_col_idx, wv_row_ptr,
        wo_values, wo_col_idx, wo_row_ptr,
        scales,
        q_val, q_col,
        k_val, k_col,
        hidden);

    residual_add(input, 
                 hidden,
                 feature_dim * tokens_dim,
                 scales[SCALE_ATT_RESIDUAL_IDX], 
                 scales[SCALE_ATT_BRANCH_RATIO_IDX],
                 hidden);
}

void encoder_mha_block_top(
    const T_Activation* input,
    int                 tokens_dim,
    int                 feature_dim,
    T_Activation* wq_values, T_MhaIndex* wq_col_idx, int* wq_row_ptr,
    T_Activation* wk_values, T_MhaIndex* wk_col_idx, int* wk_row_ptr,
    T_Activation* wv_values, T_MhaIndex* wv_col_idx, int* wv_row_ptr,
    T_Activation* wo_values, T_MhaIndex* wo_col_idx, int* wo_row_ptr,
    const T_Scale*      scales,
    T_Activation* q_val, T_MhaHeadIndex* q_col,
    T_Activation* k_val, T_MhaHeadIndex* k_col,
    T_Activation*       hidden
) {
    #pragma HLS INTERFACE s_axilite port=return      bundle=control
    #pragma HLS INTERFACE s_axilite port=tokens_dim  bundle=control
    #pragma HLS INTERFACE s_axilite port=feature_dim bundle=control
    #pragma HLS INTERFACE m_axi port=input  bundle=in_mem depth=(ENC_MHA_FEATURE_W_MAX*ENC_MHA_TOKEN_W_MAX)
    #pragma HLS INTERFACE m_axi port=scales bundle=sc_mem depth=SCALE_IDX_MAX
    #pragma HLS INTERFACE m_axi port=wq_values  bundle=wq_mem depth=MAX_NNZ
    #pragma HLS INTERFACE m_axi port=wq_col_idx bundle=wq_mem depth=MAX_NNZ
    #pragma HLS INTERFACE m_axi port=wq_row_ptr bundle=wq_mem depth=MAX_ROWS+1
    #pragma HLS INTERFACE m_axi port=wk_values  bundle=wk_mem depth=MAX_NNZ
    #pragma HLS INTERFACE m_axi port=wk_col_idx bundle=wk_mem depth=MAX_NNZ
    #pragma HLS INTERFACE m_axi port=wk_row_ptr bundle=wk_mem depth=MAX_ROWS+1
    #pragma HLS INTERFACE m_axi port=wv_values  bundle=wv_mem depth=MAX_NNZ
    #pragma HLS INTERFACE m_axi port=wv_col_idx bundle=wv_mem depth=MAX_NNZ
    #pragma HLS INTERFACE m_axi port=wv_row_ptr bundle=wv_mem depth=MAX_ROWS+1
    #pragma HLS INTERFACE m_axi port=wo_values  bundle=wo_mem depth=MAX_NNZ
    #pragma HLS INTERFACE m_axi port=wo_col_idx bundle=wo_mem depth=MAX_NNZ
    #pragma HLS INTERFACE m_axi port=wo_row_ptr bundle=wo_mem depth=MAX_ROWS+1
    #pragma HLS INTERFACE m_axi port=hidden bundle=hidden_mem depth=(ENC_MHA_FEATURE_W_MAX*ENC_MHA_TOKEN_W_MAX)
    /* DDR-resident Q'/K' val/col CSR scratch (full grid D*N); row pointers are on-chip in mha(). */
    #pragma HLS INTERFACE m_axi port=q_val bundle=qp_mem depth=MHA_MAX_QK_NNZ
    #pragma HLS INTERFACE m_axi port=q_col bundle=qp_mem depth=MHA_MAX_QK_NNZ
    #pragma HLS INTERFACE m_axi port=k_val bundle=kp_mem depth=MHA_MAX_QK_NNZ
    #pragma HLS INTERFACE m_axi port=k_col bundle=kp_mem depth=MHA_MAX_QK_NNZ

    encoder_mha_block(input, tokens_dim, feature_dim,
                      wq_values, wq_col_idx, wq_row_ptr,
                      wk_values, wk_col_idx, wk_row_ptr,
                      wv_values, wv_col_idx, wv_row_ptr,
                      wo_values, wo_col_idx, wo_row_ptr,
                      scales,
                      q_val, q_col,
                      k_val, k_col,
                      hidden);
}
