/**********************************************************************************
 * Encoder MLP Block - Implementation
 *
 * Please see the respective header file for a high-level description of the layer.
 *
 *********************************************************************************/

#include "encoder_mlp_block.h"
#include "residual.h"
#include <cmath>
#ifndef __SYNTHESIS__
#include <cassert>
#endif

// Block dims mirror the callee bounds; assert they agree so the literals can't drift.
static_assert(ENC_MLP_FEATURE_W_MAX == RMS_FEATURE_W_MAX && ENC_MLP_FEATURE_W_MAX == MLP_FEATURE_W_MAX,
              "ENC_MLP_FEATURE_W_MAX must match RMS_FEATURE_W_MAX / MLP_FEATURE_W_MAX");
static_assert(ENC_MLP_TOKEN_W_MAX == RMS_TOKEN_W_MAX && ENC_MLP_TOKEN_W_MAX == MLP_TOKEN_W_MAX,
              "ENC_MLP_TOKEN_W_MAX must match RMS_TOKEN_W_MAX / MLP_TOKEN_W_MAX");

void encoder_mlp_block(
    const T_Activation* input,
    int                 tokens_dim,
    int                 feature_dim,
    int                 intermediate_dim,
    T_Activation* w1_values, T_MlpIndex* w1_col_idx, int* w1_row_ptr,
    T_Activation* w2_values, T_MlpIndex* w2_col_idx, int* w2_row_ptr,
    const T_Scale*      scales,
    T_Activation*       output
) {
#ifndef __SYNTHESIS__
    assert(feature_dim <= ENC_MLP_FEATURE_W_MAX && "feature_dim exceeds ENC_MLP_FEATURE_W_MAX");
    assert(tokens_dim  <= ENC_MLP_TOKEN_W_MAX && "tokens_dim exceeds ENC_MLP_TOKEN_W_MAX");
#endif

    /* RMSNorm output written in URAM in packed format. */
    static T_MlpBankWord norm_input[MLP_BANK_WORDS];
    #pragma HLS BIND_STORAGE    variable=norm_input type=ram_t2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=norm_input cyclic factor=MLP_BANK_CYCLIC dim=1
    rmsnorm(input,
            tokens_dim,
            feature_dim,
            scales[SCALE_FF_RMSNORM_OUT_INV_IDX],
            (T_Activation*) nullptr,
            reinterpret_cast<T_RmsBankWord*> (norm_input));

    mlp(norm_input,
        feature_dim, 
        intermediate_dim, 
        tokens_dim,
        w1_values, 
        w1_col_idx, 
        w1_row_ptr,
        w2_values,
        w2_col_idx,
        w2_row_ptr,
        scales,
        output);

    residual_add(input,
                 output, 
                 feature_dim * tokens_dim,
                 scales[SCALE_FF_RESIDUAL_IDX], 
                 scales[SCALE_FF_BRANCH_RATIO_IDX],
                 output);
}

void encoder_mlp_block_top(
    const T_Activation* input,
    int                 tokens_dim,
    int                 feature_dim,
    int                 intermediate_dim,
    T_Activation* w1_values, T_MlpIndex* w1_col_idx, int* w1_row_ptr,
    T_Activation* w2_values, T_MlpIndex* w2_col_idx, int* w2_row_ptr,
    const T_Scale*      scales,
    T_Activation*       output
) {
    #pragma HLS INTERFACE s_axilite port=return           bundle=control
    #pragma HLS INTERFACE s_axilite port=tokens_dim       bundle=control
    #pragma HLS INTERFACE s_axilite port=feature_dim      bundle=control
    #pragma HLS INTERFACE s_axilite port=intermediate_dim bundle=control
    #pragma HLS INTERFACE m_axi port=input      bundle=x_mem    depth=(ENC_MLP_FEATURE_W_MAX*ENC_MLP_TOKEN_W_MAX)
    #pragma HLS INTERFACE m_axi port=scales     bundle=sc_mem   depth=SCALE_IDX_MAX
    #pragma HLS INTERFACE m_axi port=w1_values  bundle=w1_mem   depth=MAX_NNZ
    #pragma HLS INTERFACE m_axi port=w1_col_idx bundle=w1_mem   depth=MAX_NNZ
    #pragma HLS INTERFACE m_axi port=w1_row_ptr bundle=w1_mem   depth=MAX_ROWS+1
    #pragma HLS INTERFACE m_axi port=w2_values  bundle=w2_mem   depth=MAX_NNZ
    #pragma HLS INTERFACE m_axi port=w2_col_idx bundle=w2_mem   depth=MAX_NNZ
    #pragma HLS INTERFACE m_axi port=w2_row_ptr bundle=w2_mem   depth=MAX_ROWS+1
    #pragma HLS INTERFACE m_axi port=output     bundle=y_mem    depth=(ENC_MLP_FEATURE_W_MAX*ENC_MLP_TOKEN_W_MAX)

    encoder_mlp_block(input,
                      tokens_dim,
                      feature_dim,
                      intermediate_dim,
                      w1_values,
                      w1_col_idx,
                      w1_row_ptr,
                      w2_values,
                      w2_col_idx,
                      w2_row_ptr,
                      scales,
                      output);
}
