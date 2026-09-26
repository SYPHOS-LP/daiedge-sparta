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
    T_MhaTileWord* w1_bank, int* w1_counts,
    T_MhaTileWord* w2_bank, int* w2_counts,
    const T_Scale*      scales,
    T_Activation*       output
) {
#ifndef __SYNTHESIS__
    assert(feature_dim <= ENC_MLP_FEATURE_W_MAX && "feature_dim exceeds ENC_MLP_FEATURE_W_MAX");
    assert(tokens_dim  <= ENC_MLP_TOKEN_W_MAX && "tokens_dim exceeds ENC_MLP_TOKEN_W_MAX");
#endif

    /* RMSNorm output written in URAM in packed format (+ the spare word the SpMM's last token tile
     * reads past the final row; its value is never used). */
    static T_MlpBankWord norm_input[MLP_BANK_WORDS_ALLOC];
    #pragma HLS BIND_STORAGE    variable=norm_input type=ram_t2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=norm_input cyclic factor=MLP_BANK_CYCLIC dim=1
    rmsnorm(input,
            tokens_dim,
            feature_dim,
            scales[SCALE_FF_RMSNORM_OUT_INV_IDX],
            reinterpret_cast<T_RmsBankWord*> (norm_input));

    mlp(norm_input,
        feature_dim,
        intermediate_dim,
        tokens_dim,
        w1_bank,
        w1_counts,
        w2_bank,
        w2_counts,
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
    T_MhaTileWord* w1_bank, int* w1_counts,
    T_MhaTileWord* w2_bank, int* w2_counts,
    const T_Scale*      scales,
    T_Activation*       output
) {
    #pragma HLS INTERFACE s_axilite port=return           bundle=control
    #pragma HLS INTERFACE s_axilite port=tokens_dim       bundle=control
    #pragma HLS INTERFACE s_axilite port=feature_dim      bundle=control
    #pragma HLS INTERFACE s_axilite port=intermediate_dim bundle=control
    #pragma HLS INTERFACE m_axi port=input      bundle=x_mem    depth=(ENC_MLP_FEATURE_W_MAX*ENC_MLP_TOKEN_W_MAX)
    #pragma HLS INTERFACE m_axi port=scales     bundle=sc_mem   depth=SCALE_IDX_MAX
    #pragma HLS INTERFACE m_axi port=w1_bank  bundle=w1_mem   depth=MAX_NNZ
    #pragma HLS INTERFACE m_axi port=w1_counts bundle=w1_mem   depth=(MHA_TILES*MLP_HIDDEN_FEATURE_W_MAX)
    #pragma HLS INTERFACE m_axi port=w2_bank  bundle=w2_mem   depth=MAX_NNZ
    #pragma HLS INTERFACE m_axi port=w2_counts bundle=w2_mem   depth=(MHA_TILES*MAX_ROWS)
    #pragma HLS INTERFACE m_axi port=output     bundle=y_mem    depth=(ENC_MLP_FEATURE_W_MAX*ENC_MLP_TOKEN_W_MAX)

    encoder_mlp_block(input,
                      tokens_dim,
                      feature_dim,
                      intermediate_dim,
                      w1_bank,
                      w1_counts,
                      w2_bank,
                      w2_counts,
                      scales,
                      output);
}
