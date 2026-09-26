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
    T_MhaTileWord* wq_bank, int* wq_counts,
    T_MhaTileWord* wk_bank, int* wk_counts,
    T_MhaTileWord* wv_bank, int* wv_counts,
    T_MhaTileWord* wo_bank, int* wo_counts,
    const T_Scale*      scales,
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
            reinterpret_cast<T_RmsBankWord*>(norm_input));

    mha(norm_input,
        tokens_dim,
        wq_bank, wq_counts,
        wk_bank, wk_counts,
        wv_bank, wv_counts,
        wo_bank, wo_counts,
        scales,
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
    T_MhaTileWord* wq_bank, int* wq_counts,
    T_MhaTileWord* wk_bank, int* wk_counts,
    T_MhaTileWord* wv_bank, int* wv_counts,
    T_MhaTileWord* wo_bank, int* wo_counts,
    const T_Scale*      scales,
    T_Activation*       hidden
) {
    #pragma HLS INTERFACE s_axilite port=return      bundle=control
    #pragma HLS INTERFACE s_axilite port=tokens_dim  bundle=control
    #pragma HLS INTERFACE s_axilite port=feature_dim bundle=control
    #pragma HLS INTERFACE m_axi port=input  bundle=in_mem depth=(ENC_MHA_FEATURE_W_MAX*ENC_MHA_TOKEN_W_MAX)
    #pragma HLS INTERFACE m_axi port=scales bundle=sc_mem depth=SCALE_IDX_MAX
    #pragma HLS INTERFACE m_axi port=wq_bank  bundle=wq_mem depth=MAX_NNZ
    #pragma HLS INTERFACE m_axi port=wq_counts bundle=wq_mem depth=(MHA_TILES*MAX_ROWS)
    #pragma HLS INTERFACE m_axi port=wk_bank  bundle=wk_mem depth=MAX_NNZ
    #pragma HLS INTERFACE m_axi port=wk_counts bundle=wk_mem depth=(MHA_TILES*MAX_ROWS)
    #pragma HLS INTERFACE m_axi port=wv_bank  bundle=wv_mem depth=MAX_NNZ
    #pragma HLS INTERFACE m_axi port=wv_counts bundle=wv_mem depth=(MHA_TILES*MAX_ROWS)
    #pragma HLS INTERFACE m_axi port=wo_bank  bundle=wo_mem depth=MAX_NNZ
    #pragma HLS INTERFACE m_axi port=wo_counts bundle=wo_mem depth=(MHA_TILES*MAX_ROWS)
    #pragma HLS INTERFACE m_axi port=hidden bundle=hidden_mem depth=(ENC_MHA_FEATURE_W_MAX*ENC_MHA_TOKEN_W_MAX)

    encoder_mha_block(input, tokens_dim, feature_dim,
                      wq_bank, wq_counts,
                      wk_bank, wk_counts,
                      wv_bank, wv_counts,
                      wo_bank, wo_counts,
                      scales,
                      hidden);
}
