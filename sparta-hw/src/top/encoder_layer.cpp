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

// The layer dims are the single source of truth; assert the layer-core config literals agree so
// they cannot drift across the config headers.
static_assert(ENC_FEATURE_W_MAX == MHA_FEATURE_W_MAX && ENC_FEATURE_W_MAX == MLP_FEATURE_W_MAX,
              "ENC_FEATURE_W_MAX must match MHA_FEATURE_W_MAX / MLP_FEATURE_W_MAX");
static_assert(ENC_TOKEN_W_MAX == MHA_TOKEN_W_MAX && ENC_TOKEN_W_MAX == MLP_TOKEN_W_MAX,
              "ENC_TOKEN_W_MAX must match MHA_TOKEN_W_MAX / MLP_TOKEN_W_MAX");

void encoder_layer_top(
    const T_Activation*   input,
    int               feature_dim,
    int               tokens_dim,
    int               intermediate_dim,
    T_MhaTileWord* wq_bank, int* wq_counts,
    T_MhaTileWord* wk_bank, int* wk_counts,
    T_MhaTileWord* wv_bank, int* wv_counts,
    T_MhaTileWord* wo_bank, int* wo_counts,
    T_MhaTileWord* w1_bank, int* w1_counts,
    T_MhaTileWord* w2_bank, int* w2_counts,
    const T_Scale* scales,
    /* DDR-resident activation scratch: Q'/K' val/col (full grid D*N).  Q'/K' row pointers
     * are on-chip in mha() (produced+consumed there).  */
    T_Activation* output
) {
    #pragma HLS INTERFACE s_axilite port=return     bundle=control
    #pragma HLS INTERFACE s_axilite port=tokens_dim bundle=control
    #pragma HLS INTERFACE m_axi port=scales bundle=sc_mem depth=SCALE_IDX_MAX num_read_outstanding=16 num_write_outstanding=2 max_read_burst_length=256
    #pragma HLS INTERFACE s_axilite port=intermediate_dim bundle=control
    #pragma HLS INTERFACE s_axilite port=feature_dim      bundle=control
    #pragma HLS INTERFACE m_axi port=input     bundle=in_mem depth=(ENC_FEATURE_W_MAX*ENC_TOKEN_W_MAX) num_read_outstanding=16 num_write_outstanding=2 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=wq_bank  bundle=w_mha_mem depth=MAX_NNZ num_read_outstanding=4 num_write_outstanding=2 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=wq_counts bundle=w_mha_mem depth=(MHA_TILES*MAX_ROWS) num_read_outstanding=4 num_write_outstanding=2 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=wk_bank  bundle=w_mha_mem depth=MAX_NNZ num_read_outstanding=4 num_write_outstanding=2 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=wk_counts bundle=w_mha_mem depth=(MHA_TILES*MAX_ROWS) num_read_outstanding=4 num_write_outstanding=2 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=wv_bank  bundle=w_mha_mem depth=MAX_NNZ num_read_outstanding=4 num_write_outstanding=2 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=wv_counts bundle=w_mha_mem depth=(MHA_TILES*MAX_ROWS) num_read_outstanding=4 num_write_outstanding=2 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=wo_bank  bundle=w_mha_mem depth=MAX_NNZ num_read_outstanding=4 num_write_outstanding=2 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=wo_counts bundle=w_mha_mem depth=(MHA_TILES*MAX_ROWS) num_read_outstanding=4 num_write_outstanding=2 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=w1_bank  bundle=w1_mem depth=MAX_NNZ num_read_outstanding=16 num_write_outstanding=2 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=w1_counts bundle=w1_mem depth=(MHA_TILES*MLP_HIDDEN_FEATURE_W_MAX) num_read_outstanding=16 num_write_outstanding=2 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=w2_bank  bundle=w2_mem depth=MAX_NNZ num_read_outstanding=16 num_write_outstanding=2 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=w2_counts bundle=w2_mem depth=(MHA_TILES*MAX_ROWS) num_read_outstanding=16 num_write_outstanding=2 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=output bundle=y_mem depth=(ENC_FEATURE_W_MAX*ENC_TOKEN_W_MAX) num_read_outstanding=16 num_write_outstanding=16 max_read_burst_length=256 max_write_burst_length=256
    /* DDR-resident activation scratch: Q'/K' + H (single region each). */
    /* Each CSR array on its OWN bundle: sharing a bundle across val/col/ptr blocks HLS bursting
     * ("multiple potential writes to the same bundle in the same region"). One array per bundle
     * lets each be bursted independently. */

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

    /* On-chip intermediate ("hidden") activation. */
    static T_Activation hidden_oc[ENC_FEATURE_W_MAX * ENC_TOKEN_W_MAX];
    #pragma HLS BIND_STORAGE variable=hidden_oc type=ram_t2p impl=bram

    /* The whole layer: RMSNorm, the attention half's stages, its residual, RMSNorm again, the
     * feed-forward stages and its residual -- one stage loop, so one SpMM engine. */
    encoder_layer_stages(input, tokens_dim, feature_dim, intermediate_dim,
                         wq_bank, wq_counts,
                         wk_bank, wk_counts,
                         wv_bank, wv_counts,
                         wo_bank, wo_counts,
                         w1_bank, w1_counts,
                         w2_bank, w2_counts,
                         layer_scales,
                         hidden_oc,
                         output);
}
