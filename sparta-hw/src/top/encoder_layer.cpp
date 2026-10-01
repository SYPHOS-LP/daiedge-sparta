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
static_assert(ENC_FEATURE_W_MAX == MHA_FEATURE_W_MAX,
              "ENC_FEATURE_W_MAX must match MHA_FEATURE_W_MAX");
static_assert(ENC_TOKEN_W_MAX == MHA_TOKEN_W_MAX,
              "ENC_TOKEN_W_MAX must match MHA_TOKEN_W_MAX");

void encoder_layer_top(
    const T_MhaBankWord*  input,
    int               feature_dim,
    int               tokens_dim,
    int               intermediate_dim,
    T_MhaTileWord* wq_bank, T_MhaCountWord* wq_counts,
    T_MhaTileWord* wk_bank, T_MhaCountWord* wk_counts,
    T_MhaTileWord* wv_bank, T_MhaCountWord* wv_counts,
    T_MhaTileWord* wo_bank, T_MhaCountWord* wo_counts,
    T_MhaTileWord* w1_bank, T_MhaCountWord* w1_counts,
    T_MhaTileWord* w2_bank, T_MhaCountWord* w2_counts,
    const T_Scale* scales,
    T_MhaBankWord* output,
    int           first_layer,
    int           last_layer
) {
    /* AXI read/write sizing.  ALL THREE NUMBERS BELOW ARE FROM BOARD MEASUREMENTS, because every
     * attempt to reason about them from first principles was wrong.
     *
     * num_read_outstanding is CAPPED AT 16 by Vitis.  A pragma asking for 32 is accepted silently
     * and the generated master still reports NUM_READ_OUTSTANDING = 16 (check
     * <project>/hls/syn/verilog/*_m_axi.v if in doubt).  Do not raise it expecting an effect.
     *
     * What the measurements say, middle layers only (they do not touch the activation ports, so the
     * comparison is clean):
     *
     *   16 outstanding, 256-beat bursts   2.063 M cycles/layer
     *    4 outstanding, 256-beat bursts   2.858 M   -- starving the requests costs 795 k
     *   16 outstanding,  64-beat bursts   1.650 M   -- shortening the bursts GAINS 413 k
     *
     * So the outstanding count has a floor worth ~800 k cycles, and above that floor the burst
     * length is the lever: shorter is faster, which is the opposite of the usual advice.  The
     * mechanism is not established -- a long burst commits the path while the datapath already
     * wants the next address is the best guess, and a guess is all it is.
     *
     * Vitis sizes the data FIFO at roughly burst x outstanding, so shortening the burst also buys
     * BRAM back, which is why this direction is cheap to explore.  Those FIFOs were 56 BRAM18 of
     * the design's 136 at 256 beats. */
    #pragma HLS INTERFACE s_axilite port=return     bundle=control
    #pragma HLS INTERFACE s_axilite port=tokens_dim bundle=control
    #pragma HLS INTERFACE m_axi port=scales bundle=sc_mem depth=SCALE_IDX_MAX num_read_outstanding=16 num_write_outstanding=2 max_read_burst_length=32
    #pragma HLS INTERFACE s_axilite port=intermediate_dim bundle=control
    #pragma HLS INTERFACE s_axilite port=first_layer      bundle=control
    #pragma HLS INTERFACE s_axilite port=last_layer       bundle=control
    #pragma HLS INTERFACE s_axilite port=feature_dim      bundle=control
    #pragma HLS INTERFACE m_axi port=input     bundle=in_mem depth=MHA_BANK_WORDS num_read_outstanding=16 num_write_outstanding=2 max_read_burst_length=32
    #pragma HLS INTERFACE m_axi port=wq_bank  bundle=w_mha_mem depth=MAX_NNZ num_read_outstanding=16 num_write_outstanding=2 max_read_burst_length=32
    #pragma HLS INTERFACE m_axi port=wq_counts bundle=w_mha_mem depth=(MHA_TILES*(MAX_ROWS+1)) num_read_outstanding=16 num_write_outstanding=2 max_read_burst_length=32
    #pragma HLS INTERFACE m_axi port=wk_bank  bundle=w_mha_mem depth=MAX_NNZ num_read_outstanding=16 num_write_outstanding=2 max_read_burst_length=32
    #pragma HLS INTERFACE m_axi port=wk_counts bundle=w_mha_mem depth=(MHA_TILES*(MAX_ROWS+1)) num_read_outstanding=16 num_write_outstanding=2 max_read_burst_length=32
    #pragma HLS INTERFACE m_axi port=wv_bank  bundle=w_mha_mem depth=MAX_NNZ num_read_outstanding=16 num_write_outstanding=2 max_read_burst_length=32
    #pragma HLS INTERFACE m_axi port=wv_counts bundle=w_mha_mem depth=(MHA_TILES*(MAX_ROWS+1)) num_read_outstanding=16 num_write_outstanding=2 max_read_burst_length=32
    #pragma HLS INTERFACE m_axi port=wo_bank  bundle=w_mha_mem depth=MAX_NNZ num_read_outstanding=16 num_write_outstanding=2 max_read_burst_length=32
    #pragma HLS INTERFACE m_axi port=wo_counts bundle=w_mha_mem depth=(MHA_TILES*(MAX_ROWS+1)) num_read_outstanding=16 num_write_outstanding=2 max_read_burst_length=32
    #pragma HLS INTERFACE m_axi port=w1_bank  bundle=w1_mem depth=MAX_NNZ num_read_outstanding=16 num_write_outstanding=2 max_read_burst_length=32
    #pragma HLS INTERFACE m_axi port=w1_counts bundle=w1_mem depth=(MHA_TILES*(ENC_HIDDEN_W_MAX+1)) num_read_outstanding=16 num_write_outstanding=2 max_read_burst_length=32
    #pragma HLS INTERFACE m_axi port=w2_bank  bundle=w2_mem depth=MAX_NNZ num_read_outstanding=16 num_write_outstanding=2 max_read_burst_length=32
    #pragma HLS INTERFACE m_axi port=w2_counts bundle=w2_mem depth=(MHA_TILES*(MAX_ROWS+1)) num_read_outstanding=16 num_write_outstanding=2 max_read_burst_length=32
    #pragma HLS INTERFACE m_axi port=output bundle=y_mem depth=MHA_BANK_WORDS num_read_outstanding=16 num_write_outstanding=16 max_read_burst_length=32 max_write_burst_length=32
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
                         output,
                         first_layer,
                         last_layer);
}
