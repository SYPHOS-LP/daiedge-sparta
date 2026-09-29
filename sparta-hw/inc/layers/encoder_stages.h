/**********************************************************************************
 * Encoder layer as one sequence of row-block stages
 *
 * The whole layer -- attention and feed-forward -- is a sequence of stages, each computing
 * MHA_FEATURE_HEAD output rows of one weight matrix on the shared tiled SpMM engine
 * (see spmm_tiled.h):
 *
 *   stages                        weight        output
 *   ------------------------------------------------------------------------------------
 *   3 per head, 12 heads          W_K, W_Q, W_V  this head's K', Q' (CSR) and V (dense),
 *                                                followed by the head's linear attention
 *   MHA_ROW_BLOCKS               W_O            the attention block output
 *   hidden_dim / MHA_FEATURE_HEAD W1             H, ReLU'd into the tiles' hidden region
 *   MHA_ROW_BLOCKS               W2             the layer output
 *
 * One loop, so the engine, the weight loaders and the tails have exactly one call site each and
 * HLS builds one of each: it instantiates a function per call site, and a second engine costs
 * ~26k LUT.  RMSNorm and the two residual adds sit at the phase boundaries inside the same loop.
 *********************************************************************************/

#ifndef ENCODER_STAGES_H
#define ENCODER_STAGES_H

#include "spmm_tiled.h"
#include "encoder_scales.h"
#include "encoder_layer_cfg.h"   // ENC_HIDDEN_W_MAX, the bound on the MLP's hidden width

/**
 * @brief Run one encoder layer.
 *
 * @param[in]  input       Layer input, feature-major int8.
 * @param[in]  tokens_dim  Number of tokens.
 * @param[in]  feature_dim Feature width D.
 * @param[in]  hidden_dim  MLP hidden width.
 * @param[in]  wq_bank     Q projection weight: packed {column, value} entries, block order.
 * @param[in]  wq_counts   Q projection weight: nonzeros per (column block, row), block-major.
 * @param[in]  wk_bank     K projection weight entries.
 * @param[in]  wk_counts   K projection weight counts.
 * @param[in]  wv_bank     V projection weight entries.
 * @param[in]  wv_counts   V projection weight counts.
 * @param[in]  wo_bank     Output projection weight entries.
 * @param[in]  wo_counts   Output projection weight counts.
 * @param[in]  w1_bank     MLP fc1 weight entries.
 * @param[in]  w1_counts   MLP fc1 weight counts.
 * @param[in]  w2_bank     MLP fc2 weight entries.
 * @param[in]  w2_counts   MLP fc2 weight counts.
 * @param[in]  scales      Per-layer folded scales.
 * @param[out] output      Layer output, feature-major int8.  Written only when last_layer is set.
 * @param[in]  first_layer Non-zero to read the layer input from `input`; otherwise the activation
 *                         left on-chip by the previous layer is used.
 * @param[in]  last_layer  Non-zero to write the result to `output`; otherwise it stays on-chip for
 *                         the next layer.
 */
void encoder_layer_stages(
    const T_Activation *input,
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
    T_Activation       *output,
    int                 first_layer,
    int                 last_layer
);

#endif // ENCODER_STAGES_H
