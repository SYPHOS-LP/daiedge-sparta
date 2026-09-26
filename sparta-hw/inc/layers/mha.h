/**********************************************************************************
 * Linear Multi-Head Attention Layer
 * 
 * This is a custom implementation of the multi-head attention layer, which has
 * no softmax layer and is designed to produce sparse intermediate activations
 * wherever possible.
 * 
 * The execution flow of the layer is:
 *     Q_sparse = ReLU(input * weight_q)
 *     K_sparse = transpose(ReLU(input * weight_k))
 *     V_dense  = input * weight_v
 * 
 *     hidden_numerator   = Q_sparse * (K_sparse * V_dense)
 *     hidden_denominator = Q_sparse * sum(K_sparse)
 *     hidden = hidden_numerator / (hidden_denominator + eps)
 * 
 *     output = hidden * weight_o
 * 
 * The input activation is expected to be a dense array in feature-major order,
 * so that the Gustavson algorithm for sparse matrix multiplication can be properly 
 * used.
 *
 * This is the former half of the transformer encoder layer.
 * 
 * The output activation is also a dense array in feature-major order, as the 
 * MLP block expects the input to be in that format for the same reason.
 *********************************************************************************/

#ifndef MHA_H
#define MHA_H

#include "mha_cfg.h"
#include "csr_bounds.h"
#include "encoder_scales.h"

/**
 * @brief Linear multi-head attention (softmax-free).
 *
 * @param[in]  input_packed  Packed input activations.
 * @param[in]  tokens_dim    Number of tokens.
 * @param[in]  wq_bank       Q projection weight: packed {column, value} entries, column-block order.
 * @param[in]  wq_counts     Q projection weight: nonzeros per (column block, row), block-major.
 * @param[in]  wk_bank       K projection weight entries.
 * @param[in]  wk_counts     K projection weight counts.
 * @param[in]  wv_bank       V projection weight entries.
 * @param[in]  wv_counts     V projection weight counts.
 * @param[in]  wo_bank       Output projection weight entries.
 * @param[in]  wo_counts     Output projection weight counts.
 * @param[in]  scales        Folded scales for requantization.
 * @param[out] output        Dense attention output.
 */
void mha(
    T_MhaBankWord *input_packed,
    int            tokens_dim,
    T_MhaTileWord *wq_bank, int *wq_counts,
    T_MhaTileWord *wk_bank, int *wk_counts,
    T_MhaTileWord *wv_bank, int *wv_counts,
    T_MhaTileWord *wo_bank, int *wo_counts,
    const T_Scale *scales,
    T_Activation  *output
);

#endif // MHA_H
