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
 * @param[in]  input_packed  Packed input activations
 * @param[in]  tokens_dim    Number of tokens.
 * @param[in]  wq_values     Q projection weight values.
 * @param[in]  wq_col_idx    Q projection weight column indices.
 * @param[in]  wq_row_ptr    Q projection weight row pointer.
 * @param[in]  wk_values     K projection weight values.
 * @param[in]  wk_col_idx    K projection weight column indices.
 * @param[in]  wk_row_ptr    K projection weight row pointer.
 * @param[in]  wv_values     V projection weight values.
 * @param[in]  wv_col_idx    V projection weight column indices.
 * @param[in]  wv_row_ptr    V projection weight row pointer.
 * @param[in]  wo_values     Output projection weight values.
 * @param[in]  wo_col_idx    Output projection weight column indices.
 * @param[in]  wo_row_ptr    Output projection weight row pointer.
 * @param[in]  scales        Folded scales for requantization.
 * @param[out] output        Dense attention output.
 */
void mha(
    T_MhaBankWord *input_packed,
    int            tokens_dim,
    T_Activation *wq_values, T_MhaIndex *wq_col_idx, int *wq_row_ptr,
    T_Activation *wk_values, T_MhaIndex *wk_col_idx, int *wk_row_ptr,
    T_Activation *wv_values, T_MhaIndex *wv_col_idx, int *wv_row_ptr,
    T_Activation *wo_values, T_MhaIndex *wo_col_idx, int *wo_row_ptr,
    const T_Scale *scales,
    T_Activation *q_val, T_MhaHeadIndex *q_col,
    T_Activation *k_val, T_MhaHeadIndex *k_col,
    T_Activation  *output
);

#endif // MHA_H
