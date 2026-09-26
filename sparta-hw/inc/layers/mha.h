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
 * @brief Linear (softmax-free) attention for one head.
 *
 * A = K' . V^T over this head's features, then Z = Q' . A normalized by Q' . sum(K'), requantized
 * and written packed into the attention output at this head's feature rows.  Q' and K' arrive as
 * this head's CSR, V as its dense rows; all three are on-chip.
 *
 * @param[in]  enable      Zero returns immediately.  The caller passes a flag rather than guarding
 *                         the call, because a call inside an `if` makes HLS specialise the
 *                         enclosing loop body for that case and duplicate its other functions.
 * @param[in]  head        Head index, which selects the output's feature rows.
 * @param[in]  tokens_dim  Number of tokens.
 * @param[in]  quant_scale Folded requantization scale of the attention output.
 * @param[in]  k_val       K' values / columns / row pointers for this head (feature-major).
 * @param[in]  k_col
 * @param[in]  k_ptr
 * @param[in]  q_val       Q' values / columns / row pointers for this head (token-major).
 * @param[in]  q_col
 * @param[in]  q_ptr
 * @param[in]  v_head      This head's dense V (MHA_FEATURE_HEAD x N).
 * @param[out] out_packed  Packed attention output, all heads.
 */
void head_core (
    int             enable,
    int             head,
    int             tokens_dim,
    T_Scale         quant_scale,
    T_Activation   *k_val,
    T_MhaHeadIndex *k_col,
    int            *k_ptr,
    T_Activation   *q_val,
    T_MhaHeadIndex *q_col,
    int            *q_ptr,
    T_Activation   *v_head,
    T_MhaBankWord  *out_packed
);

#endif // MHA_H
