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

/* How V is laid out on chip: one 64-bit word per (token, group of MHA_BANK_PACK output features).
 * So one read returns MHA_BANK_PACK of a token's values, and a token's MHA_FEATURE_HEAD values take
 * MHA_V_GROUPS reads rather than MHA_FEATURE_HEAD reads.
 *
 * The reason for ordering it by token: the inner loop of K'.V takes one K' value and multiplies it
 * by all MHA_FEATURE_HEAD output features of a single token.  The values one read should return are
 * therefore consecutive output features at the same token.  When V was ordered by feature those
 * values sat a row stride apart, so each had to be read separately, and V had to be spread over
 * MHA_BANK_PACK memories to supply enough reads per cycle.  See
 * docs/plans/09-attention-rework.md.
 *
 * Defined here rather than in mha_cfg.h because that file is generated from
 * config/yml/mha.yaml. */
#define MHA_V_GROUPS (MHA_FEATURE_HEAD / MHA_BANK_PACK)
#define MHA_V_PACKED_WORDS (MHA_TOKEN_W_MAX * MHA_V_GROUPS)

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
 * @param[in]  v_packed    This head's dense V, token-major and packed.
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
    T_MhaBankWord  *v_packed,
    T_MhaBankWord  *out_packed
);

#endif // MHA_H
