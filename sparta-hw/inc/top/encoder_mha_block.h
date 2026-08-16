/**********************************************************************************
 * Encoder Multi-Head Attention Block
 *
 * This is the former half of the transormer encoder layer.
 * 
 * The execution flow of the block is:
 *     normalized_input = RMSNorm(input)
 *     attention_output = MHA(normalized_input)
 *     output           = input + attention_output
 * 
 * The input activation is expected to be a dense array in feature-major order,
 * so that the Gustavson algorithm for sparse matrix multiplicationcan be properly 
 * used in the multi-head attention block's linear projections.
 * 
 * The output activation is also a dense array in feature-major order, as the 
 * MLP block expects the input to be in that format for the same reason.
 * 
 * This block acts purely as a high-level wrapper. For more information on the
 * implementation of the layers, please see the respective header files.
 *
 *********************************************************************************/

#ifndef ENCODER_MHA_BLOCK_H
#define ENCODER_MHA_BLOCK_H

#include "mha.h"
#include "rmsnorm.h"
#include "encoder_mha_block_cfg.h"
#include "encoder_scales.h"    // EncoderScaleLUTIndex (attention-block slots of scales[])

/**
 * @brief Attention half of the transformer encoder layer.
 *
 * Computes  hidden = input + MHA(RMSNorm(input)):
 * @code
 *   normalized = RMSNorm(input)      // ln_mha
 *   attention  = MHA(normalized)     // linear softmax-free attention (mha core)
 *   hidden     = input + attention   // residual add, requantized
 * @endcode
 * All activations are feature-major int8, which is the layout the Gustavson 
 * sparse-projection MACs read, so no transpose is needed anywhere.
 *
 * @param[in]  input        Input activations, feature-major.
 * @param[in]  tokens_dim   Number of tokens.
 * @param[in]  feature_dim  Feature width.
 * @param[in]  wq_values    CSR values for the Q projection weight.
 * @param[in]  wq_col_idx   CSR column indices for the Q projection weight.
 * @param[in]  wq_row_ptr   CSR row pointers for the Q projection weight.
 * @param[in]  wk_values    CSR values for the K projection weight.
 * @param[in]  wk_col_idx   CSR column indices for the K projection weight.
 * @param[in]  wk_row_ptr   CSR row pointers for the K projection weight.
 * @param[in]  wv_values    CSR values for the V projection weight.
 * @param[in]  wv_col_idx   CSR column indices for the V projection weight.
 * @param[in]  wv_row_ptr   CSR row pointers for the V projection weight.
 * @param[in]  wo_values    CSR values for the output projection weight.
 * @param[in]  wo_col_idx   CSR column indices for the output projection weight.
 * @param[in]  wo_row_ptr   CSR row pointers for the output projection weight.
 * @param[in]  scales       Per-layer folded scales.
 * @param[out] hidden       Block output, feature-major.
 */
void encoder_mha_block(
    const T_Activation *input,
    int                 tokens_dim,
    int                 feature_dim,
    T_Activation *wq_values, T_MhaIndex *wq_col_idx, int *wq_row_ptr,
    T_Activation *wk_values, T_MhaIndex *wk_col_idx, int *wk_row_ptr,
    T_Activation *wv_values, T_MhaIndex *wv_col_idx, int *wv_row_ptr,
    T_Activation *wo_values, T_MhaIndex *wo_col_idx, int *wo_row_ptr,
    const T_Scale *scales,
    T_Activation *q_val, T_MhaHeadIndex *q_col,
    T_Activation *k_val, T_MhaHeadIndex *k_col,
    T_Activation  *hidden
);

/**
 * @brief Standalone synthesis entry point for the attention block (AXI wrapper).
 */
void encoder_mha_block_top(
    const T_Activation* input,
    int                 tokens_dim,
    int                 feature_dim,
    T_Activation* wq_values, T_MhaIndex* wq_col_idx, int* wq_row_ptr,
    T_Activation* wk_values, T_MhaIndex* wk_col_idx, int* wk_row_ptr,
    T_Activation* wv_values, T_MhaIndex* wv_col_idx, int* wv_row_ptr,
    T_Activation* wo_values, T_MhaIndex* wo_col_idx, int* wo_row_ptr,
    const T_Scale* scales,
    T_Activation* q_val, T_MhaHeadIndex* q_col,
    T_Activation* k_val, T_MhaHeadIndex* k_col,
    T_Activation*  hidden
);

#endif // ENCODER_MHA_BLOCK_H

