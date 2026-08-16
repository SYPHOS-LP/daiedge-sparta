/**********************************************************************************
 * Top-Level Encoder Layer for Vision Transformers
 *
 * High-level execution flow:
 *
 *     hidden = attention_block(input)    = input  + MHA(RMSNorm(input))
 *     output = feedforward_block(hidden) = hidden + MLP(RMSNorm(hidden))
 *
 * DESIGN NOTES
 * 
 * 1. Top-Level Layer Structure
 * 
 * The Encoder Layer is composed of two main blocks: the attention block and the feed-forward block. 
 * This is a high-level wrapper that connects the two blocks, passing the output of the attention block
 * as the input to the feed-forward block. The layer also manages the scales for quantization 
 * and dequantization, as well as the intermediate hidden activations.
 * 
 * The hidden activations are stored in DDR memory to save on-chip resources, as the intermediate
 * activations can be large (D x N, where D is the feature dimension and N is the number of tokens).
 * 
 * Input to the layer is expected to be in feature-major int8 format, and the output will also be in the same format.
 * This is to ensure that the Gustavson's algorithm for sparse matrix multiplication can be efficiently applied in the 
 * attention and feed-forward blocks. The two blocks are executed sequentially.
 * 
 * For more information on the specific implementation of the attention and feed-forward blocks, please refer to the 
 * respective header files.
 * 
 * 2. Model Quantization
 * 
 * Model is quantized to 4-bit weights and 8-bit activations in Brevitas.
 * 
 * =================
 * For Linear Layers
 * =================
 * 
 * Let's assume y is the expected output of a linear layer, then:
 *     y_float = (weight_int  *weight_scale) * (input_int * input_scale)
 *     y_float = y_int * y_scale
 * 
 * From the above, we can derive that:
 *     y_int = round((weight_int * input_int) * (weight_scale * input_scale / y_scale))
 * 
 * Therefore, to properly requantize the output of the multiplication of the quantized
 * weight and input activation, we need the folded scale factor:
 *    scale_factor = (weight_scale * input_scale) / y_scale
 * 
 * ==================
 * For RMSNorm Layers
 * ==================
 * 
 * Normalization within the RMSNorm layer is performed in floating-point, but the input 
 * and output are quantized. Therefore, we dequantize the input into a floating-point
 * representation, perform the normalization, and then requantize the output back to an 
 * integer representation.
 * 
 * ========================
 * For Residual Connections
 * ========================
 * 
 * Let's assume y is the expected output of a residual connection, then:
 *    y_float = (residual_int * residual_scale) + (branch_int * branch_scale)
 *    y_float = y_int * y_scale
 * 
 * From the above, we can derive that:
 *   y_int = round((residual_int * residual_scale / y_scale) + (branch_int * branch_scale / y_scale))
 * 
 * To properly requantize the output of the residual connection, we need the following scale factors:
 *    residual_scale_factor = residual_scale / y_scale
 *    branch_scale_factor   = branch_scale   / y_scale
 * 
 *********************************************************************************/

#ifndef ENCODER_LAYER_H
#define ENCODER_LAYER_H

#include "encoder_mha_block.h"
#include "encoder_mlp_block.h"
#include "encoder_layer_cfg.h"
#include "encoder_scales.h"    // EncoderScaleLUTIndex (the scales[] array layout)

/**
 * @brief Transformer encoder layer
 *
 * Computes  output = feedforward_block(attention_block(input)), i.e.
 * @code
 *   hidden = input  + MHA(RMSNorm(input))   // attention block
 *   output = hidden + MLP(RMSNorm(hidden))  // feed-forward block
 * @endcode
 *
 * This is a top-level AXI wrapper that runs the two sub-blocks (attention '
 * and feed-forward) in sequence. All activations are feature-major int8.
 *
 * @param[in] input            Input activations, D x N feature-major int8.
 * @param[in] feature_dim      Feature width D (= ENC_FEATURE_W_MAX).
 * @param[in] tokens_dim       Number of tokens N (columns).
 * @param[in] intermediate_dim MLP inner width (rows of fc1, cols of fc2).
 * 
 * @param[in] wq_values        CSR values for MHA Q projection weight.
 * @param[in] wq_col_idx       CSR column indices for MHA Q projection weight.
 * @param[in] wq_row_ptr       CSR row pointers for MHA Q projection weight.
 * @param[in] wk_values        CSR values for MHA K projection weight.
 * @param[in] wk_col_idx       CSR column indices for MHA K projection weight.
 * @param[in] wk_row_ptr       CSR row pointers for MHA K projection weight.
 * @param[in] wv_values        CSR values for MHA V projection weight. 
 * @param[in] wv_col_idx       CSR column indices for MHA V projection weight.
 * @param[in] wv_row_ptr       CSR row pointers for MHA V projection weight.
 * @param[in] wo_values        CSR values for MHA output projection weight.
 * @param[in] wo_col_idx       CSR column indices for MHA output projection weight.
 * @param[in] wo_row_ptr       CSR row pointers for MHA output projection weight.
 * 
 * @param[in] w1_values        CSR values for MLP FC1 weight.
 * @param[in] w1_col_idx       CSR column indices for MLP FC1 weight.
 * @param[in] w1_row_ptr       CSR row pointers for MLP FC1 weight
 * @param[in] w2_values        CSR values for MLP FC2 weight.
 * @param[in] w2_col_idx       CSR column indices for MLP FC2 weight.
 * @param[in] w2_row_ptr       CSR row pointers for MLP FC2 weight.

 * @param[in] scales           Per-layer folded scales.
 * @param[in, out] hidden      DDR scratch for the attention output / feed-forward input, D x N int8.
 * @param[out] output          Layer output, D x N feature-major int8 (DDR).
 */
void encoder_layer_top(
    const T_Activation *input,
    int                 feature_dim,
    int                 tokens_dim,
    int                 intermediate_dim,
    T_Activation *wq_values, T_MhaIndex *wq_col_idx, int *wq_row_ptr,
    T_Activation *wk_values, T_MhaIndex *wk_col_idx, int *wk_row_ptr,
    T_Activation *wv_values, T_MhaIndex *wv_col_idx, int *wv_row_ptr,
    T_Activation *wo_values, T_MhaIndex *wo_col_idx, int *wo_row_ptr,
    T_Activation *w1_values, T_MlpIndex *w1_col_idx, int *w1_row_ptr,
    T_Activation *w2_values, T_MlpIndex *w2_col_idx, int *w2_row_ptr,
    const T_Scale *scales,
    T_Activation *q_val, T_MhaHeadIndex *q_col,
    T_Activation *k_val, T_MhaHeadIndex *k_col,
    T_Activation  *hidden,
    T_Activation  *output
);

#endif // ENCODER_LAYER_H
