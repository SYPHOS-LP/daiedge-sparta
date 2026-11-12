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

#include "encoder_layer_cfg.h"
#include "encoder_stages.h"
#include "csr_bounds.h"   // MAX_NNZ / MAX_ROWS, the m_axi port depths
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
 * The four MHA weights are emitted tiled: one array of packed entries (a value and its
 * block-local column per slot, column blocks in order) plus the nonzeros per (column block, row).
 * The two MLP weights are still CSR / CSC triples.
 * @param[in] wq_bank          Q projection weight entries.
 * @param[in] wq_counts        Q projection weight counts.
 * @param[in] wk_bank          K projection weight entries.
 * @param[in] wk_counts        K projection weight counts.
 * @param[in] wv_bank          V projection weight entries.
 * @param[in] wv_counts        V projection weight counts.
 * @param[in] wo_bank          Output projection weight entries.
 * @param[in] wo_counts        Output projection weight counts.
 * @param[in] w1_bank          MLP fc1 weight entries.
 * @param[in] w1_counts        MLP fc1 weight counts.
 * @param[in] w2_bank          MLP fc2 weight entries.
 * @param[in] w2_counts        MLP fc2 weight counts.
 * 
 * 

 * @param[in] scales           Per-layer folded scales.
 * @param[out] output          Layer output, D x N feature-major int8 (DDR), written only when
 *                             last_layer is set.
 * @param[in] first_layer      Non-zero to read the input from DDR; otherwise the activation the
 *                             previous layer left on-chip is used.
 * @param[in] last_layer       Non-zero to write the result to DDR; otherwise it stays on-chip.
 *
 * Buffer contract: the weight value and column-index buffers are burst-read in wide words
 * (up to 256 bits) and the final word is read whole, so each buffer must be readable
 * (zero-padded) up to the next multiple of 32 elements.
 */
void encoder_layer_top(
    const T_MhaBankWord *input,
    int                 feature_dim,
    int                 tokens_dim,
    int                 intermediate_dim,
    T_MhaTileWord *wq_bank, T_MhaCountWord *wq_counts,
    T_MhaTileWord *wk_bank, T_MhaCountWord *wk_counts,
    T_MhaTileWord *wv_bank, T_MhaCountWord *wv_counts,
    T_MhaTileWord *wo_bank, T_MhaCountWord *wo_counts,
    T_MhaTileWord *w1_bank, T_MhaCountWord *w1_counts,
    T_MhaTileWord *w2_bank, T_MhaCountWord *w2_counts,
    const T_Scale *scales,
    T_MhaBankWord *output,
    int            first_layer,
    int            last_layer
);

#endif // ENCODER_LAYER_H
