/**********************************************************************************
 * Encoder MLP Block
 * 
 * This is the latter half of the transormer encoder layer.
 *
 * The execution flow of the block is:
 *     normalized_input = RMSNorm(input)
 *     mlp_output = MLP(normalized_input) = W2 . ReLU(W1 . normalized_input)
 *     output = input + mlp_output
 *
 * The input activation is expected to be a dense array in feature-major order,
 * which is the layout the sparse MLP already works in, so the block needs no
 * transpose: RMSNorm normalizes each token over its features, and the residual
 * add is a plain elementwise add in the same layout.
 *
 * The output activation is also a dense array in feature-major order, so that a
 * following encoder layer can consume it directly.
 *
 * Every activation carries its own int8 quantization scale. All the requantization
 * scales are runtime arguments folded on the host, so one kernel serves all 12
 * encoder layers.
 *
 * This block acts purely as a high-level wrapper. For more information on the
 * implementation of the layers, please see the respective header files.
 *
 *********************************************************************************/

#ifndef ENCODER_MLP_BLOCK_H
#define ENCODER_MLP_BLOCK_H

#include "mlp.h"
#include "rmsnorm.h"
#include "encoder_mlp_block_cfg.h"
#include "encoder_scales.h"            // EncoderScaleLUTIndex (feed-forward slots of scales[])

/**
 * @brief Feed-forward half of the transformer encoder layer.
 *
 * Computes  output = input + MLP(RMSNorm(input)):
 * @code
 *   normalized = RMSNorm(input)                  // ln_mlp
 *   mlp_output = W2 . ReLU(W1 . normalized)      // two-stage sparse MLP (mlp core)
 *   output     = input + mlp_output              // residual add, requantized
 * @endcode
 * All activations are feature-major int8, which is the layout the Gustavson 
 * sparse-projection MACs read, so no transpose is needed anywhere.
 *
 * @param[in]  input            Input activations, feature-major.
 * @param[in]  tokens_dim       Number of tokens.
 * @param[in]  feature_dim      Feature width.
 * @param[in]  intermediate_dim MLP inner width.
 * @param[in]  w1_bank          fc1 weight: packed {column, value} entries, column-block order.
 * @param[in]  w1_counts        fc1 weight: nonzeros per (column block, row), block-major.
 * @param[in]  w2_bank          fc2 weight entries.
 * @param[in]  w2_counts        fc2 weight counts.
 * @param[in]  scales           Per-layer folded scales.
 * @param[out] output           Block output, feature-major.
 */
void encoder_mlp_block(
    const T_Activation* input,
    int                 tokens_dim,
    int                 feature_dim,
    int                 intermediate_dim,
    T_MhaTileWord* w1_bank, int* w1_counts,
    T_MhaTileWord* w2_bank, int* w2_counts,
    const T_Scale*      scales,
    T_Activation*       output
);

/**
 * @brief Standalone synthesis entry point for the feed-forward block (AXI wrapper).
 */
void encoder_mlp_block_top(
    const T_Activation* input,
    int                 tokens_dim,
    int                 feature_dim,
    int                 intermediate_dim,
    T_MhaTileWord* w1_bank, int* w1_counts,
    T_MhaTileWord* w2_bank, int* w2_counts,
    const T_Scale*      scales,
    T_Activation*       output
);

#endif // ENCODER_MLP_BLOCK_H
