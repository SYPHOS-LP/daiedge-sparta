/**********************************************************************************
 * Root-Mean-Square Normalization Layer
 *
 * Ref. https://docs.pytorch.org/docs/2.13/generated/torch.nn.RMSNorm.html
 *
 * This layer performs the following operation:
 *    RMS(x) = sqrt(eps + (1/feature_dim) * sum_i(x^2))
 *    y[i] = x[i] / RMS(x)
 *
 * The gamma weight is fused into the following linear layer, so RMSNorm applies no gamma.
 *
 * RMS is computed using a fixed-point reciprocal square root (rsqrt) lookup table (LUT). The input
 * and output are both in int8 format. The input is dequantized using the input scale to enhance the
 * precision of the RMS computation. The output is then requantized using the inverse output scale.
 *
 * Both the input and output are in feature-major layout, as this is the layout used throughout the encoder.
 *
 * More information on the rsqrt LUT can be found in the rmsnorm_rsqrt.h file.
 *
 *********************************************************************************/

#ifndef RMSNORM_H
#define RMSNORM_H

#include "rmsnorm_cfg.h"

/**
 * @brief Root-Mean-Square layer normalization.
 *
 * Normalizes each token independently over its @p feature_dim features.
 *
 * Both operands hold RMS_BANK_PACK tokens of one feature per word, features in order: one read
 * covers RMS_BANK_PACK tokens, so the pass over the data costs
 * feature_dim * ceil(tokens_dim / RMS_BANK_PACK) reads rather than one per element.
 *
 * @param packed_input           Input activations, packed, feature-major.
 * @param tokens_dim             Number of tokens to normalize.
 * @param feature_dim            Features per token.
 * @param inverse_output_scale   Inverse output quantization scale (power-of-two shift).
 * @param packed_output          Output activations, same packing as the input.
 */
void rmsnorm(
    const T_RmsBankWord* packed_input,
    int                  tokens_dim,
    int                  feature_dim,
    T_Scale              inverse_output_scale,
    T_RmsBankWord*       packed_output
);

#endif // RMSNORM_H
