/**********************************************************************************
 * Sparse MLP Layer
 *
 * A two-stage sparse feed-forward layer (biases ignored):
 *     hidden = ReLU(w1 x input)
 *     output = w2 x hidden
 *
 * W1, W2 are sparse weights in CSR format, while the input is dense.  
 *
 * The hidden dense intermediate is never materialized as ReLU is directly applied
 * on the first linear layer's output as it is produced, and the results are stored
 * in CSR format as well. This allows for this intermediate result to reside in 
 * on-chip memory, and thus the DDR round-trip is avoided.
 *
 * This is the latter half of the transformer encoder layer.
 *********************************************************************************/

#ifndef MLP_H
#define MLP_H

#include "mlp_cfg.h"
#include "csr_bounds.h"
#include "encoder_scales.h"

/**
 * @brief Sparse MLP layer.
 *
 * @param[in]  input_packed  Input activations packed in URAM.
 * @param[in]  feature_dim   Output/input feature width.
 * @param[in]  hidden_dim    Hidden width.
 * @param[in]  batch         Number of samples.
 * @param[in]  w1_values     First linear layer's weight values.
 * @param[in]  w1_col_idx    First linear layer's weight column indices.
 * @param[in]  w1_row_ptr    First linear layer's weight row pointers.
 * @param[in]  w2_values     Second linear layer's weight values.
 * @param[in]  w2_col_idx    Second linear layer's weight column indices.
 * @param[in]  w2_row_ptr    Second linear layer's weight row pointers.
 * @param[in]  scales        Per-layer folded scales.
 * @param[out] output        Dense output.
 */
void mlp(
    T_MlpBankWord *input_packed,
    int            feature_dim,
    int            hidden_dim,
    int            batch,
    T_Activation  *w1_values,
    T_MlpIndex    *w1_col_idx,
    int           *w1_row_ptr,
    T_Activation  *w2_values,
    T_MlpIndex    *w2_col_idx,
    int           *w2_row_ptr,
    const T_Scale *scales,
    T_Activation  *output
);

#endif // MLP_H
