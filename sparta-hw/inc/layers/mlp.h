/**********************************************************************************
 * Sparse MLP Layer
 *
 * A two-stage sparse feed-forward layer (biases ignored):
 *     hidden = ReLU(w1 x input)
 *     output = w2 x hidden
 *
 * W1 and W2 are sparse and tiled per column block (the layout compile_model.py emits with
 * --mlp-tiles); the activations are dense.  The hidden activation H stays on-chip: the first
 * linear writes it, ReLU'd and requantized, into the hidden region of the shared activation tiles,
 * and the second linear reads it there.  It never reaches DDR and is never copied.
 *
 * This is the latter half of the transformer encoder layer.
 *********************************************************************************/

#ifndef MLP_H
#define MLP_H

#include "mlp_cfg.h"
#include "spmm_tiled.h"
#include "csr_bounds.h"
#include "encoder_scales.h"

/**
 * @brief Sparse MLP layer: output = W2 . ReLU(W1 . input).
 *
 * Both linears run on the shared tiled SpMM engine (see spmm_tiled.h), in row blocks of
 * MHA_FEATURE_HEAD output rows.  H is written into the hidden region of the activation tiles and
 * read from there by the second linear, so it never leaves the chip and is never copied.
 *
 * @param[in]  input_packed  Input activations, packed.
 * @param[in]  feature_dim   Input / output feature width.
 * @param[in]  hidden_dim    Hidden width.
 * @param[in]  batch         Number of tokens.
 * @param[in]  w1_bank       First linear weight: packed {column, value} entries, block order.
 * @param[in]  w1_counts     First linear weight: nonzeros per (column block, row), block-major.
 * @param[in]  w2_bank       Second linear weight entries.
 * @param[in]  w2_counts     Second linear weight counts.
 * @param[in]  scales        Per-layer folded scales.
 * @param[out] output        Dense output.
 */
void mlp(
    T_MlpBankWord *input_packed,
    int            feature_dim,
    int            hidden_dim,
    int            batch,
    T_MhaTileWord *w1_bank,
    int           *w1_counts,
    T_MhaTileWord *w2_bank,
    int           *w2_counts,
    const T_Scale *scales,
    T_Activation  *output
);

#endif // MLP_H
