/**********************************************************************************
 * Residual Addition Layer
 * 
 * This is a simple adder, that sums the branch activation to the residual 
 * activation. The output is requantized to 8 bits, by multiplying each
 * input with the relevant scale.
 * 
 * For more information about the scales, see encoder_layer.h
 *********************************************************************************/

#ifndef RESIDUAL_H
#define RESIDUAL_H

#include "encoder_types.h"
#include "mha_cfg.h"   // MHA_BANK_PACK / T_MhaBankWord (the shared packed-word form)

/**
 * @brief Scaled residual addition
 *
 * @param[in]  residual_input   Packed operand, MHA_BANK_PACK tokens per word.
 * @param[in]  branch_input   Packed operand, MHA_BANK_PACK tokens per word.
 * @param[in]  n_words         Number of packed words (MHA_BANK_PACK tokens each).
 * @param[in]  residual_scale  Folded scale for the residual input.
 * @param[in]  branch_scale    Folded scale for the branch input.
 * @param[out] residual_output Output activations.
 */
void residual_add(
    const T_MhaBankWord* residual_input,
    const T_MhaBankWord* branch_input,
    int                  n_words,
    T_Scale              residual_scale,
    T_Scale              branch_scale,
    T_MhaBankWord*       residual_output
);

#endif // RESIDUAL_H
