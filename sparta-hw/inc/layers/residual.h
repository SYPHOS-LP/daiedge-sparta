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

/**
 * @brief Scaled residual addition
 *
 * @param[in]  residual_input  Residual activations.
 * @param[in]  branch_input    Branch activations.
 * @param[in]  n_elem          Number of elements.
 * @param[in]  residual_scale  Folded scale for the residual input.
 * @param[in]  branch_scale    Folded scale for the branch input.
 * @param[out] residual_output Output activations.
 */
void residual_add(
    const T_Activation* residual_input,
    const T_Activation* branch_input,
    int                 n_elem,
    T_Scale             residual_scale,
    T_Scale             branch_scale,
    T_Activation*       residual_output
);

#endif // RESIDUAL_H
