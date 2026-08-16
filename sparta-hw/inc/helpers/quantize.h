/**********************************************************************************
 * Quantization helpers.
 *
 * Helper functions for quantizing values back to 8 bits.
 *
 * The folded requant scale is applied as a multiply, then round + clamp to int8.
 *********************************************************************************/
#ifndef QUANTIZE_H
#define QUANTIZE_H

#include "encoder_types.h"

/**
 * @brief Round-half-away-from-zero a fixed-point value and clamp to signed int8.
 *
 * @tparam T_Scaled  Scaled type
 *
 * @param  scaled    value already in the output scale
 * @return the saturated int8 code
 */
template <typename T_Scaled>
static inline T_Activation saturate_to_int8(T_Scaled scaled) {
#pragma HLS INLINE
    T_Scaled rounded = scaled + (scaled >= (T_Scaled) 0 ? (T_Scaled) 0.5f : (T_Scaled) -0.5f);
    int value = (int) rounded;

    if (value >  127) {
        value = 127;
    }
    else if (value < -128) {
        value = -128;
    }

    return (T_Activation) value;
}

/**
 * @brief Requantize an integer to int8: multiply by the folded requant
 *        constant, then round + clamp (saturate_to_int8).
 *
 * @tparam T_Acc     Input type (e.g. T_MhaAcc, T_MlpAcc)
 * @tparam T_Quant   Scale type (e.g. T_Scale)
 *
 * @param  acc            Input value
 * @param  requant_const  Scale
 * @return the requantized int8 value
 */
template <typename T_Acc, typename T_Quant>
static inline T_Activation requant_to_int8(T_Acc input, T_Quant requant_const) {
#pragma HLS INLINE
    return saturate_to_int8((T_Quant) input * requant_const);
}

#endif // QUANTIZE_H
