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
    /* Saturate BEFORE rounding, not after.
     *
     * The straightforward order -- add +/-0.5 across the whole value, truncate, then clamp -- puts
     * an adder and a comparator at the FULL width of the incoming value.  For the requantize that
     * value is a 32x32 fixed-point product, so the chain came to ~15 CARRY8 blocks and was this
     * design's critical path: all four paths failing 200 MHz after place-and-route were this
     * arithmetic (see docs/plans/00-open-issues.md).
     *
     * Saturating first is equivalent.  Round-half-away-from-zero then clamp to [-128, 127] yields
     * 127 for every input >= 127.5 and -128 for every input <= -128.5, which the two comparisons
     * below decide directly.  What remains is strictly inside (-128.5, 127.5), so nine integer bits
     * hold it and the rounding adder runs at that width instead of the product's.
     *
     * Comparisons against a constant are also cheaper than an add: the carry chain only has to
     * resolve a magnitude, not produce a sum.
     *
     * Every fractional bit is kept -- the rounding decision depends on all of them -- and no clamp
     * is needed afterwards: an input below 127.5 cannot round above 127, and one above -128.5
     * cannot round below -128 (truncation is toward zero). */
    if (scaled >= (T_Scaled) 127.5f) {
        return (T_Activation) 127;
    }
    if (scaled <= (T_Scaled) -128.5f) {
        return (T_Activation) -128;
    }

    typedef ap_fixed<T_Scaled::width - T_Scaled::iwidth + 9, 9> T_Narrow;
    const T_Narrow narrow  = scaled;
    const T_Narrow rounded = narrow + (narrow >= (T_Narrow) 0 ? (T_Narrow) 0.5f
                                                              : (T_Narrow) -0.5f);
    return (T_Activation) (int) rounded;
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
