/**********************************************************************************
 * Reciprocal Square Root for RMSNorm layer
 *
 * This file implements the reciprocal square root (rsqrt) function for the RMSNorm layer. 
 * The implementation uses a lookup table to approximate the function, as this is 
 * more efficient from a hardware perspective than computing the function directly. 
 * 
 * The lookup table is generated at compile time and is stored in a dedicated ROM.
 * 
 * The file contains three different variants of the reciprocal square root function,
 * which can be selected at compile time using the RMS_RSQRT_VARIANT macro. The three
 * variants are:
 *     1. UNIFORM:
 *        A uniform lookup table that approximates the function over a fixed range.
 *     2. LOG: 
 *        A lookup table with logarithmic spacing for better accuracy over a wide dynamic range.
 *     3. EXPMANT: 
 *        A lookup table with exponent-and-mantissa-based addressing for optimal hardware efficiency.
 *
 * The following table provides insight on the accuracy of the different variants,
 * with respect to the number of address bits used for the lookup table:
 * 
 * TODO: Update this table with the actual accuracy values for each variant.
 *  -----------------------------------------------------
 * | Address Bits | Entries |     Max Relative Error     |
 * |--------------|---------| Uniform |  Log  | Exp-Mant |
 * |      5       |   32    |  ~1.5%  | ~1.5% |  ~1.5%   |
 * |      6       |   64    |  ~0.8%  | ~0.8% |  ~0.8%   |
 * |      7       |  128    |  ~0.4%  | ~0.4% |  ~0.4%   |
 * |      8       |  256    |  ~0.2%  | ~0.2% |  ~0.2%   |
 *  -----------------------------------------------------
 *********************************************************************************/

#ifndef RMSNORM_RSQRT_H
#define RMSNORM_RSQRT_H

#include <cmath>
#include "rmsnorm_cfg.h"

#if RMS_RSQRT_VARIANT == RMS_RSQRT_UNIFORM
 /* Uniform variant cannot resolve rsqrt near zero reliably. */
 #define RMS_MS_LO (1.0f)
#else
 #define RMS_MS_LO ((float) RMS_EPS)
#endif

#define RMS_MS_HI (1.10f * 127.0f * 127.0f)


#if RMS_RSQRT_VARIANT == RMS_RSQRT_UNIFORM

struct RmsUniformLut {
    T_RmsRsqrt val[RMS_LUT_SIZE];
};

/*
 * Uniformly distributed lookup table
 *
 * The uniform variant of the reciprocal square root function uses a lookup table 
 * that is uniformly distributed over the range [RMS_MS_LO, RMS_MS_HI]. 
 * 
 * The step used for each entry is equal to:
 *     step = (RMS_MS_HI - RMS_MS_LO) / RMS_LUT_SIZE
 * 
 * For better accuracy, the lookup table stores the value of rsqrt at the center of each step:
 *     entry[k] = 1 / sqrt(RMS_MS_LO + (k + 0.5) * step)
 * 
 * At runtime, the input mean-square value is clamped to the range [RMS_MS_LO, RMS_MS_HI] 
 * and mapped to the corresponding entry in the lookup table using:
 *     index = floor((mean_square - RMS_MS_LO) / step)
 */
static RmsUniformLut rms_rsqrt_populate() {
    RmsUniformLut lut;

    const float step = (RMS_MS_HI - RMS_MS_LO) / (float) RMS_LUT_SIZE;

    for (int k = 0; k < RMS_LUT_SIZE; k++) {
        float center = RMS_MS_LO + ((float)k + 0.5f) * step;
        lut.val[k]   = (T_RmsRsqrt)(1.0f / std::sqrt(center));
    }
    return lut;
}

static inline T_RmsRsqrt rms_rsqrt(T_RmsSq mean_square) {
#pragma HLS INLINE
    static const RmsUniformLut lut = rms_rsqrt_populate();
#pragma HLS BIND_STORAGE variable=lut.val type=rom_1p impl=bram

    float msf = (float) mean_square;

    /* Saturate to the valid range */
    if (msf < RMS_MS_LO) {
        msf = RMS_MS_LO;
    }
    else if (msf > RMS_MS_HI) {
        msf = RMS_MS_HI;
    }

    const float inv_step = (float) RMS_LUT_SIZE / (RMS_MS_HI - RMS_MS_LO);
    int idx = (int) ((msf - RMS_MS_LO) * inv_step);
    
    if (idx < 0) {
        idx = 0;
    }
    else if (idx >= RMS_LUT_SIZE) {
        idx = RMS_LUT_SIZE - 1;
    }

    return lut.val[idx];
}

#elif RMS_RSQRT_VARIANT == RMS_RSQRT_LOG

struct RmsLogLut {
    T_RmsSq    key[RMS_LUT_SIZE];
    T_RmsRsqrt val[RMS_LUT_SIZE];
};

/* 
 * Logarithmically distributed lookup table
 * 
 * Every entry stores both the key (mean-square) and the corresponding rsqrt value.
 * 
 * The relationship between consecutive keys is:
 *      key[i] = ratio * key[i-1]
 *             = key[0] * ratio^i
 * 
 * Since key[RMS_LUT_SIZE - 1] = RMS_MS_HI and key[0] = RMS_MS_LO, we derive:
 *     ratio = (RMS_MS_HI / RMS_MS_LO)^(1/(RMS_LUT_SIZE - 1))
 * 
 * For each key[i], instead of storing 1/sqrt(key[i]), we store the inverse 
 * square root at the geometric mean of key[i] and key[i+1]:
 *     geometric_mean[i] = sqrt(key[i] * key[i+1])
 *                       = sqrt(key[i] * key[i] * ratio)
 *                       = key[i] * sqrt(ratio)
 * 
 *     val[i] = 1 / sqrt(geometric_mean[i])
 *            = 1 / (key[i] * sqrt(ratio))
 * 
 */
static RmsLogLut rms_rsqrt_populate() {
    RmsLogLut lut;
    /* Ratio to be used. */
    const float ratio = std::pow(RMS_MS_HI / RMS_MS_LO, 1.0f / (float)(RMS_LUT_SIZE - 1));
    
    /* Square root of the ratio, for usage within the geometric mean. */
    const float sqrt_ratio = std::sqrt(ratio);

    float key = RMS_MS_LO;
    for (int k = 0; k < RMS_LUT_SIZE; k++) {
        lut.key[k] = (T_RmsSq)key;
        lut.val[k] = (T_RmsRsqrt)(1.0f / std::sqrt(key * sqrt_ratio));
        key       *= ratio;
    }
    
    return lut;
}

static inline T_RmsRsqrt rms_rsqrt(T_RmsSq mean_square) {
#pragma HLS INLINE
    static const RmsLogLut lut = rms_rsqrt_populate();
#pragma HLS BIND_STORAGE variable=lut.key type=rom_1p impl=bram
#pragma HLS BIND_STORAGE variable=lut.val type=rom_1p impl=bram

    /* Binary search for the largest k with key[k] <= mean_square */
    int low_idx  = 0;
    int high_idx = RMS_LUT_SIZE - 1;
    int curr_idx = 0;
    BSEARCH:
    for (int step = 0; step < RMS_LUT_ADDR_BITS + 1; step++) {
#pragma HLS PIPELINE
        if (low_idx > high_idx) {
            break;
        }
        
        int mid = (low_idx + high_idx) >> 1;

        if (lut.key[mid] <= mean_square) { 
            curr_idx = mid; 
            low_idx  = mid + 1; 
        }
        else { 
            high_idx = mid - 1;
        }
    }

    return lut.val[curr_idx];
}

#elif RMS_RSQRT_VARIANT == RMS_RSQRT_EXPMANT

/* Number of fractional bits in T_RmsSq, used to recover the exponent of the
 * mean-square from the bit position of its leading one. */
#define RMS_SQ_FRAC (T_RmsSq::width - T_RmsSq::iwidth)

struct RmsExpMantLut {
    T_RmsRsqrt even[RMS_LUT_SIZE];   /* rsqrt(m) for m in [1,2) */
    T_RmsRsqrt odd[RMS_LUT_SIZE];    /* rsqrt(m) / sqrt(2), for odd exponents */
};

static const float RMS_INV_SQRT2 = 0.70710678118654752440f;  /* 1/sqrt(2) */

/*
 * Exponent-and-mantissa lookup table
 *
 * Write the mean-square as ms = m * 2^e with the mantissa m in [1,2).  Then:
 *     rsqrt(ms) = rsqrt(m) * 2^(-e/2)
 *
 * Only rsqrt(m) for m in [1,2) is tabulated (RMS_LUT_SIZE entries, addressed by
 * the top mantissa bits); the 2^(-e/2) factor is applied at runtime as a shift.
 * The half-power 2^(-e/2) is exact only for even e, so two tables are stored:
 *     even[k] = 1 / sqrt(m)               (applied for even e)
 *     odd[k]  = 1 / sqrt(m) * 1/sqrt(2)   (applied for odd  e, folding in 2^(-1/2))
 *
 * Each entry uses the bucket center for address k:
 *     m = 1 + (k + 0.5) / RMS_LUT_SIZE
 */
static RmsExpMantLut rms_rsqrt_populate() {
    RmsExpMantLut lut;
    for (int k = 0; k < RMS_LUT_SIZE; k++) {
        float m = 1.0f + ((float)k + 0.5f) / (float)RMS_LUT_SIZE;
        lut.even[k] = (T_RmsRsqrt)(1.0f / std::sqrt(m));
        lut.odd[k]  = (T_RmsRsqrt)(1.0f / std::sqrt(m) * RMS_INV_SQRT2);
    }
    return lut;
}

static inline T_RmsRsqrt rms_rsqrt(T_RmsSq mean_square) {
#pragma HLS INLINE
    static const RmsExpMantLut lut = rms_rsqrt_populate();
#pragma HLS BIND_STORAGE variable=lut.even type=rom_1p impl=bram
#pragma HLS BIND_STORAGE variable=lut.odd  type=rom_1p impl=bram

    /* Raw integer view of the fixed-point mean-square (value = raw * 2^-RMS_SQ_FRAC). */
    ap_uint<T_RmsSq::width> raw = mean_square.range();
    if (raw == 0) {
        return (T_RmsRsqrt)0;   /* guarded; mean_square >= eps in practice */
    }

    /* Leading-one position gives the exponent: ms = m * 2^e, e = lead_pos - RMS_SQ_FRAC. */
    int lead_pos = 0;
    LEADING_ONE:
    for (int b = T_RmsSq::width - 1; b >= 0; b--) {
#pragma HLS PIPELINE
        if (raw[b]) {
            lead_pos = b;
            break;
        }
    }
    int exponent = lead_pos - RMS_SQ_FRAC;

    /* Mantissa address = the RMS_LUT_ADDR_BITS bits just below the leading one (the
     * leading one is the implicit integer 1 and is dropped by the ADDR_BITS width). */
    int shift = lead_pos - RMS_LUT_ADDR_BITS;
    ap_uint<RMS_LUT_ADDR_BITS> addr;
    if (shift >= 0) {
        addr = (ap_uint<RMS_LUT_ADDR_BITS>)(raw >> shift);
    }
    else {
        addr = (ap_uint<RMS_LUT_ADDR_BITS>)(raw << (-shift));
    }

    /* Select table by exponent parity (odd folds in 1/sqrt(2)); apply 2^(-half)
     * as a shift.  For odd e, (e-1)/2 is exact. */
    bool       is_odd = (exponent & 1) != 0;
    int        half   = is_odd ? (exponent - 1) / 2 : exponent / 2;
    T_RmsRsqrt base   = is_odd ? lut.odd[addr] : lut.even[addr];

    T_RmsRsqrt out;
    if (half >= 0) {
        out = (T_RmsRsqrt)(base >> half);
    }
    else {
        out = (T_RmsRsqrt)(base << (-half));
    }
    return out;
}

#else
#error "RMS_RSQRT_VARIANT must be 1 (UNIFORM), 2 (LOG), or 3 (EXPMANT)."
#endif

#endif // RMSNORM_RSQRT_H
