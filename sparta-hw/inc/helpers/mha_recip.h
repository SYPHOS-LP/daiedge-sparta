/**********************************************************************************
 * Reciprocal Look-Up Table for the division within the MHA layer.
 *
 * The implementation follows the same approach as the log-distributed LUT of
 * RMSNorm, since they have the same shape near zero. It is stored on-chip as
 * a ROM, whose size is configurable.
 * 
 * The input received is the multiplication between Q and sum(K). Q is greater
 * than 0 as it is the result of a ReLU operation or equal to 0 if the entire 
 * Q row has been zeroed away. This means that the numerator would also be 0,
 * so we return the first LUT entry to force a zero result. The same principle 
 * applies for sum(K), it is equal to 0 only if an entire K row is zeroed, 
 * which means that KxV in the numerator is also 0.
 * 
 *  ---------------------------------------------
 * | Address Bits | Entries | Max Relative Error |
 * |--------------|---------|--------------------|
 * |      6       |   64    |       ~0.8%        |
 * |      7       |  128    |       ~0.4%        |
 * |      8       |  256    |       ~0.2%        | 
 *  ---------------------------------------------
 *
 *********************************************************************************/

#ifndef MHA_RECIP_H
#define MHA_RECIP_H

#include <cmath>
#include "mha_cfg.h"


#define MHA_RECIP_LO    1.0f

/*
 * EXPONENT-AND-MANTISSA addressing (search-free) — mirrors rmsnorm_rsqrt.h's
 * RMS_RSQRT_EXPMANT variant.  The denominator is a positive integer (T_MhaAcc =
 * ap_int<32>).  Write it as  d = m * 2^e  with mantissa m in [1,2):
 *     1/d = (1/m) * 2^(-e)
 * Tabulate 1/m for m in [1,2) only (MHA_RECIP_LUT_SIZE entries, addressed by the
 * top mantissa bits below the leading one); apply 2^(-e) at runtime as a shift.
 *   e    = leading-one bit position of d      (a priority encoder in HW)
 *   addr = the ADDR_BITS bits just below the leading one
 * This removes the binary-search critical path (loop-carried BRAM-read+compare per
 * step) that failed timing: the whole lookup is now one priority-encode + one shift +
 * one ROM read + one shift, all feed-forward.  See docs/exploration/expmant_lut_explained.md.
 *
 * Accuracy on integer denominators (ablation): max rel err 0.195% at 8 addr bits,
 * far under the old log-search's 9.08% max — verified in the HW replica (24/24, 0 flips).
 */
struct MhaRecipLut {
    T_MhaRecip val[MHA_RECIP_LUT_SIZE];   /* 1/m for m in [1,2), bucket centers */
};

static MhaRecipLut populate_mha_reciprocal_lut () {
    MhaRecipLut lut;
    for (int k = 0; k < MHA_RECIP_LUT_SIZE; k++) {
        /* Bucket center: m = 1 + (k + 0.5)/SIZE  (matches rms_rsqrt_populate). */
        float m = 1.0f + ((float) k + 0.5f) / (float) MHA_RECIP_LUT_SIZE;
        lut.val[k] = (T_MhaRecip) (1.0f / m);
    }
    return lut;
}

static inline T_MhaRecip mha_recip(T_MhaAcc denominator) {
#pragma HLS INLINE
    static const MhaRecipLut lut = populate_mha_reciprocal_lut();
#pragma HLS BIND_STORAGE variable=lut.val type=rom_1p impl=bram

    /* If denominator <= 1 the numerator is also 0 (all-zero Q'/K' row): return the
     * m=1 entry (bucket 0) so 0/0 forces a 0 result, avoiding the undefined divide. */
    if (denominator <= (T_MhaAcc) MHA_RECIP_LO) {
        return lut.val[0];
    }

    /* Raw non-negative integer bits (d > 1 here, so the sign bit is 0). */
    ap_uint<T_MhaAcc::width> raw = (ap_uint<T_MhaAcc::width>) denominator;

    /* Leading-one position = exponent e (d = m * 2^e, T_MhaAcc has 0 frac bits). */
    int lead_pos = 0;
    LEADING_ONE:
    for (int b = T_MhaAcc::width - 1; b >= 0; b--) {
#pragma HLS PIPELINE
        if (raw[b]) {
            lead_pos = b;
            break;
        }
    }

    /* Mantissa address = the ADDR_BITS bits just below the leading one. */
    int shift = lead_pos - MHA_RECIP_LUT_ADDR_BITS;
    ap_uint<MHA_RECIP_LUT_ADDR_BITS> addr;
    if (shift >= 0) {
        addr = (ap_uint<MHA_RECIP_LUT_ADDR_BITS>) (raw >> shift);
    } else {
        addr = (ap_uint<MHA_RECIP_LUT_ADDR_BITS>) (raw << (-shift));
    }

    /* 1/d = (1/m) * 2^(-e): the table gives 1/m, the exponent is a right-shift by e. */
    return (T_MhaRecip) (lut.val[addr] >> lead_pos);
}

#endif // MHA_RECIP_H
