// ============================================================================
// POT4 test scales + reference requant — shared by the encoder block/layer TBs.
//
// The POT4 model's folded scales are powers of two (2^s).  The kernel applies
// them as a MULTIPLY (a runtime shift synthesizes to a deep barrel shifter, worse
// than the DSP multiply).  These constants are the real scale VALUES 2^s that the
// DUT's scales[] array carries, plus a reference requant that mirrors quantize.h
// (multiply + round-half-away + clamp) so the double-precision references match.
//
// The values are self-consistent power-of-two scales for the identity-projection
// tests (their magnitudes are arbitrary; DUT and reference must apply the SAME
// scale, which these guarantee).
// ============================================================================
#ifndef TB_POT_SCALES_H
#define TB_POT_SCALES_H

#include <cmath>

// Power-of-two scale value 2^s for a given signed exponent s.
static inline double pot_scale(int s) { return std::ldexp(1.0, s); }

// ---- MHA block (attention) scale exponents s (real scale = 2^s) ----
#define S_MHA_Q       (-3)   // Q' projection requant
#define S_MHA_K       (-3)   // K' projection requant
#define S_MHA_V       (-3)   // V projection requant
#define S_MHA_O       (-4)   // output projection requant
#define S_MHA_DIV     (3)    // divide output requant
#define S_MHA_RMS_INV (3)    // RMSNorm output inverse scale
#define S_MHA_RES     (0)    // residual: sX/sOUT
#define S_MHA_BRANCH  (0)    // branch:   sM/sOUT

// ---- MLP block (feed-forward) scale exponents ----
#define S_MLP_FC1     (-2)   // fc1 requant
#define S_MLP_FC2     (-4)   // fc2 requant
#define S_MLP_RMS_INV (3)    // RMSNorm output inverse scale
#define S_MLP_RES     (0)    // residual: sX/sOUT
#define S_MLP_BRANCH  (1)    // branch:   sM/sOUT

// ---------------------------------------------------------------------------
// Reference requant: multiply an integer acc by the power-of-two scale (2^s),
// round-half-away-from-zero, clamp to int8.  Mirrors quantize.h::requant_to_int8.
// ---------------------------------------------------------------------------
static inline int ref_requant(long acc, int s) {
    double v = (double) acc * pot_scale(s);
    long r = std::lround(v);
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int) r;
}

#endif // TB_POT_SCALES_H
