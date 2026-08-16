// AUTO-GENERATED FILE - DO NOT EDIT MANUALLY
// Generated from config/yml/rmsnorm.yaml by scripts/gen_layer_config.py
// ============================================================================

#ifndef RMSNORM_CFG_H
#define RMSNORM_CFG_H

#include "ap_int.h"
#include "ap_fixed.h"
#include "encoder_types.h"

// ---- Defines --------------------------------------------------------------
#define RMS_FEATURE_W_MAX 768   // maximum feature width (rows, feature-major layout)
#define RMS_TOKEN_W_MAX 50   // maximum token width (columns)
#define RMS_BANK_PACK 8   // int8 lanes per packed URAM word (= MHA/MLP_BANK_PACK)
#define RMS_BANK_WORD_BITS (RMS_BANK_PACK * 8)   // 64-bit packed word
#define RMS_BANK_PAD_TOK (((RMS_TOKEN_W_MAX + RMS_BANK_PACK - 1) / RMS_BANK_PACK) * RMS_BANK_PACK)   // tokens padded to multiple of PACK (197->200)
#define RMS_BANK_ROW_WORDS (RMS_BANK_PAD_TOK / RMS_BANK_PACK)   // packed words per feature row (25)
#define RMS_EPS 1e-6f   // epsilon added to mean-square before rsqrt
#define RMS_RSQRT_UNIFORM 1
#define RMS_RSQRT_LOG 2
#define RMS_RSQRT_EXPMANT 3
#define RMS_RSQRT_VARIANT RMS_RSQRT_EXPMANT   // active variant: exponent-mantissa (search-free; removes the binary-search critical path). See docs/exploration/expmant_lut_explained.md
#define RMS_LUT_ADDR_BITS 8   // 256 entries (Exp-Mant mantissa addr). Reduced 16->8: max rel err 0.045%->0.098%, still < the old log-LUT's 0.205% and far under the architectural floor. Ablation: docs/exploration/expmant_lut_explained.md
#define RMS_LUT_SIZE (1 << RMS_LUT_ADDR_BITS)

// ---- Types ----------------------------------------------------------------
typedef ap_uint<RMS_BANK_WORD_BITS> T_RmsBankWord;   // packed URAM bank word: RMS_BANK_PACK int8 (encoder packed-output path)
typedef ap_ufixed<56, 26> T_RmsSq;   // x^2 + mean-square accumulator (30 frac bits; eps=1e-6 representable)
typedef ap_ufixed<28, 12> T_RmsRsqrt;   // 1/sqrt(ms) in [~0.25,~1000]
typedef ap_fixed<32, 16> T_RmsNorm;   // x*rsqrt working value

#endif // RMSNORM_CFG_H
