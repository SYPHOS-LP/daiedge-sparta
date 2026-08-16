// AUTO-GENERATED FILE - DO NOT EDIT MANUALLY
// Generated from config/yml/mha.yaml by scripts/gen_layer_config.py
// ============================================================================

#ifndef MHA_CFG_H
#define MHA_CFG_H

#include "ap_int.h"
#include "ap_fixed.h"
#include "encoder_types.h"

// ---- Defines --------------------------------------------------------------
#define MHA_FEATURE_W_MAX 768   // maximum feature width (rows, feature-major layout)
#define MHA_TOKEN_W_MAX 50   // maximum token width (columns)
#define MHA_N_HEADS 12   // attention heads
#define MHA_FEATURE_HEAD 64   // per-head feature width (MHA_FEATURE_W_MAX / MHA_N_HEADS)
#define MHA_MAX_QK_NNZ 38400   // Q'/K' CSR cache capacity
#define MHA_KP_STREAM_NNZ (MHA_FEATURE_HEAD * MHA_TOKEN_W_MAX)   // per-head K' slice buffer = fully-dense slice (64*197); never overflows
#define MHA_QP_STREAM_NNZ (MHA_FEATURE_HEAD * MHA_TOKEN_W_MAX)   // per-head Q' slice buffer = fully-dense slice (64*197); never overflows
#define MHA_MAX_W_NNZ 256   // per-row weight-decode buffer depth (measured max 195)
#define MHA_MAX_W_TOTAL_NNZ 58980   // on-chip full-weight CSR capacity for a dense projection
#define MHA_SPARSE_FORMAT_CSR 1   // CSR format guard for the MHA decode paths
#define MHA_RECIP_LUT_ADDR_BITS 8   // 256 entries
#define MHA_RECIP_LUT_SIZE (1 << MHA_RECIP_LUT_ADDR_BITS)
#define MHA_BANK_PACK 8   // int8 lanes packed per URAM word (64-bit)
#define MHA_ACT_ELEM_BITS 8   // activation bit width (int8)
#define MHA_BANK_WORD_BITS (MHA_BANK_PACK * MHA_ACT_ELEM_BITS)   // 64-bit packed bank word
#define MHA_BANK_PAD_TOK (((MHA_TOKEN_W_MAX + MHA_BANK_PACK - 1) / MHA_BANK_PACK) * MHA_BANK_PACK)   // tokens padded up to a multiple of MHA_BANK_PACK (197->200)
#define MHA_BANK_ROW_WORDS (MHA_BANK_PAD_TOK / MHA_BANK_PACK)   // packed words per feature row (200/8=25)
#define MHA_BANK_WORDS (MHA_FEATURE_W_MAX * MHA_BANK_ROW_WORDS)   // total packed words in the bank (768*25)
#define MHA_MAC_UNROLL 16   // on-chip MAC lanes over tokens + accumulator partition
#define MHA_BANK_CYCLIC (MHA_MAC_UNROLL / MHA_BANK_PACK)   // word-array cyclic factor = words per MAC-wide read (16/8=2)
#define MHA_SCR_QP_VAL_CNT MHA_MAX_QK_NNZ   // Q' values (int8); mirrors MHA_MAX_QK_NNZ
#define MHA_SCR_QP_COL_CNT MHA_MAX_QK_NNZ   // Q' col indices (head features 0..63)
#define MHA_SCR_QP_PTR_CNT (MHA_N_HEADS * (MHA_TOKEN_W_MAX + 1))   // Q' HEAD-MAJOR row_ptr: 12 blocks of (n+1); head h token-major spans in block h
#define MHA_SCR_KP_VAL_CNT MHA_MAX_QK_NNZ   // K' values (int8); mirrors MHA_MAX_QK_NNZ
#define MHA_SCR_KP_COL_CNT MHA_MAX_QK_NNZ   // K' col indices (tokens 0..196)
#define MHA_SCR_KP_PTR_CNT (MHA_FEATURE_W_MAX + 1)   // K' feature-major row_ptr (D+1)
#define MHA_SCR_V_CNT (MHA_FEATURE_W_MAX * MHA_TOKEN_W_MAX)   // V dense D x N (int8)
#define MHA_SCR_A_CNT (MHA_FEATURE_HEAD * MHA_FEATURE_HEAD)   // per-head A = K'V^T (int32)
#define MHA_TC_FEATURE 768   // feature rows (projections)
#define MHA_TC_TOKEN 50   // tokens / dense width
#define MHA_TC_FEATURE_HEAD 64   // per-head width
#define MHA_TC_W_NNZ 77   // nonzeros per projection weight row (~10% of D)
#define MHA_TC_QP_NNZ 7   // Q' nonzeros per token-row (~10% of d_head=64)
#define MHA_TC_KP_NNZ 5   // K' nonzeros per feature-row (~10% of n=50)

// ---- Types ----------------------------------------------------------------
typedef ap_int<4> T_MhaWVal;   // on-chip decoded W value buffer (w4 weight, -7..+7)
typedef ap_uint<16> T_MhaIndex;   // DDR weight-CSR col index (dims < 65536; 16-bit keeps the AXI burst byte-aligned)
typedef ap_uint<10> T_MhaWCol;   // on-chip decoded W col buffer (feature < MHA_FEATURE_W_MAX=768)
typedef ap_uint<8> T_MhaHeadIndex;   // Q'/K' CSR col idx: K'=token(0..196), Q'=BAND-LOCAL feature(0..63) after head-major emit -> both fit 8 bits
typedef ap_uint<MHA_BANK_WORD_BITS> T_MhaBankWord;   // packed bank URAM word: MHA_BANK_PACK int8 activations (64-bit)
typedef ap_int<32> T_MhaAcc;   // MAC / scatter accumulator (int8xint8 summed)
typedef ap_ufixed<32, 1> T_MhaRecip;   // 1/(d+eps) reciprocal LUT output; d is large so recip ~1e-6, needs many frac bits
typedef ap_fixed<32, 16> T_MhaDiv;   // N*recip ~= N/d ratio (O(1)) before requant

#endif // MHA_CFG_H
