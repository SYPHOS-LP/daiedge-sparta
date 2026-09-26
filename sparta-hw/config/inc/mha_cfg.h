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
#define MHA_HEAD_NNZ_MAX (MHA_FEATURE_HEAD * MHA_TOKEN_W_MAX)   // per-head Q'/K' CSR capacity = fully dense head slice (64*50)
#define MHA_SCR_V_HEAD_CNT (MHA_FEATURE_HEAD * MHA_TOKEN_W_MAX)   // one head's dense V, and the Q' transpose scratch (int8)
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
#define MHA_BANK_WORDS_ALLOC (MHA_BANK_WORDS + MHA_BANK_CYCLIC - 1)   // bank array size read by the SpMM: + the spare word(s) the last token tile reads past the final row (value unused)
#define MHA_TILES 12   // column blocks of the feature axis (= heads at D=768)
#define MHA_TILE_FEATURES (MHA_FEATURE_W_MAX / MHA_TILES)   // features per block (64)
#define MHA_TILE_WORDS (MHA_TILE_FEATURES * MHA_BANK_ROW_WORDS)   // packed activation words per tile (64*7=448, one BRAM36)
#define MHA_ROWS_PER_STEP 1   // output rows the lanes advance together (amortises per-row block imbalance)
#define MHA_TILE_ENTRY_BITS 16   // packed weight entry: {block-local column (6 b), value (4 b)}
#define MHA_TILE_PACK 4   // weight entries per 64-bit bank word
#define MHA_ROW_BLOCKS (MHA_FEATURE_W_MAX / MHA_FEATURE_HEAD)   // row blocks per weight (12; a block = one head's rows)
#define MHA_TILE_NNZ_MAX 1024   // per-bank capacity for one row block; measured worst cell = 977 + up to 3 lead-in entries
#define MHA_TILE_BANK_WORDS (MHA_TILE_NNZ_MAX / MHA_TILE_PACK)   // bank depth in packed words (256)
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
typedef ap_uint<(MHA_TILE_PACK * MHA_TILE_ENTRY_BITS)> T_MhaTileWord;   // packed weight-bank word: MHA_TILE_PACK entries of {col, val} (64-bit)
typedef ap_uint<6> T_MhaTileCol;   // block-local column index (0..MHA_TILE_FEATURES-1)
typedef ap_uint<8> T_MhaTileCount;   // nonzeros of one output row inside one block (measured max 62)
typedef ap_int<32> T_MhaAcc;   // MAC / scatter accumulator (int8xint8 summed)
typedef ap_ufixed<32, 1> T_MhaRecip;   // 1/(d+eps) reciprocal LUT output; d is large so recip ~1e-6, needs many frac bits
typedef ap_fixed<32, 16> T_MhaDiv;   // N*recip ~= N/d ratio (O(1)) before requant

#endif // MHA_CFG_H
