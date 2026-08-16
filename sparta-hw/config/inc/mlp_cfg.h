// AUTO-GENERATED FILE - DO NOT EDIT MANUALLY
// Generated from config/yml/mlp.yaml by scripts/gen_layer_config.py
// ============================================================================

#ifndef MLP_CFG_H
#define MLP_CFG_H

#include "ap_int.h"
#include "ap_fixed.h"
#include "encoder_types.h"

// ---- Defines --------------------------------------------------------------
#define MLP_FEATURE_W_MAX 768   // maximum feature width (rows, feature-major layout)
#define MLP_TOKEN_W_MAX 50   // maximum token width (columns)
#define MLP_HIDDEN_FEATURE_W_MAX 3072   // maximum hidden width (rows of H)
#define MLP_HIDDEN_NNZ_MAX 153600   // on-chip H cache capacity
#define MLP_FEATURE_OUT_W_MAX 768   // output feature width (rows of Y = cols of W2^T)
#define MLP_SPARSE_FORMAT_CSR 1   // CSR format guard for the MLP decode path
#define MLP_MAC_UNROLL 16   // on-chip MAC lanes over batch + bank partition (>=2, power of two)
#define MLP_ROW_PARALLEL 4   // hidden-row PEs computed concurrently (fused MAC loop)
#define MLP_OUT_ROW_PARALLEL 2   // output-row scatter PEs (own acc bank + own per-slice W2 CSC; H broadcast)
#define MLP_OUT_ROW_SLICE ((MLP_FEATURE_OUT_W_MAX + MLP_OUT_ROW_PARALLEL - 1) / MLP_OUT_ROW_PARALLEL)   // out_rows owned per PE (768/2=384; ceil so P need not divide d_out)
#define MLP_W2_WORDS_PE MLP_W2_WORDS   // packed W2 words per per-slice CSC (sized to full W2 -> overflow-proof for any out_row imbalance)
#define MLP_MAX_W1_NNZ 512   // W1 per-row decode-buffer depth
#define MLP_MAX_W2_NNZ 3072   // W2 per-row decode-buffer depth
#define MLP_MAX_W2_TOTAL_NNZ 235920   // full W2 CSC nnz staged on-chip (exact measured max; static weight)
#define MLP_ACC_PACK 3   // int20 accumulators packed per acc URAM word (token axis; 3*20=60b)
#define MLP_ACC_BITS 20   // packed acc lane width (max |acc| ~273k < 2^19)
#define MLP_ACC_WORD_BITS (MLP_ACC_PACK * MLP_ACC_BITS)   // packed acc URAM word width (3*20=60b)
#define MLP_ACC_PAD_TOK (((MLP_TOKEN_W_MAX + MLP_ACC_PACK - 1) / MLP_ACC_PACK) * MLP_ACC_PACK)   // batch padded to a multiple of MLP_ACC_PACK (50->51)
#define MLP_ACC_ROW_WORDS (MLP_ACC_PAD_TOK / MLP_ACC_PACK)   // packed acc words per output row (51/3=17)
#define MLP_W2_PACK 5   // W2 CSC nonzeros packed per URAM word (5*14=70b fills the 72b URAM word)
#define MLP_W2_LANE_BITS 14   // bits per packed W2 nonzero: {row(10b) : val(4b)}
#define MLP_W2_WORD_BITS (MLP_W2_PACK * MLP_W2_LANE_BITS)   // packed W2 URAM word width (5*14=70b)
#define MLP_W2_WORDS ((MLP_MAX_W2_TOTAL_NNZ + MLP_W2_PACK - 1) / MLP_W2_PACK)   // packed W2 word count (ceil nnz/PACK)
#define MLP_BANK_PACK 8   // int8 lanes packed per URAM word (64-bit)
#define MLP_ACT_ELEM_BITS 8   // activation bit width (int8)
#define MLP_BANK_WORD_BITS (MLP_BANK_PACK * MLP_ACT_ELEM_BITS)   // 64-bit packed bank word
#define MLP_BANK_PAD_TOK (((MLP_TOKEN_W_MAX + MLP_BANK_PACK - 1) / MLP_BANK_PACK) * MLP_BANK_PACK)   // batch padded up to a multiple of MLP_BANK_PACK (50->56)
#define MLP_BANK_ROW_WORDS (MLP_BANK_PAD_TOK / MLP_BANK_PACK)   // packed words per feature row (56/8=7)
#define MLP_BANK_WORDS (MLP_FEATURE_W_MAX * MLP_BANK_ROW_WORDS)   // total packed words in the bank (768*7)
#define MLP_BANK_CYCLIC (MLP_MAC_UNROLL / MLP_BANK_PACK)   // word-array cyclic factor = words per MAC-wide read (16/8=2)
#define MLP_TC_HIDDEN_FEATURE 3072   // hidden rows (H_ROW_LOOP)
#define MLP_TC_FEATURE_OUT 768   // output rows (Y_ROW_LOOP) = d_out
#define MLP_TC_BATCH 50   // samples / dense row width
#define MLP_TC_W1_NNZ 77   // nonzeros per W1 row (~10% of d_in=768)
#define MLP_TC_W2_NNZ 307   // nonzeros per W2 ROW (~10% of d_h=3072); W2-driven decode
#define MLP_TC_W2COL_NNZ 77   // nonzeros per W2 COLUMN (~10% of d_out=768); the H-driven scatter walks a column
#define MLP_TC_HIDDEN_NNZ 5   // nonzeros per H row (~10% of batch=50)

// ---- Types ----------------------------------------------------------------
typedef ap_int<4> T_MlpWVal;   // on-chip decoded W value buffer (w4 weight, -7..+7)
typedef ap_uint<16> T_MlpIndex;   // WEIGHT CSR column index (W1 spans MLP_FEATURE_W_MAX=768, W2 spans MLP_HIDDEN_FEATURE_W_MAX=3072); also the DDR element type — 16-bit keeps the DDR side byte-aligned
typedef ap_uint<8> T_MlpHCol;   // on-chip H-cache column index: spans the token/batch dim (<= MLP_TOKEN_W_MAX=50 < 256), so 8-bit halves the H-cache col BRAM vs T_MlpIndex
typedef ap_uint<10> T_MlpW1Col;   // on-chip decoded-W1-row col buffer: W1 col = input feature (< MLP_FEATURE_W_MAX=768 <= 1024), 10-bit shrinks the decode BRAM. DDR col stays 16-bit (T_MlpIndex)
typedef ap_uint<12> T_MlpW2Col;   // on-chip decoded-W2-row col buffer: W2 col = hidden feature (< MLP_HIDDEN_FEATURE_W_MAX=3072 <= 4096), 12-bit shrinks the decode BRAM. DDR col stays 16-bit (T_MlpIndex)
typedef ap_uint<10> T_MlpW2Row;   // on-chip W2 CSC row index (H-driven stage-2): the output row a W2 nonzero targets (< MLP_FEATURE_OUT_W_MAX=768 <= 1024)
typedef ap_uint<MLP_W2_WORD_BITS> T_MlpW2Word;   // packed W2 CSC URAM word: MLP_W2_PACK lanes of {row(10b):val(4b)}
typedef ap_int<MLP_ACC_BITS> T_MlpAccLane;   // one packed acc accumulator lane (int20)
typedef ap_uint<MLP_ACC_WORD_BITS> T_MlpAccWord;   // packed acc URAM word: MLP_ACC_PACK int20 lanes (token axis)
typedef ap_int<32> T_MlpAcc;   // MAC / scatter accumulator
typedef ap_uint<MLP_BANK_WORD_BITS> T_MlpBankWord;   // packed bank URAM word: MLP_BANK_PACK int8 activations (64-bit)

#endif // MLP_CFG_H
