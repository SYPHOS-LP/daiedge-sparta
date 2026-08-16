// AUTO-GENERATED FILE - DO NOT EDIT MANUALLY
// Generated from config/yml/encoder_types.yaml by scripts/gen_layer_config.py
// ============================================================================

#ifndef ENCODER_TYPES_H
#define ENCODER_TYPES_H

#include "ap_int.h"
#include "ap_fixed.h"

// ---- Types ----------------------------------------------------------------
typedef ap_int<8> T_Activation;   // int8 activation shared by every encoder layer (in/out, weights values, residual I/O)
typedef ap_fixed<32, 16> T_Scale;   // folded quantization scale: per-layer scales[] element + residual-add working value

#endif // ENCODER_TYPES_H
