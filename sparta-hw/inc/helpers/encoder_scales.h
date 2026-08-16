/**********************************************************************************
 * Encoder Scale Lookup-Table Layout
 *
 * Quantization is performed offline and per-tensor, so the required folded scales are
 * pre-computed by the host and passed to the encoder as one fixed-point array in DDR.
 * This enum is holds the array's layout so that the encoder layer can index the correct slots 
 * for each block.
 *
 * Slots 0-8 are the attention block's scales, while slots 9-14 are the feed-forward block's.
 *********************************************************************************/

#ifndef ENCODER_SCALES_H
#define ENCODER_SCALES_H

enum EncoderScaleLUTIndex {
    /* Attention block */
    SCALE_Q_IDX                   = 0,  // Folded scale for Q projection
    SCALE_K_IDX                   = 1,  // Folded scale for K projection
    SCALE_V_IDX                   = 2,  // Folded scale for V projection
    SCALE_LINEAR_OUT_IDX          = 3,  // Folded scale for output projection
    SCALE_DIV_OUT_IDX             = 4,  // Folded scale for the division output within the attention block
    SCALE_ATT_RMSNORM_IN_IDX      = 5,  // Dequantization scale for RMSNorm input
    SCALE_ATT_RMSNORM_OUT_INV_IDX = 6,  // Inverse quantization scale for RMSNorm output
    SCALE_ATT_RESIDUAL_IDX        = 7,  // Input scale for residual connection
    SCALE_ATT_BRANCH_RATIO_IDX    = 8,  // Branch scale for residual connection

    /* Feed-forward block */
    SCALE_FC1_IDX                 = 9,  // Folded scale for first FC layer
    SCALE_FC2_IDX                 = 10, // Folded scale for second FC layer
    SCALE_FF_RMSNORM_IN_IDX       = 11, // Dequantization scale for RMSNorm input
    SCALE_FF_RMSNORM_OUT_INV_IDX  = 12, // Inverse quantization scale for RMSNorm output
    SCALE_FF_RESIDUAL_IDX         = 13, // Input scale for residual connection
    SCALE_FF_BRANCH_RATIO_IDX     = 14, // Branch scale for residual connection
    SCALE_IDX_MAX                 = 15
};

#endif // ENCODER_SCALES_H
