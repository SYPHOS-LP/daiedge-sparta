/**********************************************************************************
 * CSR Bounds
 *
 * Depth bounds for the kernel's weight ports.  They are AXI port depths only -- HLS uses them to
 * size bursts, nothing on chip is allocated from them -- so they bound the largest matrix in the
 * model rather than any one transfer:
 * MAX_ROWS bounds a weight's output-row count (the MLP's first linear, d_h = 3072),
 * MAX_NNZ bounds a weight's packed entry count.
 * Named for the CSR format the weights used before they were packed into block banks.
 *********************************************************************************/

#ifndef CSR_BOUNDS_H
#define CSR_BOUNDS_H

#define MAX_ROWS 3072   /* output rows of the largest weight (d_h) */
/* Packed WORDS of the largest weight, not entries: the MLP's first linear holds 235 944 entries in
 * the shipping payload, 58 986 words of MHA_TILE_PACK.  Only co-simulation reads this -- it copies
 * exactly `depth` elements per m_axi port, so too small means the kernel reads past what was copied
 * and gets garbage.  Verified to leave synthesis byte-identical (same LUT/FF/BRAM/URAM/DSP and
 * latency), so it cannot move the bitstream.  Raise it if a denser model is compiled. */
#define MAX_NNZ  65536  /* packed words of the largest weight, with headroom */

#endif // CSR_BOUNDS_H
