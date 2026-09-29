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
#define MAX_NNZ  3072   /* packed entries of the largest weight */

#endif // CSR_BOUNDS_H
