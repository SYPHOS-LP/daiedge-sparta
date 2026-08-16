/**********************************************************************************
 * CSR Bounds
 *
 * Shared dimension bounds for the CSR weight ports of the encoder layers, sized
 * for the ViT-MLP workload (W1 768x3072, W2 3072x768): 
 * MAX_ROWS bounds any matrix's row count / row_ptr length (d_h = 3072),
 * MAX_NNZ bounds the nonzeros of a single CSR row (a W2 row spans up to d_h).
 *********************************************************************************/

#ifndef CSR_BOUNDS_H
#define CSR_BOUNDS_H

#define MAX_ROWS 3072   /* max rows in any matrix (d_h); also row_ptr length */
#define MAX_NNZ  3072   /* max nonzeros buffered for a single CSR row */

#endif // CSR_BOUNDS_H
