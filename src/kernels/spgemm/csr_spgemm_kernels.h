#ifndef RVSP_CSR_SPGEMM_KERNELS_H
#define RVSP_CSR_SPGEMM_KERNELS_H

#include "rv_sparse.h"

rvsp_status_t rvsp_spgemm_csr_scalar_f32(
    const rvsp_csr_matrix_t *A,
    const rvsp_csr_matrix_t *B,
    rvsp_csr_matrix_t *C
);

#endif
