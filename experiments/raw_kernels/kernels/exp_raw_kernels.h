/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Experimental raw kernel declarations.
 */

#ifndef EXP_RAW_KERNELS_H
#define EXP_RAW_KERNELS_H

#include <stdint.h>

#include "rv_sparse_types.h"

rvsp_status_t exp_spgemm_csr_scalar_i8_reference_raw(
    int32_t a_rows,
    int32_t a_cols,
    int32_t b_cols,
    const int32_t *a_row_ptr,
    const int32_t *a_col_idx,
    const int8_t *a_values,
    const int32_t *b_row_ptr,
    const int32_t *b_col_idx,
    const int8_t *b_values,
    int32_t **c_row_ptr_out,
    int32_t **c_col_idx_out,
    int32_t **c_values_out,
    int32_t *c_nnz_out);

rvsp_status_t exp_accumulate_row_i8_rvv(
    int8_t a_val,
    int32_t b_nnz,
    const int32_t *b_col_idx,
    const int8_t *b_values,
    int32_t b_cols,
    int32_t *acc);

#endif /* EXP_RAW_KERNELS_H */