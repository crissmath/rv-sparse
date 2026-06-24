/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Correctness test for experimental indexed RVV CSR SpGEMM INT8 raw kernel.
 */

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "exp_raw_kernels.h"

static void assert_same_csr_i32(int32_t rows, const int32_t *ref_row_ptr, const int32_t *ref_col_idx,
                                const int32_t *ref_values, int32_t ref_nnz, const int32_t *test_row_ptr,
                                const int32_t *test_col_idx, const int32_t *test_values, int32_t test_nnz)
{
    assert(ref_nnz == test_nnz);

    for (int32_t i = 0; i < rows + 1; i++)
    {
        assert(ref_row_ptr[i] == test_row_ptr[i]);
    }

    for (int32_t i = 0; i < ref_nnz; i++)
    {
        assert(ref_col_idx[i] == test_col_idx[i]);
        assert(ref_values[i] == test_values[i]);
    }
}

int main(void)
{
    /*
     * A =
     * [ 1  2  0  4]
     * [ 0 -3  5  0]
     * [ 2  0 -1  3]
     */
    int32_t a_rows = 3;
    int32_t a_cols = 4;

    int32_t a_row_ptr[] = {0, 3, 5, 8};
    int32_t a_col_idx[] = {0, 1, 3, 1, 2, 0, 2, 3};
    int8_t a_values[] = {1, 2, 4, -3, 5, 2, -1, 3};

    /*
     * B =
     * [ 1  0  2  0  3]
     * [ 0  3  0 -1  0]
     * [ 4  0  5  0 -2]
     * [ 0  6  7  0  1]
     */
    int32_t b_cols = 5;

    int32_t b_row_ptr[] = {0, 3, 5, 8, 11};
    int32_t b_col_idx[] = {
        0, 2, 4,
        1, 3,
        0, 2, 4,
        1, 2, 4};
    int8_t b_values[] = {
        1, 2, 3,
        3, -1,
        4, 5, -2,
        6, 7, 1};

    int32_t *c_ref_row_ptr = NULL;
    int32_t *c_ref_col_idx = NULL;
    int32_t *c_ref_values = NULL;
    int32_t c_ref_nnz = 0;

    int32_t *c_rvv_row_ptr = NULL;
    int32_t *c_rvv_col_idx = NULL;
    int32_t *c_rvv_values = NULL;
    int32_t c_rvv_nnz = 0;

    rvsp_status_t status;

    status = exp_spgemm_csr_scalar_i8_reference_raw(a_rows, a_cols, b_cols, a_row_ptr, a_col_idx, a_values,
                                                    b_row_ptr, b_col_idx, b_values, &c_ref_row_ptr, &c_ref_col_idx, &c_ref_values, &c_ref_nnz);

    assert(status == RVSP_SUCCESS);

    status = exp_spgemm_csr_rvv_i8_indexed_raw(a_rows, a_cols, b_cols, a_row_ptr, a_col_idx, a_values,
                                               b_row_ptr, b_col_idx, b_values, &c_rvv_row_ptr, &c_rvv_col_idx, &c_rvv_values, &c_rvv_nnz);

    assert(status == RVSP_SUCCESS);

    assert_same_csr_i32(a_rows, c_ref_row_ptr, c_ref_col_idx, c_ref_values, c_ref_nnz,
                        c_rvv_row_ptr, c_rvv_col_idx, c_rvv_values, c_rvv_nnz);

    free(c_ref_row_ptr);
    free(c_ref_col_idx);
    free(c_ref_values);

    free(c_rvv_row_ptr);
    free(c_rvv_col_idx);
    free(c_rvv_values);

#if defined(__riscv_vector)
    printf("test_csr_rvv_i8_indexed_raw: PASS [RVV indexed path]\n");
#else
    printf("test_csr_rvv_i8_indexed_raw: PASS [scalar fallback path]\n");
#endif

    return 0;
}