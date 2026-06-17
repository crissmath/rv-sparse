/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Correctness test for the experimental RVV INT8 row accumulation kernel.
 */

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "exp_raw_kernels.h"

// Golden reference for test kernel
static void scalar_reference(int8_t a_val, int32_t b_nnz, const int32_t *b_col_idx, const int8_t *b_values, int32_t *acc)
{
    for (int32_t p = 0; p < b_nnz; p++)
    {
        int32_t col = b_col_idx[p];
        acc[col] += (int32_t)a_val * (int32_t)b_values[p];
    }
}

int main(void)
{
    int8_t a_val = 3;

    int32_t b_cols = 10;
    int32_t b_nnz = 10;

    int32_t b_col_idx[] = {
        0, 2, 4, 6, 8,
        1, 3, 5, 7, 9};

    int8_t b_values[] = {
        1, -2, 3, -4, 5,
        6, -7, 8, -9, 10};

    int32_t acc_ref[10] = {0};
    int32_t acc_rvv[10] = {0};

    scalar_reference(a_val, b_nnz, b_col_idx, b_values, acc_ref);

    rvsp_status_t status = exp_accumulate_row_i8_rvv(a_val, b_nnz, b_col_idx, b_values, b_cols, acc_rvv);

    assert(status == RVSP_SUCCESS);

    for (int32_t i = 0; i < b_cols; i++)
    {
        assert(acc_ref[i] == acc_rvv[i]);
    }

#if defined(__riscv_vector)
    printf("test_accumulate_row_i8_rvv: PASS [RVV path]\n");
#else
    printf("test_accumulate_row_i8_rvv: PASS [scalar fallback path]\n");
#endif

    return 0;
}