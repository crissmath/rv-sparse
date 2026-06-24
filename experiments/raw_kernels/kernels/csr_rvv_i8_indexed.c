
// Copyright (C) 2026 rv-sparse contributors
//
//  SPDX-License-Identifier: GPL-3.0-only
//  Experimental CSR SpGEMM RVV INT8 raw kernel using indexed accumulation.
//  Computes INT8 x INT8 with INT32 accumulation/output.
//

#include <stdint.h>
#include <stdlib.h>

#include "exp_raw_kernels.h"

rvsp_status_t exp_spgemm_csr_rvv_i8_indexed_raw(int32_t a_rows, int32_t a_cols, int32_t b_cols,
                                                const int32_t *a_row_ptr, const int32_t *a_col_idx, const int8_t *a_values,
                                                const int32_t *b_row_ptr,
                                                const int32_t *b_col_idx,
                                                const int8_t *b_values,
                                                int32_t **c_row_ptr_out,
                                                int32_t **c_col_idx_out,
                                                int32_t **c_values_out,
                                                int32_t *c_nnz_out)
{
    if (!a_row_ptr || !a_col_idx || !a_values ||
        !b_row_ptr || !b_col_idx || !b_values ||
        !c_row_ptr_out || !c_col_idx_out ||
        !c_values_out || !c_nnz_out)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    if (a_rows <= 0 || a_cols <= 0 || b_cols <= 0)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    int32_t max_nnz = a_rows * b_cols;

    int32_t *c_row_ptr = (int32_t *)calloc((size_t)a_rows + 1, sizeof(int32_t));
    int32_t *c_col_idx = (int32_t *)malloc((size_t)max_nnz * sizeof(int32_t));
    int32_t *c_values = (int32_t *)malloc((size_t)max_nnz * sizeof(int32_t));

    if (!c_row_ptr || !c_col_idx || !c_values)
    {
        free(c_row_ptr);
        free(c_col_idx);
        free(c_values);
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    int32_t nnz_count = 0;

    for (int32_t row = 0; row < a_rows; row++)
    {
        c_row_ptr[row] = nnz_count;

        int32_t *acc = (int32_t *)calloc((size_t)b_cols, sizeof(int32_t));

        if (!acc)
        {
            free(c_row_ptr);
            free(c_col_idx);
            free(c_values);
            return RVSP_ERROR_ALLOCATION_FAILED;
        }

        int32_t a_start = a_row_ptr[row];
        int32_t a_end = a_row_ptr[row + 1];

        for (int32_t a_pos = a_start; a_pos < a_end; a_pos++)
        {
            int32_t k = a_col_idx[a_pos];

            if (k < 0 || k >= a_cols)
            {
                free(acc);
                free(c_row_ptr);
                free(c_col_idx);
                free(c_values);
                return RVSP_ERROR_INVALID_CSR;
            }

            int8_t a_val = a_values[a_pos];

            int32_t b_start = b_row_ptr[k];
            int32_t b_end = b_row_ptr[k + 1];
            int32_t b_nnz = b_end - b_start;

            rvsp_status_t status = exp_accumulate_row_i8_rvv_indexed(a_val, b_nnz, &b_col_idx[b_start], &b_values[b_start], b_cols, acc);

            if (status != RVSP_SUCCESS)
            {
                free(acc);
                free(c_row_ptr);
                free(c_col_idx);
                free(c_values);
                return status;
            }
        }

        for (int32_t col = 0; col < b_cols; col++)
        {
            if (acc[col] != 0)
            {
                c_col_idx[nnz_count] = col;
                c_values[nnz_count] = acc[col];
                nnz_count++;
            }
        }

        free(acc);
    }

    c_row_ptr[a_rows] = nnz_count;

    *c_row_ptr_out = c_row_ptr;
    *c_col_idx_out = c_col_idx;
    *c_values_out = c_values;
    *c_nnz_out = nnz_count;

    return RVSP_SUCCESS;
}