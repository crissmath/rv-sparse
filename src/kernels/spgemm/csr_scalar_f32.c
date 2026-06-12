/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * This file is part of rv-sparse.
 *
 * This file implements the initial scalar FP32 CSR SpGEMM kernel:
 *
 *     C = A * B
 *
 * where A, B, and C are stored in Compressed Sparse Row (CSR) format.
 *
 * The implementation is intended as a correctness-first baseline.
 */
#include <stdlib.h>
#include "rv_sparse.h"
#include "csr_spgemm_kernels.h"

rvsp_status_t rvsp_spgemm_csr_scalar_f32_raw(int32_t a_rows, int32_t a_cols, int32_t b_cols, const int32_t *restrict a_row_ptr,
                                             const int32_t *restrict a_col_idx, const float *restrict a_values, const int32_t *restrict b_row_ptr,
                                             const int32_t *restrict b_col_idx, const float *restrict b_values, int32_t **c_row_ptr_out, int32_t **c_col_idx_out,
                                             float **c_values_out, int32_t *c_nnz_out)
{
    int32_t max_nnz = a_rows * b_cols;

    int32_t *c_row_ptr = (int32_t *)calloc((size_t)a_rows + 1, sizeof(int32_t));
    int32_t *c_col_idx = (int32_t *)malloc((size_t)max_nnz * sizeof(int32_t));
    float *c_values = (float *)malloc((size_t)max_nnz * sizeof(float));

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

        float *acc = (float *)calloc((size_t)b_cols, sizeof(float));
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
            float a_val = a_values[a_pos];

            int32_t b_start = b_row_ptr[k];
            int32_t b_end = b_row_ptr[k + 1];

            for (int32_t b_pos = b_start; b_pos < b_end; b_pos++)
            {
                int32_t col = b_col_idx[b_pos];
                float b_val = b_values[b_pos];

                acc[col] += a_val * b_val;
            }
        }

        for (int32_t col = 0; col < b_cols; col++)
        {
            if (acc[col] != 0.0f)
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

    (void)a_cols;

    return RVSP_SUCCESS;
}

// wrapper
rvsp_status_t rvsp_spgemm_csr_scalar_f32(const rvsp_csr_matrix_t *A, const rvsp_csr_matrix_t *B, rvsp_csr_matrix_t *C)
{
    rvsp_status_t status;

    if (!A || !B || !C)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    status = rvsp_csr_validate(A);
    if (status != RVSP_SUCCESS)
    {
        return status;
    }

    status = rvsp_csr_validate(B);
    if (status != RVSP_SUCCESS)
    {
        return status;
    }

    if (A->dtype != RVSP_DTYPE_FP32 || B->dtype != RVSP_DTYPE_FP32)
    {
        return RVSP_ERROR_UNSUPPORTED_DTYPE;
    }

    if (A->cols != B->rows)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    int32_t *c_row_ptr = NULL;
    int32_t *c_col_idx = NULL;
    float *c_values = NULL;
    int32_t c_nnz = 0;

    status = rvsp_spgemm_csr_scalar_f32_raw(
        A->rows,
        A->cols,
        B->cols,
        A->row_ptr,
        A->col_idx,
        (const float *)A->values,
        B->row_ptr,
        B->col_idx,
        (const float *)B->values,
        &c_row_ptr,
        &c_col_idx,
        &c_values,
        &c_nnz);

    if (status != RVSP_SUCCESS)
    {
        return status;
    }

    C->rows = A->rows;
    C->cols = B->cols;
    C->nnz = c_nnz;
    C->row_ptr = c_row_ptr;
    C->col_idx = c_col_idx;
    C->values = c_values;
    C->dtype = RVSP_DTYPE_FP32;
    C->format = RVSP_FORMAT_CSR;
    C->owns_data = 1;

    return RVSP_SUCCESS;
}