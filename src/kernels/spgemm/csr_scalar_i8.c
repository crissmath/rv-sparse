/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Scalar INT8 CSR SpGEMM kernel.
 * Computes INT8 x INT8 with INT32 accumulation/output.
 *
 * Row-wise Gustavson with a sparse accumulator (SPA):
 *
 *   - Workspace (accumulator, occupancy marks, touched-column list) is
 *     allocated ONCE, sized b_cols, and reused across rows. Resetting
 *     between rows walks only the touched-column list, so per-row cost
 *     is O(nnz(row)) instead of O(b_cols). Total complexity is
 *     O(flops + nnz(C) log nnz_row(C)), not O(a_rows * b_cols).
 *
 *   - Output size is determined by a symbolic pass and allocated
 *     exactly. This replaces the previous dense worst-case allocation
 *     `a_rows * b_cols`, which (a) overflowed int32 for any matrix
 *     with more than 46,340 rows and (b) requested O(M*N) memory.
 *     All size arithmetic is 64-bit; if nnz(C) would exceed INT32_MAX
 *     (the API's index type), the kernel fails cleanly.
 *
 *   - Column indices within each output row are emitted in ascending
 *     order (canonical CSR), preserving bit-identical output with the
 *     previous implementation.
 *
 *   - No allocation or deallocation occurs inside the row loop. Besides
 *     performance, this avoids exercising allocator/brk churn, which is
 *     unreliable under gem5 SE-mode syscall emulation.
 */

#include <stdlib.h>
#include <string.h>

#include "rv_sparse.h"
#include "csr_spgemm_kernels.h"

/* ascending insertion of touched columns at emit time via per-row sort */
static int cmp_i32(const void *a, const void *b)
{
    int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
    return (x > y) - (x < y);
}

rvsp_status_t rvsp_spgemm_csr_scalar_i8_raw(int32_t a_rows, int32_t a_cols, int32_t b_cols, const int32_t *restrict a_row_ptr,
                                            const int32_t *restrict a_col_idx, const int8_t *restrict a_values,
                                            const int32_t *restrict b_row_ptr, const int32_t *restrict b_col_idx,
                                            const int8_t *restrict b_values,
                                            int32_t **c_row_ptr_out, int32_t **c_col_idx_out, int32_t **c_values_out,
                                            int32_t *c_nnz_out)
{
    if (!a_row_ptr || !a_col_idx || !a_values ||
        !b_row_ptr || !b_col_idx || !b_values ||
        !c_row_ptr_out || !c_col_idx_out || !c_values_out || !c_nnz_out)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    if (a_rows <= 0 || a_cols <= 0 || b_cols <= 0)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

    /* ---- persistent workspace, allocated once ---- */
    int32_t *acc     = (int32_t *)malloc((size_t)b_cols * sizeof(int32_t));
    int32_t *touched = (int32_t *)malloc((size_t)b_cols * sizeof(int32_t));
    uint8_t *mark    = (uint8_t *)calloc((size_t)b_cols, sizeof(uint8_t));
    int32_t *c_row_ptr = (int32_t *)calloc((size_t)a_rows + 1, sizeof(int32_t));

    if (!acc || !touched || !mark || !c_row_ptr)
    {
        free(acc); free(touched); free(mark); free(c_row_ptr);
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    /* ---- symbolic pass: exact nnz per row (validates CSR as it goes) ---- */
    int64_t total_nnz = 0;

    for (int32_t row = 0; row < a_rows; row++)
    {
        int32_t n_touched = 0;

        for (int32_t a_pos = a_row_ptr[row]; a_pos < a_row_ptr[row + 1]; a_pos++)
        {
            int32_t k = a_col_idx[a_pos];

            if (k < 0 || k >= a_cols)
            {
                free(acc); free(touched); free(mark); free(c_row_ptr);
                return RVSP_ERROR_INVALID_CSR;
            }

            for (int32_t b_pos = b_row_ptr[k]; b_pos < b_row_ptr[k + 1]; b_pos++)
            {
                int32_t col = b_col_idx[b_pos];
                if (!mark[col])
                {
                    mark[col] = 1;
                    touched[n_touched++] = col;
                }
            }
        }

        c_row_ptr[row] = n_touched; /* per-row count; prefix-summed below */
        total_nnz += n_touched;

        for (int32_t t = 0; t < n_touched; t++)
        {
            mark[touched[t]] = 0;
        }
    }

    if (total_nnz > INT32_MAX)
    {
        /* output does not fit the API's int32 index type */
        free(acc); free(touched); free(mark); free(c_row_ptr);
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    /* exclusive prefix sum: counts -> row pointers */
    {
        int32_t running = 0;
        for (int32_t row = 0; row < a_rows; row++)
        {
            int32_t cnt = c_row_ptr[row];
            c_row_ptr[row] = running;
            running += cnt;
        }
        c_row_ptr[a_rows] = running;
    }

    /* ---- exact output allocation ---- */
    size_t alloc_nnz = total_nnz > 0 ? (size_t)total_nnz : 1;
    int32_t *c_col_idx = (int32_t *)malloc(alloc_nnz * sizeof(int32_t));
    int32_t *c_values  = (int32_t *)malloc(alloc_nnz * sizeof(int32_t));

    if (!c_col_idx || !c_values)
    {
        free(acc); free(touched); free(mark); free(c_row_ptr);
        free(c_col_idx); free(c_values);
        return RVSP_ERROR_ALLOCATION_FAILED;
    }

    /* ---- numeric pass ---- */
    for (int32_t row = 0; row < a_rows; row++)
    {
        int32_t n_touched = 0;

        for (int32_t a_pos = a_row_ptr[row]; a_pos < a_row_ptr[row + 1]; a_pos++)
        {
            int32_t k = a_col_idx[a_pos];
            int32_t a_val = (int32_t)a_values[a_pos];

            for (int32_t b_pos = b_row_ptr[k]; b_pos < b_row_ptr[k + 1]; b_pos++)
            {
                int32_t col = b_col_idx[b_pos];
                int32_t b_val = (int32_t)b_values[b_pos];

                if (!mark[col])
                {
                    mark[col] = 1;
                    acc[col] = 0;
                    touched[n_touched++] = col;
                }

                acc[col] += a_val * b_val;
            }
        }

        /* canonical CSR: ascending column order within the row */
        qsort(touched, (size_t)n_touched, sizeof(int32_t), cmp_i32);

        int32_t dst = c_row_ptr[row];
        for (int32_t t = 0; t < n_touched; t++)
        {
            int32_t col = touched[t];
            /* keep explicit zeros out, matching previous behavior
             * (a symbolic slot can numerically cancel to zero) */
            if (acc[col] != 0)
            {
                c_col_idx[dst] = col;
                c_values[dst] = acc[col];
                dst++;
            }
            mark[col] = 0;
        }

        /* if cancellation dropped entries, compact the row pointer */
        c_row_ptr[row + 1] = dst;
    }

    free(acc); free(touched); free(mark);

    *c_row_ptr_out = c_row_ptr;
    *c_col_idx_out = c_col_idx;
    *c_values_out = c_values;
    *c_nnz_out = c_row_ptr[a_rows];

    return RVSP_SUCCESS;
}