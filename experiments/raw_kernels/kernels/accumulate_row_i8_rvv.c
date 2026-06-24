/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Experimental RVV INT8 row accumulation microkernel.
 */

#include <stdint.h>

#include "exp_raw_kernels.h"

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

rvsp_status_t exp_accumulate_row_i8_rvv(int8_t a_val, int32_t b_nnz,
                                        const int32_t *b_col_idx,
                                        const int8_t *b_values,
                                        int32_t b_cols,
                                        int32_t *acc)
{
    if (!b_col_idx || !b_values || !acc)
    {
        return RVSP_ERROR_NULL_POINTER;
    }

    if (b_nnz < 0 || b_cols <= 0)
    {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

#if defined(__riscv_vector)
    int32_t p = 0;

    while (p < b_nnz)
    {
        size_t vl = __riscv_vsetvl_e8m1((size_t)(b_nnz - p));

        vint8m1_t vb = __riscv_vle8_v_i8m1(&b_values[p], vl);
        vint16m2_t vprod = __riscv_vwmul_vx_i16m2(vb, a_val, vl);

        int16_t tmp[vl];
        __riscv_vse16_v_i16m2(tmp, vprod, vl);

        for (size_t t = 0; t < vl; t++)
        {
            int32_t col = b_col_idx[p + (int32_t)t];

            if (col < 0 || col >= b_cols)
            {
                return RVSP_ERROR_INVALID_CSR;
            }

            acc[col] += (int32_t)tmp[t];
        }

        p += (int32_t)vl;
    }
#else
    for (int32_t p = 0; p < b_nnz; p++)
    {
        int32_t col = b_col_idx[p];

        if (col < 0 || col >= b_cols)
        {
            return RVSP_ERROR_INVALID_CSR;
        }

        acc[col] += (int32_t)a_val * (int32_t)b_values[p];
    }
#endif

    return RVSP_SUCCESS;
}