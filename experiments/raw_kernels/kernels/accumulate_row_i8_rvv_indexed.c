/*
 * Copyright (C) 2026 rv-sparse contributors
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Experimental RVV INT8 row accumulation microkernel using indexed gather/scatter.
 */
#include <stddef.h>
#include <stdint.h>

#include "exp_raw_kernels.h"

#if defined(__riscv_vector) && defined(__riscv_v_intrinsic) // sugest in RV intricsics
#include <riscv_vector.h>
#define EXP_HAVE_RVV_INTRINSICS 1
#endif

rvsp_status_t exp_accumulate_row_i8_rvv_indexed(int8_t a_val, int32_t b_nnz, const int32_t *b_col_idx,
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

#if defined(EXP_HAVE_RVV_INTRINSICS)
    int32_t p = 0;

    while (p < b_nnz)
    {
        size_t vl = __riscv_vsetvl_e8m1((size_t)(b_nnz - p));

        // This scalar validation is intentionally kept in the experimental
        //  if test =  pass can retired(wen you move to API).
        for (size_t t = 0; t < vl; t++)
        {
            int32_t col = b_col_idx[p + (int32_t)t];

            if (col < 0 || col >= b_cols)
            {
                return RVSP_ERROR_INVALID_CSR;
            }
        }
        vint8m1_t vb_i8 = __riscv_vle8_v_i8m1(&b_values[p], vl);

        vint16m2_t vprod_i16 = __riscv_vwmul_vx_i16m2(vb_i8, a_val, vl);

        vint32m4_t vprod_i32 = __riscv_vwadd_vx_i32m4(vprod_i16, (int16_t)0, vl);

        vint32m4_t vcols_i32 = __riscv_vle32_v_i32m4(&b_col_idx[p], vl);

        vuint32m4_t vcols_u32 = __riscv_vreinterpret_v_i32m4_u32m4(vcols_i32);

        vuint32m4_t voffsets = __riscv_vsll_vx_u32m4(vcols_u32, 2, vl);

        // vacc = acc[col]
        vint32m4_t vacc = __riscv_vluxei32_v_i32m4(acc, voffsets, vl);

        // acc[col] += a_val * b_values[p]
        vint32m4_t vsum = __riscv_vadd_vv_i32m4(vacc, vprod_i32, vl);
        __riscv_vsuxei32_v_i32m4(acc, voffsets, vsum, vl);

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