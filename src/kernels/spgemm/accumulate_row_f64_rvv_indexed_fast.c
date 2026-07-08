#include <stdint.h>
#include <stddef.h>

#include "csr_spgemm_kernels.h"

#if defined(__riscv_vector)
#include <riscv_vector.h>
#define RVSP_HAVE_RVV_INTRINSICS 1
#endif

rvsp_status_t rvsp_accumulate_row_f64_rvv_indexed_fast(
    double a_val,
    int32_t b_nnz,
    const int32_t *b_col_idx,
    const double *b_values,
    double *acc
)
{
    if (b_nnz < 0 || b_col_idx == NULL || b_values == NULL || acc == NULL) {
        return RVSP_ERROR_INVALID_ARGUMENT;
    }

#if defined(RVSP_HAVE_RVV_INTRINSICS)
    int32_t p = 0;

    while (p < b_nnz) {
        size_t vl = __riscv_vsetvl_e64m4((size_t)(b_nnz - p));

        vfloat64m4_t vb = __riscv_vle64_v_f64m4(&b_values[p], vl);
        vfloat64m4_t vprod = __riscv_vfmul_vf_f64m4(vb, a_val, vl);

        vint32m2_t vidx_i32 = __riscv_vle32_v_i32m2(&b_col_idx[p], vl);
        vint32m2_t voff_i32 = __riscv_vsll_vx_i32m2(vidx_i32, 3, vl);
        vuint32m2_t voff_u32 = __riscv_vreinterpret_v_i32m2_u32m2(voff_i32);

        vfloat64m4_t vacc = __riscv_vluxei32_v_f64m4(acc, voff_u32, vl);
        vacc = __riscv_vfadd_vv_f64m4(vacc, vprod, vl);
        __riscv_vsuxei32_v_f64m4(acc, voff_u32, vacc, vl);

        p += (int32_t)vl;
    }
#else
    for (int32_t p = 0; p < b_nnz; p++) {
        int32_t col = b_col_idx[p];
        acc[col] += a_val * b_values[p];
    }
#endif

    return RVSP_SUCCESS;
}
