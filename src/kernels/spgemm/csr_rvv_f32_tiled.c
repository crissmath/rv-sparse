#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "csr_spgemm_kernels.h"

#ifndef RVSP_RVV_F32_TILE_COLS
#define RVSP_RVV_F32_TILE_COLS 4096
#endif

#ifndef RVSP_RVV_F32_TILE_MIN_B_COLS
#define RVSP_RVV_F32_TILE_MIN_B_COLS 8192
#endif

#ifndef RVSP_SORT_INSERTION_LIMIT
#define RVSP_SORT_INSERTION_LIMIT 32
#endif

static int rvsp_i32_cmp(const void* a, const void* b) {
  const int32_t x = *(const int32_t*)a;
  const int32_t y = *(const int32_t*)b;
  return (x > y) - (x < y);
}

static void rvsp_insertion_sort_i32(int32_t* x, int32_t n) {
  for (int32_t i = 1; i < n; i++) {
    const int32_t key = x[i];
    int32_t j = i - 1;

    while (j >= 0 && x[j] > key) {
      x[j + 1] = x[j];
      j--;
    }

    x[j + 1] = key;
  }
}

static void rvsp_sort_i32(int32_t* x, int32_t n) {
  if (n <= 1) {
    return;
  }

  if (n <= RVSP_SORT_INSERTION_LIMIT) {
    rvsp_insertion_sort_i32(x, n);
    return;
  }

  qsort(x, (size_t)n, sizeof(x[0]), rvsp_i32_cmp);
}

static int32_t rvsp_min_i32(int32_t a, int32_t b) { return a < b ? a : b; }

static int32_t rvsp_lower_bound_i32(const int32_t* x, int32_t lo, int32_t hi,
                                    int32_t key) {
  while (lo < hi) {
    const int32_t mid = lo + ((hi - lo) >> 1);

    if (x[mid] < key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  return lo;
}

static rvsp_status_t rvsp_validate_b_rows_for_tiling(int32_t b_rows,
                                                     int32_t b_cols,
                                                     const int32_t* b_row_ptr,
                                                     const int32_t* b_col_idx,
                                                     int* rows_sorted_out) {
  if (rows_sorted_out == NULL || b_row_ptr == NULL || b_col_idx == NULL) {
    return RVSP_ERROR_INVALID_ARGUMENT;
  }

  *rows_sorted_out = 1;

  if (b_rows < 0 || b_cols < 0 || b_row_ptr[0] < 0) {
    return RVSP_ERROR_INVALID_ARGUMENT;
  }

  for (int32_t r = 0; r < b_rows; r++) {
    const int32_t begin = b_row_ptr[r];
    const int32_t end = b_row_ptr[r + 1];

    if (begin < 0 || end < begin) {
      return RVSP_ERROR_INVALID_ARGUMENT;
    }

    for (int32_t p = begin; p < end; p++) {
      const int32_t col = b_col_idx[p];

      if (col < 0 || col >= b_cols) {
        return RVSP_ERROR_INVALID_ARGUMENT;
      }

      if (p > begin && b_col_idx[p - 1] > col) {
        *rows_sorted_out = 0;
      }
    }
  }

  return RVSP_SUCCESS;
}

rvsp_status_t rvsp_spgemm_csr_rvv_f32_tiled_raw(
    int32_t a_rows, int32_t a_cols, int32_t b_cols, const int32_t* a_row_ptr,
    const int32_t* a_col_idx, const float* a_values, const int32_t* b_row_ptr,
    const int32_t* b_col_idx, const float* b_values, int32_t** c_row_ptr_out,
    int32_t** c_col_idx_out, float** c_values_out, int32_t* c_nnz_out) {
  if (a_rows < 0 || a_cols < 0 || b_cols < 0 || a_row_ptr == NULL ||
      b_row_ptr == NULL || c_row_ptr_out == NULL || c_col_idx_out == NULL ||
      c_values_out == NULL || c_nnz_out == NULL) {
    return RVSP_ERROR_INVALID_ARGUMENT;
  }

  *c_row_ptr_out = NULL;
  *c_col_idx_out = NULL;
  *c_values_out = NULL;
  *c_nnz_out = 0;

  if (a_rows == 0) {
    int32_t* empty_row_ptr = (int32_t*)calloc(1, sizeof(int32_t));
    if (empty_row_ptr == NULL) {
      return RVSP_ERROR_ALLOCATION_FAILED;
    }

    *c_row_ptr_out = empty_row_ptr;
    return RVSP_SUCCESS;
  }

  if (a_cols == 0 || b_cols == 0) {
    int32_t* empty_row_ptr =
        (int32_t*)calloc((size_t)a_rows + 1u, sizeof(int32_t));
    if (empty_row_ptr == NULL) {
      return RVSP_ERROR_ALLOCATION_FAILED;
    }

    *c_row_ptr_out = empty_row_ptr;
    return RVSP_SUCCESS;
  }

  if (a_col_idx == NULL || a_values == NULL || b_col_idx == NULL ||
      b_values == NULL) {
    return RVSP_ERROR_INVALID_ARGUMENT;
  }

  // Benchmark can force tiling with:
  //  -DRVSP_RVV_F32_TILE_MIN_B_COLS=0
  if (b_cols < RVSP_RVV_F32_TILE_MIN_B_COLS) {
    return rvsp_spgemm_csr_rvv_f32_indexed_marked_raw(
        a_rows, a_cols, b_cols, a_row_ptr, a_col_idx, a_values, b_row_ptr,
        b_col_idx, b_values, c_row_ptr_out, c_col_idx_out, c_values_out,
        c_nnz_out);
  }

  int b_rows_sorted = 1;
  const rvsp_status_t b_status = rvsp_validate_b_rows_for_tiling(
      a_cols, b_cols, b_row_ptr, b_col_idx, &b_rows_sorted);

  if (b_status != RVSP_SUCCESS) {
    return b_status;
  }

  if (!b_rows_sorted) {
    return rvsp_spgemm_csr_rvv_f32_indexed_marked_raw(
        a_rows, a_cols, b_cols, a_row_ptr, a_col_idx, a_values, b_row_ptr,
        b_col_idx, b_values, c_row_ptr_out, c_col_idx_out, c_values_out,
        c_nnz_out);
  }

  int32_t tile_cols = RVSP_RVV_F32_TILE_COLS;
  if (tile_cols <= 0) {
    tile_cols = b_cols;
  }

  if (tile_cols > b_cols) {
    tile_cols = b_cols;
  }

  int32_t* row_counts = (int32_t*)calloc((size_t)a_rows, sizeof(int32_t));
  int32_t* c_row_ptr =
      (int32_t*)malloc(((size_t)a_rows + 1u) * sizeof(int32_t));
  unsigned char* mark =
      (unsigned char*)calloc((size_t)tile_cols, sizeof(unsigned char));
  float* acc = (float*)calloc((size_t)tile_cols, sizeof(float));
  int32_t* touched = (int32_t*)malloc((size_t)tile_cols * sizeof(int32_t));

  if (row_counts == NULL || c_row_ptr == NULL || mark == NULL || acc == NULL ||
      touched == NULL) {
    free(row_counts);
    free(c_row_ptr);
    free(mark);
    free(acc);
    free(touched);
    return RVSP_ERROR_ALLOCATION_FAILED;
  }

  int64_t c_nnz_est = 0;

  for (int32_t tile_start = 0; tile_start < b_cols; tile_start += tile_cols) {
    const int32_t tile_end = rvsp_min_i32(tile_start + tile_cols, b_cols);

    for (int32_t i = 0; i < a_rows; i++) {
      int32_t touched_count = 0;

      for (int32_t pa = a_row_ptr[i]; pa < a_row_ptr[i + 1]; pa++) {
        const int32_t brow = a_col_idx[pa];

        if (brow < 0 || brow >= a_cols) {
          free(row_counts);
          free(c_row_ptr);
          free(mark);
          free(acc);
          free(touched);
          return RVSP_ERROR_INVALID_ARGUMENT;
        }

        const int32_t pb_begin = rvsp_lower_bound_i32(
            b_col_idx, b_row_ptr[brow], b_row_ptr[brow + 1], tile_start);
        const int32_t pb_end = rvsp_lower_bound_i32(
            b_col_idx, pb_begin, b_row_ptr[brow + 1], tile_end);

        for (int32_t pb = pb_begin; pb < pb_end; pb++) {
          const int32_t local_col = b_col_idx[pb] - tile_start;

          if (!mark[local_col]) {
            mark[local_col] = 1u;
            touched[touched_count++] = local_col;
            row_counts[i]++;

            if (row_counts[i] < 0) {
              free(row_counts);
              free(c_row_ptr);
              free(mark);
              free(acc);
              free(touched);
              return RVSP_ERROR_INVALID_ARGUMENT;
            }
          }
        }
      }

      for (int32_t t = 0; t < touched_count; t++) {
        mark[touched[t]] = 0u;
      }
    }
  }

  c_row_ptr[0] = 0;

  for (int32_t i = 0; i < a_rows; i++) {
    c_nnz_est += (int64_t)row_counts[i];

    if (c_nnz_est > INT_MAX) {
      free(row_counts);
      free(c_row_ptr);
      free(mark);
      free(acc);
      free(touched);
      return RVSP_ERROR_INVALID_ARGUMENT;
    }

    c_row_ptr[i + 1] = (int32_t)c_nnz_est;
  }

  int32_t* c_col_idx = NULL;
  float* c_values = NULL;

  if (c_nnz_est > 0) {
    c_col_idx = (int32_t*)malloc((size_t)c_nnz_est * sizeof(int32_t));
    c_values = (float*)malloc((size_t)c_nnz_est * sizeof(float));

    if (c_col_idx == NULL || c_values == NULL) {
      free(row_counts);
      free(c_row_ptr);
      free(c_col_idx);
      free(c_values);
      free(mark);
      free(acc);
      free(touched);
      return RVSP_ERROR_ALLOCATION_FAILED;
    }
  }

  int32_t c_nnz_actual = 0;

  for (int32_t i = 0; i < a_rows; i++) {
    c_row_ptr[i] = c_nnz_actual;

    for (int32_t tile_start = 0; tile_start < b_cols; tile_start += tile_cols) {
      const int32_t tile_end = rvsp_min_i32(tile_start + tile_cols, b_cols);
      int32_t touched_count = 0;

      for (int32_t pa = a_row_ptr[i]; pa < a_row_ptr[i + 1]; pa++) {
        const int32_t brow = a_col_idx[pa];
        const float a_val = a_values[pa];

        if (a_val == 0.0f) {
          continue;
        }

        const int32_t pb_begin = rvsp_lower_bound_i32(
            b_col_idx, b_row_ptr[brow], b_row_ptr[brow + 1], tile_start);
        const int32_t pb_end = rvsp_lower_bound_i32(
            b_col_idx, pb_begin, b_row_ptr[brow + 1], tile_end);

        for (int32_t pb = pb_begin; pb < pb_end; pb++) {
          const int32_t local_col = b_col_idx[pb] - tile_start;

          if (!mark[local_col]) {
            mark[local_col] = 1u;
            acc[local_col] = 0.0f;
            touched[touched_count++] = local_col;
          }

          acc[local_col] += a_val * b_values[pb];
        }
      }

      rvsp_sort_i32(touched, touched_count);

      for (int32_t t = 0; t < touched_count; t++) {
        const int32_t local_col = touched[t];
        const float value = acc[local_col];

        if (value != 0.0f) {
          if ((int64_t)c_nnz_actual >= c_nnz_est) {
            free(row_counts);
            free(c_row_ptr);
            free(c_col_idx);
            free(c_values);
            free(mark);
            free(acc);
            free(touched);
            return RVSP_ERROR_INVALID_ARGUMENT;
          }

          c_col_idx[c_nnz_actual] = tile_start + local_col;
          c_values[c_nnz_actual] = value;
          c_nnz_actual++;
        }

        mark[local_col] = 0u;
        acc[local_col] = 0.0f;
      }
    }
  }

  c_row_ptr[a_rows] = c_nnz_actual;

  if (c_nnz_actual == 0) {
    free(c_col_idx);
    free(c_values);
    c_col_idx = NULL;
    c_values = NULL;
  } else if ((int64_t)c_nnz_actual < c_nnz_est) {
    int32_t* new_col_idx =
        (int32_t*)realloc(c_col_idx, (size_t)c_nnz_actual * sizeof(int32_t));
    float* new_values =
        (float*)realloc(c_values, (size_t)c_nnz_actual * sizeof(float));

    if (new_col_idx != NULL) {
      c_col_idx = new_col_idx;
    }

    if (new_values != NULL) {
      c_values = new_values;
    }
  }

  free(row_counts);
  free(mark);
  free(acc);
  free(touched);

  *c_row_ptr_out = c_row_ptr;
  *c_col_idx_out = c_col_idx;
  *c_values_out = c_values;
  *c_nnz_out = c_nnz_actual;

  return RVSP_SUCCESS;
}
