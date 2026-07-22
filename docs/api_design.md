# rv-sparse API Design

## Purpose

The initial `rv-sparse` API provides a clean C interface for sparse matrix operations while keeping the low-level kernel implementations separated from the public interface.

The current implementation focuses on CSR sparse matrix-matrix multiplication supporting multiple data types and optimized backends (Scalar and RVV).

## Current Supported Operation

The currently implemented operation is:

```text
C = A * B
```

where `A`, `B`, and `C` are sparse matrices stored in Compressed Sparse Row (CSR) format.

## Current Supported Data Types

The implemented data type combinations are:

```text
FP32 x FP32 -> FP32
FP64 x FP64 -> FP64
INT8 x INT8 -> INT8
```

This means that:

* `values` arrays must contain `float`, `double`, or `int8_t` types respectively.
* The API dynamically manages these types through the `rvsp_dtype_t` enumeration (`RVSP_DTYPE_FP32`, `RVSP_DTYPE_FP64`, `RVSP_DTYPE_INT8`).

## Current Supported Backends

The currently implemented backends are:

```text
RVSP_BACKEND_SCALAR
RVSP_BACKEND_RVV
```

The engine provides specific kernel implementations for these backends:

* **Scalar**: Baseline implementations (`csr_scalar_*`) and optimized unrolled versions (e.g., `csr_scalar_unroll4_f32`).
* **RVV (RISC-V Vector)**: Highly optimized vector kernels using indexed and marked approaches along with fast row accumulators (`csr_rvv_*_indexed_marked`, `accumulate_row_*_rvv_indexed_fast`).

## Architecture

The library currently follows a layered architecture composed of four main components.

### 1. Public API layer

The public API is exposed through:

```text
include/rv_sparse.h
include/rv_sparse_types.h
```

This layer defines the public matrix descriptors, status codes, data types, backend options, and user-facing functions.

Users interact with the library through structs such as:

```c
rvsp_csr_matrix_t
rvsp_spgemm_options_t
```

### 2. Dispatch layer

The dispatch layer is implemented in:

```text
src/core/spgemm.c
```

This layer selects the correct implementation according to:

* selected backend (Scalar or RVV)
* input data type
* output data type

Currently, the dispatcher supports routing to both Scalar and RVV execution paths for all supported data types (FP32, FP64, INT8).

### 3. Internal wrapper layer

The wrapper layer is implemented in:

```text
src/kernels/spgemm/csr_spgemm_wrappers.c
```

This layer connects the public API with the raw kernel implementation.

The wrapper is responsible for:

* validating input pointers
* validating CSR matrices
* checking matrix dimensions
* checking data types
* casting `void *values` to the correct typed pointer
* calling the specific raw pointer kernel based on dispatch rules
* filling the output `rvsp_csr_matrix_t`

### 4. Raw kernel layer

The raw kernels are implemented in `src/kernels/spgemm/` and include files such as:

```text
csr_scalar_f32.c
csr_scalar_f64.c
csr_scalar_i8.c
csr_rvv_f32_indexed_marked.c
...
```

This layer contains the actual computational engines.

The raw kernels use plain pointers and dimensions instead of public structs. This design keeps kernels simple, portable, and easier to optimize using specific RISC-V intrinsics or inline assembly.

## Main Public Types

### `rvsp_csr_matrix_t`

Represents a sparse matrix in CSR format.

Main fields:

```c
int32_t rows;
int32_t cols;
int32_t nnz;

int32_t *row_ptr;
int32_t *col_idx;
void    *values;

rvsp_dtype_t dtype;
rvsp_format_t format;

int owns_data;
```

CSR layout:

```text
row_ptr size: rows + 1
col_idx size: nnz
values  size: nnz
```

The `values` field is stored as `void *` so the same matrix descriptor can be reused for different data types (`float *`, `double *`, `int8_t *`).

### `rvsp_spgemm_options_t`

Controls the backend and data type selection for SpGEMM.

```c
typedef struct {
    rvsp_backend_t backend;
    rvsp_dtype_t input_dtype;
    rvsp_dtype_t output_dtype;
    int sort_output_indices;
} rvsp_spgemm_options_t;
```

Example configuration for an RVV FP64 execution:

```c
rvsp_spgemm_options_t options = {
    .backend = RVSP_BACKEND_RVV,
    .input_dtype = RVSP_DTYPE_FP64,
    .output_dtype = RVSP_DTYPE_FP64,
    .sort_output_indices = 1
};
```

## Public Functions

### `rvsp_csr_create()`

Initializes a CSR matrix descriptor.

```c
rvsp_status_t rvsp_csr_create(
    rvsp_csr_matrix_t *A,
    int32_t rows,
    int32_t cols,
    int32_t nnz,
    int32_t *row_ptr,
    int32_t *col_idx,
    void *values,
    rvsp_dtype_t dtype
);
```

This function does not allocate matrix data. It only initializes the descriptor using arrays provided by the caller.

### `rvsp_csr_validate()`

Validates the structure of a CSR matrix.

```c
rvsp_status_t rvsp_csr_validate(const rvsp_csr_matrix_t *A);
```

The validation checks:

* non-null matrix pointer
* non-null `row_ptr`, `col_idx`, and `values`
* valid dimensions
* `row_ptr[0] == 0`
* `row_ptr[rows] == nnz`
* non-decreasing `row_ptr`
* column indices within matrix bounds

### `rvsp_csr_destroy()`

Resets a CSR matrix descriptor and frees owned memory when applicable.

```c
void rvsp_csr_destroy(rvsp_csr_matrix_t *A);
```

Ownership behavior:

* If `owns_data == 0`, the arrays are not freed.
* If `owns_data == 1`, `row_ptr`, `col_idx`, and `values` are freed.

Matrices created with `rvsp_csr_create()` do not own their input arrays.

Matrices returned by `rvsp_spgemm_csr()` currently own their output arrays.

### `rvsp_spgemm_csr()`

Computes sparse matrix-matrix multiplication in CSR format.

```c
rvsp_status_t rvsp_spgemm_csr(
    const rvsp_csr_matrix_t *A,
    const rvsp_csr_matrix_t *B,
    rvsp_csr_matrix_t *C,
    const rvsp_spgemm_options_t *options
);
```

If an unsupported backend or data type combination is requested, the function returns an error status.

## Ownership Model

The current ownership model is intentionally simple.

Input matrices:

* The caller owns `row_ptr`, `col_idx`, and `values`.
* `rvsp_csr_create()` does not copy or free those arrays.
* Input matrix descriptors have `owns_data = 0`.

Output matrix:

* `rvsp_spgemm_csr()` allocates the CSR arrays for `C`.
* The output matrix has `owns_data = 1`.
* The caller must release the output matrix with `rvsp_csr_destroy(&C)`.

## Current Design Rule

The public API should remain stable as new kernels are added.

New implementations should be added internally by providing:

1. a raw pointer kernel
2. an internal wrapper integration
3. a dispatcher branch
4. an example implementation
5. a correctness test in the test suite

User code should continue to call the same public function:

```c
rvsp_spgemm_csr(&A, &B, &C, &options);
```
