Plaintext

# rv-sparse Directory Structure

The project follows a standard C library organization, separating the public API, core library logic, sparse formats, optimized kernels, benchmarks, examples, tests, and documentation.

## Project Layout

```text
rv-sparse/
├── include/
│   ├── rv_sparse.h
│   └── rv_sparse_types.h
│
├── src/
│   ├── core/
│   ├── formats/
│   └── kernels/
│       └── spgemm/
│
├── benchmarks/
├── examples/
├── tests/
└── docs/
```

## Directory Roles

### include/

Public C API headers exposed to users of the library.

### src/core/

Core implementation layer for validation, error handling, dispatching, and common library utilities.

### src/formats/

Sparse matrix format utilities. CSR is the initial format.

### src/kernels/spgemm/

Sparse GEMM kernels organized by backend and data type.

### benchmarks/

Performance evaluation programs.

### examples/

Minimal programs showing how to use the public API.

### tests/

Correctness tests.

### docs/

Project documentation, API design notes, and timeline.
