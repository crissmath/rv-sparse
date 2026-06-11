# rv-sparse Directory Structure

The project is organized using an OpenBLAS/MKL-inspired separation between the public API, core library logic, sparse formats, optimized kernels, benchmarks, examples, tests, and documentation.

## Proposed Layout

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
