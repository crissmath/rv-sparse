# Raw Kernel Experiments

This directory contains experimental raw kernels for `rv-sparse`.

The purpose of this workspace is to develop, test, and compare low-level kernels before integrating them into the public API.

These kernels are intentionally kept outside the public dispatcher while they are still experimental.

## Current Target

The initial optimization target is:

```text
INT8 x INT8 -> INT32
```

This target is used to evaluate scalar, unrolled, GCC auto-vectorized, and RISC-V Vector implementations.

## Current Scope

The current experimental scope includes:

```text
CSR SpGEMM
INT8 input values
INT32 accumulation/output
Raw pointer-based kernels
Correctness tests against a scalar reference
RISC-V/QEMU validation
```

## Development Flow

1. Implement a raw kernel.
2. Compare the result against the scalar reference kernel.
3. Run native correctness tests.
4. Run RISC-V/QEMU correctness tests.
5. Benchmark the kernel.
6. Compare performance against the other experimental kernels.
7. Integrate into the public API only after correctness and performance validation.

## Rule

Experimental kernels must not be exposed through the public dispatcher until they are correct, tested, benchmarked, and selected for integration.

## Directory Structure

```text
├── Makefile
├── README.md
├── bin
│   ├── test_accumulate_row_i8_rvv
│   ├── test_csr_rvv_i8_indexed_marked_raw
│   ├── test_csr_rvv_i8_indexed_raw
│   └── test_csr_rvv_i8_raw
├── kernels
│   ├── accumulate_row_i8_rvv.c
│   ├── accumulate_row_i8_rvv_indexed.c
│   ├── accumulate_row_i8_rvv_indexed_fast.c
│   ├── csr_rvv_i8.c
│   ├── csr_rvv_i8_indexed.c
│   ├── csr_rvv_i8_indexed_marked.c
│   ├── csr_scalar_i8_reference.c
│   └── exp_raw_kernels.h
├── obj
│   └── kernels
└── tests
    ├── test_accumulate_row_i8_rvv.c
    ├── test_csr_rvv_i8_indexed_marked_raw.c
    ├── test_csr_rvv_i8_indexed_raw.c
    └── test_csr_rvv_i8_raw.c
```

## Build and Test

From this directory:

```bash
make clean
make
make test
```

To test with the RISC-V cross-compiler and QEMU:

```bash
make clean
make test TARGET_ARCH=riscv
```

## Actual test output

```bash
---> Target Architecture: NATIVE Host
---> Build Type: RELEASE

--- Running Raw Kernel Tests ---
Executing bin/test_csr_rvv_i8_raw...
test_csr_rvv_i8_raw: PASS [scalar fallback path]
Executing bin/test_accumulate_row_i8_rvv...
test_accumulate_row_i8_rvv: PASS [scalar fallback path]
Executing bin/test_csr_rvv_i8_indexed_raw...
test_csr_rvv_i8_indexed_raw: PASS [scalar fallback path]
Executing bin/test_csr_rvv_i8_indexed_marked_raw...
test_csr_rvv_i8_indexed_marked_raw: PASS [scalar fallback path]
--- All Raw Kernel Tests Passed ---
```

```bash
---> Target Architecture: RISC-V
---> Build Type: RELEASE
  CC      kernels/accumulate_row_i8_rvv_indexed_fast.c
  CC      kernels/csr_scalar_i8_reference.c
  CC      kernels/csr_rvv_i8_indexed.c
  CC      kernels/accumulate_row_i8_rvv_indexed.c
  CC      kernels/csr_rvv_i8_indexed_marked.c
  CC      kernels/accumulate_row_i8_rvv.c
  CC      kernels/csr_rvv_i8.c
  CCLD    bin/test_csr_rvv_i8_raw
  CCLD    bin/test_accumulate_row_i8_rvv
  CCLD    bin/test_csr_rvv_i8_indexed_raw
  CCLD    bin/test_csr_rvv_i8_indexed_marked_raw

--- Running Raw Kernel Tests ---
Executing bin/test_csr_rvv_i8_raw...
vector version is not specified, use the default value v1.0
test_csr_rvv_i8_raw: PASS [RVV path]
Executing bin/test_accumulate_row_i8_rvv...
vector version is not specified, use the default value v1.0
test_accumulate_row_i8_rvv: PASS [RVV path]
Executing bin/test_csr_rvv_i8_indexed_raw...
vector version is not specified, use the default value v1.0
test_csr_rvv_i8_indexed_raw: PASS [RVV indexed path]
Executing bin/test_csr_rvv_i8_indexed_marked_raw...
vector version is not specified, use the default value v1.0
test_csr_rvv_i8_indexed_marked_raw: PASS [RVV optimized path]
--- All Raw Kernel Tests Passed ---
```

## Integration Policy

Only the best validated kernel should be migrated to the main library backend.

The migration path is:

```text
experiments/raw_kernels/
src/kernels/spgemm/
wrapper layer
dispatcher layer
public API
```

The public API should remain stable while experimental kernels are being evaluated.
