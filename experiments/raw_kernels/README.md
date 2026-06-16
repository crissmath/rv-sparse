# Raw Kernel Experiments

This directory contains experimental raw kernels for `rv-sparse`.

The goal is to develop and test scalar, unrolled, GCC auto-vectorized, and RISC-V Vector kernels before integrating them into the public API.

## Current Target

The initial optimization target is:

```text
INT8 x INT8 -> INT32
```

1. Development Flow
2. Implement a raw kernel.
3. Compare against the scalar reference.
4. Run correctness tests.
5. Run RISC-V/QEMU tests.
6. Benchmark the result.
7. Integrate into the public API only after validation.
Rule

> Experimental kernels should not be exposed through the public dispatcher until they are correct and tested
