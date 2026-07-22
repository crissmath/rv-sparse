# rv-sparse Documentation

This directory contains the core design documentation, architectural decisions, and API guidelines for the `rv-sparse` project.

## Current Focus

The active development and documentation efforts are currently centered around the following areas:

| Focus Area | Status | Comments / Next Steps |
| :--- | :--- | :--- |
| **Public API Design** | Established | Stable interface defined for users (`rv_sparse.h`). |
| **Directory Structure** | Established | Standard C library layout implemented. |
| **CSR SpGEMM Interface** | Implemented | Baseline kernels for sparse matrix multiplication standardized. |
| **Backend Selection Model** | Work in Progress | Refining the dispatch mechanism (Scalar vs. GCC auto-vectorized vs. RVV). |
| **Project Timeline** | Active | Tracking milestones and upcoming feature integrations. |

## Library Architecture

The library structure ensures a strict separation of concerns. Below is the visual representation of the architecture followed by its detailed components:

```text
+-------------------------------------------------------------+
|                   rv-sparse Architecture                    |
+-------------------------------------------------------------+
|                                                             |
|  [rv-sparse/]                                               |
|   |                                                         |
|   +-- [include/]    <-- Public API Headers                  |
|   |                                                         |
|   +-- [src/]                                                |
|   |    +-- [core/]     <-- Logic, Context & Error Handling  |
|   |    +-- [formats/]  <-- Data Structures (CSR, COO)       |
|   |    `-- [kernels/]  <-- Compute Engines (Scalar, RVV)    |
|   |                                                         |
|   +-- [benchmarks/] <-- Performance Measurement             |
|   +-- [examples/]   <-- API Usage Samples                   |
|   +-- [tests/]      <-- Correctness Validation              |
|   `-- [docs/]       <-- Architecture Specs & Guidelines     |
|                                                             |
+-------------------------------------------------------------+
```
