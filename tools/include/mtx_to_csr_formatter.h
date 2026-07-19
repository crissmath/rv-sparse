#ifndef MTX_TO_CSR_FORMATTER_H
#define MTX_TO_CSR_FORMATTER_H

#include "vec.h"

struct CSR {
    vec_t *row_ptr;
    vec_t *col_ind;
    vec_t *val;
};

struct CSR assemble_csr_matrix(const char *filePath);
void printArray_int(vec_t *v);
void printArray_float(vec_t *v);

#endif // MTX_TO_CSR_FORMATTER_H
