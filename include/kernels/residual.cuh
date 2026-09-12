#pragma once

#include <cuda_runtime.h>

// Residual Add: out = a + b
void residual_add(
    const float* a,
    const float* b,
    float* out,
    int N,
    cudaStream_t stream = 0
);

// Accumulate: a += b
void residual_accumulate(
    float* a,
    const float* b,
    int N,
    cudaStream_t stream = 0
);
