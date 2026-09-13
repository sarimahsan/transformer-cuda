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

// Fused Bias Addition + Residual Add: out = residual + in + bias
// Evaluated in-register with 128-bit float4 memory access
void add_bias_residual(
    const float* residual,
    const float* in,
    const float* bias,
    float* out,
    int M,
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
