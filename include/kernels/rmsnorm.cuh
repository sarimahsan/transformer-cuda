#pragma once

#include <cuda_runtime.h>

// Vectorized float4 RMSNorm forward kernel
void launch_rmsnorm_forward(
    const float* x,
    const float* gamma,
    float* out,
    float* rstd_cache,
    int N,
    int C,
    float eps,
    cudaStream_t stream = 0
);

// Vectorized float4 RMSNorm backward kernel
void launch_rmsnorm_backward(
    const float* d_out,
    const float* x,
    const float* gamma,
    const float* rstd_cache,
    float* d_x,
    float* d_gamma,
    int N,
    int C,
    cudaStream_t stream = 0
);
