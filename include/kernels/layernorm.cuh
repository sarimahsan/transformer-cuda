#pragma once

#include <cuda_runtime.h>

// Vectorized Float4 LayerNorm Forward:
// y = (x - mean) * rstd * gamma + beta
// Saves mean and rstd (1.0f / sqrt(var + eps)) for backward pass.
// Shapes: x, out: (N, C), gamma, beta: (C,), mean, rstd: (N,)
void layernorm_forward(
    const float* x,
    const float* gamma,
    const float* beta,
    float* out,
    float* mean,
    float* rstd,
    int N, int C,
    float eps,
    cudaStream_t stream = 0
);

// Analytical LayerNorm Backward:
// Computes dx, dgamma, dbeta using saved x, out, mean, rstd, and dout.
void layernorm_backward(
    const float* dout,
    const float* x,
    const float* gamma,
    const float* mean,
    const float* rstd,
    float* dx,
    float* dgamma,
    float* dbeta,
    int N, int C,
    cudaStream_t stream = 0
);
