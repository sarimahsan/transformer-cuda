#pragma once

#include <cuda_runtime.h>

// GELU Activation Forward:
// y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void gelu_forward(
    const float* x,
    float* y,
    int N,
    cudaStream_t stream = 0
);

// Fused Bias Addition + GELU Activation Forward:
// y = GELU(x + bias) evaluated in-register using vectorized float4 memory access
void add_bias_gelu_forward(
    const float* x,
    const float* bias,
    float* y,
    int M,
    int N,
    cudaStream_t stream = 0
);

// GELU Activation Backward:
// dx = dy * gelu'(x)
void gelu_backward(
    const float* dy,
    const float* x,
    float* dx,
    int N,
    cudaStream_t stream = 0
);
