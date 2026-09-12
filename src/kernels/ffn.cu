#include "kernels/ffn.cuh"
#include "common.h"
#include <cmath>

#define SQRT_2_OVER_PI 0.7978845608028654f
#define COEFF 0.044715f

__global__ void gelu_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float val = x[idx];
    float u = SQRT_2_OVER_PI * (val + COEFF * val * val * val);
    float t = tanhf(u);
    y[idx] = 0.5f * val * (1.0f + t);
}

__global__ void gelu_backward_kernel(
    const float* __restrict__ dy,
    const float* __restrict__ x,
    float* __restrict__ dx,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float val = x[idx];
    float u = SQRT_2_OVER_PI * (val + COEFF * val * val * val);
    float t = tanhf(u);
    float dt = 1.0f - t * t;
    float du = SQRT_2_OVER_PI * (1.0f + 3.0f * COEFF * val * val);
    float dgelu = 0.5f * (1.0f + t) + 0.5f * val * dt * du;

    dx[idx] = dy[idx] * dgelu;
}

void gelu_forward(
    const float* x,
    float* y,
    int N,
    cudaStream_t stream
) {
    int block_dim = 256;
    int grid_dim = (N + block_dim - 1) / block_dim;
    gelu_forward_kernel<<<grid_dim, block_dim, 0, stream>>>(x, y, N);
}

void gelu_backward(
    const float* dy,
    const float* x,
    float* dx,
    int N,
    cudaStream_t stream
) {
    int block_dim = 256;
    int grid_dim = (N + block_dim - 1) / block_dim;
    gelu_backward_kernel<<<grid_dim, block_dim, 0, stream>>>(dy, x, dx, N);
}
