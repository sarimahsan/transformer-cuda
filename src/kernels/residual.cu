#include "kernels/residual.cuh"
#include "common.h"

__global__ void residual_add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    out[idx] = a[idx] + b[idx];
}

__global__ void residual_accumulate_kernel(
    float* __restrict__ a,
    const float* __restrict__ b,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    a[idx] += b[idx];
}

void residual_add(
    const float* a,
    const float* b,
    float* out,
    int N,
    cudaStream_t stream
) {
    int block_dim = 256;
    int grid_dim = (N + block_dim - 1) / block_dim;
    residual_add_kernel<<<grid_dim, block_dim, 0, stream>>>(a, b, out, N);
}

void residual_accumulate(
    float* a,
    const float* b,
    int N,
    cudaStream_t stream
) {
    int block_dim = 256;
    int grid_dim = (N + block_dim - 1) / block_dim;
    residual_accumulate_kernel<<<grid_dim, block_dim, 0, stream>>>(a, b, N);
}
