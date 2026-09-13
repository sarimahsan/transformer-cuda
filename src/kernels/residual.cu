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

// Fused Bias + Residual Vectorized Kernel (float4)
__global__ void add_bias_residual_vec4_kernel(
    const float4* __restrict__ residual,
    const float4* __restrict__ in,
    const float4* __restrict__ bias,
    float4* __restrict__ out,
    int num_vec4,
    int n_vec4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vec4) return;

    int b_idx = idx % n_vec4;
    float4 r = residual[idx];
    float4 i = in[idx];
    float4 b = bias[b_idx];

    float4 res;
    res.x = r.x + i.x + b.x;
    res.y = r.y + i.y + b.y;
    res.z = r.z + i.z + b.z;
    res.w = r.w + i.w + b.w;

    out[idx] = res;
}

// Fused Bias + Residual Scalar Fallback Kernel
__global__ void add_bias_residual_kernel(
    const float* __restrict__ residual,
    const float* __restrict__ in,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int total_elements,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int col = idx % N;
    out[idx] = residual[idx] + in[idx] + bias[col];
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

void add_bias_residual(
    const float* residual,
    const float* in,
    const float* bias,
    float* out,
    int M,
    int N,
    cudaStream_t stream
) {
    int total_elements = M * N;
    if (N % 4 == 0 && total_elements % 4 == 0) {
        int num_vec4 = total_elements / 4;
        int n_vec4 = N / 4;
        int block_dim = 256;
        int grid_dim = (num_vec4 + block_dim - 1) / block_dim;
        add_bias_residual_vec4_kernel<<<grid_dim, block_dim, 0, stream>>>(
            reinterpret_cast<const float4*>(residual),
            reinterpret_cast<const float4*>(in),
            reinterpret_cast<const float4*>(bias),
            reinterpret_cast<float4*>(out),
            num_vec4,
            n_vec4
        );
    } else {
        int block_dim = 256;
        int grid_dim = (total_elements + block_dim - 1) / block_dim;
        add_bias_residual_kernel<<<grid_dim, block_dim, 0, stream>>>(
            residual, in, bias, out, total_elements, N
        );
    }
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
