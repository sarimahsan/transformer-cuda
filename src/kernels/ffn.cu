#include "kernels/ffn.cuh"
#include "common.h"
#include <cmath>

#define SQRT_2_OVER_PI 0.7978845608028654f
#define COEFF 0.044715f

__device__ __forceinline__ float gelu_compute(float val) {
    float u = SQRT_2_OVER_PI * (val + COEFF * val * val * val);
    float t = tanhf(u);
    return 0.5f * val * (1.0f + t);
}

__device__ __forceinline__ float gelu_bwd_compute(float val, float dy) {
    float u = SQRT_2_OVER_PI * (val + COEFF * val * val * val);
    float t = tanhf(u);
    float dt = 1.0f - t * t;
    float du = SQRT_2_OVER_PI * (1.0f + 3.0f * COEFF * val * val);
    float dgelu = 0.5f * (1.0f + t) + 0.5f * val * dt * du;
    return dy * dgelu;
}

__global__ void gelu_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    y[idx] = gelu_compute(x[idx]);
}

// Fused Bias + GELU Vectorized Kernel (128-bit float4)
__global__ void add_bias_gelu_forward_vec4_kernel(
    const float4* __restrict__ x,
    const float4* __restrict__ bias,
    float4* __restrict__ y,
    int num_vec4,
    int n_vec4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vec4) return;

    int b_idx = idx % n_vec4;
    float4 x_val = x[idx];
    float4 b_val = bias[b_idx];

    float4 out_val;
    out_val.x = gelu_compute(x_val.x + b_val.x);
    out_val.y = gelu_compute(x_val.y + b_val.y);
    out_val.z = gelu_compute(x_val.z + b_val.z);
    out_val.w = gelu_compute(x_val.w + b_val.w);

    y[idx] = out_val;
}

// Fused Bias + GELU Scalar Fallback Kernel
__global__ void add_bias_gelu_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int total_elements,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int col = idx % N;
    y[idx] = gelu_compute(x[idx] + bias[col]);
}

__global__ void gelu_backward_kernel(
    const float* __restrict__ dy,
    const float* __restrict__ x,
    float* __restrict__ dx,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    dx[idx] = gelu_bwd_compute(x[idx], dy[idx]);
}

// Vectorized GELU Backward Kernel (float4)
__global__ void gelu_backward_vec4_kernel(
    const float4* __restrict__ dy,
    const float4* __restrict__ x,
    float4* __restrict__ dx,
    int num_vec4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vec4) return;

    float4 dy_val = dy[idx];
    float4 x_val = x[idx];

    float4 out_val;
    out_val.x = gelu_bwd_compute(x_val.x, dy_val.x);
    out_val.y = gelu_bwd_compute(x_val.y, dy_val.y);
    out_val.z = gelu_bwd_compute(x_val.z, dy_val.z);
    out_val.w = gelu_bwd_compute(x_val.w, dy_val.w);

    dx[idx] = out_val;
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

void add_bias_gelu_forward(
    const float* x,
    const float* bias,
    float* y,
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
        add_bias_gelu_forward_vec4_kernel<<<grid_dim, block_dim, 0, stream>>>(
            reinterpret_cast<const float4*>(x),
            reinterpret_cast<const float4*>(bias),
            reinterpret_cast<float4*>(y),
            num_vec4,
            n_vec4
        );
    } else {
        int block_dim = 256;
        int grid_dim = (total_elements + block_dim - 1) / block_dim;
        add_bias_gelu_forward_kernel<<<grid_dim, block_dim, 0, stream>>>(x, bias, y, total_elements, N);
    }
}

void gelu_backward(
    const float* dy,
    const float* x,
    float* dx,
    int N,
    cudaStream_t stream
) {
    if (N % 4 == 0) {
        int num_vec4 = N / 4;
        int block_dim = 256;
        int grid_dim = (num_vec4 + block_dim - 1) / block_dim;
        gelu_backward_vec4_kernel<<<grid_dim, block_dim, 0, stream>>>(
            reinterpret_cast<const float4*>(dy),
            reinterpret_cast<const float4*>(x),
            reinterpret_cast<float4*>(dx),
            num_vec4
        );
    } else {
        int block_dim = 256;
        int grid_dim = (N + block_dim - 1) / block_dim;
        gelu_backward_kernel<<<grid_dim, block_dim, 0, stream>>>(dy, x, dx, N);
    }
}
