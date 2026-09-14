#include "kernels/rmsnorm.cuh"
#include "common.h"

#define WARP_SIZE 32

__device__ __forceinline__ float rms_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float rms_block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = rms_warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
    if (wid == 0) val = rms_warp_reduce_sum(val);
    return val;
}

// Vectorized Float4 RMSNorm Forward Kernel
__global__ void rmsnorm_forward_vec4_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float* __restrict__ out,
    float* __restrict__ rstd_cache,
    int N, int C, float eps
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float4* x_vec = reinterpret_cast<const float4*>(x + row * C);
    float4* out_vec = reinterpret_cast<float4*>(out + row * C);
    const float4* gamma_vec = (gamma != nullptr) ? reinterpret_cast<const float4*>(gamma) : nullptr;

    extern __shared__ float s_mem[];
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int num_vec4 = C / 4;

    // 1. Sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < num_vec4; i += nthreads) {
        float4 val = x_vec[i];
        sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }
    float block_sum = rms_block_reduce_sum(sum_sq, s_mem);
    if (tid == 0) {
        float rstd = rsqrtf(block_sum / (float)C + eps);
        s_mem[0] = rstd;
        if (rstd_cache) rstd_cache[row] = rstd;
    }
    __syncthreads();
    float rstd = s_mem[0];

    // 2. Normalize and scale
    for (int i = tid; i < num_vec4; i += nthreads) {
        float4 val = x_vec[i];
        float4 g = (gamma_vec != nullptr) ? gamma_vec[i] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        float4 res;
        res.x = val.x * rstd * g.x;
        res.y = val.y * rstd * g.y;
        res.z = val.z * rstd * g.z;
        res.w = val.w * rstd * g.w;
        out_vec[i] = res;
    }
}

// Vectorized Float4 RMSNorm Backward Kernel
__global__ void rmsnorm_backward_vec4_kernel(
    const float* __restrict__ d_out,
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ rstd_cache,
    float* __restrict__ d_x,
    int N, int C
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float4* dout_vec = reinterpret_cast<const float4*>(d_out + row * C);
    const float4* x_vec = reinterpret_cast<const float4*>(x + row * C);
    float4* dx_vec = reinterpret_cast<float4*>(d_x + row * C);
    const float4* gamma_vec = (gamma != nullptr) ? reinterpret_cast<const float4*>(gamma) : nullptr;

    extern __shared__ float s_mem[];
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int num_vec4 = C / 4;
    float rstd = rstd_cache[row];

    // Compute dot(d_out * gamma, x)
    float sum_dx_x = 0.0f;
    for (int i = tid; i < num_vec4; i += nthreads) {
        float4 dout = dout_vec[i];
        float4 xv = x_vec[i];
        float4 g = (gamma_vec != nullptr) ? gamma_vec[i] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        sum_dx_x += (dout.x * g.x * xv.x) + (dout.y * g.y * xv.y) + (dout.z * g.z * xv.z) + (dout.w * g.w * xv.w);
    }
    float block_sum_dx_x = rms_block_reduce_sum(sum_dx_x, s_mem);
    if (tid == 0) s_mem[0] = block_sum_dx_x / (float)C;
    __syncthreads();
    float scale = s_mem[0];

    // dx_i = rstd * (dout_i * gamma_i - scale * x_i * rstd^2)
    float rstd3 = rstd * rstd * rstd;
    for (int i = tid; i < num_vec4; i += nthreads) {
        float4 dout = dout_vec[i];
        float4 xv = x_vec[i];
        float4 g = (gamma_vec != nullptr) ? gamma_vec[i] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        float4 res;
        res.x = rstd * dout.x * g.x - scale * xv.x * rstd3;
        res.y = rstd * dout.y * g.y - scale * xv.y * rstd3;
        res.z = rstd * dout.z * g.z - scale * xv.z * rstd3;
        res.w = rstd * dout.w * g.w - scale * xv.w * rstd3;
        dx_vec[i] = res;
    }
}

// Gamma gradient accumulation kernel
__global__ void rmsnorm_gamma_grad_kernel(
    const float* __restrict__ d_out,
    const float* __restrict__ x,
    const float* __restrict__ rstd_cache,
    float* __restrict__ d_gamma,
    int N, int C
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= C) return;

    float sum_g = 0.0f;
    for (int row = 0; row < N; ++row) {
        float rstd = rstd_cache[row];
        sum_g += d_out[row * C + col] * (x[row * C + col] * rstd);
    }
    atomicAdd(&d_gamma[col], sum_g);
}

void launch_rmsnorm_forward(
    const float* x,
    const float* gamma,
    float* out,
    float* rstd_cache,
    int N,
    int C,
    float eps,
    cudaStream_t stream
) {
    int threads = (C >= 1024) ? 256 : ((C >= 256) ? 128 : 64);
    int shared_bytes = ((threads + WARP_SIZE - 1) / WARP_SIZE) * sizeof(float);

    rmsnorm_forward_vec4_kernel<<<N, threads, shared_bytes, stream>>>(
        x, gamma, out, rstd_cache, N, C, eps
    );
}

void launch_rmsnorm_backward(
    const float* d_out,
    const float* x,
    const float* gamma,
    const float* rstd_cache,
    float* d_x,
    float* d_gamma,
    int N,
    int C,
    cudaStream_t stream
) {
    int threads = (C >= 1024) ? 256 : ((C >= 256) ? 128 : 64);
    int shared_bytes = ((threads + WARP_SIZE - 1) / WARP_SIZE) * sizeof(float);

    rmsnorm_backward_vec4_kernel<<<N, threads, shared_bytes, stream>>>(
        d_out, x, gamma, rstd_cache, d_x, N, C
    );

    if (d_gamma != nullptr) {
        int b_dim = 256;
        int g_dim = (C + b_dim - 1) / b_dim;
        rmsnorm_gamma_grad_kernel<<<g_dim, b_dim, 0, stream>>>(
            d_out, x, rstd_cache, d_gamma, N, C
        );
    }
}
