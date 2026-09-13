#include "kernels/layernorm.cuh"
#include "common.h"

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

// Vectorized Float4 LayerNorm Forward Kernel
__global__ void layernorm_forward_vec4_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    float* __restrict__ mean_cache,
    float* __restrict__ rstd_cache,
    int N, int C, float eps
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float4* x_vec = reinterpret_cast<const float4*>(x + row * C);
    float4* out_vec = reinterpret_cast<float4*>(out + row * C);
    const float4* gamma_vec = (gamma != nullptr) ? reinterpret_cast<const float4*>(gamma) : nullptr;
    const float4* beta_vec = (beta != nullptr) ? reinterpret_cast<const float4*>(beta) : nullptr;

    extern __shared__ float s_mem[];
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int num_vec4 = C / 4;

    // 1. Mean
    float sum = 0.0f;
    for (int i = tid; i < num_vec4; i += nthreads) {
        float4 val = x_vec[i];
        sum += val.x + val.y + val.z + val.w;
    }
    float mean = block_reduce_sum(sum, s_mem);
    if (tid == 0) s_mem[0] = mean / (float)C;
    __syncthreads();
    mean = s_mem[0];

    // 2. Variance
    float var_sum = 0.0f;
    for (int i = tid; i < num_vec4; i += nthreads) {
        float4 val = x_vec[i];
        float dx = val.x - mean;
        float dy = val.y - mean;
        float dz = val.z - mean;
        float dw = val.w - mean;
        var_sum += dx * dx + dy * dy + dz * dz + dw * dw;
    }
    float var = block_reduce_sum(var_sum, s_mem);
    if (tid == 0) {
        float rstd = rsqrtf(var / (float)C + eps);
        s_mem[0] = rstd;
        mean_cache[row] = mean;
        rstd_cache[row] = rstd;
    }
    __syncthreads();
    float rstd = s_mem[0];

    // 3. Normalize & scale/shift
    for (int i = tid; i < num_vec4; i += nthreads) {
        float4 val = x_vec[i];
        float4 g = gamma_vec ? gamma_vec[i] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        float4 b = beta_vec  ? beta_vec[i]  : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        float4 res;
        res.x = ((val.x - mean) * rstd) * g.x + b.x;
        res.y = ((val.y - mean) * rstd) * g.y + b.y;
        res.z = ((val.z - mean) * rstd) * g.z + b.z;
        res.w = ((val.w - mean) * rstd) * g.w + b.w;
        out_vec[i] = res;
    }
}

// Fallback scalar kernel for non-multiple-of-4 dimensions
__global__ void layernorm_forward_scalar_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    float* __restrict__ mean_cache,
    float* __restrict__ rstd_cache,
    int N, int C, float eps
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* x_row = x + row * C;
    float* out_row = out + row * C;

    extern __shared__ float s_mem[];
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    float sum = 0.0f;
    for (int i = tid; i < C; i += nthreads) sum += x_row[i];
    float mean = block_reduce_sum(sum, s_mem);
    if (tid == 0) s_mem[0] = mean / (float)C;
    __syncthreads();
    mean = s_mem[0];

    float var_sum = 0.0f;
    for (int i = tid; i < C; i += nthreads) {
        float diff = x_row[i] - mean;
        var_sum += diff * diff;
    }
    float var = block_reduce_sum(var_sum, s_mem);
    if (tid == 0) {
        float rstd = rsqrtf(var / (float)C + eps);
        s_mem[0] = rstd;
        mean_cache[row] = mean;
        rstd_cache[row] = rstd;
    }
    __syncthreads();
    float rstd = s_mem[0];

    for (int i = tid; i < C; i += nthreads) {
        float g = gamma ? gamma[i] : 1.0f;
        float b = beta  ? beta[i]  : 0.0f;
        out_row[i] = ((x_row[i] - mean) * rstd) * g + b;
    }
}

// LayerNorm Backward Kernel for dx
__global__ void layernorm_backward_dx_kernel(
    const float* __restrict__ dout,
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ mean_cache,
    const float* __restrict__ rstd_cache,
    float* __restrict__ dx,
    int N, int C,
    const float* __restrict__ residual_add,
    bool accumulate
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* dout_row = dout + row * C;
    const float* x_row = x + row * C;
    float* dx_row = dx + row * C;
    const float* res_row = residual_add ? (residual_add + row * C) : nullptr;

    float mean = mean_cache[row];
    float rstd = rstd_cache[row];

    extern __shared__ float s_mem[];
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // sum1 = sum(dout * gamma)
    // sum2 = sum(dout * gamma * (x - mean))
    float thread_sum1 = 0.0f;
    float thread_sum2 = 0.0f;

    for (int i = tid; i < C; i += nthreads) {
        float g = gamma ? gamma[i] : 1.0f;
        float dy = dout_row[i];
        thread_sum1 += dy * g;
        thread_sum2 += dy * g * (x_row[i] - mean);
    }

    float s1 = block_reduce_sum(thread_sum1, s_mem);
    if (tid == 0) s_mem[0] = s1;
    __syncthreads();
    s1 = s_mem[0];

    float s2 = block_reduce_sum(thread_sum2, s_mem);
    if (tid == 0) s_mem[0] = s2;
    __syncthreads();
    s2 = s_mem[0];

    float inv_C = 1.0f / (float)C;
    for (int i = tid; i < C; i += nthreads) {
        float g = gamma ? gamma[i] : 1.0f;
        float dy = dout_row[i];
        float x_hat = (x_row[i] - mean) * rstd;
        float val = rstd * (dy * g - inv_C * s1 - inv_C * x_hat * s2 * rstd);
        if (res_row != nullptr) {
            val += res_row[i];
        }
        if (accumulate) {
            dx_row[i] += val;
        } else {
            dx_row[i] = val;
        }
    }
}

// LayerNorm Backward Kernel for dgamma and dbeta (2D Grid across rows to saturate GPU SMs)
__global__ void layernorm_backward_params_kernel(
    const float* __restrict__ dout,
    const float* __restrict__ x,
    const float* __restrict__ mean_cache,
    const float* __restrict__ rstd_cache,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int N, int C
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk_id = blockIdx.y;
    int num_chunks = gridDim.y;
    if (col >= C) return;

    int rows_per_chunk = (N + num_chunks - 1) / num_chunks;
    int start_row = chunk_id * rows_per_chunk;
    int end_row = min(N, start_row + rows_per_chunk);

    float dg = 0.0f;
    float db = 0.0f;

    for (int row = start_row; row < end_row; ++row) {
        float dy = dout[row * C + col];
        float mean = mean_cache[row];
        float rstd = rstd_cache[row];
        float x_hat = (x[row * C + col] - mean) * rstd;
        dg += dy * x_hat;
        db += dy;
    }

    if (num_chunks == 1) {
        if (dgamma) dgamma[col] += dg;
        if (dbeta)  dbeta[col]  += db;
    } else {
        if (dgamma) atomicAdd(&dgamma[col], dg);
        if (dbeta)  atomicAdd(&dbeta[col], db);
    }
}

void layernorm_forward(
    const float* x,
    const float* gamma,
    const float* beta,
    float* out,
    float* mean,
    float* rstd,
    int N, int C,
    float eps,
    cudaStream_t stream
) {
    int threads = (C >= 1024) ? 1024 : ((C >= 512) ? 512 : 256);
    size_t shared_size = (threads / WARP_SIZE) * sizeof(float);

    if (C % 4 == 0) {
        layernorm_forward_vec4_kernel<<<N, threads, shared_size, stream>>>(
            x, gamma, beta, out, mean, rstd, N, C, eps
        );
    } else {
        layernorm_forward_scalar_kernel<<<N, threads, shared_size, stream>>>(
            x, gamma, beta, out, mean, rstd, N, C, eps
        );
    }
}

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
    const float* residual_add,
    bool accumulate,
    cudaStream_t stream
) {
    int threads = (C >= 1024) ? 1024 : ((C >= 512) ? 512 : 256);
    size_t shared_size = (threads / WARP_SIZE) * sizeof(float);

    layernorm_backward_dx_kernel<<<N, threads, shared_size, stream>>>(
        dout, x, gamma, mean, rstd, dx, N, C, residual_add, accumulate
    );

    int block_dim = 256;
    dim3 grid_dim((C + block_dim - 1) / block_dim, 16);
    layernorm_backward_params_kernel<<<grid_dim, block_dim, 0, stream>>>(
        dout, x, mean, rstd, dgamma, dbeta, N, C
    );
}
