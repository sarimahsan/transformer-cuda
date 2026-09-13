#include "optimizer.h"
#include "common.h"
#include <cmath>
#include <vector>

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum_opt(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void adamw_step_kernel(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float* __restrict__ m,
    float* __restrict__ v,
    size_t num_params,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_params) return;

    float p = params[idx];
    float g = grads[idx];

    // Update biased first moment estimate
    float m_t = beta1 * m[idx] + (1.0f - beta1) * g;
    m[idx] = m_t;

    // Update biased second raw moment estimate
    float v_t = beta2 * v[idx] + (1.0f - beta2) * g * g;
    v[idx] = v_t;

    // Compute bias-corrected moments
    float m_hat = m_t / bias_correction1;
    float v_hat = v_t / bias_correction2;

    // AdamW decoupled weight decay update
    p = p - lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * p);
    params[idx] = p;
}

__global__ void sum_sq_kernel(
    const float* __restrict__ grads,
    float* __restrict__ block_sums,
    size_t num_params
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < num_params) ? grads[idx] : 0.0f;
    float sq = val * val;

    sq = warp_reduce_sum_opt(sq);

    extern __shared__ float s_opt[];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    if (lane == 0) s_opt[wid] = sq;
    __syncthreads();

    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    sq = (threadIdx.x < num_warps) ? s_opt[lane] : 0.0f;
    if (wid == 0) sq = warp_reduce_sum_opt(sq);
    if (threadIdx.x == 0) block_sums[blockIdx.x] = sq;
}

__global__ void reduce_total_norm_kernel(
    const float* __restrict__ block_sums,
    int num_blocks,
    float* __restrict__ d_total_norm
) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        sum += block_sums[i];
    }
    sum = warp_reduce_sum_opt(sum);

    extern __shared__ float s_red[];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    if (lane == 0) s_red[wid] = sum;
    __syncthreads();

    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    sum = (threadIdx.x < num_warps) ? s_red[lane] : 0.0f;
    if (wid == 0) sum = warp_reduce_sum_opt(sum);
    if (threadIdx.x == 0) {
        *d_total_norm = sqrtf(sum);
    }
}

__global__ void scale_grads_device_kernel(
    float* __restrict__ grads,
    const float* __restrict__ d_total_norm,
    float max_norm,
    size_t num_params
) {
    float norm = *d_total_norm;
    if (norm <= max_norm || norm <= 1e-6f) return;
    float scale = max_norm / norm;

    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_params) {
        grads[idx] *= scale;
    }
}

__global__ void fused_clip_adamw_zero_kernel(
    float* __restrict__ params,
    float* __restrict__ grads,
    float* __restrict__ m,
    float* __restrict__ v,
    const float* __restrict__ d_total_norm,
    size_t num_params,
    float lr, float beta1, float beta2, float eps,
    float weight_decay, float max_norm,
    float bias_correction1, float bias_correction2
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_params) return;

    float norm = *d_total_norm;
    float g = grads[idx];

    // Clip gradient in-register
    if (norm > max_norm && norm > 1e-6f) {
        g *= max_norm / norm;
    }

    float p = params[idx];

    // AdamW update
    float m_t = beta1 * m[idx] + (1.0f - beta1) * g;
    m[idx] = m_t;
    float v_t = beta2 * v[idx] + (1.0f - beta2) * g * g;
    v[idx] = v_t;

    float m_hat = m_t / bias_correction1;
    float v_hat = v_t / bias_correction2;
    p = p - lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * p);
    params[idx] = p;

    // Zero gradient
    grads[idx] = 0.0f;
}

AdamW::AdamW(float* params, float* grads, size_t num_params, const TransformerConfig& config)
    : d_params(params),
      d_grads(grads),
      num_params(num_params),
      step_count(0),
      beta1(config.beta1),
      beta2(config.beta2),
      eps(config.adam_eps),
      weight_decay(config.weight_decay)
{
    CUDA_CHECK(cudaMalloc(&d_m, num_params * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, num_params * sizeof(float)));
    reset();

    int block_dim = 256;
    int grid_dim = (num_params + block_dim - 1) / block_dim;
    CUDA_CHECK(cudaMalloc(&d_norm_buffer, (grid_dim + 1) * sizeof(float)));
}

AdamW::~AdamW() {
    cudaFree(d_m);
    cudaFree(d_v);
    cudaFree(d_norm_buffer);
}

void AdamW::reset() {
    CUDA_CHECK(cudaMemset(d_m, 0, num_params * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v, 0, num_params * sizeof(float)));
    step_count = 0;
}

void AdamW::zero_grad(cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(d_grads, 0, num_params * sizeof(float), stream));
}

float AdamW::clip_grad_norm(float max_norm, cudaStream_t stream) {
    NVTX_PUSH("Clip_Grad_Norm");
    int block_dim = 256;
    int grid_dim = (num_params + block_dim - 1) / block_dim;
    size_t shared_size = (block_dim / WARP_SIZE) * sizeof(float);

    float* d_block_sums = d_norm_buffer;
    float* d_total_norm = d_norm_buffer + grid_dim;

    sum_sq_kernel<<<grid_dim, block_dim, shared_size, stream>>>(d_grads, d_block_sums, num_params);
    reduce_total_norm_kernel<<<1, 256, (256 / WARP_SIZE) * sizeof(float), stream>>>(d_block_sums, grid_dim, d_total_norm);
    scale_grads_device_kernel<<<grid_dim, block_dim, 0, stream>>>(d_grads, d_total_norm, max_norm, num_params);

    NVTX_POP();
    return 0.0f;
}

void AdamW::step(float lr, cudaStream_t stream) {
    NVTX_PUSH("AdamW_Step");
    step_count++;
    float bias_correction1 = 1.0f - powf(beta1, (float)step_count);
    float bias_correction2 = 1.0f - powf(beta2, (float)step_count);

    int block_dim = 256;
    int grid_dim = (num_params + block_dim - 1) / block_dim;

    adamw_step_kernel<<<grid_dim, block_dim, 0, stream>>>(
        d_params, d_grads, d_m, d_v, num_params,
        lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2
    );
    NVTX_POP();
}

void AdamW::fused_step(float lr, float max_norm, cudaStream_t stream) {
    NVTX_PUSH("Fused_Clip_AdamW_Zero");
    step_count++;
    float bias_correction1 = 1.0f - powf(beta1, (float)step_count);
    float bias_correction2 = 1.0f - powf(beta2, (float)step_count);

    int block_dim = 256;
    int grid_dim = (num_params + block_dim - 1) / block_dim;
    size_t shared_size = (block_dim / WARP_SIZE) * sizeof(float);

    float* d_block_sums = d_norm_buffer;
    float* d_total_norm = d_norm_buffer + grid_dim;

    // Stage 1: Per-block squared gradient norm reduction
    sum_sq_kernel<<<grid_dim, block_dim, shared_size, stream>>>(d_grads, d_block_sums, num_params);
    // Stage 2: Final reduction to scalar norm
    reduce_total_norm_kernel<<<1, 256, (256 / WARP_SIZE) * sizeof(float), stream>>>(d_block_sums, grid_dim, d_total_norm);
    // Stage 3: Fused clip + AdamW update + zero grad (single pass over gradient buffer)
    fused_clip_adamw_zero_kernel<<<grid_dim, block_dim, 0, stream>>>(
        d_params, d_grads, d_m, d_v, d_total_norm, num_params,
        lr, beta1, beta2, eps, weight_decay, max_norm,
        bias_correction1, bias_correction2
    );
    NVTX_POP();
}

