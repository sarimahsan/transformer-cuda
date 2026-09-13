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

__global__ void scale_grads_kernel(
    float* __restrict__ grads,
    float scale,
    size_t num_params
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_params) {
        grads[idx] *= scale;
    }
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
    CUDA_CHECK(cudaMalloc(&d_norm_buffer, grid_dim * sizeof(float)));
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

    sum_sq_kernel<<<grid_dim, block_dim, shared_size, stream>>>(d_grads, d_norm_buffer, num_params);

    std::vector<float> h_sums(grid_dim);
    CUDA_CHECK(cudaMemcpyAsync(h_sums.data(), d_norm_buffer, grid_dim * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float total_sum_sq = 0.0f;
    for (int i = 0; i < grid_dim; ++i) total_sum_sq += h_sums[i];
    float norm = sqrtf(total_sum_sq);

    if (norm > max_norm && norm > 1e-6f) {
        float scale = max_norm / norm;
        scale_grads_kernel<<<grid_dim, block_dim, 0, stream>>>(d_grads, scale, num_params);
    }
    NVTX_POP();
    return norm;
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
