#include "kernels/loss.cuh"
#include "common.h"
#include <cfloat>
#include <vector>

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_max_loss(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum_loss(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void cross_entropy_kernel(
    const float* __restrict__ logits,
    const int* __restrict__ targets,
    float* __restrict__ dlogits,
    float* __restrict__ per_row_loss,
    int N, int V, float scale
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* l_row = logits + row * V;
    float* dl_row = dlogits + row * V;
    int target_idx = targets[row];

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // 1. Max
    float max_val = -FLT_MAX;
    for (int v = tid; v < V; v += nthreads) {
        max_val = fmaxf(max_val, l_row[v]);
    }
    max_val = warp_reduce_max_loss(max_val);

    extern __shared__ float s_mem[];
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    if (lane == 0) s_mem[wid] = max_val;
    __syncthreads();

    int num_warps = (nthreads + WARP_SIZE - 1) / WARP_SIZE;
    max_val = (tid < num_warps) ? s_mem[lane] : -FLT_MAX;
    if (wid == 0) max_val = warp_reduce_max_loss(max_val);
    if (tid == 0) s_mem[0] = max_val;
    __syncthreads();
    max_val = s_mem[0];

    // 2. Sum exp
    float sum_exp = 0.0f;
    for (int v = tid; v < V; v += nthreads) {
        sum_exp += expf(l_row[v] - max_val);
    }
    sum_exp = warp_reduce_sum_loss(sum_exp);
    if (lane == 0) s_mem[wid] = sum_exp;
    __syncthreads();

    sum_exp = (tid < num_warps) ? s_mem[lane] : 0.0f;
    if (wid == 0) sum_exp = warp_reduce_sum_loss(sum_exp);
    if (tid == 0) s_mem[0] = sum_exp;
    __syncthreads();
    sum_exp = s_mem[0];

    // 3. Loss & dlogits
    float inv_sum = 1.0f / (sum_exp + 1e-12f);
    if (tid == 0) {
        float target_logit = l_row[target_idx];
        float row_loss = logf(sum_exp) - (target_logit - max_val);
        if (per_row_loss != nullptr) {
            per_row_loss[row] = row_loss;
        }
    }

    for (int v = tid; v < V; v += nthreads) {
        float p = expf(l_row[v] - max_val) * inv_sum;
        float indicator = (v == target_idx) ? 1.0f : 0.0f;
        dl_row[v] = (p - indicator) * scale;
    }
}

void cross_entropy_forward_backward(
    const float* logits,
    const int* targets,
    float* dlogits,
    float* host_loss,
    int B, int T, int V,
    cudaStream_t stream
) {
    int N = B * T;
    float scale = 1.0f / (float)N;

    float* d_row_loss = nullptr;
    if (host_loss != nullptr) {
        CUDA_CHECK(cudaMallocAsync(&d_row_loss, N * sizeof(float), stream));
    }

    int block_dim = (V <= 64) ? 64 : ((V <= 128) ? 128 : ((V <= 256) ? 256 : 512));
    size_t shared_size = (block_dim / WARP_SIZE) * sizeof(float);

    cross_entropy_kernel<<<N, block_dim, shared_size, stream>>>(
        logits, targets, dlogits, d_row_loss, N, V, scale
    );

    if (host_loss != nullptr) {
        std::vector<float> h_row_loss(N);
        CUDA_CHECK(cudaMemcpyAsync(h_row_loss.data(), d_row_loss, N * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        float total_loss = 0.0f;
        for (int i = 0; i < N; ++i) total_loss += h_row_loss[i];
        *host_loss = total_loss / (float)N;
        CUDA_CHECK(cudaFreeAsync(d_row_loss, stream));
    }
}
