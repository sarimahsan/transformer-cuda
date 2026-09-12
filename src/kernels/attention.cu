#include "kernels/attention.cuh"
#include "common.h"
#include <cfloat>

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// QKV Split & Transpose Forward: (B, T, 3 * H * d_head) -> 3 x (B, H, T, d_head)
__global__ void qkv_split_transpose_fwd_kernel(
    const float* __restrict__ qkv,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    int B, int T, int H, int d_head
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = B * T * H * d_head;
    if (idx >= total) return;

    int d = idx % d_head;
    int h = (idx / d_head) % H;
    int t = (idx / (d_head * H)) % T;
    int b = idx / (d_head * H * T);

    int in_qkv_stride = 3 * H * d_head;
    int in_base = b * T * in_qkv_stride + t * in_qkv_stride;

    int q_offset = in_base + (0 * H + h) * d_head + d;
    int k_offset = in_base + (1 * H + h) * d_head + d;
    int v_offset = in_base + (2 * H + h) * d_head + d;

    int out_idx = b * (H * T * d_head) + h * (T * d_head) + t * d_head + d;

    q[out_idx] = qkv[q_offset];
    k[out_idx] = qkv[k_offset];
    v[out_idx] = qkv[v_offset];
}

// QKV Split & Transpose Backward
__global__ void qkv_split_transpose_bwd_kernel(
    const float* __restrict__ dq,
    const float* __restrict__ dk,
    const float* __restrict__ dv,
    float* __restrict__ dqkv,
    int B, int T, int H, int d_head
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = B * T * H * d_head;
    if (idx >= total) return;

    int d = idx % d_head;
    int h = (idx / d_head) % H;
    int t = (idx / (d_head * H)) % T;
    int b = idx / (d_head * H * T);

    int in_qkv_stride = 3 * H * d_head;
    int in_base = b * T * in_qkv_stride + t * in_qkv_stride;

    int q_offset = in_base + (0 * H + h) * d_head + d;
    int k_offset = in_base + (1 * H + h) * d_head + d;
    int v_offset = in_base + (2 * H + h) * d_head + d;

    int out_idx = b * (H * T * d_head) + h * (T * d_head) + t * d_head + d;

    dqkv[q_offset] = dq[out_idx];
    dqkv[k_offset] = dk[out_idx];
    dqkv[v_offset] = dv[out_idx];
}

// Fused Scaled Causal Softmax Forward:
// probs[b, h, i, j] = softmax(scores[b, h, i, j] * scale) for j <= i, else 0
__global__ void causal_softmax_fwd_kernel(
    const float* __restrict__ scores,
    float* __restrict__ probs,
    int B, int H, int T, float scale
) {
    int row_idx = blockIdx.x; // in [0, B * H * T - 1]
    int i = row_idx % T;      // query position
    const float* s_row = scores + row_idx * T;
    float* p_row = probs + row_idx * T;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // 1. Find max for numerical stability over j <= i
    float max_val = -FLT_MAX;
    for (int j = tid; j <= i; j += nthreads) {
        max_val = fmaxf(max_val, s_row[j] * scale);
    }
    max_val = warp_reduce_max(max_val);

    extern __shared__ float s_data[];
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    if (lane == 0) s_data[wid] = max_val;
    __syncthreads();

    int num_warps = (nthreads + WARP_SIZE - 1) / WARP_SIZE;
    max_val = (tid < num_warps) ? s_data[lane] : -FLT_MAX;
    if (wid == 0) max_val = warp_reduce_max(max_val);
    if (tid == 0) s_data[0] = max_val;
    __syncthreads();
    max_val = s_data[0];

    // 2. Exponentiate and sum
    float sum_exp = 0.0f;
    for (int j = tid; j < T; j += nthreads) {
        if (j <= i) {
            float e = expf(s_row[j] * scale - max_val);
            p_row[j] = e;
            sum_exp += e;
        } else {
            p_row[j] = 0.0f;
        }
    }
    sum_exp = warp_reduce_sum(sum_exp);
    if (lane == 0) s_data[wid] = sum_exp;
    __syncthreads();

    sum_exp = (tid < num_warps) ? s_data[lane] : 0.0f;
    if (wid == 0) sum_exp = warp_reduce_sum(sum_exp);
    if (tid == 0) s_data[0] = sum_exp;
    __syncthreads();
    sum_exp = s_data[0];

    // 3. Normalize
    float inv_sum = 1.0f / (sum_exp + 1e-12f);
    for (int j = tid; j <= i; j += nthreads) {
        p_row[j] *= inv_sum;
    }
}

// Fused Scaled Causal Softmax Backward:
// dscores = scale * probs * (dprobs - sum(dprobs * probs)) for j <= i, else 0
__global__ void causal_softmax_bwd_kernel(
    const float* __restrict__ dprobs,
    const float* __restrict__ probs,
    float* __restrict__ dscores,
    int B, int H, int T, float scale
) {
    int row_idx = blockIdx.x;
    int i = row_idx % T;
    const float* dp_row = dprobs + row_idx * T;
    const float* p_row = probs + row_idx * T;
    float* ds_row = dscores + row_idx * T;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // sum(dprobs * probs)
    float dot = 0.0f;
    for (int j = tid; j <= i; j += nthreads) {
        dot += dp_row[j] * p_row[j];
    }
    dot = warp_reduce_sum(dot);

    extern __shared__ float s_data[];
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    if (lane == 0) s_data[wid] = dot;
    __syncthreads();

    int num_warps = (nthreads + WARP_SIZE - 1) / WARP_SIZE;
    dot = (tid < num_warps) ? s_data[lane] : 0.0f;
    if (wid == 0) dot = warp_reduce_sum(dot);
    if (tid == 0) s_data[0] = dot;
    __syncthreads();
    dot = s_data[0];

    for (int j = tid; j < T; j += nthreads) {
        if (j <= i) {
            ds_row[j] = scale * p_row[j] * (dp_row[j] - dot);
        } else {
            ds_row[j] = 0.0f;
        }
    }
}

// Merge Heads Transpose Forward: (B, H, T, d_head) -> (B, T, H * d_head)
__global__ void head_merge_transpose_fwd_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int B, int H, int T, int d_head
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = B * H * T * d_head;
    if (idx >= total) return;

    int d = idx % d_head;
    int t = (idx / d_head) % T;
    int h = (idx / (d_head * T)) % H;
    int b = idx / (d_head * T * H);

    int out_idx = b * (T * H * d_head) + t * (H * d_head) + h * d_head + d;
    out[out_idx] = in[idx];
}

// Merge Heads Transpose Backward: (B, T, H * d_head) -> (B, H, T, d_head)
__global__ void head_merge_transpose_bwd_kernel(
    const float* __restrict__ dout,
    float* __restrict__ din,
    int B, int H, int T, int d_head
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = B * H * T * d_head;
    if (idx >= total) return;

    int d = idx % d_head;
    int t = (idx / d_head) % T;
    int h = (idx / (d_head * T)) % H;
    int b = idx / (d_head * T * H);

    int out_idx = b * (T * H * d_head) + t * (H * d_head) + h * d_head + d;
    din[idx] = dout[out_idx];
}

void qkv_split_transpose_forward(
    const float* qkv,
    float* q, float* k, float* v,
    int B, int T, int H, int d_head,
    cudaStream_t stream
) {
    int total = B * T * H * d_head;
    int block_dim = 256;
    int grid_dim = (total + block_dim - 1) / block_dim;
    qkv_split_transpose_fwd_kernel<<<grid_dim, block_dim, 0, stream>>>(
        qkv, q, k, v, B, T, H, d_head
    );
}

void qkv_split_transpose_backward(
    const float* dq, const float* dk, const float* dv,
    float* dqkv,
    int B, int T, int H, int d_head,
    cudaStream_t stream
) {
    int total = B * T * H * d_head;
    int block_dim = 256;
    int grid_dim = (total + block_dim - 1) / block_dim;
    qkv_split_transpose_bwd_kernel<<<grid_dim, block_dim, 0, stream>>>(
        dq, dk, dv, dqkv, B, T, H, d_head
    );
}

void causal_softmax_forward(
    const float* scores,
    float* probs,
    int B, int H, int T,
    float scale,
    cudaStream_t stream
) {
    int grid_dim = B * H * T;
    int block_dim = (T <= 64) ? 64 : ((T <= 128) ? 128 : ((T <= 256) ? 256 : 512));
    size_t shared_size = (block_dim / WARP_SIZE) * sizeof(float);
    causal_softmax_fwd_kernel<<<grid_dim, block_dim, shared_size, stream>>>(
        scores, probs, B, H, T, scale
    );
}

void causal_softmax_backward(
    const float* dprobs,
    const float* probs,
    float* dscores,
    int B, int H, int T,
    float scale,
    cudaStream_t stream
) {
    int grid_dim = B * H * T;
    int block_dim = (T <= 64) ? 64 : ((T <= 128) ? 128 : ((T <= 256) ? 256 : 512));
    size_t shared_size = (block_dim / WARP_SIZE) * sizeof(float);
    causal_softmax_bwd_kernel<<<grid_dim, block_dim, shared_size, stream>>>(
        dprobs, probs, dscores, B, H, T, scale
    );
}

void head_merge_transpose_forward(
    const float* in,
    float* out,
    int B, int H, int T, int d_head,
    cudaStream_t stream
) {
    int total = B * H * T * d_head;
    int block_dim = 256;
    int grid_dim = (total + block_dim - 1) / block_dim;
    head_merge_transpose_fwd_kernel<<<grid_dim, block_dim, 0, stream>>>(
        in, out, B, H, T, d_head
    );
}

void head_merge_transpose_backward(
    const float* dout,
    float* din,
    int B, int H, int T, int d_head,
    cudaStream_t stream
) {
    int total = B * H * T * d_head;
    int block_dim = 256;
    int grid_dim = (total + block_dim - 1) / block_dim;
    head_merge_transpose_bwd_kernel<<<grid_dim, block_dim, 0, stream>>>(
        dout, din, B, H, T, d_head
    );
}
