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

    extern __shared__ float s_data[];
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    int num_warps = (nthreads + WARP_SIZE - 1) / WARP_SIZE;

    // Single-pass register-cached path when T <= nthreads (true for standard T <= 512)
    if (T <= nthreads) {
        float s_val = (tid <= i && tid < T) ? (s_row[tid] * scale) : -FLT_MAX;
        float max_val = warp_reduce_max(s_val);
        if (lane == 0) s_data[wid] = max_val;
        __syncthreads();

        max_val = (tid < num_warps) ? s_data[lane] : -FLT_MAX;
        if (wid == 0) max_val = warp_reduce_max(max_val);
        if (tid == 0) s_data[0] = max_val;
        __syncthreads();
        max_val = s_data[0];

        float e = (tid <= i && tid < T) ? expf(s_val - max_val) : 0.0f;
        float sum_exp = warp_reduce_sum(e);
        if (lane == 0) s_data[wid] = sum_exp;
        __syncthreads();

        sum_exp = (tid < num_warps) ? s_data[lane] : 0.0f;
        if (wid == 0) sum_exp = warp_reduce_sum(sum_exp);
        if (tid == 0) s_data[0] = sum_exp;
        __syncthreads();
        sum_exp = s_data[0];

        float inv_sum = 1.0f / (sum_exp + 1e-12f);
        if (tid < T) {
            p_row[tid] = (tid <= i) ? (e * inv_sum) : 0.0f;
        }
        return;
    }

    // General fallback for T > nthreads
    // 1. Find max for numerical stability over j <= i
    float max_val = -FLT_MAX;
    for (int j = tid; j <= i; j += nthreads) {
        max_val = fmaxf(max_val, s_row[j] * scale);
    }
    max_val = warp_reduce_max(max_val);
    if (lane == 0) s_data[wid] = max_val;
    __syncthreads();

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

    extern __shared__ float s_data[];
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    int num_warps = (nthreads + WARP_SIZE - 1) / WARP_SIZE;

    // Single-pass register-cached path when T <= nthreads (no re-reading DRAM!)
    if (T <= nthreads) {
        float p_val = (tid <= i && tid < T) ? p_row[tid] : 0.0f;
        float dp_val = (tid <= i && tid < T) ? dp_row[tid] : 0.0f;
        float dot = p_val * dp_val;
        dot = warp_reduce_sum(dot);
        if (lane == 0) s_data[wid] = dot;
        __syncthreads();

        dot = (tid < num_warps) ? s_data[lane] : 0.0f;
        if (wid == 0) dot = warp_reduce_sum(dot);
        if (tid == 0) s_data[0] = dot;
        __syncthreads();
        dot = s_data[0];

        if (tid < T) {
            ds_row[tid] = (tid <= i) ? (scale * p_val * (dp_val - dot)) : 0.0f;
        }
        return;
    }

    // General fallback for T > nthreads
    float dot = 0.0f;
    for (int j = tid; j <= i; j += nthreads) {
        dot += dp_row[j] * p_row[j];
    }
    dot = warp_reduce_sum(dot);
    if (lane == 0) s_data[wid] = dot;
    __syncthreads();

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

// FlashAttention-Style Tiled Causal Attention Forward Kernel
// Br: Query tile size (blockDim.x), Bc: Key/Value tile size (shared memory), Dh: Head dimension compile-time constant
template <int Bc, int Dh>
__global__ void tiled_causal_attention_fwd_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int B, int H, int T,
    float scale,
    int Br
) {
    int bh = blockIdx.y; // batch and head index: [0, B * H - 1]
    int query_tile_idx = blockIdx.x;
    int i = query_tile_idx * Br + threadIdx.x; // query sequence position

    int head_stride = T * Dh;
    const float* q_head = q + bh * head_stride;
    const float* k_head = k + bh * head_stride;
    const float* v_head = v + bh * head_stride;
    float* out_head = out + bh * head_stride;

    extern __shared__ float smem[];
    float* s_k = smem;             // size: Bc * Dh
    float* s_v = smem + Bc * Dh;   // size: Bc * Dh

    // Thread-local hardware registers (fully unrolled, zero local memory spill)
    float q_reg[Dh];
    float o_reg[Dh];
    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    if (i < T) {
        #pragma unroll
        for (int d = 0; d < Dh; ++d) {
            q_reg[d] = q_head[i * Dh + d];
            o_reg[d] = 0.0f;
        }
    }

    int max_query_in_tile = min(T - 1, (query_tile_idx + 1) * Br - 1);
    int num_kv_tiles = (T + Bc - 1) / Bc;

    for (int c = 0; c < num_kv_tiles; ++c) {
        // Causal skip: if this entire KV tile starts after the maximum query in this block
        if (c * Bc > max_query_in_tile) break;

        // Cooperatively load K and V tile into Shared Memory
        int total_kv_elements = Bc * Dh;
        for (int idx = threadIdx.x; idx < total_kv_elements; idx += blockDim.x) {
            int k_t = idx / Dh;
            int k_d = idx % Dh;
            int global_k_pos = c * Bc + k_t;
            if (global_k_pos < T) {
                s_k[idx] = k_head[global_k_pos * Dh + k_d];
                s_v[idx] = v_head[global_k_pos * Dh + k_d];
            } else {
                s_k[idx] = 0.0f;
                s_v[idx] = 0.0f;
            }
        }
        __syncthreads();

        // Compute dot-product attention and online softmax in registers
        if (i < T) {
            int max_j = min(i, (c + 1) * Bc - 1);
            int start_j = c * Bc;

            for (int j = start_j; j <= max_j; ++j) {
                int j_rel = j - start_j;
                const float* k_vec = s_k + j_rel * Dh;
                const float* v_vec = s_v + j_rel * Dh;

                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < Dh; ++d) {
                    score += q_reg[d] * k_vec[d];
                }
                score *= scale;

                // Online Softmax update
                float m_new = fmaxf(m_i, score);
                float alpha = expf(m_i - m_new);
                float beta = expf(score - m_new);
                l_i = l_i * alpha + beta;

                #pragma unroll
                for (int d = 0; d < Dh; ++d) {
                    o_reg[d] = o_reg[d] * alpha + beta * v_vec[d];
                }
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    // Write final normalized outputs to global memory
    if (i < T) {
        float inv_l = (l_i > 0.0f) ? (1.0f / (l_i + 1e-12f)) : 0.0f;
        #pragma unroll
        for (int d = 0; d < Dh; ++d) {
            out_head[i * Dh + d] = o_reg[d] * inv_l;
        }
    }
}

void tiled_causal_attention_forward(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int B, int H, int T, int d_head,
    float scale,
    cudaStream_t stream
) {
    const int Br = 64; // Query tile size
    const int Bc = 32; // Key/Value tile size
    dim3 grid((T + Br - 1) / Br, B * H);
    dim3 block(Br);

    size_t shared_mem_bytes = 2 * Bc * d_head * sizeof(float);
    if (d_head == 32) {
        tiled_causal_attention_fwd_kernel<Bc, 32><<<grid, block, shared_mem_bytes, stream>>>(
            q, k, v, out, B, H, T, scale, Br
        );
    } else if (d_head == 64) {
        tiled_causal_attention_fwd_kernel<Bc, 64><<<grid, block, shared_mem_bytes, stream>>>(
            q, k, v, out, B, H, T, scale, Br
        );
    } else {
        tiled_causal_attention_fwd_kernel<Bc, 128><<<grid, block, shared_mem_bytes, stream>>>(
            q, k, v, out, B, H, T, scale, Br
        );
    }
}
