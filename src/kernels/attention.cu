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

    // Write final normalized outputs and optional softmax stats for backward pass
    if (i < T) {
        float inv_l = (l_i > 0.0f) ? (1.0f / (l_i + 1e-12f)) : 0.0f;
        #pragma unroll
        for (int d = 0; d < Dh; ++d) {
            out_head[i * Dh + d] = o_reg[d] * inv_l;
        }
        if (m_out != nullptr) m_out[bh * T + i] = m_i;
        if (l_out != nullptr) l_out[bh * T + i] = l_i;
    }
}

void tiled_causal_attention_forward(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int B, int H, int T, int d_head,
    float scale,
    float* m_out,
    float* l_out,
    cudaStream_t stream
) {
    const int Br = 64; // Query tile size
    const int Bc = 32; // Key/Value tile size
    dim3 grid((T + Br - 1) / Br, B * H);
    dim3 block(Br);

    size_t shared_mem_bytes = 2 * Bc * d_head * sizeof(float);
    if (d_head == 32) {
        tiled_causal_attention_fwd_kernel<Bc, 32><<<grid, block, shared_mem_bytes, stream>>>(
            q, k, v, out, B, H, T, scale, Br, m_out, l_out
        );
    } else if (d_head == 64) {
        tiled_causal_attention_fwd_kernel<Bc, 64><<<grid, block, shared_mem_bytes, stream>>>(
            q, k, v, out, B, H, T, scale, Br, m_out, l_out
        );
    } else {
        tiled_causal_attention_fwd_kernel<Bc, 128><<<grid, block, shared_mem_bytes, stream>>>(
            q, k, v, out, B, H, T, scale, Br, m_out, l_out
        );
    }
}

// Precompute D_i = sum_d (dO_{i,d} * O_{i,d})
__global__ void attention_precompute_dot_do_o_kernel(
    const float* __restrict__ do_ptr,
    const float* __restrict__ o_ptr,
    float* __restrict__ d_out,
    int total_rows,
    int d_head
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows) return;

    const float* do_row = do_ptr + row * d_head;
    const float* o_row = o_ptr + row * d_head;

    float sum = 0.0f;
    for (int d = 0; d < d_head; ++d) {
        sum += do_row[d] * o_row[d];
    }
    d_out[row] = sum;
}

void attention_precompute_dot_do_o(
    const float* dO,
    const float* O,
    float* D,
    int B, int H, int T, int d_head,
    cudaStream_t stream
) {
    int total_rows = B * H * T;
    int block_dim = 256;
    int grid_dim = (total_rows + block_dim - 1) / block_dim;
    attention_precompute_dot_do_o_kernel<<<grid_dim, block_dim, 0, stream>>>(
        dO, O, D, total_rows, d_head
    );
}

// FlashAttention Backward Kernel 1: Query-Parallel dQ accumulation in SRAM/registers
// Br: Query tile size, Bc: KV tile size, Dh: Head dim
template <int Bc, int Dh>
__global__ void tiled_causal_attention_bwd_dq_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ do_ptr,
    const float* __restrict__ m_ptr,
    const float* __restrict__ l_ptr,
    const float* __restrict__ d_ptr,
    float* __restrict__ dq,
    int B, int H, int T,
    float scale,
    int Br
) {
    int bh = blockIdx.y;
    int query_tile_idx = blockIdx.x;
    int i = query_tile_idx * Br + threadIdx.x;

    int head_stride = T * Dh;
    const float* q_head = q + bh * head_stride;
    const float* k_head = k + bh * head_stride;
    const float* v_head = v + bh * head_stride;
    const float* do_head = do_ptr + bh * head_stride;
    float* dq_head = dq + bh * head_stride;

    const float* m_head = m_ptr + bh * T;
    const float* l_head = l_ptr + bh * T;
    const float* d_head_ptr = d_ptr + bh * T;

    extern __shared__ float smem[];
    float* s_k = smem;             // size: Bc * Dh
    float* s_v = smem + Bc * Dh;   // size: Bc * Dh

    float q_reg[Dh];
    float do_reg[Dh];
    float dq_reg[Dh];
    float m_i = -FLT_MAX;
    float l_i = 1.0f;
    float d_i = 0.0f;

    if (i < T) {
        #pragma unroll
        for (int d = 0; d < Dh; ++d) {
            q_reg[d] = q_head[i * Dh + d];
            do_reg[d] = do_head[i * Dh + d];
            dq_reg[d] = 0.0f;
        }
        m_i = m_head[i];
        l_i = l_head[i];
        d_i = d_head_ptr[i];
    }

    float inv_l = (l_i > 0.0f) ? (1.0f / (l_i + 1e-12f)) : 0.0f;
    int max_query_in_tile = min(T - 1, (query_tile_idx + 1) * Br - 1);
    int num_kv_tiles = (T + Bc - 1) / Bc;

    for (int c = 0; c < num_kv_tiles; ++c) {
        if (c * Bc > max_query_in_tile) break;

        // Load K and V tile into shared memory
        int total_kv = Bc * Dh;
        for (int idx = threadIdx.x; idx < total_kv; idx += blockDim.x) {
            int k_t = idx / Dh;
            int k_d = idx % Dh;
            int global_k = c * Bc + k_t;
            if (global_k < T) {
                s_k[idx] = k_head[global_k * Dh + k_d];
                s_v[idx] = v_head[global_k * Dh + k_d];
            } else {
                s_k[idx] = 0.0f;
                s_v[idx] = 0.0f;
            }
        }
        __syncthreads();

        if (i < T) {
            int start_j = c * Bc;
            int max_j = min(i, (c + 1) * Bc - 1);

            for (int j = start_j; j <= max_j; ++j) {
                int j_rel = j - start_j;
                const float* k_vec = s_k + j_rel * Dh;
                const float* v_vec = s_v + j_rel * Dh;

                float s_ij = 0.0f;
                #pragma unroll
                for (int d = 0; d < Dh; ++d) {
                    s_ij += q_reg[d] * k_vec[d];
                }
                s_ij *= scale;

                float p_ij = expf(s_ij - m_i) * inv_l;

                float dp_ij = 0.0f;
                #pragma unroll
                for (int d = 0; d < Dh; ++d) {
                    dp_ij += do_reg[d] * v_vec[d];
                }

                float ds_ij = scale * p_ij * (dp_ij - d_i);

                #pragma unroll
                for (int d = 0; d < Dh; ++d) {
                    dq_reg[d] += ds_ij * k_vec[d];
                }
            }
        }
        __syncthreads();
    }

    if (i < T) {
        #pragma unroll
        for (int d = 0; d < Dh; ++d) {
            dq_head[i * Dh + d] = dq_reg[d];
        }
    }
}

// FlashAttention Backward Kernel 2: Key/Value-Parallel dK and dV accumulation in SRAM/registers
// Br: Query tile size in smem, Bc: KV tile size, Dh: Head dim
template <int Br, int Dh>
__global__ void tiled_causal_attention_bwd_dkv_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ do_ptr,
    const float* __restrict__ m_ptr,
    const float* __restrict__ l_ptr,
    const float* __restrict__ d_ptr,
    float* __restrict__ dk,
    float* __restrict__ dv,
    int B, int H, int T,
    float scale,
    int Bc
) {
    int bh = blockIdx.y;
    int kv_tile_idx = blockIdx.x;
    int j = kv_tile_idx * Bc + threadIdx.x;

    int head_stride = T * Dh;
    const float* q_head = q + bh * head_stride;
    const float* k_head = k + bh * head_stride;
    const float* v_head = v + bh * head_stride;
    const float* do_head = do_ptr + bh * head_stride;
    float* dk_head = dk + bh * head_stride;
    float* dv_head = dv + bh * head_stride;

    const float* m_head = m_ptr + bh * T;
    const float* l_head = l_ptr + bh * T;
    const float* d_head_ptr = d_ptr + bh * T;

    extern __shared__ float smem[];
    float* s_q = smem;              // size: Br * Dh
    float* s_do = smem + Br * Dh;   // size: Br * Dh
    float* s_m = smem + 2 * Br * Dh; // size: Br
    float* s_l = s_m + Br;          // size: Br
    float* s_d = s_l + Br;          // size: Br

    float k_reg[Dh];
    float v_reg[Dh];
    float dk_reg[Dh];
    float dv_reg[Dh];

    if (j < T) {
        #pragma unroll
        for (int d = 0; d < Dh; ++d) {
            k_reg[d] = k_head[j * Dh + d];
            v_reg[d] = v_head[j * Dh + d];
            dk_reg[d] = 0.0f;
            dv_reg[d] = 0.0f;
        }
    }

    int min_key_in_tile = kv_tile_idx * Bc;
    int num_query_tiles = (T + Br - 1) / Br;

    for (int r = 0; r < num_query_tiles; ++r) {
        // Causal skip: if the entire query tile is before this KV block's keys
        if ((r + 1) * Br - 1 < min_key_in_tile) continue;

        // Load query tile and statistics into shared memory
        int total_q = Br * Dh;
        for (int idx = threadIdx.x; idx < total_q; idx += blockDim.x) {
            int q_t = idx / Dh;
            int q_d = idx % Dh;
            int global_q = r * Br + q_t;
            if (global_q < T) {
                s_q[idx] = q_head[global_q * Dh + q_d];
                s_do[idx] = do_head[global_q * Dh + q_d];
            } else {
                s_q[idx] = 0.0f;
                s_do[idx] = 0.0f;
            }
        }
        for (int idx = threadIdx.x; idx < Br; idx += blockDim.x) {
            int global_q = r * Br + idx;
            if (global_q < T) {
                s_m[idx] = m_head[global_q];
                s_l[idx] = l_head[global_q];
                s_d[idx] = d_head_ptr[global_q];
            } else {
                s_m[idx] = -FLT_MAX;
                s_l[idx] = 1.0f;
                s_d[idx] = 0.0f;
            }
        }
        __syncthreads();

        if (j < T) {
            int start_i = max(j, r * Br);
            int end_i = min(T - 1, (r + 1) * Br - 1);

            for (int i = start_i; i <= end_i; ++i) {
                int i_rel = i - r * Br;
                const float* q_vec = s_q + i_rel * Dh;
                const float* do_vec = s_do + i_rel * Dh;
                float m_i = s_m[i_rel];
                float l_i = s_l[i_rel];
                float d_i = s_d[i_rel];
                float inv_l = (l_i > 0.0f) ? (1.0f / (l_i + 1e-12f)) : 0.0f;

                float s_ij = 0.0f;
                #pragma unroll
                for (int d = 0; d < Dh; ++d) {
                    s_ij += q_vec[d] * k_reg[d];
                }
                s_ij *= scale;

                float p_ij = expf(s_ij - m_i) * inv_l;

                #pragma unroll
                for (int d = 0; d < Dh; ++d) {
                    dv_reg[d] += p_ij * do_vec[d];
                }

                float dp_ij = 0.0f;
                #pragma unroll
                for (int d = 0; d < Dh; ++d) {
                    dp_ij += do_vec[d] * v_reg[d];
                }

                float ds_ij = scale * p_ij * (dp_ij - d_i);

                #pragma unroll
                for (int d = 0; d < Dh; ++d) {
                    dk_reg[d] += ds_ij * q_vec[d];
                }
            }
        }
        __syncthreads();
    }

    if (j < T) {
        #pragma unroll
        for (int d = 0; d < Dh; ++d) {
            dk_head[j * Dh + d] = dk_reg[d];
            dv_head[j * Dh + d] = dv_reg[d];
        }
    }
}

void tiled_causal_attention_backward(
    const float* q,
    const float* k,
    const float* v,
    const float* o,
    const float* do_ptr,
    const float* m_ptr,
    const float* l_ptr,
    float* dq,
    float* dk,
    float* dv,
    float* d_workspace,
    int B, int H, int T, int d_head,
    float scale,
    cudaStream_t stream
) {
    // 1. Precompute D_i = sum_d (dO_{i,d} * O_{i,d})
    attention_precompute_dot_do_o(do_ptr, o, d_workspace, B, H, T, d_head, stream);

    const int Br = 64; // Tile size
    const int Bc = 32;

    if (d_head == 32) {
        // 2. Launch dQ query-parallel kernel
        dim3 grid_dq((T + Br - 1) / Br, B * H);
        dim3 block_dq(Br);
        size_t smem_dq = 2 * Bc * 32 * sizeof(float);
        tiled_causal_attention_bwd_dq_kernel<Bc, 32><<<grid_dq, block_dq, smem_dq, stream>>>(
            q, k, v, do_ptr, m_ptr, l_ptr, d_workspace, dq, B, H, T, scale, Br
        );

        // 3. Launch dK & dV key/value-parallel kernel
        dim3 grid_dkv((T + Bc - 1) / Bc, B * H);
        dim3 block_dkv(Bc);
        size_t smem_dkv = (2 * Br * 32 + 3 * Br) * sizeof(float);
        tiled_causal_attention_bwd_dkv_kernel<Br, 32><<<grid_dkv, block_dkv, smem_dkv, stream>>>(
            q, k, v, do_ptr, m_ptr, l_ptr, d_workspace, dk, dv, B, H, T, scale, Bc
        );
    } else if (d_head == 64) {
        dim3 grid_dq((T + Br - 1) / Br, B * H);
        dim3 block_dq(Br);
        size_t smem_dq = 2 * Bc * 64 * sizeof(float);
        tiled_causal_attention_bwd_dq_kernel<Bc, 64><<<grid_dq, block_dq, smem_dq, stream>>>(
            q, k, v, do_ptr, m_ptr, l_ptr, d_workspace, dq, B, H, T, scale, Br
        );

        dim3 grid_dkv((T + Bc - 1) / Bc, B * H);
        dim3 block_dkv(Bc);
        size_t smem_dkv = (2 * Br * 64 + 3 * Br) * sizeof(float);
        tiled_causal_attention_bwd_dkv_kernel<Br, 64><<<grid_dkv, block_dkv, smem_dkv, stream>>>(
            q, k, v, do_ptr, m_ptr, l_ptr, d_workspace, dk, dv, B, H, T, scale, Bc
        );
    } else {
        // Fallback or general Dh
    }
}
