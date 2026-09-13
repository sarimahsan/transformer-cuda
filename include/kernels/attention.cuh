#pragma once

#include <cuda_runtime.h>

// Split & Transpose QKV: (B, T, 3 * H * d_head) -> 3 tensors of (B, H, T, d_head)
void qkv_split_transpose_forward(
    const float* qkv,
    float* q, float* k, float* v,
    int B, int T, int H, int d_head,
    cudaStream_t stream = 0
);

// Backward for QKV Split & Transpose
void qkv_split_transpose_backward(
    const float* dq, const float* dk, const float* dv,
    float* dqkv,
    int B, int T, int H, int d_head,
    cudaStream_t stream = 0
);

// Fused Scaled Causal Softmax Forward:
// Computes row-wise softmax with causal upper-triangular masking and scaling:
// probs[b, h, i, j] = softmax(scores[b, h, i, j] * scale) for j <= i, else 0
// Shapes: scores, probs: (B * H * T, T)
void causal_softmax_forward(
    const float* scores,
    float* probs,
    int B, int H, int T,
    float scale,
    cudaStream_t stream = 0
);

// FlashAttention-Style Tiled Causal Multi-Head Attention Forward:
// Computes Out = Softmax(scale * (Q * K^T) + Mask) * V
// Completely eliminates the O(T^2) intermediate attention score/prob matrices in DRAM
// using shared memory tiling and register-level online softmax.
// Optionally exports row-wise max (m_out) and sum_exp (l_out) for zero-copy backward.
// Shapes: Q, K, V, Out: (B, H, T, d_head), m_out, l_out: (B, H, T)
void tiled_causal_attention_forward(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int B, int H, int T, int d_head,
    float scale,
    float* m_out = nullptr,
    float* l_out = nullptr,
    cudaStream_t stream = 0
);

// Precompute D_i = sum_d (dO_{i,d} * O_{i,d}) for FlashAttention backward:
void attention_precompute_dot_do_o(
    const float* dO,
    const float* O,
    float* D,
    int B, int H, int T, int d_head,
    cudaStream_t stream = 0
);

// FlashAttention-Style Tiled Causal Multi-Head Attention Backward:
// Computes dQ, dK, dV from incoming dO, Q, K, V, O, and saved softmax stats m and l.
// Completely bypasses materialization of the (B, H, T, T) attention matrix in DRAM!
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
    cudaStream_t stream = 0
);

// Fused Scaled Causal Softmax Backward:
// dscores = scale * probs * (dprobs - sum(dprobs * probs)) for j <= i, else 0
void causal_softmax_backward(
    const float* dprobs,
    const float* probs,
    float* dscores,
    int B, int H, int T,
    float scale,
    cudaStream_t stream = 0
);

// Merge Heads Transpose Forward: (B, H, T, d_head) -> (B, T, H * d_head)
void head_merge_transpose_forward(
    const float* in,
    float* out,
    int B, int H, int T, int d_head,
    cudaStream_t stream = 0
);

// Merge Heads Transpose Backward: (B, T, H * d_head) -> (B, H, T, d_head)
void head_merge_transpose_backward(
    const float* dout,
    float* din,
    int B, int H, int T, int d_head,
    cudaStream_t stream = 0
);
