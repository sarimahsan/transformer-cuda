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
