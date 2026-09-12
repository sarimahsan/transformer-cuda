#pragma once

#include <cuda_runtime.h>

// Embedding Forward:
// out[b, t, c] = tok_emb[tokens[b, t], c] + pos_emb[t, c]
// Shapes:
//   tokens: (B, T)
//   tok_emb: (V, C)
//   pos_emb: (T, C)
//   out: (B, T, C)
void embedding_forward(
    const int* tokens,
    const float* tok_emb,
    const float* pos_emb,
    float* out,
    int B, int T, int C,
    cudaStream_t stream = 0
);

// Embedding Backward:
// Accumulates gradients into d_tok_emb and d_pos_emb using atomicAdd
void embedding_backward(
    const float* dout,
    const int* tokens,
    float* d_tok_emb,
    float* d_pos_emb,
    int B, int T, int C,
    cudaStream_t stream = 0
);
