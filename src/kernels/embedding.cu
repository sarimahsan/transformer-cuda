#include "kernels/embedding.cuh"
#include "common.h"

__global__ void embedding_forward_kernel(
    const int* __restrict__ tokens,
    const float* __restrict__ tok_emb,
    const float* __restrict__ pos_emb,
    float* __restrict__ out,
    int B, int T, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * C;
    if (idx >= total) return;

    int c = idx % C;
    int t = (idx / C) % T;
    int b = idx / (C * T);

    int tok_id = tokens[b * T + t];
    float val = tok_emb[tok_id * C + c] + pos_emb[t * C + c];
    out[idx] = val;
}

__global__ void embedding_backward_kernel(
    const float* __restrict__ dout,
    const int* __restrict__ tokens,
    float* __restrict__ d_tok_emb,
    float* __restrict__ d_pos_emb,
    int B, int T, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * C;
    if (idx >= total) return;

    int c = idx % C;
    int t = (idx / C) % T;
    int b = idx / (C * T);

    int tok_id = tokens[b * T + t];
    float dy = dout[idx];

    atomicAdd(&d_tok_emb[tok_id * C + c], dy);
    atomicAdd(&d_pos_emb[t * C + c], dy);
}

void embedding_forward(
    const int* tokens,
    const float* tok_emb,
    const float* pos_emb,
    float* out,
    int B, int T, int C,
    cudaStream_t stream
) {
    int total = B * T * C;
    int block_dim = 256;
    int grid_dim = (total + block_dim - 1) / block_dim;
    embedding_forward_kernel<<<grid_dim, block_dim, 0, stream>>>(
        tokens, tok_emb, pos_emb, out, B, T, C
    );
}

void embedding_backward(
    const float* dout,
    const int* tokens,
    float* d_tok_emb,
    float* d_pos_emb,
    int B, int T, int C,
    cudaStream_t stream
) {
    int total = B * T * C;
    int block_dim = 256;
    int grid_dim = (total + block_dim - 1) / block_dim;
    embedding_backward_kernel<<<grid_dim, block_dim, 0, stream>>>(
        dout, tokens, d_tok_emb, d_pos_emb, B, T, C
    );
}
