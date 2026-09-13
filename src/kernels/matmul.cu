#include "kernels/matmul.cuh"
#include "common.h"

void matmul_forward(
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    bool transA,
    bool transB,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    int lda = transA ? M : K;
    int ldb = transB ? K : N;
    int ldc = N;

    // In cuBLAS column-major: C^T = B^T * A^T
    CUBLAS_CHECK(cublasSgemm(
        handle,
        opB, opA,
        N, M, K,
        &alpha,
        B, ldb,
        A, lda,
        &beta,
        C, ldc
    ));
}

void matmul_batched_strided(
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    long long int strideA,
    long long int strideB,
    long long int strideC,
    int batch_count,
    bool transA,
    bool transB,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    int lda = transA ? M : K;
    int ldb = transB ? K : N;
    int ldc = N;

    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle,
        opB, opA,
        N, M, K,
        &alpha,
        B, ldb, strideB,
        A, lda, strideA,
        &beta,
        C, ldc, strideC,
        batch_count
    ));
}

__global__ void add_bias_kernel(
    float* __restrict__ Y,
    const float* __restrict__ bias,
    int M, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;

    int col = idx % N;
    Y[idx] += bias[col];
}

void add_bias(
    float* Y,
    const float* bias,
    int M, int N,
    cudaStream_t stream
) {
    int total = M * N;
    int block_dim = 256;
    int grid_dim = (total + block_dim - 1) / block_dim;
    add_bias_kernel<<<grid_dim, block_dim, 0, stream>>>(Y, bias, M, N);
}

__global__ void bias_backward_2d_kernel(
    const float* __restrict__ dY,
    float* __restrict__ dbias,
    int M, int N
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row_start = blockIdx.y * blockDim.y + threadIdx.y;
    int row_stride = gridDim.y * blockDim.y;

    if (col >= N) return;

    float sum = 0.0f;
    for (int row = row_start; row < M; row += row_stride) {
        sum += dY[row * N + col];
    }

    atomicAdd(&dbias[col], sum);
}

void bias_backward(
    const float* dY,
    float* dbias,
    int M, int N,
    cudaStream_t stream
) {
    dim3 block_dim(32, 8); // 256 threads per block, 32 coalesced columns per warp
    int num_row_blocks = (M >= 1024) ? 32 : ((M >= 256) ? 16 : 1);
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x, num_row_blocks);

    bias_backward_2d_kernel<<<grid_dim, block_dim, 0, stream>>>(dY, dbias, M, N);
}
