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

__global__ void bias_backward_kernel(
    const float* __restrict__ dY,
    float* __restrict__ dbias,
    int M, int N
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;

    float sum = 0.0f;
    for (int row = 0; row < M; ++row) {
        sum += dY[row * N + col];
    }
    dbias[col] += sum;
}

void bias_backward(
    const float* dY,
    float* dbias,
    int M, int N,
    cudaStream_t stream
) {
    int block_dim = 256;
    int grid_dim = (N + block_dim - 1) / block_dim;
    bias_backward_kernel<<<grid_dim, block_dim, 0, stream>>>(dY, dbias, M, N);
}
