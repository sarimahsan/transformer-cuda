#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

// Row-Major Matrix Multiplication using cuBLAS:
// C = alpha * (op_A(A) * op_B(B)) + beta * C
// A, B, C are assumed to be row-major in memory!
void matmul_forward(
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    bool transA = false,
    bool transB = false,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = 0
);

// Batched Strided GEMM for Multi-Head Attention:
// C[i] = alpha * (op_A(A[i]) * op_B(B[i])) + beta * C[i]
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
    bool transA = false,
    bool transB = false,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = 0
);

// Fused Bias Addition: Y = Y + bias (broadcasted over M rows)
// Y: (M, N), bias: (N,)
void add_bias(
    float* Y,
    const float* bias,
    int M, int N,
    cudaStream_t stream = 0
);

// Bias Backward: dbias = sum(dY, axis=0)
// dY: (M, N), dbias: (N,)
void bias_backward(
    const float* dY,
    float* dbias,
    int M, int N,
    cudaStream_t stream = 0
);
