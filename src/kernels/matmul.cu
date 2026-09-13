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

void matmul_cublaslt(
    cublasLtHandle_t lt_handle,
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    cublasLtEpilogue_t epilogue,
    const float* bias,
    void* workspace,
    size_t workspace_size,
    bool transA,
    bool transB,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    int lda = transA ? M : K;
    int ldb = transB ? K : N;
    int ldc = N;

    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t layoutB = NULL;
    cublasLtMatrixLayout_t layoutA = NULL;
    cublasLtMatrixLayout_t layoutC = NULL;

    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opB, sizeof(opB));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA));

    if (epilogue != CUBLASLT_EPILOGUE_DEFAULT) {
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
        if (bias != nullptr) {
            cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
        }
    }

    int rowsB = transB ? K : N;
    int colsB = transB ? N : K;
    cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_32F, rowsB, colsB, ldb);

    int rowsA = transA ? M : K;
    int colsA = transA ? K : M;
    cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_32F, rowsA, colsA, lda);

    cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32F, N, M, ldc);

    cublasLtMatmulPreference_t preference = NULL;
    cublasLtMatmulPreferenceCreate(&preference);
    if (workspace != nullptr && workspace_size > 0) {
        cublasLtMatmulPreferenceSetAttribute(
            preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_size,
            sizeof(workspace_size)
        );
    }

    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        lt_handle,
        operationDesc,
        layoutB,
        layoutA,
        layoutC,
        layoutC,
        preference,
        1,
        &heuristicResult,
        &returnedResults
    );

    if (status == CUBLAS_STATUS_SUCCESS && returnedResults > 0) {
        cublasLtMatmul(
            lt_handle,
            operationDesc,
            &alpha,
            B, layoutB,
            A, layoutA,
            &beta,
            C, layoutC,
            C, layoutC,
            &heuristicResult.algo,
            workspace,
            workspace_size,
            stream
        );
    } else {
        cublasLtMatmul(
            lt_handle,
            operationDesc,
            &alpha,
            B, layoutB,
            A, layoutA,
            &beta,
            C, layoutC,
            C, layoutC,
            NULL,
            workspace,
            workspace_size,
            stream
        );
    }

    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (layoutC) cublasLtMatrixLayoutDestroy(layoutC);
    if (layoutA) cublasLtMatrixLayoutDestroy(layoutA);
    if (layoutB) cublasLtMatrixLayoutDestroy(layoutB);
    if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);
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
