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
    cudaStream_t cur_stream = nullptr;
    cublasGetStream(handle, &cur_stream);
    if (cur_stream != stream) {
        CUBLAS_CHECK(cublasSetStream(handle, stream));
    }

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

#include <map>

struct CublasLtPlanKey {
    int M, N, K;
    bool transA, transB;
    int epilogue;
    bool operator<(const CublasLtPlanKey& o) const {
        if (M != o.M) return M < o.M;
        if (N != o.N) return N < o.N;
        if (K != o.K) return K < o.K;
        if (transA != o.transA) return transA < o.transA;
        if (transB != o.transB) return transB < o.transB;
        return epilogue < o.epilogue;
    }
};

struct CublasLtCachedPlan {
    cublasLtMatmulDesc_t opDesc;
    cublasLtMatrixLayout_t layA, layB, layC;
    cublasLtMatmulAlgo_t algo;
    bool hasAlgo;
};

static std::map<CublasLtPlanKey, CublasLtCachedPlan> g_cublaslt_plan_cache;

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
    CublasLtPlanKey key{M, N, K, transA, transB, (int)epilogue};
    auto it = g_cublaslt_plan_cache.find(key);

    if (it == g_cublaslt_plan_cache.end()) {
        // First call for this config: create and cache the full plan
        CublasLtCachedPlan plan;

        cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
        int lda = transA ? M : K;
        int ldb = transB ? K : N;
        int ldc = N;

        cublasLtMatmulDescCreate(&plan.opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasLtMatmulDescSetAttribute(plan.opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opB, sizeof(opB));
        cublasLtMatmulDescSetAttribute(plan.opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA));

        if (epilogue != CUBLASLT_EPILOGUE_DEFAULT) {
            cublasLtMatmulDescSetAttribute(plan.opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
        }

        int rowsB = transB ? K : N;
        int colsB = transB ? N : K;
        cublasLtMatrixLayoutCreate(&plan.layB, CUDA_R_32F, rowsB, colsB, ldb);

        int rowsA = transA ? M : K;
        int colsA = transA ? K : M;
        cublasLtMatrixLayoutCreate(&plan.layA, CUDA_R_32F, rowsA, colsA, lda);

        cublasLtMatrixLayoutCreate(&plan.layC, CUDA_R_32F, N, M, ldc);

        // Query heuristic once and cache the algorithm
        cublasLtMatmulPreference_t preference = NULL;
        cublasLtMatmulPreferenceCreate(&preference);
        if (workspace != nullptr && workspace_size > 0) {
            cublasLtMatmulPreferenceSetAttribute(
                preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &workspace_size, sizeof(workspace_size)
            );
        }

        cublasLtMatmulHeuristicResult_t heuristicResult = {};
        int returnedResults = 0;
        cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
            lt_handle, plan.opDesc, plan.layB, plan.layA, plan.layC, plan.layC,
            preference, 1, &heuristicResult, &returnedResults
        );

        plan.hasAlgo = (status == CUBLAS_STATUS_SUCCESS && returnedResults > 0);
        if (plan.hasAlgo) {
            plan.algo = heuristicResult.algo;
        }

        cublasLtMatmulPreferenceDestroy(preference);

        g_cublaslt_plan_cache[key] = plan;
        it = g_cublaslt_plan_cache.find(key);
    }

    CublasLtCachedPlan& plan = it->second;

    // Update bias pointer per-call (only for EPILOGUE_BIAS)
    if (bias != nullptr) {
        cublasLtMatmulDescSetAttribute(plan.opDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    }

    if (plan.hasAlgo) {
        cublasLtMatmul(
            lt_handle, plan.opDesc, &alpha,
            B, plan.layB, A, plan.layA, &beta,
            C, plan.layC, C, plan.layC,
            &plan.algo, workspace, workspace_size, stream
        );
    } else {
        cublasLtMatmul(
            lt_handle, plan.opDesc, &alpha,
            B, plan.layB, A, plan.layA, &beta,
            C, plan.layC, C, plan.layC,
            NULL, workspace, workspace_size, stream
        );
    }
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
    cudaStream_t cur_stream = nullptr;
    cublasGetStream(handle, &cur_stream);
    if (cur_stream != stream) {
        CUBLAS_CHECK(cublasSetStream(handle, stream));
    }

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
