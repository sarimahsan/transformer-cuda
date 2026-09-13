#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <chrono>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t status = call;                                            \
        if (status != cudaSuccess) {                                          \
            std::stringstream ss;                                             \
            ss << "CUDA Error at " << __FILE__ << ":" << __LINE__             \
               << " - " << cudaGetErrorString(status);                        \
            std::cerr << ss.str() << std::endl;                               \
            throw std::runtime_error(ss.str());                               \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t status = call;                                         \
        if (status != CUBLAS_STATUS_SUCCESS) {                                \
            std::stringstream ss;                                             \
            ss << "cuBLAS Error at " << __FILE__ << ":" << __LINE__           \
               << " - Code: " << status;                                      \
            std::cerr << ss.str() << std::endl;                               \
            throw std::runtime_error(ss.str());                               \
        }                                                                     \
    } while (0)

// High-precision GPU timer using CUDA Events
struct GpuTimer {
    cudaEvent_t start_event, stop_event;

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }

    ~GpuTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start_event, stream));
    }

    void stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_event, stream));
    }

    float elapsed_ms() {
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
        return ms;
    }
};

// High-precision CPU timer
struct CpuTimer {
    std::chrono::high_resolution_clock::time_point t_start, t_stop;

    void start() {
        t_start = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        t_stop = std::chrono::high_resolution_clock::now();
    }

    float elapsed_ms() const {
        return std::chrono::duration<float, std::milli>(t_stop - t_start).count();
    }
};

// NVTX Profiling Support for Nsight Systems
#if !defined(DISABLE_NVTX)
#include <nvtx3/nvToolsExt.h>
#define NVTX_PUSH(name) nvtxRangePushA(name)
#define NVTX_POP() nvtxRangePop()
#else
#define NVTX_PUSH(name) ((void)0)
#define NVTX_POP() ((void)0)
#endif

struct NvtxScopedRange {
    explicit NvtxScopedRange(const char* name) {
        NVTX_PUSH(name);
    }
    ~NvtxScopedRange() {
        NVTX_POP();
    }
};
