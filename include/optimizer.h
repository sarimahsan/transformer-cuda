#pragma once

#include "config.h"
#include <cuda_runtime.h>
#include <cstddef>

class AdamW {
public:
    AdamW(float* params, float* grads, size_t num_params, const TransformerConfig& config);
    ~AdamW();

    void step(float lr, cudaStream_t stream = 0);
    void zero_grad(cudaStream_t stream = 0);
    float clip_grad_norm(float max_norm, cudaStream_t stream = 0);
    void reset();

    size_t get_step_count() const { return step_count; }

private:
    float* d_params;
    float* d_grads;
    float* d_m; // First moment
    float* d_v; // Second moment
    size_t num_params;
    size_t step_count;

    float beta1;
    float beta2;
    float eps;
    float weight_decay;

    float* d_norm_buffer;
};
