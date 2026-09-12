#pragma once

#include <cuda_runtime.h>

// Fused Softmax Cross-Entropy Loss Forward & Backward
// Computes scalar loss on host and produces dlogits on device:
// dlogits[i, v] = (softmax(logits)[i, v] - (v == targets[i])) / (B * T)
void cross_entropy_forward_backward(
    const float* logits,
    const int* targets,
    float* dlogits,
    float* host_loss,
    int B, int T, int V,
    cudaStream_t stream = 0
);
