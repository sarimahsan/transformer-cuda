#include "common.h"
#include "kernels/layernorm.cuh"
#include "kernels/embedding.cuh"
#include "kernels/attention.cuh"
#include "kernels/ffn.cuh"
#include "kernels/residual.cuh"
#include "kernels/loss.cuh"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

bool test_layernorm() {
    std::cout << "[Test] Running LayerNorm kernel verification...";
    int N = 8;
    int C = 64;
    std::vector<float> h_x(N * C, 1.0f);
    std::vector<float> h_gamma(C, 1.0f);
    std::vector<float> h_beta(C, 0.0f);
    std::vector<float> h_out(N * C);
    std::vector<float> h_mean(N);
    std::vector<float> h_rstd(N);

    // Give each row different values
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < C; ++j) {
            h_x[i * C + j] = static_cast<float>(j % 10);
        }
    }

    float *d_x, *d_gamma, *d_beta, *d_out, *d_mean, *d_rstd;
    CUDA_CHECK(cudaMalloc(&d_x, N * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rstd, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * C * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), C * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), C * sizeof(float), cudaMemcpyHostToDevice));

    layernorm_forward(d_x, d_gamma, d_beta, d_out, d_mean, d_rstd, N, C, 1e-5f);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * C * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mean.data(), d_mean, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify row mean is ~4.5 and output mean is ~0
    float sum_out_first_row = 0.0f;
    for (int j = 0; j < C; ++j) sum_out_first_row += h_out[j];
    assert(std::fabs(sum_out_first_row / C) < 1e-4f);

    cudaFree(d_x);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_out);
    cudaFree(d_mean);
    cudaFree(d_rstd);

    std::cout << " PASSED\n";
    return true;
}

bool test_gelu() {
    std::cout << "[Test] Running GELU kernel verification...";
    int N = 128;
    std::vector<float> h_x(N);
    std::vector<float> h_y(N);
    for (int i = 0; i < N; ++i) h_x[i] = (i - 64) * 0.1f;

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    gelu_forward(d_x, d_y, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    // GELU(0) should be 0
    assert(std::fabs(h_y[64]) < 1e-5f);

    cudaFree(d_x);
    cudaFree(d_y);

    std::cout << " PASSED\n";
    return true;
}

bool test_causal_softmax() {
    std::cout << "[Test] Running Causal Softmax kernel verification...";
    int B = 1, H = 1, T = 4;
    std::vector<float> h_scores(T * T, 1.0f);
    std::vector<float> h_probs(T * T, 0.0f);

    float *d_scores, *d_probs;
    CUDA_CHECK(cudaMalloc(&d_scores, T * T * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_probs, T * T * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_scores, h_scores.data(), T * T * sizeof(float), cudaMemcpyHostToDevice));

    causal_softmax_forward(d_scores, d_probs, B, H, T, 1.0f);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_probs.data(), d_probs, T * T * sizeof(float), cudaMemcpyDeviceToHost));

    // Row 0 should have 1.0 at pos 0, and 0 everywhere else
    assert(std::fabs(h_probs[0] - 1.0f) < 1e-5f);
    assert(std::fabs(h_probs[1]) < 1e-5f);

    // Row 1 should have 0.5 at pos 0 and pos 1
    assert(std::fabs(h_probs[T + 0] - 0.5f) < 1e-4f);
    assert(std::fabs(h_probs[T + 1] - 0.5f) < 1e-4f);
    assert(std::fabs(h_probs[T + 2]) < 1e-5f);

    cudaFree(d_scores);
    cudaFree(d_probs);

    std::cout << " PASSED\n";
    return true;
}

int main() {
    std::cout << "========================================\n";
    std::cout << " Isolated CUDA Kernel Test Suite\n";
    std::cout << "========================================\n";
    test_layernorm();
    test_gelu();
    test_causal_softmax();
    std::cout << "========================================\n";
    std::cout << " All isolated kernel tests passed!\n";
    std::cout << "========================================\n";
    return 0;
}
