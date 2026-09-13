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

bool test_fused_add_bias_gelu() {
    std::cout << "[Test] Running Fused AddBias+GELU kernel verification...";
    int M = 16;
    int N = 64; // multiple of 4 for float4 vectorization
    int total = M * N;

    std::vector<float> h_x(total);
    std::vector<float> h_bias(N);
    std::vector<float> h_out(total);
    std::vector<float> h_ref(total);

    for (int i = 0; i < total; ++i) h_x[i] = (i % 50 - 25) * 0.05f;
    for (int j = 0; j < N; ++j) h_bias[j] = (j % 10 - 5) * 0.02f;

    // Reference compute
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float val = h_x[m * N + n] + h_bias[n];
            float u = 0.7978845608028654f * (val + 0.044715f * val * val * val);
            float t = std::tanh(u);
            h_ref[m * N + n] = 0.5f * val * (1.0f + t);
        }
    }

    float *d_x, *d_bias, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    add_bias_gelu_forward(d_x, d_bias, d_out, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, total * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < total; ++i) {
        float diff = std::fabs(h_out[i] - h_ref[i]);
        assert(diff < 1e-4f);
    }

    cudaFree(d_x);
    cudaFree(d_bias);
    cudaFree(d_out);

    std::cout << " PASSED\n";
    return true;
}

bool test_fused_add_bias_residual() {
    std::cout << "[Test] Running Fused AddBias+Residual kernel verification...";
    int M = 16;
    int N = 64;
    int total = M * N;

    std::vector<float> h_res(total, 2.0f);
    std::vector<float> h_in(total, 1.5f);
    std::vector<float> h_bias(N, 0.25f);
    std::vector<float> h_out(total, 0.0f);

    float *d_res, *d_in, *d_bias, *d_out;
    CUDA_CHECK(cudaMalloc(&d_res, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_in, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    add_bias_residual(d_res, d_in, d_bias, d_out, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, total * sizeof(float), cudaMemcpyDeviceToHost));

    // 2.0 + 1.5 + 0.25 = 3.75
    for (int i = 0; i < total; ++i) {
        assert(std::fabs(h_out[i] - 3.75f) < 1e-5f);
    }

    cudaFree(d_res);
    cudaFree(d_in);
    cudaFree(d_bias);
    cudaFree(d_out);

    std::cout << " PASSED\n";
    return true;
}

bool test_tiled_causal_attention() {
    std::cout << "[Test] Running Tiled Causal Attention forward & backward kernel verification...";
    int B = 1, H = 1, T = 8, d_head = 32;
    int total = B * H * T * d_head;
    int stats_total = B * H * T;

    std::vector<float> h_q(total, 0.1f);
    std::vector<float> h_k(total, 0.1f);
    std::vector<float> h_v(total, 1.0f);
    std::vector<float> h_out(total, 0.0f);
    std::vector<float> h_do(total, 0.5f);
    std::vector<float> h_dq(total, 0.0f);
    std::vector<float> h_dk(total, 0.0f);
    std::vector<float> h_dv(total, 0.0f);

    float *d_q, *d_k, *d_v, *d_out, *d_m, *d_l;
    float *d_do, *d_dq, *d_dk, *d_dv, *d_workspace;
    CUDA_CHECK(cudaMalloc(&d_q, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m, stats_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_l, stats_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_do, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dq, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dk, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dv, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_workspace, stats_total * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_do, h_do.data(), total * sizeof(float), cudaMemcpyHostToDevice));

    float scale = 1.0f / std::sqrt(static_cast<float>(d_head));
    tiled_causal_attention_forward(d_q, d_k, d_v, d_out, B, H, T, d_head, scale, d_m, d_l);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, total * sizeof(float), cudaMemcpyDeviceToHost));

    // When V is all 1.0, convex combination sum(softmax_j * 1.0) == 1.0 for all queries
    for (int i = 0; i < total; ++i) {
        assert(std::fabs(h_out[i] - 1.0f) < 1e-3f);
    }

    tiled_causal_attention_backward(
        d_q, d_k, d_v, d_out, d_do, d_m, d_l,
        d_dq, d_dk, d_dv, d_workspace,
        B, H, T, d_head, scale
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_dq.data(), d_dq, total * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dk.data(), d_dk, total * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dv.data(), d_dv, total * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < total; ++i) {
        assert(!std::isnan(h_dq[i]) && !std::isinf(h_dq[i]));
        assert(!std::isnan(h_dk[i]) && !std::isinf(h_dk[i]));
        assert(!std::isnan(h_dv[i]) && !std::isinf(h_dv[i]));
    }

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);
    cudaFree(d_m);
    cudaFree(d_l);
    cudaFree(d_do);
    cudaFree(d_dq);
    cudaFree(d_dk);
    cudaFree(d_dv);
    cudaFree(d_workspace);

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
    test_fused_add_bias_gelu();
    test_fused_add_bias_residual();
    test_tiled_causal_attention();
    std::cout << "========================================\n";
    std::cout << " All isolated kernel tests passed!\n";
    std::cout << "========================================\n";
    return 0;
}
