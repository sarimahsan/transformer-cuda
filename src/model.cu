#include "model.h"
#include "common.h"
#include "kernels/embedding.cuh"
#include "kernels/layernorm.cuh"
#include "kernels/matmul.cuh"
#include "kernels/attention.cuh"
#include "kernels/ffn.cuh"
#include "kernels/residual.cuh"
#include "kernels/loss.cuh"
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>

TransformerModel::TransformerModel(const TransformerConfig& cfg) : config(cfg) {
    config.validate();
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    allocate_memory();
    init_parameters(42);
}

TransformerModel::~TransformerModel() {
    free_memory();
    if (cublas_handle != nullptr) {
        cublasDestroy(cublas_handle);
    }
}

void TransformerModel::allocate_memory() {
    size_t V = config.vocab_size;
    size_t T = config.max_seq_len;
    size_t C = config.d_model;
    size_t L = config.num_layers;
    size_t H = config.num_heads;
    size_t d_head = config.d_head;
    size_t d_ff = config.d_ff;
    size_t B = config.batch_size;

    // 1. Calculate parameter sizes
    size_t tok_emb_size = V * C;
    size_t pos_emb_size = T * C;
    size_t ln1_size = 2 * C;
    size_t qkv_size = C * (3 * C) + (3 * C);
    size_t proj_size = C * C + C;
    size_t ln2_size = 2 * C;
    size_t ffn1_size = C * d_ff + d_ff;
    size_t ffn2_size = d_ff * C + C;
    size_t per_layer_params = ln1_size + qkv_size + proj_size + ln2_size + ffn1_size + ffn2_size;
    size_t ln_f_size = 2 * C;
    size_t head_size = C * V;

    num_parameters = tok_emb_size + pos_emb_size + L * per_layer_params + ln_f_size + head_size;

    // Allocate contiguous parameter and gradient buffers
    CUDA_CHECK(cudaMalloc(&d_params_memory, num_parameters * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grads_memory, num_parameters * sizeof(float)));

    map_parameter_pointers(params, d_params_memory);
    map_parameter_pointers(grads.params, d_grads_memory);

    // 2. Allocate forward activations
    CUDA_CHECK(cudaMalloc(&acts.emb_out, B * T * C * sizeof(float)));
    acts.layers.resize(L);
    for (size_t l = 0; l < L; ++l) {
        CUDA_CHECK(cudaMalloc(&acts.layers[l].ln1_out, B * T * C * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts.layers[l].ln1_mean, B * T * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts.layers[l].ln1_rstd, B * T * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts.layers[l].qkv, B * T * (3 * C) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts.layers[l].q, B * H * T * d_head * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts.layers[l].k, B * H * T * d_head * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts.layers[l].v, B * H * T * d_head * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts.layers[l].attn_scores, B * H * T * T * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts.layers[l].attn_probs, B * H * T * T * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts.layers[l].attn_out, B * H * T * d_head * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts.layers[l].head_merged, B * T * C * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts.layers[l].proj_out, B * T * C * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts.layers[l].res1_out, B * T * C * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts.layers[l].ln2_out, B * T * C * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts.layers[l].ln2_mean, B * T * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts.layers[l].ln2_rstd, B * T * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts.layers[l].ffn1_out, B * T * d_ff * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts.layers[l].ffn_gelu, B * T * d_ff * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts.layers[l].ffn2_out, B * T * C * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts.layers[l].block_out, B * T * C * sizeof(float)));
    }
    CUDA_CHECK(cudaMalloc(&acts.ln_f_out, B * T * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&acts.ln_f_mean, B * T * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&acts.ln_f_rstd, B * T * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&acts.logits, B * T * V * sizeof(float)));

    // 3. Allocate backward scratch gradients
    CUDA_CHECK(cudaMalloc(&grads.d_logits, B * T * V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grads.d_ln_f_out, B * T * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grads.d_block_out, B * T * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grads.d_res1_out, B * T * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grads.d_ffn2_out, B * T * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grads.d_ffn_gelu, B * T * d_ff * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grads.d_ffn1_out, B * T * d_ff * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grads.d_ln2_out, B * T * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grads.d_proj_out, B * T * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grads.d_head_merged, B * T * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grads.d_attn_out, B * H * T * d_head * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grads.d_attn_probs, B * H * T * T * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grads.d_attn_scores, B * H * T * T * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grads.d_q, B * H * T * d_head * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grads.d_k, B * H * T * d_head * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grads.d_v, B * H * T * d_head * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grads.d_qkv, B * T * (3 * C) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grads.d_ln1_out, B * T * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grads.d_emb_out, B * T * C * sizeof(float)));
}

void TransformerModel::free_memory() {
    cudaFree(d_params_memory);
    cudaFree(d_grads_memory);

    cudaFree(acts.emb_out);
    for (size_t l = 0; l < acts.layers.size(); ++l) {
        cudaFree(acts.layers[l].ln1_out);
        cudaFree(acts.layers[l].ln1_mean);
        cudaFree(acts.layers[l].ln1_rstd);
        cudaFree(acts.layers[l].qkv);
        cudaFree(acts.layers[l].q);
        cudaFree(acts.layers[l].k);
        cudaFree(acts.layers[l].v);
        cudaFree(acts.layers[l].attn_scores);
        cudaFree(acts.layers[l].attn_probs);
        cudaFree(acts.layers[l].attn_out);
        cudaFree(acts.layers[l].head_merged);
        cudaFree(acts.layers[l].proj_out);
        cudaFree(acts.layers[l].res1_out);
        cudaFree(acts.layers[l].ln2_out);
        cudaFree(acts.layers[l].ln2_mean);
        cudaFree(acts.layers[l].ln2_rstd);
        cudaFree(acts.layers[l].ffn1_out);
        cudaFree(acts.layers[l].ffn_gelu);
        cudaFree(acts.layers[l].ffn2_out);
        cudaFree(acts.layers[l].block_out);
    }
    cudaFree(acts.ln_f_out);
    cudaFree(acts.ln_f_mean);
    cudaFree(acts.ln_f_rstd);
    cudaFree(acts.logits);

    cudaFree(grads.d_logits);
    cudaFree(grads.d_ln_f_out);
    cudaFree(grads.d_block_out);
    cudaFree(grads.d_res1_out);
    cudaFree(grads.d_ffn2_out);
    cudaFree(grads.d_ffn_gelu);
    cudaFree(grads.d_ffn1_out);
    cudaFree(grads.d_ln2_out);
    cudaFree(grads.d_proj_out);
    cudaFree(grads.d_head_merged);
    cudaFree(grads.d_attn_out);
    cudaFree(grads.d_attn_probs);
    cudaFree(grads.d_attn_scores);
    cudaFree(grads.d_q);
    cudaFree(grads.d_k);
    cudaFree(grads.d_v);
    cudaFree(grads.d_qkv);
    cudaFree(grads.d_ln1_out);
    cudaFree(grads.d_emb_out);
}

void TransformerModel::map_parameter_pointers(TransformerParameters& p, float* base) {
    size_t V = config.vocab_size;
    size_t T = config.max_seq_len;
    size_t C = config.d_model;
    size_t L = config.num_layers;
    size_t d_ff = config.d_ff;

    float* ptr = base;
    p.token_emb = ptr; ptr += V * C;
    p.pos_emb = ptr; ptr += T * C;

    p.layers.resize(L);
    for (size_t l = 0; l < L; ++l) {
        p.layers[l].ln1_gamma = ptr; ptr += C;
        p.layers[l].ln1_beta = ptr; ptr += C;
        p.layers[l].qkv_w = ptr; ptr += C * (3 * C);
        p.layers[l].qkv_b = ptr; ptr += 3 * C;
        p.layers[l].proj_w = ptr; ptr += C * C;
        p.layers[l].proj_b = ptr; ptr += C;
        p.layers[l].ln2_gamma = ptr; ptr += C;
        p.layers[l].ln2_beta = ptr; ptr += C;
        p.layers[l].ffn1_w = ptr; ptr += C * d_ff;
        p.layers[l].ffn1_b = ptr; ptr += d_ff;
        p.layers[l].ffn2_w = ptr; ptr += d_ff * C;
        p.layers[l].ffn2_b = ptr; ptr += C;
    }
    p.ln_f_gamma = ptr; ptr += C;
    p.ln_f_beta = ptr; ptr += C;
    p.head_w = ptr; ptr += C * V;
}

void TransformerModel::init_parameters(unsigned long long seed) {
    std::mt19937_64 rng(seed);
    std::vector<float> h_params(num_parameters);

    size_t C = config.d_model;
    float std_dev = 0.02f;
    std::normal_distribution<float> norm_dist(0.0f, std_dev);

    // Default initialization: weights ~ N(0, 0.02), biases = 0, gamma = 1, beta = 0
    for (size_t i = 0; i < num_parameters; ++i) {
        h_params[i] = norm_dist(rng);
    }

    TransformerParameters h_p;
    map_parameter_pointers(h_p, h_params.data());

    // Initialize LayerNorm gammas to 1, betas to 0, and biases to 0
    for (size_t l = 0; l < config.num_layers; ++l) {
        std::fill(h_p.layers[l].ln1_gamma, h_p.layers[l].ln1_gamma + C, 1.0f);
        std::fill(h_p.layers[l].ln1_beta, h_p.layers[l].ln1_beta + C, 0.0f);
        std::fill(h_p.layers[l].qkv_b, h_p.layers[l].qkv_b + 3 * C, 0.0f);
        std::fill(h_p.layers[l].proj_b, h_p.layers[l].proj_b + C, 0.0f);
        std::fill(h_p.layers[l].ln2_gamma, h_p.layers[l].ln2_gamma + C, 1.0f);
        std::fill(h_p.layers[l].ln2_beta, h_p.layers[l].ln2_beta + C, 0.0f);
        std::fill(h_p.layers[l].ffn1_b, h_p.layers[l].ffn1_b + config.d_ff, 0.0f);
        std::fill(h_p.layers[l].ffn2_b, h_p.layers[l].ffn2_b + C, 0.0f);
    }
    std::fill(h_p.ln_f_gamma, h_p.ln_f_gamma + C, 1.0f);
    std::fill(h_p.ln_f_beta, h_p.ln_f_beta + C, 0.0f);

    CUDA_CHECK(cudaMemcpy(d_params_memory, h_params.data(), num_parameters * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_grads_memory, 0, num_parameters * sizeof(float)));
}

void TransformerModel::forward(const int* d_tokens, cudaStream_t stream) {
    NVTX_PUSH("Transformer_Forward");
    int B = static_cast<int>(config.batch_size);
    int T = static_cast<int>(config.max_seq_len);
    int C = static_cast<int>(config.d_model);
    int H = static_cast<int>(config.num_heads);
    int d_head = static_cast<int>(config.d_head);
    int d_ff = static_cast<int>(config.d_ff);
    int V = static_cast<int>(config.vocab_size);
    int L = static_cast<int>(config.num_layers);

    // 1. Embedding
    NVTX_PUSH("Embedding_Fwd");
    embedding_forward(d_tokens, params.token_emb, params.pos_emb, acts.emb_out, B, T, C, stream);
    NVTX_POP();

    const float* curr_x = acts.emb_out;

    // 2. Transformer Blocks
    for (int l = 0; l < L; ++l) {
        NVTX_PUSH("Layer_Fwd");
        LayerParameters& lp = params.layers[l];
        LayerActivations& la = acts.layers[l];

        // Pre-LN 1
        NVTX_PUSH("LN1");
        layernorm_forward(
            curr_x, lp.ln1_gamma, lp.ln1_beta,
            la.ln1_out, la.ln1_mean, la.ln1_rstd,
            B * T, C, config.layernorm_eps, stream
        );
        NVTX_POP();

        // QKV Projection: (B * T, C) x (C, 3 * C) -> (B * T, 3 * C)
        NVTX_PUSH("QKV_Proj");
        matmul_forward(cublas_handle, la.ln1_out, lp.qkv_w, la.qkv, B * T, 3 * C, C, false, false, 1.0f, 0.0f, stream);
        add_bias(la.qkv, lp.qkv_b, B * T, 3 * C, stream);
        NVTX_POP();

        // Split & Transpose QKV
        NVTX_PUSH("Attention");
        qkv_split_transpose_forward(la.qkv, la.q, la.k, la.v, B, T, H, d_head, stream);

        float scale = 1.0f / sqrtf(static_cast<float>(d_head));

        if (config.use_tiled_attention) {
            // FlashAttention-Style Tiled Online Softmax in SRAM/registers
            tiled_causal_attention_forward(la.q, la.k, la.v, la.attn_out, B, H, T, d_head, scale, stream);
        } else {
            // Batched GEMM: Attention Scores = Q x K^T
            matmul_batched_strided(
                cublas_handle,
                la.q, la.k, la.attn_scores,
                T, T, d_head,
                T * d_head, T * d_head, T * T,
                B * H, false, true, 1.0f, 0.0f, stream
            );

            // Scaled Causal Softmax
            causal_softmax_forward(la.attn_scores, la.attn_probs, B, H, T, scale, stream);

            // Batched GEMM: Attention Output = Probs x V
            matmul_batched_strided(
                cublas_handle,
                la.attn_probs, la.v, la.attn_out,
                T, d_head, T,
                T * T, T * d_head, T * d_head,
                B * H, false, false, 1.0f, 0.0f, stream
            );
        }

        // Merge Heads Transpose: (B, H, T, d_head) -> (B, T, C)
        head_merge_transpose_forward(la.attn_out, la.head_merged, B, H, T, d_head, stream);
        NVTX_POP();

        // Out Projection & Fused Residual Add: res1 = curr_x + (proj_out + bias)
        NVTX_PUSH("Attn_Out_Proj");
        matmul_forward(cublas_handle, la.head_merged, lp.proj_w, la.proj_out, B * T, C, C, false, false, 1.0f, 0.0f, stream);
        add_bias_residual(curr_x, la.proj_out, lp.proj_b, la.res1_out, B * T, C, stream);
        NVTX_POP();

        // Pre-LN 2
        NVTX_PUSH("LN2");
        layernorm_forward(
            la.res1_out, lp.ln2_gamma, lp.ln2_beta,
            la.ln2_out, la.ln2_mean, la.ln2_rstd,
            B * T, C, config.layernorm_eps, stream
        );
        NVTX_POP();

        // FFN Layer 1 & Fused Bias-GELU
        NVTX_PUSH("FFN1_GELU");
        matmul_forward(cublas_handle, la.ln2_out, lp.ffn1_w, la.ffn1_out, B * T, d_ff, C, false, false, 1.0f, 0.0f, stream);
        add_bias_gelu_forward(la.ffn1_out, lp.ffn1_b, la.ffn_gelu, B * T, d_ff, stream);
        NVTX_POP();

        // FFN Layer 2 & Fused Residual Add
        NVTX_PUSH("FFN2_Residual");
        matmul_forward(cublas_handle, la.ffn_gelu, lp.ffn2_w, la.ffn2_out, B * T, C, d_ff, false, false, 1.0f, 0.0f, stream);
        add_bias_residual(la.res1_out, la.ffn2_out, lp.ffn2_b, la.block_out, B * T, C, stream);
        NVTX_POP();

        curr_x = la.block_out;
        NVTX_POP(); // Layer_Fwd
    }

    // 3. Final LayerNorm
    NVTX_PUSH("LN_Final");
    layernorm_forward(
        curr_x, params.ln_f_gamma, params.ln_f_beta,
        acts.ln_f_out, acts.ln_f_mean, acts.ln_f_rstd,
        B * T, C, config.layernorm_eps, stream
    );
    NVTX_POP();

    // 4. Head Projection to Logits: (B * T, C) x (C, V) -> (B * T, V)
    NVTX_PUSH("Head_Proj");
    matmul_forward(cublas_handle, acts.ln_f_out, params.head_w, acts.logits, B * T, V, C, false, false, 1.0f, 0.0f, stream);
    NVTX_POP();

    NVTX_POP(); // Transformer_Forward
}

void TransformerModel::backward(const int* d_tokens, const int* d_targets, float* host_loss, cudaStream_t stream) {
    NVTX_PUSH("Transformer_Backward");
    int B = static_cast<int>(config.batch_size);
    int T = static_cast<int>(config.max_seq_len);
    int C = static_cast<int>(config.d_model);
    int H = static_cast<int>(config.num_heads);
    int d_head = static_cast<int>(config.d_head);
    int d_ff = static_cast<int>(config.d_ff);
    int V = static_cast<int>(config.vocab_size);
    int L = static_cast<int>(config.num_layers);

    // 1. Fused Cross-Entropy Loss & d_logits
    NVTX_PUSH("Loss_Head_Bwd");
    cross_entropy_forward_backward(acts.logits, d_targets, grads.d_logits, host_loss, B, T, V, stream);

    // 2. Head Projection Backward:
    matmul_forward(cublas_handle, acts.ln_f_out, grads.d_logits, grads.params.head_w, C, V, B * T, true, false, 1.0f, 0.0f, stream);
    matmul_forward(cublas_handle, grads.d_logits, params.head_w, grads.d_ln_f_out, B * T, C, V, false, true, 1.0f, 0.0f, stream);

    // 3. Final LayerNorm Backward
    const float* prev_block_out = (L > 0) ? acts.layers[L - 1].block_out : acts.emb_out;
    layernorm_backward(
        grads.d_ln_f_out, prev_block_out, params.ln_f_gamma,
        acts.ln_f_mean, acts.ln_f_rstd,
        grads.d_block_out, grads.params.ln_f_gamma, grads.params.ln_f_beta,
        B * T, C, stream
    );
    NVTX_POP();

    float scale = 1.0f / sqrtf(static_cast<float>(d_head));

    // 4. Reverse Block Iteration
    for (int l = L - 1; l >= 0; --l) {
        NVTX_PUSH("Layer_Bwd");
        LayerParameters& lp = params.layers[l];
        LayerParameters& d_lp = grads.params.layers[l];
        LayerActivations& la = acts.layers[l];
        const float* block_input = (l > 0) ? acts.layers[l - 1].block_out : acts.emb_out;

        // Residual 2: d_block_out branches into d_res1_out and d_ffn2_out
        CUDA_CHECK(cudaMemcpyAsync(grads.d_res1_out, grads.d_block_out, B * T * C * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(grads.d_ffn2_out, grads.d_block_out, B * T * C * sizeof(float), cudaMemcpyDeviceToDevice, stream));

        // FFN2 Backward
        NVTX_PUSH("FFN2_Bwd");
        matmul_forward(cublas_handle, la.ffn_gelu, grads.d_ffn2_out, d_lp.ffn2_w, d_ff, C, B * T, true, false, 1.0f, 0.0f, stream);
        bias_backward(grads.d_ffn2_out, d_lp.ffn2_b, B * T, C, stream);
        matmul_forward(cublas_handle, grads.d_ffn2_out, lp.ffn2_w, grads.d_ffn_gelu, B * T, d_ff, C, false, true, 1.0f, 0.0f, stream);
        NVTX_POP();

        // GELU Backward
        NVTX_PUSH("GELU_Bwd");
        gelu_backward(grads.d_ffn_gelu, la.ffn1_out, grads.d_ffn1_out, B * T * d_ff, stream);
        NVTX_POP();

        // FFN1 Backward
        NVTX_PUSH("FFN1_Bwd");
        matmul_forward(cublas_handle, la.ln2_out, grads.d_ffn1_out, d_lp.ffn1_w, C, d_ff, B * T, true, false, 1.0f, 0.0f, stream);
        bias_backward(grads.d_ffn1_out, d_lp.ffn1_b, B * T, d_ff, stream);
        matmul_forward(cublas_handle, grads.d_ffn1_out, lp.ffn1_w, grads.d_ln2_out, B * T, C, d_ff, false, true, 1.0f, 0.0f, stream);
        NVTX_POP();

        // LayerNorm 2 Backward
        NVTX_PUSH("LN2_Bwd");
        layernorm_backward(
            grads.d_ln2_out, la.res1_out, lp.ln2_gamma,
            la.ln2_mean, la.ln2_rstd,
            grads.d_emb_out, d_lp.ln2_gamma, d_lp.ln2_beta,
            B * T, C, stream
        );
        residual_accumulate(grads.d_res1_out, grads.d_emb_out, B * T * C, stream);
        NVTX_POP();

        // Residual 1: d_res1_out branches into d_in and d_proj_out
        CUDA_CHECK(cudaMemcpyAsync(grads.d_proj_out, grads.d_res1_out, B * T * C * sizeof(float), cudaMemcpyDeviceToDevice, stream));

        // Proj Backward
        NVTX_PUSH("Attn_Out_Bwd");
        matmul_forward(cublas_handle, la.head_merged, grads.d_proj_out, d_lp.proj_w, C, C, B * T, true, false, 1.0f, 0.0f, stream);
        bias_backward(grads.d_proj_out, d_lp.proj_b, B * T, C, stream);
        matmul_forward(cublas_handle, grads.d_proj_out, lp.proj_w, grads.d_head_merged, B * T, C, C, false, true, 1.0f, 0.0f, stream);
        head_merge_transpose_backward(grads.d_head_merged, grads.d_attn_out, B, H, T, d_head, stream);
        NVTX_POP();

        // Attention GEMMs Backward
        NVTX_PUSH("Attention_Bwd");
        matmul_batched_strided(
            cublas_handle,
            grads.d_attn_out, la.v, grads.d_attn_probs,
            T, T, d_head,
            T * d_head, T * d_head, T * T,
            B * H, false, true, 1.0f, 0.0f, stream
        );

        matmul_batched_strided(
            cublas_handle,
            la.attn_probs, grads.d_attn_out, grads.d_v,
            T, d_head, T,
            T * T, T * d_head, T * d_head,
            B * H, true, false, 1.0f, 0.0f, stream
        );

        causal_softmax_backward(grads.d_attn_probs, la.attn_probs, grads.d_attn_scores, B, H, T, scale, stream);

        matmul_batched_strided(
            cublas_handle,
            grads.d_attn_scores, la.k, grads.d_q,
            T, d_head, T,
            T * T, T * d_head, T * d_head,
            B * H, false, false, 1.0f, 0.0f, stream
        );

        matmul_batched_strided(
            cublas_handle,
            grads.d_attn_scores, la.q, grads.d_k,
            T, d_head, T,
            T * T, T * d_head, T * d_head,
            B * H, true, false, 1.0f, 0.0f, stream
        );

        qkv_split_transpose_backward(grads.d_q, grads.d_k, grads.d_v, grads.d_qkv, B, T, H, d_head, stream);
        NVTX_POP();

        // QKV Projection Backward
        NVTX_PUSH("QKV_Proj_Bwd");
        matmul_forward(cublas_handle, la.ln1_out, grads.d_qkv, d_lp.qkv_w, C, 3 * C, B * T, true, false, 1.0f, 0.0f, stream);
        bias_backward(grads.d_qkv, d_lp.qkv_b, B * T, 3 * C, stream);
        matmul_forward(cublas_handle, grads.d_qkv, lp.qkv_w, grads.d_ln1_out, B * T, C, 3 * C, false, true, 1.0f, 0.0f, stream);
        NVTX_POP();

        // LayerNorm 1 Backward
        NVTX_PUSH("LN1_Bwd");
        layernorm_backward(
            grads.d_ln1_out, block_input, lp.ln1_gamma,
            la.ln1_mean, la.ln1_rstd,
            grads.d_emb_out, d_lp.ln1_gamma, d_lp.ln1_beta,
            B * T, C, stream
        );
        residual_add(grads.d_res1_out, grads.d_emb_out, grads.d_block_out, B * T * C, stream);
        NVTX_POP();

        NVTX_POP(); // Layer_Bwd
    }

    // 5. Embedding Backward: accum into d_tok_emb, d_pos_emb
    NVTX_PUSH("Embedding_Bwd");
    embedding_backward(grads.d_block_out, d_tokens, grads.params.token_emb, grads.params.pos_emb, B, T, C, stream);
    NVTX_POP();

    NVTX_POP(); // Transformer_Backward
}

void TransformerModel::save_parameters(const std::string& filename) const {
    std::vector<float> h_params(num_parameters);
    CUDA_CHECK(cudaMemcpy(h_params.data(), d_params_memory, num_parameters * sizeof(float), cudaMemcpyDeviceToHost));

    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Failed to open " << filename << " for saving parameters\n";
        return;
    }
    out.write(reinterpret_cast<const char*>(h_params.data()), num_parameters * sizeof(float));
    std::cout << "[Model] Saved " << num_parameters << " parameters to " << filename << "\n";
}

void TransformerModel::load_parameters(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Failed to open " << filename << " for loading parameters\n";
        return;
    }
    std::vector<float> h_params(num_parameters);
    in.read(reinterpret_cast<char*>(h_params.data()), num_parameters * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_params_memory, h_params.data(), num_parameters * sizeof(float), cudaMemcpyHostToDevice));
    std::cout << "[Model] Loaded " << num_parameters << " parameters from " << filename << "\n";
}

void TransformerModel::save_gradients(const std::string& filename) const {
    std::vector<float> h_grads(num_parameters);
    CUDA_CHECK(cudaMemcpy(h_grads.data(), d_grads_memory, num_parameters * sizeof(float), cudaMemcpyDeviceToHost));

    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Failed to open " << filename << " for saving gradients\n";
        return;
    }
    out.write(reinterpret_cast<const char*>(h_grads.data()), num_parameters * sizeof(float));
    std::cout << "[Model] Saved " << num_parameters << " gradients to " << filename << "\n";
}
