#pragma once

#include "config.h"
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>

// Pointers to weights & biases for a single Transformer block
struct LayerParameters {
    float* ln1_gamma;   // (C,)
    float* ln1_beta;    // (C,)
    float* qkv_w;       // (C, 3 * C)
    float* qkv_b;       // (3 * C,)
    float* proj_w;      // (C, C)
    float* proj_b;      // (C,)
    float* ln2_gamma;   // (C,)
    float* ln2_beta;    // (C,)
    float* ffn1_w;      // (C, d_ff)
    float* ffn1_b;      // (d_ff,)
    float* ffn2_w;      // (d_ff, C)
    float* ffn2_b;      // (C,)
};

// Pointers to all model weights in contiguous GPU memory
struct TransformerParameters {
    float* token_emb;   // (V, C)
    float* pos_emb;     // (T, C)
    std::vector<LayerParameters> layers;
    float* ln_f_gamma;  // (C,)
    float* ln_f_beta;   // (C,)
    float* head_w;      // (C, V)
};

// Forward activations cached for analytical backward pass
struct LayerActivations {
    float* ln1_out;      // (B, T, C)
    float* ln1_mean;     // (B, T)
    float* ln1_rstd;     // (B, T)
    float* qkv;          // (B, T, 3 * C)
    float* q;            // (B, H, T, d_head)
    float* k;            // (B, H, T, d_head)
    float* v;            // (B, H, T, d_head)
    float* attn_scores;  // (B, H, T, T)
    float* attn_probs;   // (B, H, T, T)
    float* attn_out;     // (B, H, T, d_head)
    float* attn_m;       // (B, H, T) for FlashAttention backward
    float* attn_l;       // (B, H, T) for FlashAttention backward
    float* head_merged;  // (B, T, C)
    float* proj_out;     // (B, T, C)
    float* res1_out;     // (B, T, C)
    float* ln2_out;      // (B, T, C)
    float* ln2_mean;     // (B, T)
    float* ln2_rstd;     // (B, T)
    float* ffn1_out;     // (B, T, d_ff)
    float* ffn_gelu;     // (B, T, d_ff)
    float* ffn2_out;     // (B, T, C)
    float* block_out;    // (B, T, C)
};

struct TransformerActivations {
    float* emb_out;      // (B, T, C)
    std::vector<LayerActivations> layers;
    float* ln_f_out;     // (B, T, C)
    float* ln_f_mean;    // (B, T)
    float* ln_f_rstd;    // (B, T)
    float* logits;       // (B, T, V)
};

// Gradient buffers for analytical backward pass
struct TransformerGradients {
    TransformerParameters params; // Mirrors parameter layout for d_params
    float* d_logits;              // (B, T, V)
    float* d_ln_f_out;            // (B, T, C)
    float* d_block_out;           // Scratch buffer (B, T, C)
    float* d_res1_out;            // Scratch buffer (B, T, C)
    float* d_ffn2_out;            // Scratch buffer (B, T, C)
    float* d_ffn_gelu;            // Scratch buffer (B, T, d_ff)
    float* d_ffn1_out;            // Scratch buffer (B, T, d_ff)
    float* d_ln2_out;             // Scratch buffer (B, T, C)
    float* d_proj_out;            // Scratch buffer (B, T, C)
    float* d_head_merged;         // Scratch buffer (B, T, C)
    float* d_attn_out;            // Scratch buffer (B, H, T, d_head)
    float* d_attn_probs;          // Scratch buffer (B, H, T, T)
    float* d_attn_scores;         // Scratch buffer (B, H, T, T)
    float* d_q;                   // Scratch buffer (B, H, T, d_head)
    float* d_k;                   // Scratch buffer (B, H, T, d_head)
    float* d_v;                   // Scratch buffer (B, H, T, d_head)
    float* d_qkv;                 // Scratch buffer (B, T, 3 * C)
    float* d_ln1_out;             // Scratch buffer (B, T, C)
    float* d_emb_out;             // Scratch buffer (B, T, C)
    float* d_attn_d_scratch;      // Scratch buffer (B, H, T) for FlashAttention backward D_i
};

class TransformerModel {
public:
    explicit TransformerModel(const TransformerConfig& cfg);
    ~TransformerModel();

    void forward(const int* d_tokens, cudaStream_t stream = 0);
    void backward(const int* d_tokens, const int* d_targets, float* host_loss, cudaStream_t stream = 0);

    void init_parameters(unsigned long long seed = 42);
    void load_parameters(const std::string& filename);
    void save_parameters(const std::string& filename) const;
    void save_gradients(const std::string& filename) const;

    float* get_params_memory() { return d_params_memory; }
    float* get_grads_memory()  { return d_grads_memory; }
    const TransformerParameters& get_params() const { return params; }
    const TransformerActivations& get_activations() const { return acts; }
    const TransformerGradients& get_gradients() const { return grads; }
    size_t get_num_parameters() const { return num_parameters; }
    const TransformerConfig& get_config() const { return config; }

    cublasHandle_t get_cublas_handle() const { return cublas_handle; }
    cublasLtHandle_t get_cublaslt_handle() const { return cublaslt_handle; }

private:
    TransformerConfig config;
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    void* d_cublaslt_workspace;
    size_t cublaslt_workspace_size;

    size_t num_parameters;
    float* d_params_memory; // Contiguous buffer
    float* d_grads_memory;  // Contiguous buffer

    TransformerParameters params;
    TransformerActivations acts;
    TransformerGradients grads;

    void allocate_memory();
    void free_memory();
    void map_parameter_pointers(TransformerParameters& p, float* base);
};
