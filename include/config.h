#pragma once

#include <cstddef>
#include <iostream>
#include <string>

struct TransformerConfig {
    size_t batch_size = 32;       // B
    size_t max_seq_len = 256;     // T
    size_t vocab_size = 65;       // V (default TinyShakespeare character vocab)
    size_t d_model = 256;         // C or d_model
    size_t num_layers = 6;        // L
    size_t num_heads = 8;         // H
    size_t d_head = 32;           // d_k = d_model / num_heads
    size_t d_ff = 1024;           // 4 * d_model
    float layernorm_eps = 1e-5f;  // LayerNorm epsilon
    float learning_rate = 3e-4f;  // AdamW learning rate
    float weight_decay = 0.01f;   // AdamW weight decay
    float beta1 = 0.9f;           // AdamW beta1
    float beta2 = 0.999f;         // AdamW beta2
    float adam_eps = 1e-8f;       // AdamW epsilon
    float grad_clip = 1.0f;       // Gradient norm clipping threshold
    bool use_tiled_attention = false; // FlashAttention-style tiled online softmax
    bool is_fast_arch = false;        // FastTransformer architecture (MQA + RMSNorm + Lean MLP)
    size_t num_kv_heads = 8;          // Number of Key/Value heads (1 for MQA)

    void validate() const {
        if (d_model % num_heads != 0) {
            std::cerr << "Error: d_model (" << d_model << ") must be divisible by num_heads (" << num_heads << ")\n";
            exit(1);
        }
        if (d_model / num_heads != d_head) {
            std::cerr << "Warning: d_head (" << d_head << ") mismatch with d_model / num_heads ("
                      << (d_model / num_heads) << "). Updating d_head.\n";
        }
    }

    void print() const {
        std::cout << "==================================================\n";
        std::cout << " Transformer Model Configuration\n";
        std::cout << "==================================================\n";
        std::cout << " Batch Size (B)       : " << batch_size << "\n";
        std::cout << " Sequence Length (T)  : " << max_seq_len << "\n";
        std::cout << " Vocabulary Size (V)  : " << vocab_size << "\n";
        std::cout << " Hidden Dim (d_model) : " << d_model << "\n";
        std::cout << " Layers (L)           : " << num_layers << "\n";
        std::cout << " Heads (H)            : " << num_heads << "\n";
        std::cout << " Head Dim (d_k)       : " << d_head << "\n";
        std::cout << " FFN Dim (d_ff)       : " << d_ff << "\n";
        std::cout << " LayerNorm Epsilon    : " << layernorm_eps << "\n";
        std::cout << " Learning Rate        : " << learning_rate << "\n";
        std::cout << " Weight Decay         : " << weight_decay << "\n";
        std::cout << "==================================================\n";
    }
};
