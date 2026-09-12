#include "common.h"
#include "config.h"
#include "model.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

int main(int argc, char** argv) {
    std::string data_dir = "tests/parity_data";
    if (argc > 1) data_dir = argv[1];

    std::string cfg_path = data_dir + "/config.txt";
    std::ifstream cfg_in(cfg_path);
    if (!cfg_in.is_open()) {
        std::cerr << "Error: Could not open " << cfg_path << "\n";
        return 1;
    }

    TransformerConfig config;
    cfg_in >> config.batch_size >> config.max_seq_len >> config.d_model
           >> config.num_layers >> config.num_heads >> config.vocab_size;
    config.d_head = config.d_model / config.num_heads;
    config.d_ff = 4 * config.d_model;

    std::cout << "[Parity Audit] Loading config: B=" << config.batch_size
              << ", T=" << config.max_seq_len << ", C=" << config.d_model
              << ", L=" << config.num_layers << ", H=" << config.num_heads
              << ", V=" << config.vocab_size << "\n";

    TransformerModel model(config);

    // Load reference parameters exported from PyTorch
    std::string weights_path = data_dir + "/pytorch_params.bin";
    model.load_parameters(weights_path);

    size_t total_tokens = config.batch_size * config.max_seq_len;
    std::vector<int> h_inputs(total_tokens);
    std::vector<int> h_targets(total_tokens);

    std::ifstream in_f(data_dir + "/inputs.bin", std::ios::binary);
    in_f.read(reinterpret_cast<char*>(h_inputs.data()), total_tokens * sizeof(int));
    std::ifstream tgt_f(data_dir + "/targets.bin", std::ios::binary);
    tgt_f.read(reinterpret_cast<char*>(h_targets.data()), total_tokens * sizeof(int));

    int* d_inputs;
    int* d_targets;
    CUDA_CHECK(cudaMalloc(&d_inputs, total_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_targets, total_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_inputs, h_inputs.data(), total_tokens * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets.data(), total_tokens * sizeof(int), cudaMemcpyHostToDevice));

    // Forward pass
    model.forward(d_inputs);

    // Save forward logits
    size_t logits_size = total_tokens * config.vocab_size;
    std::vector<float> h_logits(logits_size);
    CUDA_CHECK(cudaMemcpy(h_logits.data(), model.get_activations().logits, logits_size * sizeof(float), cudaMemcpyDeviceToHost));

    std::ofstream out_logits(data_dir + "/cuda_logits.bin", std::ios::binary);
    out_logits.write(reinterpret_cast<const char*>(h_logits.data()), logits_size * sizeof(float));

    // Backward pass
    float host_loss = 0.0f;
    model.backward(d_inputs, d_targets, &host_loss);

    std::ofstream out_loss(data_dir + "/cuda_loss.bin", std::ios::binary);
    out_loss.write(reinterpret_cast<const char*>(&host_loss), sizeof(float));

    // Save analytical gradients
    model.save_gradients(data_dir + "/cuda_grads.bin");

    std::cout << "[Parity Audit] Completed forward & backward pass. Loss: " << host_loss << "\n";
    std::cout << "[Parity Audit] Golden tensors dumped to " << data_dir << "\n";

    cudaFree(d_inputs);
    cudaFree(d_targets);
    return 0;
}
