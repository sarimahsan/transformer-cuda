#include "common.h"
#include "config.h"
#include "model.h"
#include "optimizer.h"
#include "dataloader.h"
#include <iostream>
#include <iomanip>
#include <string>

int main(int argc, char** argv) {
    TransformerConfig config;
    std::string data_file = "data/input.bin";
    int num_epochs = 5;
    int log_interval = 10;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) {
            data_file = argv[++i];
        } else if (arg == "--epochs" && i + 1 < argc) {
            num_epochs = std::stoi(argv[++i]);
        } else if (arg == "--batch_size" || arg == "-b") {
            config.batch_size = std::stoull(argv[++i]);
        } else if (arg == "--seq_len" || arg == "-t") {
            config.max_seq_len = std::stoull(argv[++i]);
        } else if (arg == "--d_model" || arg == "-d") {
            config.d_model = std::stoull(argv[++i]);
        } else if (arg == "--num_layers" || arg == "-l") {
            config.num_layers = std::stoull(argv[++i]);
        } else if (arg == "--num_heads" || arg == "-h") {
            config.num_heads = std::stoull(argv[++i]);
        } else if (arg == "--lr") {
            config.learning_rate = std::stof(argv[++i]);
        }
    }
    config.d_head = config.d_model / config.num_heads;
    config.d_ff = 4 * config.d_model;

    config.print();

    std::cout << "[Train] Initializing DataLoader from: " << data_file << "\n";
    DataLoader loader(data_file, config.batch_size, config.max_seq_len);

    std::cout << "[Train] Initializing Pure CUDA Transformer (" << config.num_layers << " layers, d_model=" << config.d_model << ")...\n";
    TransformerModel model(config);
    AdamW optimizer(model.get_params_memory(), model.get_grads_memory(), model.get_num_parameters(), config);

    std::cout << "[Train] Total Model Parameters: " << model.get_num_parameters() << "\n";

    int* d_inputs;
    int* d_targets;
    size_t batch_tokens = config.batch_size * config.max_seq_len;
    CUDA_CHECK(cudaMalloc(&d_inputs, batch_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_targets, batch_tokens * sizeof(int)));

    GpuTimer step_timer;
    float running_loss = 0.0f;
    int step = 0;

    std::cout << "[Train] Starting training loop across " << num_epochs << " epochs...\n";

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        loader.reset();
        while (loader.next_batch(d_inputs, d_targets)) {
            step_timer.start();

            // Forward
            model.forward(d_inputs);

            // Backward
            float loss = 0.0f;
            model.backward(d_inputs, d_targets, &loss);

            // Optimizer step
            float grad_norm = optimizer.clip_grad_norm(config.grad_clip);
            optimizer.step(config.learning_rate);
            optimizer.zero_grad();

            step_timer.stop();
            float dt_ms = step_timer.elapsed_ms();

            running_loss += loss;
            step++;

            if (step % log_interval == 0) {
                float avg_loss = running_loss / log_interval;
                float tok_per_sec = (dt_ms > 0) ? (batch_tokens / (dt_ms / 1000.0f)) : 0.0f;
                std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "] "
                          << "Step " << std::setw(5) << step << " | "
                          << "Loss: " << std::fixed << std::setprecision(4) << avg_loss << " | "
                          << "GradNorm: " << std::setprecision(2) << grad_norm << " | "
                          << "Step: " << std::setprecision(2) << dt_ms << " ms | "
                          << "Throughput: " << std::setprecision(1) << tok_per_sec << " tok/s\n";
                running_loss = 0.0f;
            }
        }
    }

    std::cout << "[Train] Training complete. Saving final weights to checkpoint.bin...\n";
    model.save_parameters("checkpoint.bin");

    cudaFree(d_inputs);
    cudaFree(d_targets);
    return 0;
}
