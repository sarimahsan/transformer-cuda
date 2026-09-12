#include "dataloader.h"
#include "common.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <cstdlib>

DataLoader::DataLoader(const std::string& filename, size_t batch_size, size_t seq_len)
    : batch_size(batch_size), seq_len(seq_len), current_batch_idx(0)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        // Generate synthetic random tokens if file does not exist
        std::cerr << "[DataLoader] Warning: " << filename << " not found. Generating synthetic tokens.\n";
        total_tokens = 100000;
        tokens.resize(total_tokens);
        for (size_t i = 0; i < total_tokens; ++i) {
            tokens[i] = rand() % 65;
        }
    } else {
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        // Assume uint16 tokens (or int32 if file_size is divisible by 4)
        if (file_size % 4 == 0) {
            total_tokens = file_size / 4;
            tokens.resize(total_tokens);
            file.read(reinterpret_cast<char*>(tokens.data()), file_size);
        } else {
            total_tokens = file_size / 2;
            std::vector<uint16_t> u16_tokens(total_tokens);
            file.read(reinterpret_cast<char*>(u16_tokens.data()), file_size);
            tokens.resize(total_tokens);
            for (size_t i = 0; i < total_tokens; ++i) {
                tokens[i] = static_cast<int>(u16_tokens[i]);
            }
        }
    }

    size_t tokens_per_batch = batch_size * seq_len;
    num_batches = (total_tokens - 1) / tokens_per_batch;

    h_inputs = new int[tokens_per_batch];
    h_targets = new int[tokens_per_batch];
}

DataLoader::~DataLoader() {
    delete[] h_inputs;
    delete[] h_targets;
}

void DataLoader::reset() {
    current_batch_idx = 0;
}

bool DataLoader::next_batch(int* d_inputs, int* d_targets) {
    size_t tokens_per_batch = batch_size * seq_len;
    if (current_batch_idx >= num_batches) {
        return false;
    }

    size_t offset = current_batch_idx * tokens_per_batch;
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            size_t idx = b * seq_len + t;
            size_t tok_pos = offset + idx;
            h_inputs[idx] = tokens[tok_pos];
            h_targets[idx] = tokens[tok_pos + 1];
        }
    }

    CUDA_CHECK(cudaMemcpy(d_inputs, h_inputs, tokens_per_batch * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets, tokens_per_batch * sizeof(int), cudaMemcpyHostToDevice));

    current_batch_idx++;
    return true;
}
