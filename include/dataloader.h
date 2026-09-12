#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>

class DataLoader {
public:
    DataLoader(const std::string& filename, size_t batch_size, size_t seq_len);
    ~DataLoader();

    // Loads next (B, T) batch of input tokens and target tokens (shifted by 1)
    // Copies directly to GPU pointers d_inputs and d_targets
    bool next_batch(int* d_inputs, int* d_targets);
    void reset();

    size_t get_total_tokens() const { return total_tokens; }
    size_t get_num_batches() const { return num_batches; }

private:
    std::vector<int> tokens;
    size_t batch_size;
    size_t seq_len;
    size_t total_tokens;
    size_t num_batches;
    size_t current_batch_idx;

    int* h_inputs;
    int* h_targets;
};
