#include "common.h"
#include "config.h"
#include "model.h"
#include "optimizer.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <string>

struct TimingStats {
    float mean_ms;
    float median_ms;
    float min_ms;
    float max_ms;
    float std_ms;
};

TimingStats compute_stats(std::vector<float>& times) {
    if (times.empty()) return {0, 0, 0, 0, 0};
    std::sort(times.begin(), times.end());
    float sum = 0.0f;
    for (float t : times) sum += t;
    float mean = sum / times.size();
    float median = (times.size() % 2 == 0)
        ? (times[times.size() / 2 - 1] + times[times.size() / 2]) * 0.5f
        : times[times.size() / 2];
    float min_val = times.front();
    float max_val = times.back();

    float var = 0.0f;
    for (float t : times) {
        float diff = t - mean;
        var += diff * diff;
    }
    float std_val = std::sqrt(var / times.size());

    return {mean, median, min_val, max_val, std_val};
}

void print_help(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  -b, --batch_size   <int>   Batch size B (default: 32)\n"
              << "  -t, --seq_len      <int>   Sequence length T (default: 256)\n"
              << "  -d, --d_model      <int>   Hidden dimension d_model (default: 256)\n"
              << "  -l, --num_layers   <int>   Number of layers L (default: 6)\n"
              << "  -h, --num_heads    <int>   Number of attention heads H (default: 8)\n"
              << "      --d_ff         <int>   FFN dimension (default: 4 * d_model)\n"
              << "  -v, --vocab_size   <int>   Vocabulary size V (default: 65)\n"
              << "      --warmup       <int>   Warmup steps (default: 10)\n"
              << "      --steps        <int>   Benchmark measurement steps (default: 50)\n"
              << "      --json                 Emit machine-readable JSON metrics\n"
              << "      --help                 Display this help\n";
}

int main(int argc, char** argv) {
    TransformerConfig config;
    int warmup_steps = 10;
    int bench_steps = 50;
    bool json_output = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-b" || arg == "--batch_size") && i + 1 < argc) {
            config.batch_size = std::stoull(argv[++i]);
        } else if ((arg == "-t" || arg == "--seq_len") && i + 1 < argc) {
            config.max_seq_len = std::stoull(argv[++i]);
        } else if ((arg == "-d" || arg == "--d_model") && i + 1 < argc) {
            config.d_model = std::stoull(argv[++i]);
        } else if ((arg == "-l" || arg == "--num_layers") && i + 1 < argc) {
            config.num_layers = std::stoull(argv[++i]);
        } else if ((arg == "-h" || arg == "--num_heads") && i + 1 < argc) {
            config.num_heads = std::stoull(argv[++i]);
        } else if (arg == "--d_ff" && i + 1 < argc) {
            config.d_ff = std::stoull(argv[++i]);
        } else if ((arg == "-v" || arg == "--vocab_size") && i + 1 < argc) {
            config.vocab_size = std::stoull(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup_steps = std::stoi(argv[++i]);
        } else if (arg == "--steps" && i + 1 < argc) {
            bench_steps = std::stoi(argv[++i]);
        } else if (arg == "--json") {
            json_output = true;
        } else if (arg == "--help") {
            print_help(argv[0]);
            return 0;
        }
    }
    config.d_head = config.d_model / config.num_heads;
    if (config.d_ff == 0) config.d_ff = 4 * config.d_model;

    if (!json_output) {
        config.print();
        std::cout << "[Benchmark] Initializing Pure CUDA Transformer Model...\n";
    }

    TransformerModel model(config);
    AdamW optimizer(model.get_params_memory(), model.get_grads_memory(), model.get_num_parameters(), config);

    int B = static_cast<int>(config.batch_size);
    int T = static_cast<int>(config.max_seq_len);
    size_t total_tokens = B * T;

    // Allocate synthetic inputs and targets
    int* d_tokens;
    int* d_targets;
    CUDA_CHECK(cudaMalloc(&d_tokens, total_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_targets, total_tokens * sizeof(int)));

    std::vector<int> h_tokens(total_tokens);
    std::vector<int> h_targets(total_tokens);
    for (size_t i = 0; i < total_tokens; ++i) {
        h_tokens[i] = rand() % config.vocab_size;
        h_targets[i] = rand() % config.vocab_size;
    }
    CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), total_tokens * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets.data(), total_tokens * sizeof(int), cudaMemcpyHostToDevice));

    float host_loss = 0.0f;

    // Warmup
    if (!json_output) std::cout << "[Benchmark] Running " << warmup_steps << " warmup iterations...\n";
    for (int i = 0; i < warmup_steps; ++i) {
        model.forward(d_tokens);
        model.backward(d_tokens, d_targets, &host_loss);
        optimizer.clip_grad_norm(config.grad_clip);
        optimizer.step(config.learning_rate);
        optimizer.zero_grad();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measurement
    if (!json_output) std::cout << "[Benchmark] Running " << bench_steps << " measured iterations...\n";
    std::vector<float> fwd_times, bwd_times, opt_times, step_times;
    fwd_times.reserve(bench_steps);
    bwd_times.reserve(bench_steps);
    opt_times.reserve(bench_steps);
    step_times.reserve(bench_steps);

    GpuTimer timer_fwd, timer_bwd, timer_opt, timer_step;

    for (int i = 0; i < bench_steps; ++i) {
        timer_step.start();

        // 1. Forward
        timer_fwd.start();
        model.forward(d_tokens);
        timer_fwd.stop();

        // 2. Backward
        timer_bwd.start();
        model.backward(d_tokens, d_targets, &host_loss);
        timer_bwd.stop();

        // 3. Optimizer Step
        timer_opt.start();
        optimizer.clip_grad_norm(config.grad_clip);
        optimizer.step(config.learning_rate);
        optimizer.zero_grad();
        timer_opt.stop();

        timer_step.stop();

        fwd_times.push_back(timer_fwd.elapsed_ms());
        bwd_times.push_back(timer_bwd.elapsed_ms());
        opt_times.push_back(timer_opt.elapsed_ms());
        step_times.push_back(timer_step.elapsed_ms());
    }

    TimingStats s_fwd = compute_stats(fwd_times);
    TimingStats s_bwd = compute_stats(bwd_times);
    TimingStats s_opt = compute_stats(opt_times);
    TimingStats s_step = compute_stats(step_times);

    float tokens_per_sec = (s_step.mean_ms > 0) ? (total_tokens / (s_step.mean_ms / 1000.0f)) : 0.0f;

    if (json_output) {
        std::cout << "{\n"
                  << "  \"framework\": \"Pure CUDA\",\n"
                  << "  \"batch_size\": " << config.batch_size << ",\n"
                  << "  \"seq_len\": " << config.max_seq_len << ",\n"
                  << "  \"d_model\": " << config.d_model << ",\n"
                  << "  \"num_layers\": " << config.num_layers << ",\n"
                  << "  \"num_heads\": " << config.num_heads << ",\n"
                  << "  \"tokens_per_sec\": " << std::fixed << std::setprecision(2) << tokens_per_sec << ",\n"
                  << "  \"forward\": {\"mean_ms\": " << s_fwd.mean_ms << ", \"std_ms\": " << s_fwd.std_ms << ", \"median_ms\": " << s_fwd.median_ms << "},\n"
                  << "  \"backward\": {\"mean_ms\": " << s_bwd.mean_ms << ", \"std_ms\": " << s_bwd.std_ms << ", \"median_ms\": " << s_bwd.median_ms << "},\n"
                  << "  \"optimizer\": {\"mean_ms\": " << s_opt.mean_ms << ", \"std_ms\": " << s_opt.std_ms << ", \"median_ms\": " << s_opt.median_ms << "},\n"
                  << "  \"step\": {\"mean_ms\": " << s_step.mean_ms << ", \"std_ms\": " << s_step.std_ms << ", \"median_ms\": " << s_step.median_ms << "}\n"
                  << "}\n";
    } else {
        std::cout << "\n======================================================================\n";
        std::cout << " Pure CUDA Transformer Benchmark Results\n";
        std::cout << "======================================================================\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << " Phase          | Mean (ms) | Median (ms) | Std (ms)  | Min (ms)  | Max (ms)  \n";
        std::cout << "----------------+-----------+-------------+-----------+-----------+-----------\n";
        std::cout << " Forward        | " << std::setw(9) << s_fwd.mean_ms  << " | " << std::setw(11) << s_fwd.median_ms  << " | " << std::setw(9) << s_fwd.std_ms  << " | " << std::setw(9) << s_fwd.min_ms  << " | " << std::setw(9) << s_fwd.max_ms  << "\n";
        std::cout << " Backward       | " << std::setw(9) << s_bwd.mean_ms  << " | " << std::setw(11) << s_bwd.median_ms  << " | " << std::setw(9) << s_bwd.std_ms  << " | " << std::setw(9) << s_bwd.min_ms  << " | " << std::setw(9) << s_bwd.max_ms  << "\n";
        std::cout << " AdamW Step     | " << std::setw(9) << s_opt.mean_ms  << " | " << std::setw(11) << s_opt.median_ms  << " | " << std::setw(9) << s_opt.std_ms  << " | " << std::setw(9) << s_opt.min_ms  << " | " << std::setw(9) << s_opt.max_ms  << "\n";
        std::cout << " Full Step      | " << std::setw(9) << s_step.mean_ms << " | " << std::setw(11) << s_step.median_ms << " | " << std::setw(9) << s_step.std_ms << " | " << std::setw(9) << s_step.min_ms << " | " << std::setw(9) << s_step.max_ms << "\n";
        std::cout << "----------------+-----------+-------------+-----------+-----------+-----------\n";
        std::cout << " Throughput     : " << std::setprecision(1) << tokens_per_sec << " tokens/sec\n";
        std::cout << " Final Loss     : " << std::setprecision(5) << host_loss << "\n";
        std::cout << "======================================================================\n\n";
    }

    cudaFree(d_tokens);
    cudaFree(d_targets);
    return 0;
}
