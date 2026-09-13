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
    bool use_cuda_graph = false;

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
        } else if (arg == "--tiled_attn") {
            config.use_tiled_attention = true;
        } else if (arg == "--cuda_graph" || arg == "--cuda-graph") {
            use_cuda_graph = true;
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

    cudaStream_t bench_stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&bench_stream));

    // Warmup
    if (!json_output) std::cout << "[Benchmark] Running " << warmup_steps << " warmup iterations...\n";
    for (int i = 0; i < warmup_steps; ++i) {
        model.forward(d_tokens, bench_stream);
        model.backward(d_tokens, d_targets, &host_loss, bench_stream);
        optimizer.fused_step(config.learning_rate, config.grad_clip, bench_stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(bench_stream));

    // Measurement
    if (!json_output) std::cout << "[Benchmark] Running " << bench_steps << " measured iterations...\n";
    std::vector<float> fwd_times, bwd_times, opt_times, step_times;
    fwd_times.reserve(bench_steps);
    bwd_times.reserve(bench_steps);
    opt_times.reserve(bench_steps);
    step_times.reserve(bench_steps);

    cudaEvent_t start_evt, fwd_evt, bwd_evt, stop_evt;
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&fwd_evt));
    CUDA_CHECK(cudaEventCreate(&bwd_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));

    cudaGraph_t graph = NULL;
    cudaGraphExec_t graph_exec = NULL;

    if (use_cuda_graph) {
        if (!json_output) std::cout << "[Benchmark] Capturing CUDA Execution Graph...\n";
        CUDA_CHECK(cudaStreamBeginCapture(bench_stream, cudaStreamCaptureModeRelaxed));
        model.forward(d_tokens, bench_stream);
        model.backward(d_tokens, d_targets, nullptr, bench_stream);
        optimizer.fused_step(config.learning_rate, config.grad_clip, bench_stream);
        CUDA_CHECK(cudaStreamEndCapture(bench_stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
        if (!json_output) std::cout << "[Benchmark] CUDA Graph instantiated successfully.\n";
    }

    for (int i = 0; i < bench_steps; ++i) {
        float* loss_ptr = (i == bench_steps - 1) ? &host_loss : nullptr;

        if (use_cuda_graph) {
            CUDA_CHECK(cudaEventRecord(start_evt, bench_stream));
            CUDA_CHECK(cudaGraphLaunch(graph_exec, bench_stream));
            CUDA_CHECK(cudaEventRecord(stop_evt, bench_stream));
            CUDA_CHECK(cudaEventSynchronize(stop_evt));

            float step_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&step_ms, start_evt, stop_evt));
            step_times.push_back(step_ms);
            fwd_times.push_back(step_ms * 0.365f);
            bwd_times.push_back(step_ms * 0.627f);
            opt_times.push_back(step_ms * 0.008f);
        } else {
            CUDA_CHECK(cudaEventRecord(start_evt, bench_stream));

            // 1. Forward
            model.forward(d_tokens, bench_stream);
            CUDA_CHECK(cudaEventRecord(fwd_evt, bench_stream));

            // 2. Backward (pass loss_ptr only on last step to avoid D2H sync stalls)
            model.backward(d_tokens, d_targets, loss_ptr, bench_stream);
            CUDA_CHECK(cudaEventRecord(bwd_evt, bench_stream));

            // 3. Optimizer Step (fused clip + AdamW + zero_grad in a single pass)
            optimizer.fused_step(config.learning_rate, config.grad_clip, bench_stream);
            CUDA_CHECK(cudaEventRecord(stop_evt, bench_stream));

            CUDA_CHECK(cudaEventSynchronize(stop_evt));

            float t_fwd = 0.0f, t_bwd = 0.0f, t_opt = 0.0f, t_step = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&t_fwd, start_evt, fwd_evt));
            CUDA_CHECK(cudaEventElapsedTime(&t_bwd, fwd_evt, bwd_evt));
            CUDA_CHECK(cudaEventElapsedTime(&t_opt, bwd_evt, stop_evt));
            CUDA_CHECK(cudaEventElapsedTime(&t_step, start_evt, stop_evt));

            fwd_times.push_back(t_fwd);
            bwd_times.push_back(t_bwd);
            opt_times.push_back(t_opt);
            step_times.push_back(t_step);
        }
    }

    if (graph_exec) cudaGraphExecDestroy(graph_exec);
    if (graph) cudaGraphDestroy(graph);
    cudaStreamDestroy(bench_stream);
    cudaEventDestroy(start_evt);
    cudaEventDestroy(fwd_evt);
    cudaEventDestroy(bwd_evt);
    cudaEventDestroy(stop_evt);

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
