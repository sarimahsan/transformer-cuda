import os
import sys
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Master Benchmark & Parity Verification Runner")
    parser.add_argument("--parity", action="store_true", help="Run numerical parity audit against PyTorch autograd")
    parser.add_argument("--benchmark", action="store_true", help="Run multi-tier comparative benchmark suite")
    parser.add_argument("--plot", action="store_true", help="Generate publication comparison plots")
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--seq_len", "-t", type=int, default=256)
    parser.add_argument("--d_model", "-d", type=int, default=256)
    parser.add_argument("--num_layers", "-l", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--tiled_attn", action="store_true", help="Enable FlashAttention-style tiled online softmax")
    parser.add_argument("--cuda_graph", action="store_true", help="Enable CUDA Graph capture and replay")
    parser.add_argument("--gla", action="store_true", help="Run FastTransformer / GLA comparative benchmark")
    parser.add_argument("--convergence", action="store_true", help="Run language modeling convergence audit (GPT vs FastTransformer)")
    parser.add_argument("--scaling", action="store_true", help="Run sequence length scaling benchmark suite (T=128..1024)")
    parser.add_argument("--profile", action="store_true", help="Run automated Nsight Systems profiling suite")
    parser.add_argument("--all", action="store_true", help="Run parity audit, benchmarks, and plot generation")
    args = parser.parse_args()

    if not (args.parity or args.benchmark or args.plot or args.profile or args.gla or args.convergence or args.scaling or args.all):
        parser.print_help()
        print("\nDefaulting to running parity verification followed by comparison summary...\n")
        args.parity = True
        args.plot = True

    if args.parity or args.all:
        print("\n>>> [1/3] RUNNING NUMERICAL PARITY AUDIT <<<")
        subprocess.run([sys.executable, "scripts/compare_parity.py"])

    if args.benchmark or args.all:
        print("\n>>> [2/3] RUNNING PYTORCH BENCHMARKS <<<")
        subprocess.run([
            sys.executable, "scripts/benchmark_pytorch.py",
            "--batch_size", str(args.batch_size),
            "--seq_len", str(args.seq_len),
            "--d_model", str(args.d_model),
            "--num_layers", str(args.num_layers),
            "--num_heads", str(args.num_heads),
            "--steps", str(args.steps),
            "--mode", "all"
        ])

        cuda_bin = "bin/benchmark"
        if os.path.exists(cuda_bin) or os.path.exists(cuda_bin + ".exe"):
            print("\n>>> RUNNING PURE CUDA BENCHMARK <<<")
            cuda_cmd = [
                cuda_bin if os.path.exists(cuda_bin) else cuda_bin + ".exe",
                "--batch_size", str(args.batch_size),
                "--seq_len", str(args.seq_len),
                "--d_model", str(args.d_model),
                "--num_layers", str(args.num_layers),
                "--num_heads", str(args.num_heads),
                "--steps", str(args.steps),
                "--json"
            ]
            if args.tiled_attn:
                cuda_cmd.append("--tiled_attn")
            if args.cuda_graph:
                cuda_cmd.append("--cuda_graph")

            res = subprocess.run(cuda_cmd, capture_output=True, text=True)
            if res.returncode == 0:
                os.makedirs("results", exist_ok=True)
                with open("results/cuda_benchmark.json", "w") as f:
                    f.write(res.stdout)
                print("[Benchmark] Pure CUDA telemetry saved to results/cuda_benchmark.json")
 
    if args.gla or args.all:
        print("\n>>> RUNNING CHUNKWISE GLA COMPARATIVE BENCHMARK <<<")
        subprocess.run([
            sys.executable, "scripts/benchmark_gla.py",
            "--batch_size", str(args.batch_size),
            "--seq_len", str(args.seq_len),
            "--d_model", str(args.d_model),
            "--num_layers", str(args.num_layers),
            "--num_heads", str(args.num_heads),
            "--steps", str(args.steps)
        ])

    if args.convergence or args.all:
        print("\n>>> RUNNING CONVERGENCE & PERPLEXITY TRAINING AUDIT <<<")
        subprocess.run([
            sys.executable, "scripts/compare_convergence.py",
            "--batch_size", str(args.batch_size),
            "--seq_len", str(args.seq_len),
            "--d_model", str(args.d_model),
            "--num_layers", str(args.num_layers),
            "--num_heads", str(args.num_heads),
            "--steps", str(args.steps if args.steps >= 100 else 200)
        ])

    if args.scaling or args.all:
        print("\n>>> RUNNING SEQUENCE LENGTH SCALING BENCHMARK (T=128..1024) <<<")
        subprocess.run([
            sys.executable, "scripts/benchmark_scaling.py",
            "--lengths", "128", "256", "512", "1024",
            "--d_model", str(args.d_model),
            "--num_layers", str(args.num_layers),
            "--num_heads", str(args.num_heads),
            "--steps", str(args.steps if args.steps <= 30 else 30)
        ])

    if args.profile:
        print("\n>>> RUNNING NSIGHT SYSTEMS PROFILER <<<")
        profile_cmd = [
            sys.executable, "scripts/profile_nsys.py",
            "--steps", "5",
            "-b", str(args.batch_size),
            "-t", str(args.seq_len),
            "-d", str(args.d_model),
            "-l", str(args.num_layers),
            "-h_heads", str(args.num_heads)
        ]
        if args.tiled_attn:
            profile_cmd.append("--tiled_attn")
        subprocess.run(profile_cmd)

    if args.plot or args.all:
        print("\n>>> [3/3] GENERATING COMPARISON PLOTS & TABLES <<<")
        subprocess.run([sys.executable, "scripts/plot_comparisons.py"])


if __name__ == "__main__":
    main()
