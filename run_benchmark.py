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
    parser.add_argument("--all", action="store_true", help="Run parity audit, benchmarks, and plot generation")
    args = parser.parse_args()

    if not (args.parity or args.benchmark or args.plot or args.all):
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
            res = subprocess.run([
                cuda_bin if os.path.exists(cuda_bin) else cuda_bin + ".exe",
                "--batch_size", str(args.batch_size),
                "--seq_len", str(args.seq_len),
                "--d_model", str(args.d_model),
                "--num_layers", str(args.num_layers),
                "--num_heads", str(args.num_heads),
                "--steps", str(args.steps),
                "--json"
            ], capture_output=True, text=True)
            if res.returncode == 0:
                os.makedirs("results", exist_ok=True)
                with open("results/cuda_benchmark.json", "w") as f:
                    f.write(res.stdout)
                print("[Benchmark] Pure CUDA telemetry saved to results/cuda_benchmark.json")

    if args.plot or args.all:
        print("\n>>> [3/3] GENERATING COMPARISON PLOTS & TABLES <<<")
        subprocess.run([sys.executable, "scripts/plot_comparisons.py"])


if __name__ == "__main__":
    main()
