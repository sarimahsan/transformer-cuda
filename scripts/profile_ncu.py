#!/usr/bin/env python3
"""
profile_ncu.py: Automated Nsight Compute (ncu) kernel microarchitecture profiling script.
Inspects SM Throughput, Memory Throughput, Warp Occupancy, and Register limits
to identify why individual kernels take time.
"""

import argparse
import os
import shutil
import subprocess
import sys


def find_ncu():
    ncu_path = shutil.which("ncu")
    if not ncu_path:
        candidates = [
            "/usr/local/cuda/bin/ncu",
            "/opt/nvidia/nsight-compute/bin/ncu",
            "/usr/bin/ncu"
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
    return ncu_path


def main():
    parser = argparse.ArgumentParser(description="Nsight Compute Kernel Profiling")
    parser.add_argument("--kernel_regex", type=str, default=".*(layernorm|tiled_causal|add_bias|gelu|sgemm).*",
                        help="Regex pattern of kernel names to profile")
    parser.add_argument("--metrics", type=str, default="SpeedOfLight,Occupancy,LaunchStats",
                        help="NCU sections or metric sets (e.g. SpeedOfLight, MemoryWorkloadAnalysis, Occupancy)")
    parser.add_argument("--target", type=str, default="pure_cuda", choices=["pure_cuda", "pytorch"],
                        help="Target application to profile")
    parser.add_argument("--out_dir", type=str, default="results/nsight", help="Output directory")
    args = parser.parse_args()

    ncu_bin = find_ncu()
    if not ncu_bin:
        print("[Error] NVIDIA Nsight Compute CLI ('ncu') not found.")
        print("In Google Colab, install or verify with: !which ncu or /usr/local/cuda/bin/ncu")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    out_rep = os.path.join(args.out_dir, f"ncu_{args.target}")

    sections = [f"--section={s.strip()}" for s in args.metrics.split(",") if s.strip()]

    if args.target == "pure_cuda":
        bin_path = "./bin/benchmark" if os.path.exists("./bin/benchmark") else "bin/benchmark.exe"
        target_cmd = [bin_path, "-b", "32", "-t", "256", "-d", "256", "-l", "2", "-h", "8", "--steps", "1", "--tiled_attn"]
    else:
        target_cmd = [sys.executable, "scripts/benchmark_pytorch.py", "--mode", "compile", "--steps", "1", "--num_layers", "2"]

    cmd = [
        ncu_bin,
        "-k", f"regex:{args.kernel_regex}",
        "-o", out_rep,
        "--force-overwrite"
    ] + sections + target_cmd

    print(f"\n[NCU Runner] Launching Nsight Compute on {args.target}...")
    print(" ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)

    print(f"\n[NCU] Report saved to: {out_rep}.ncu-rep")
    print("Inspect in NVIDIA Nsight Compute GUI to view Roofline charts, memory bandwidth utilization, and stall analysis.")


if __name__ == "__main__":
    main()
