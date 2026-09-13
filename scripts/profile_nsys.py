#!/usr/bin/env python3
"""
profile_nsys.py: Automated Nsight Systems (nsys) profiling and comparative diagnostic suite.
Profiles Pure CUDA vs. PyTorch Multi-Tier Transformers, extracts GPU kernel metrics,
and analyzes why specific kernels take more time.
"""

import argparse
import os
import shutil
import subprocess
import sys
import re


def find_nsys():
    nsys_path = shutil.which("nsys")
    if not nsys_path:
        # Check standard Linux/CUDA install paths
        candidates = [
            "/usr/local/cuda/bin/nsys",
            "/opt/nvidia/nsight-systems/bin/nsys",
            "/usr/bin/nsys"
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
    return nsys_path


def run_command(cmd, desc="Running"):
    print(f"\n[NSYS Runner] {desc}:")
    print(" ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        print(f"[Warning] Command returned non-zero exit code {res.returncode}:\n{res.stdout}")
    return res.stdout


def profile_pure_cuda(nsys_bin, out_prefix, steps, batch_size, seq_len, d_model, num_layers, num_heads, tiled_attn):
    bin_path = "./bin/benchmark"
    if not os.path.exists(bin_path) and os.path.exists("bin/benchmark.exe"):
        bin_path = "bin/benchmark.exe"

    if not os.path.exists(bin_path):
        print(f"[Error] Benchmark binary {bin_path} not found. Please compile first ('make' or 'cmake').")
        return None

    cmd = [
        nsys_bin, "profile",
        "-t", "cuda,nvtx,osrt",
        "-o", f"{out_prefix}_pure_cuda",
        "--force-overwrite=true",
        bin_path,
        "-b", str(batch_size),
        "-t", str(seq_len),
        "-d", str(d_model),
        "-l", str(num_layers),
        "-h", str(num_heads),
        "--warmup", "3",
        "--steps", str(steps)
    ]
    if tiled_attn:
        cmd.append("--tiled_attn")

    run_command(cmd, f"Profiling Pure CUDA Engine (Tiled Attn: {tiled_attn})")
    rep_file = f"{out_prefix}_pure_cuda.nsys-rep"
    return rep_file if os.path.exists(rep_file) else None


def profile_pytorch(nsys_bin, out_prefix, mode, steps, batch_size, seq_len, d_model, num_layers, num_heads):
    cmd = [
        nsys_bin, "profile",
        "-t", "cuda,nvtx,osrt",
        "-o", f"{out_prefix}_pytorch_{mode}",
        "--force-overwrite=true",
        sys.executable, "scripts/benchmark_pytorch.py",
        "--mode", mode,
        "--batch_size", str(batch_size),
        "--seq_len", str(seq_len),
        "--d_model", str(d_model),
        "--num_layers", str(num_layers),
        "--num_heads", str(num_heads),
        "--warmup", "3",
        "--steps", str(steps)
    ]
    run_command(cmd, f"Profiling PyTorch {mode.upper()}")
    rep_file = f"{out_prefix}_pytorch_{mode}.nsys-rep"
    return rep_file if os.path.exists(rep_file) else None


def extract_kernel_summary(nsys_bin, rep_file):
    if not rep_file or not os.path.exists(rep_file):
        return ""
    cmd = [nsys_bin, "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", rep_file]
    out = run_command(cmd, f"Extracting kernel summary for {rep_file}")
    return out


def parse_kernel_csv(csv_text, max_rows=10):
    lines = csv_text.strip().split("\n")
    data_rows = []
    header = None

    for line in lines:
        if "Time (%)" in line or "Total Time (ns)" in line:
            header = [c.strip().strip('"') for c in line.split(",")]
            continue
        if header and line.strip() and not line.startswith("#"):
            parts = [p.strip().strip('"') for p in line.split(",")]
            if len(parts) >= len(header):
                data_rows.append(parts)

    return header, data_rows[:max_rows]


def main():
    parser = argparse.ArgumentParser(description="Automated Nsight Systems Profiling for Pure CUDA vs PyTorch")
    parser.add_argument("--steps", type=int, default=5, help="Number of benchmark measurement steps")
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-t", "--seq_len", type=int, default=256)
    parser.add_argument("-d", "--d_model", type=int, default=256)
    parser.add_argument("-l", "--num_layers", type=int, default=6)
    parser.add_argument("-h_heads", "--num_heads", type=int, default=8)
    parser.add_argument("--tiled_attn", action="store_true", help="Enable FlashAttention-style tiled online softmax")
    parser.add_argument("--pytorch_modes", nargs="+", default=["eager", "compile"], help="PyTorch modes to profile")
    parser.add_argument("--out_dir", type=str, default="results/nsight", help="Directory to save profile reports")
    args = parser.parse_args()

    nsys_bin = find_nsys()
    if not nsys_bin:
        print("[Error] NVIDIA Nsight Systems CLI ('nsys') was not found in PATH or standard directories.")
        print("Please ensure CUDA Toolkit with Nsight Systems is installed.")
        print("In Google Colab, nsys is preinstalled under /usr/local/cuda/bin/nsys.")
        sys.exit(1)

    print(f"[NSYS] Found nsys binary at: {nsys_bin}")
    os.makedirs(args.out_dir, exist_ok=True)
    out_prefix = os.path.join(args.out_dir, "profile")

    # 1. Profile Pure CUDA
    cuda_rep = profile_pure_cuda(
        nsys_bin, out_prefix, args.steps,
        args.batch_size, args.seq_len, args.d_model, args.num_layers, args.num_heads, args.tiled_attn
    )

    # 2. Profile PyTorch Modes
    pt_reps = {}
    for mode in args.pytorch_modes:
        pt_reps[mode] = profile_pytorch(
            nsys_bin, out_prefix, mode, args.steps,
            args.batch_size, args.seq_len, args.d_model, args.num_layers, args.num_heads
        )

    # 3. Generate Comparative Report
    report_md_path = os.path.join(args.out_dir, "nsys_kernel_breakdown.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Nsight Systems Comparative Kernel Diagnostic Report\n\n")
        f.write(f"- Configuration: $B={args.batch_size}, T={args.seq_len}, C={args.d_model}, L={args.num_layers}, H={args.num_heads}$\n")
        f.write(f"- Pure CUDA Tiled Attention: **{args.tiled_attn}**\n\n")

        # Pure CUDA Kernels
        f.write("## 1. Pure CUDA Top GPU Kernels\n\n")
        cuda_csv = extract_kernel_summary(nsys_bin, cuda_rep)
        header, rows = parse_kernel_csv(cuda_csv)
        if header and rows:
            f.write("| " + " | ".join(header[:6]) + " |\n")
            f.write("| " + " | ".join(["---"] * 6) + " |\n")
            for r in rows:
                f.write("| " + " | ".join(r[:6]) + " |\n")
            f.write("\n")
        else:
            f.write("*(Summary could not be extracted directly; inspect .nsys-rep in Nsight Systems GUI)*\n\n")

        # PyTorch Modes
        for mode, rep in pt_reps.items():
            f.write(f"## 2. PyTorch {mode.upper()} Top GPU Kernels\n\n")
            pt_csv = extract_kernel_summary(nsys_bin, rep)
            p_header, p_rows = parse_kernel_csv(pt_csv)
            if p_header and p_rows:
                f.write("| " + " | ".join(p_header[:6]) + " |\n")
                f.write("| " + " | ".join(["---"] * 6) + " |\n")
                for r in p_rows:
                    f.write("| " + " | ".join(r[:6]) + " |\n")
                f.write("\n")

        f.write("## 3. Diagnostic Takeaways\n\n")
        f.write("1. **Kernel Dispatches**: Notice total kernel calls. PyTorch Eager executes dozens of discrete launches, whereas Pure CUDA v2 with fused epilogues cuts launches significantly.\n")
        f.write("2. **GEMM Saturation**: Verify whether `sgemm` dominates runtime or whether pointwise ops (`add_bias`, `gelu`) are taking non-negligible percentages.\n")
        f.write("3. **Tiled Attention**: Inspect whether intermediate attention score read/writes are completely absent under tiled attention.\n")

    print(f"\n[NSYS] Profiling complete! Analysis report saved to:\n  {os.path.abspath(report_md_path)}")
    print(f"Profile traces (.nsys-rep) can be opened in the NVIDIA Nsight Systems desktop GUI.")


if __name__ == "__main__":
    main()
