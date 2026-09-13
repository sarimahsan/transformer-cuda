#!/usr/bin/env python3
"""
profile_torch.py: Profiling Transformer execution using PyTorch's built-in torch.profiler.
No external Nsight installation required. Works in Colab and local environments.
Generates:
1. Console summary table sorted by CUDA kernel execution time.
2. Breakdown by operator category (GEMM/cuBLAS, Attention, LayerNorm, Pointwise, Memory Copy).
3. Chrome trace JSON (viewable directly in chrome://tracing or https://ui.perfetto.dev).
"""

import argparse
import os
import sys
import torch
import torch.nn as nn

# Add parent directory to path to import pytorch_src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pytorch_src.model import PyTorchGPT


def profile_model(
    mode: str = "eager",
    batch_size: int = 32,
    seq_len: int = 256,
    d_model: int = 256,
    num_layers: int = 6,
    num_heads: int = 8,
    vocab_size: int = 65,
    warmup_steps: int = 2,
    active_steps: int = 3,
    out_dir: str = "results/torch_profile"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(f" Running torch.profiler | Mode: {mode.upper()} | Device: {device}")
    print(f" Config: B={batch_size}, T={seq_len}, C={d_model}, L={num_layers}, H={num_heads}")
    print("=" * 80)

    if device != "cuda":
        print("[Error] CUDA GPU is required for CUDA kernel profiling.")
        return

    os.makedirs(out_dir, exist_ok=True)

    # Initialize model
    model = PyTorchGPT(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=4 * d_model
    ).to(device)

    optimizer = model.configure_optimizers(lr=3e-4)

    if mode == "compile":
        print("[Profile] Compiling model with torch.compile...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"[Warning] torch.compile failed: {e}. Falling back to standard compile.")
            model = torch.compile(model)

    # Synthetic input batch
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Initial warmup to stabilize GPU clocks and JIT compilation
    print(f"[Profile] Running warmup iterations...")
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    trace_file = os.path.join(out_dir, f"trace_{mode}.json")

    print(f"[Profile] Starting torch.profiler capture ({active_steps} active steps)...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=warmup_steps, active=active_steps, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as prof:
        total_steps = 1 + warmup_steps + active_steps
        for step in range(total_steps):
            optimizer.zero_grad(set_to_none=True)
            with torch.profiler.record_function("## 1. FORWARD_PASS ##"):
                logits, loss = model(x, y)
            with torch.profiler.record_function("## 2. BACKWARD_PASS ##"):
                loss.backward()
            with torch.profiler.record_function("## 3. OPTIMIZER_STEP ##"):
                optimizer.step()
            torch.cuda.synchronize()
            prof.step()

    # Export Chrome trace
    try:
        prof.export_chrome_trace(trace_file)
        print(f"\n[Success] Chrome trace exported to: {trace_file}")
        print(f"          (Open in browser at: chrome://tracing or https://ui.perfetto.dev)\n")
    except Exception as e:
        print(f"\n[Notice] Trace export note: {e}\n")

    # Display Top CUDA kernels by total CUDA time
    print("=" * 100)
    print(f" TOP 20 CUDA KERNELS BY EXECUTION TIME ({mode.upper()})")
    print("=" * 100)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Display Top Operators by Self CUDA time
    print("=" * 100)
    print(f" TOP 15 OPERATORS BY SELF CUDA TIME ({mode.upper()})")
    print("=" * 100)
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))

    # Category Breakdown Analysis
    events = prof.key_averages()
    total_cuda_us = sum(e.cuda_time_total for e in events)

    categories = {
        "GEMM / Matrix Multiplication (cuBLAS / Ampere / Turing)": ["gemm", "sgemm", "cutlass", "cublas", "mm", "matmul", "linear", "bmm"],
        "Attention / Softmax": ["attention", "softmax", "baddbmm", "sdpa", "flash"],
        "Normalization (LayerNorm)": ["layernorm", "native_layer_norm", "norm"],
        "Pointwise / Activation / Residual (GELU, Add)": ["gelu", "add", "bias", "mul", "elementwise"],
        "Memory Transfers / Copies (DtoD, HtoD)": ["memcpy", "to_copy", "copy_"]
    }

    cat_times = {cat: 0.0 for cat in categories}
    uncat_time = 0.0

    for e in events:
        if e.cuda_time_total <= 0:
            continue
        matched = False
        name_lower = e.key.lower()
        for cat, keywords in categories.items():
            if any(kw in name_lower for kw in keywords):
                cat_times[cat] += e.cuda_time_total
                matched = True
                break
        if not matched:
            uncat_time += e.cuda_time_total

    print("\n" + "=" * 80)
    print(f" HIGH-LEVEL GPU TIME BREAKDOWN ({mode.upper()})")
    print("=" * 80)
    print(f"{'Category':<55} | {'CUDA Time (ms)':<15} | {'Share (%)'}")
    print("-" * 80)
    for cat, time_us in cat_times.items():
        time_ms = time_us / 1000.0
        pct = (time_us / max(total_cuda_us, 1.0)) * 100.0
        print(f"{cat:<55} | {time_ms:>13.2f} ms | {pct:>6.1f}%")
    if uncat_time > 0:
        print(f"{'Other / Uncategorized':<55} | {uncat_time / 1000.0:>13.2f} ms | {(uncat_time / max(total_cuda_us, 1.0)) * 100.0:>6.1f}%")
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Profile Transformer with torch.profiler")
    parser.add_argument("--mode", type=str, default="eager", choices=["eager", "compile", "all"],
                        help="Execution mode to profile: eager, compile, or all")
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--seq_len", "-t", type=int, default=256)
    parser.add_argument("--d_model", "-d", type=int, default=256)
    parser.add_argument("--num_layers", "-l", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--out_dir", type=str, default="results/torch_profile")
    args = parser.parse_args()

    modes = ["eager", "compile"] if args.mode == "all" else [args.mode]
    for m in modes:
        profile_model(
            mode=m,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            out_dir=args.out_dir
        )


if __name__ == "__main__":
    main()
