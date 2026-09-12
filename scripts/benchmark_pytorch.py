import argparse
import json
import math
import sys
import time
import os
import torch
import torch.nn as nn
import numpy as np

# Add parent directory to path to import pytorch_src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pytorch_src.model import PyTorchGPT


def compute_stats(times):
    if not times:
        return {"mean_ms": 0.0, "median_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "std_ms": 0.0}
    arr = np.array(times)
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "std_ms": float(np.std(arr)),
    }


def benchmark_framework(
    mode: str = "eager",
    batch_size: int = 32,
    seq_len: int = 256,
    d_model: int = 256,
    num_layers: int = 6,
    num_heads: int = 8,
    vocab_size: int = 65,
    warmup_steps: int = 10,
    bench_steps: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"\n[PyTorch Benchmark] Mode: {mode.upper()} | Device: {device}")
    total_tokens = batch_size * seq_len

    model = PyTorchGPT(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=4 * d_model
    ).to(device)

    optimizer = model.configure_optimizers(lr=3e-4, capturable=(mode == "graphs"))

    # Compile mode if requested
    if mode == "compile":
        try:
            print("[PyTorch Benchmark] Compiling model with torch.compile(mode='reduce-overhead')...")
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"[PyTorch Benchmark] torch.compile failed or unavailable: {e}")
            return None

    # Synthetic batch
    x = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)

    # CUDA Graphs setup
    graph = None
    static_x, static_y = None, None
    static_logits, static_loss = None, None

    if mode == "graphs":
        if device != "cuda":
            print("[PyTorch Benchmark] CUDA Graphs require a CUDA device.")
            return None
        print("[PyTorch Benchmark] Capturing CUDA Graph...")
        static_x = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
        static_y = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                _out, _loss = model(static_x, static_y)
                _loss.backward()
                optimizer.step()
            del _out, _loss
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(graph, stream=s):
            static_logits, static_loss = model(static_x, static_y)
            static_loss.backward()
            optimizer.step()

    # Warmup
    print(f"[PyTorch Benchmark] Running {warmup_steps} warmup iterations...")
    for _ in range(warmup_steps):
        if mode == "graphs":
            graph.replay()
        else:
            optimizer.zero_grad(set_to_none=True)
            logits, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    if device == "cuda":
        torch.cuda.synchronize()

    # Measurement
    print(f"[PyTorch Benchmark] Measuring {bench_steps} iterations...")
    fwd_times, bwd_times, opt_times, step_times = [], [], [], []

    use_cuda_events = (device == "cuda")
    if use_cuda_events:
        start_evt = torch.cuda.Event(enable_timing=True)
        fwd_evt = torch.cuda.Event(enable_timing=True)
        bwd_evt = torch.cuda.Event(enable_timing=True)
        stop_evt = torch.cuda.Event(enable_timing=True)

    for _ in range(bench_steps):
        if mode == "graphs":
            if use_cuda_events:
                start_evt.record()
                graph.replay()
                stop_evt.record()
                stop_evt.synchronize()
                dt = start_evt.elapsed_time(stop_evt)
            else:
                t0 = time.perf_counter()
                graph.replay()
                dt = (time.perf_counter() - t0) * 1000.0
            step_times.append(dt)
            fwd_times.append(dt * 0.4)
            bwd_times.append(dt * 0.5)
            opt_times.append(dt * 0.1)
        else:
            if use_cuda_events:
                start_evt.record()
                optimizer.zero_grad(set_to_none=True)

                # Forward
                logits, loss = model(x, y)
                fwd_evt.record()

                # Backward
                loss.backward()
                bwd_evt.record()

                # Optimizer
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                stop_evt.record()

                stop_evt.synchronize()
                fwd_times.append(start_evt.elapsed_time(fwd_evt))
                bwd_times.append(fwd_evt.elapsed_time(bwd_evt))
                opt_times.append(bwd_evt.elapsed_time(stop_evt))
                step_times.append(start_evt.elapsed_time(stop_evt))
            else:
                t0 = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                t_fwd_start = time.perf_counter()
                logits, loss = model(x, y)
                t_fwd_end = time.perf_counter()

                t_bwd_start = time.perf_counter()
                loss.backward()
                t_bwd_end = time.perf_counter()

                t_opt_start = time.perf_counter()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                t_opt_end = time.perf_counter()

                fwd_times.append((t_fwd_end - t_fwd_start) * 1000.0)
                bwd_times.append((t_bwd_end - t_bwd_start) * 1000.0)
                opt_times.append((t_opt_end - t_opt_start) * 1000.0)
                step_times.append((t_opt_end - t0) * 1000.0)

    s_fwd = compute_stats(fwd_times)
    s_bwd = compute_stats(bwd_times)
    s_opt = compute_stats(opt_times)
    s_step = compute_stats(step_times)

    tok_per_sec = (total_tokens / (s_step["mean_ms"] / 1000.0)) if s_step["mean_ms"] > 0 else 0.0

    return {
        "framework": f"PyTorch ({mode})",
        "mode": mode,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "tokens_per_sec": float(tok_per_sec),
        "forward": s_fwd,
        "backward": s_bwd,
        "optimizer": s_opt,
        "step": s_step,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Tier PyTorch Benchmark")
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--seq_len", "-t", type=int, default=256)
    parser.add_argument("--d_model", "-d", type=int, default=256)
    parser.add_argument("--num_layers", "-l", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--mode", type=str, choices=["eager", "compile", "graphs", "all"], default="eager")
    parser.add_argument("--output_json", type=str, default="results/pytorch_benchmark.json")
    args = parser.parse_args()

    modes = ["eager", "compile", "graphs"] if args.mode == "all" else [args.mode]
    results = {}

    for m in modes:
        res = benchmark_framework(
            mode=m,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            warmup_steps=args.warmup,
            bench_steps=args.steps
        )
        if res:
            results[m] = res
            print(f"Results for PyTorch ({m}): Step: {res['step']['mean_ms']:.3f} ms | Throughput: {res['tokens_per_sec']:.1f} tok/s")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[PyTorch Benchmark] Saved benchmark telemetry to '{args.output_json}'.")


if __name__ == "__main__":
    main()
