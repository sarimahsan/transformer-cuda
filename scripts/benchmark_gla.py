import argparse
import json
import math
import os
import sys
import time
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pytorch_src.model import PyTorchGPT
from pytorch_src.gla_model import PyTorchGLA


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


def benchmark_single(
    model_name: str = "gla",
    mode: str = "compile",
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
    print(f"\n==================================================")
    print(f"[{model_name.upper()}] Mode: {mode.upper()} | Device: {device}")
    print(f"Config: B={batch_size}, T={seq_len}, C={d_model}, L={num_layers}, H={num_heads}")
    print(f"==================================================")

    total_tokens = batch_size * seq_len

    if model_name.lower() == "gpt":
        model = PyTorchGPT(
            vocab_size=vocab_size,
            max_seq_len=seq_len,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=4 * d_model
        ).to(device)
    elif model_name.lower() == "gla":
        model = PyTorchGLA(
            vocab_size=vocab_size,
            max_seq_len=seq_len,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=3 * d_model,
            chunk_size=64
        ).to(device)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {param_count:,}")

    optimizer = model.configure_optimizers(lr=3e-4)

    if mode == "compile":
        try:
            print("[Compiling] torch.compile(model, mode='reduce-overhead')...")
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"[Warning] torch.compile unavailable: {e}. Falling back to eager.")
            mode = "eager"

    # Synthetic batch
    x = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)

    # Warmup
    print(f"Running {warmup_steps} warmup iterations...")
    for _ in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()

    if device == "cuda":
        torch.cuda.synchronize()

    # Measurement
    print(f"Measuring {bench_steps} iterations...")
    fwd_times, bwd_times, opt_times, step_times = [], [], [], []
    use_cuda_events = (device == "cuda")

    if use_cuda_events:
        start_evt = torch.cuda.Event(enable_timing=True)
        fwd_evt = torch.cuda.Event(enable_timing=True)
        bwd_evt = torch.cuda.Event(enable_timing=True)
        stop_evt = torch.cuda.Event(enable_timing=True)

    for _ in range(bench_steps):
        if use_cuda_events:
            start_evt.record()
            optimizer.zero_grad(set_to_none=True)

            logits, loss = model(x, y)
            fwd_evt.record()

            loss.backward()
            bwd_evt.record()

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

            t_fwd_0 = time.perf_counter()
            logits, loss = model(x, y)
            t_fwd_1 = time.perf_counter()

            loss.backward()
            t_bwd_1 = time.perf_counter()

            optimizer.step()
            t_opt_1 = time.perf_counter()

            fwd_times.append((t_fwd_1 - t_fwd_0) * 1000.0)
            bwd_times.append((t_bwd_1 - t_fwd_1) * 1000.0)
            opt_times.append((t_opt_1 - t_bwd_1) * 1000.0)
            step_times.append((t_opt_1 - t0) * 1000.0)

    fwd_stats = compute_stats(fwd_times)
    bwd_stats = compute_stats(bwd_times)
    opt_stats = compute_stats(opt_times)
    step_stats = compute_stats(step_times)

    tok_per_sec = total_tokens / (step_stats["mean_ms"] / 1000.0)

    print(f"Results for [{model_name.upper()} - {mode.upper()}]:")
    print(f"  Forward:   {fwd_stats['mean_ms']:.2f} ms (±{fwd_stats['std_ms']:.2f})")
    print(f"  Backward:  {bwd_stats['mean_ms']:.2f} ms (±{bwd_stats['std_ms']:.2f})")
    print(f"  Optimizer: {opt_stats['mean_ms']:.2f} ms (±{opt_stats['std_ms']:.2f})")
    print(f"  Step:      {step_stats['mean_ms']:.2f} ms (±{step_stats['std_ms']:.2f})")
    print(f"  Throughput: {tok_per_sec:,.2f} tokens/sec")

    return {
        "model": model_name,
        "mode": mode,
        "param_count": param_count,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "tokens_per_sec": tok_per_sec,
        "forward": fwd_stats,
        "backward": bwd_stats,
        "optimizer": opt_stats,
        "step": step_stats
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Standard Transformer vs. Chunkwise GLA")
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--seq_len", "-t", type=int, default=256)
    parser.add_argument("--d_model", "-d", type=int, default=256)
    parser.add_argument("--num_layers", "-l", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {}

    configs = [
        ("gpt", "eager"),
        ("gpt", "compile"),
        ("gla", "eager"),
        ("gla", "compile"),
    ]

    for model_name, mode in configs:
        tag = f"{model_name}_{mode}"
        try:
            res = benchmark_single(
                model_name=model_name,
                mode=mode,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                d_model=args.d_model,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                warmup_steps=args.warmup,
                bench_steps=args.steps
            )
            summary[tag] = res
        except Exception as e:
            print(f"[Error] Failed running {tag}: {e}")

    out_file = os.path.join(args.out_dir, "gla_comparison.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Benchmark Complete] Telemetry written to {out_file}")

    print("\n" + "=" * 80)
    print("ARCHITECTURAL BENCHMARK SUMMARY SCORECARD")
    print("=" * 80)
    print(f"{'Configuration':<25} | {'Forward (ms)':<14} | {'Backward (ms)':<14} | {'Step (ms)':<12} | {'Tokens / Sec':<15}")
    print("-" * 80)
    for tag, res in summary.items():
        if res is not None:
            fwd = f"{res['forward']['mean_ms']:.2f}"
            bwd = f"{res['backward']['mean_ms']:.2f}"
            stp = f"{res['step']['mean_ms']:.2f}"
            tps = f"{res['tokens_per_sec']:,.1f}"
            print(f"{tag:<25} | {fwd:<14} | {bwd:<14} | {stp:<12} | {tps:<15}")
    print("=" * 80)


if __name__ == "__main__":
    main()
