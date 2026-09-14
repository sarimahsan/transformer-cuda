import argparse
import json
import os
import sys
import time
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pytorch_src.model import PyTorchGPT
from pytorch_src.gla_model import PyTorchGLA


def benchmark_config(
    model_cls,
    model_kwargs,
    batch_size: int,
    seq_len: int,
    warmup: int = 10,
    steps: int = 30,
    compile_model: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    model = model_cls(**model_kwargs).to(device)
    optimizer = model.configure_optimizers(lr=3e-4)

    if compile_model and device == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"[Warning] torch.compile failed: {e}. Falling back to eager.")

    vocab_size = model_kwargs.get("vocab_size", 65)
    x = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)

    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()

    if device == "cuda":
        torch.cuda.synchronize()

    # Measurement
    times = []
    use_cuda_events = (device == "cuda")
    if use_cuda_events:
        start_evt = torch.cuda.Event(enable_timing=True)
        stop_evt = torch.cuda.Event(enable_timing=True)

    for _ in range(steps):
        if use_cuda_events:
            start_evt.record()
            optimizer.zero_grad(set_to_none=True)
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            stop_evt.record()
            stop_evt.synchronize()
            times.append(start_evt.elapsed_time(stop_evt))
        else:
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            times.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(times))
    total_tokens = batch_size * seq_len
    tokens_per_sec = total_tokens / (mean_ms / 1000.0)
    return {"mean_ms": mean_ms, "tokens_per_sec": tokens_per_sec}


def main():
    parser = argparse.ArgumentParser(description="Sequence Length Scaling Benchmark Suite")
    parser.add_argument("--lengths", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--total_tokens", type=int, default=8192)
    parser.add_argument("--d_model", "-d", type=int, default=256)
    parser.add_argument("--num_layers", "-l", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Sequence Length Scaling Benchmark (Device: {device.upper()}) ===")
    print(f"Sweeping T in {args.lengths} with constant batch tokens = {args.total_tokens}")

    results = {"lengths": args.lengths, "gpt_compile": [], "fast_compile": []}

    for T in args.lengths:
        B = max(1, args.total_tokens // T)
        print(f"\n--- Testing Sequence Length T = {T} (Batch Size B = {B}) ---")

        # 1. Standard GPT-2 (compiled)
        gpt_kwargs = {
            "vocab_size": 65,
            "max_seq_len": T,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "d_ff": 4 * args.d_model
        }
        try:
            print("  Running Standard GPT-2 (torch.compile)...")
            gpt_res = benchmark_config(
                PyTorchGPT, gpt_kwargs, batch_size=B, seq_len=T,
                warmup=args.warmup, steps=args.steps, compile_model=True, device=device
            )
            print(f"    GPT-2: {gpt_res['mean_ms']:.2f} ms | {gpt_res['tokens_per_sec']:,.1f} tok/s")
            results["gpt_compile"].append(gpt_res)
        except Exception as e:
            print(f"    GPT-2 failed at T={T}: {e}")
            results["gpt_compile"].append(None)

        # 2. FastTransformer (compiled)
        fast_kwargs = {
            "vocab_size": 65,
            "max_seq_len": T,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "d_ff": 2 * args.d_model
        }
        try:
            print("  Running FastTransformer (torch.compile)...")
            fast_res = benchmark_config(
                PyTorchGLA, fast_kwargs, batch_size=B, seq_len=T,
                warmup=args.warmup, steps=args.steps, compile_model=True, device=device
            )
            print(f"    FastTransformer: {fast_res['mean_ms']:.2f} ms | {fast_res['tokens_per_sec']:,.1f} tok/s")
            results["fast_compile"].append(fast_res)
        except Exception as e:
            print(f"    FastTransformer failed at T={T}: {e}")
            results["fast_compile"].append(None)

    out_file = os.path.join(args.out_dir, "scaling_comparison.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Telemetry Saved] Scaling telemetry written to {out_file}")

    print("\n" + "=" * 80)
    print("SEQUENCE LENGTH SCALING SCORECARD")
    print("=" * 80)
    print(f"{'Seq Length (T)':<16} | {'Batch Size (B)':<16} | {'GPT-2 (tok/s)':<18} | {'FastTransformer':<18} | {'Speedup':<10}")
    print("-" * 80)
    for i, T in enumerate(args.lengths):
        B = max(1, args.total_tokens // T)
        gpt = results["gpt_compile"][i]
        fast = results["fast_compile"][i]
        gpt_str = f"{gpt['tokens_per_sec']:,.1f}" if gpt else "OOM/Fail"
        fast_str = f"{fast['tokens_per_sec']:,.1f}" if fast else "OOM/Fail"
        speedup = f"{fast['tokens_per_sec'] / gpt['tokens_per_sec']:.2f}x" if (gpt and fast) else "N/A"
        print(f"{T:<16} | {B:<16} | {gpt_str:<18} | {fast_str:<18} | {speedup:<10}")
    print("=" * 80)


if __name__ == "__main__":
    main()
