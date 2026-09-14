import argparse
import json
import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pytorch_src.model import PyTorchGPT
from pytorch_src.gla_model import PyTorchGLA  # FastTransformer


def load_tokens(data_path: str, vocab_size: int = 65):
    if not os.path.exists(data_path):
        print(f"[Warning] {data_path} not found. Generating synthetic dataset of 200,000 tokens.")
        return torch.tensor(np.random.randint(0, vocab_size, size=200000, dtype=np.int32), dtype=torch.long)
    
    file_size = os.path.getsize(data_path)
    if file_size % 4 == 0:
        tokens = np.fromfile(data_path, dtype=np.int32)
    else:
        tokens = np.fromfile(data_path, dtype=np.uint16).astype(np.int32)
    return torch.tensor(tokens, dtype=torch.long)


def train_model(
    model,
    model_name: str,
    tokens: torch.Tensor,
    batch_size: int = 32,
    seq_len: int = 256,
    steps: int = 200,
    lr: float = 3e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    compile_model: bool = True
):
    print(f"\n" + "=" * 60)
    print(f"Training [{model_name.upper()}] on {device.upper()} for {steps} steps")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {param_count:,}")
    print("=" * 60)

    model = model.to(device)
    optimizer = model.configure_optimizers(lr=lr)

    if compile_model and device == "cuda":
        try:
            print(f"Compiling {model_name} with torch.compile(mode='reduce-overhead')...")
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"torch.compile failed: {e}. Running in eager mode.")

    batch_tokens = batch_size * seq_len
    num_available_batches = (len(tokens) - 1) // batch_tokens

    step_history = []
    loss_history = []
    dt_history = []

    model.train()
    start_time = time.perf_counter()

    for step in range(1, steps + 1):
        batch_idx = (step - 1) % num_available_batches
        offset = batch_idx * batch_tokens

        x = tokens[offset : offset + batch_tokens].view(batch_size, seq_len).to(device)
        y = tokens[offset + 1 : offset + 1 + batch_tokens].view(batch_size, seq_len).to(device)

        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()

        dt_ms = (time.perf_counter() - t0) * 1000.0

        step_history.append(step)
        loss_history.append(float(loss.item()))
        dt_history.append(dt_ms)

        if step % 25 == 0 or step == 1:
            tok_per_sec = batch_tokens / (dt_ms / 1000.0) if dt_ms > 0 else 0.0
            perplexity = float(np.exp(loss.item()))
            print(f"Step {step:4d}/{steps} | Loss: {loss.item():.4f} | PPL: {perplexity:6.2f} | "
                  f"Step Time: {dt_ms:6.2f} ms | Throughput: {tok_per_sec:9.1f} tok/s")

    total_time = time.perf_counter() - start_time
    avg_step_ms = float(np.mean(dt_history[10:])) if len(dt_history) > 10 else float(np.mean(dt_history))
    avg_tok_s = batch_tokens / (avg_step_ms / 1000.0)

    print("-" * 60)
    print(f"[{model_name.upper()} Summary] Final Loss: {loss_history[-1]:.4f} | "
          f"Avg Step: {avg_step_ms:.2f} ms | Avg Throughput: {avg_tok_s:,.1f} tok/s")

    return {
        "model": model_name,
        "parameters": param_count,
        "final_loss": loss_history[-1],
        "final_ppl": float(np.exp(loss_history[-1])),
        "avg_step_ms": avg_step_ms,
        "avg_tokens_per_sec": avg_tok_s,
        "total_time_sec": total_time,
        "losses": loss_history[::10],  # Subsample for JSON compactness
        "steps": step_history[::10]
    }


def main():
    parser = argparse.ArgumentParser(description="Convergence & Perplexity Verification: GPT vs FastTransformer")
    parser.add_argument("--data", type=str, default="data/input.bin")
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--seq_len", "-t", type=int, default=256)
    parser.add_argument("--d_model", "-d", type=int, default=256)
    parser.add_argument("--num_layers", "-l", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tokens = load_tokens(args.data, vocab_size=65)

    # 1. Baseline GPT-2
    torch.manual_seed(42)
    gpt_model = PyTorchGPT(
        vocab_size=65,
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=4 * args.d_model
    )
    gpt_res = train_model(
        gpt_model,
        model_name="Standard_GPT2",
        tokens=tokens,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        steps=args.steps,
        lr=args.lr
    )

    # 2. FastTransformer (MQA + Fused SDPA + Lean MLP)
    torch.manual_seed(42)
    fast_model = PyTorchGLA(
        vocab_size=65,
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=2 * args.d_model
    )
    fast_res = train_model(
        fast_model,
        model_name="FastTransformer",
        tokens=tokens,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        steps=args.steps,
        lr=args.lr
    )

    out_file = os.path.join(args.out_dir, "convergence_comparison.json")
    with open(out_file, "w") as f:
        json.dump({"gpt2": gpt_res, "fast_transformer": fast_res}, f, indent=2)
    print(f"\n[Telemetry Saved] Convergence comparison written to {out_file}")

    print("\n" + "=" * 80)
    print("CONVERGENCE & PERPLEXITY SCORECARD")
    print("=" * 80)
    print(f"{'Architecture':<20} | {'Parameters':<12} | {'Final Loss':<12} | {'Perplexity':<12} | {'Step Time':<12} | {'Throughput':<15}")
    print("-" * 80)
    for res in [gpt_res, fast_res]:
        print(f"{res['model']:<20} | {res['parameters']:<12,} | {res['final_loss']:<12.4f} | "
              f"{res['final_ppl']:<12.2f} | {res['avg_step_ms']:<10.2f} ms | {res['avg_tokens_per_sec']:<12,.1f} tok/s")
    print("=" * 80)


if __name__ == "__main__":
    main()
