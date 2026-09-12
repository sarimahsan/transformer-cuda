import argparse
import time
import os
import torch
import torch.nn as nn
import numpy as np
from .model import PyTorchGPT


def train_pytorch(
    data_path: str = "data/input.bin",
    batch_size: int = 32,
    seq_len: int = 256,
    d_model: int = 256,
    num_layers: int = 6,
    num_heads: int = 8,
    lr: float = 3e-4,
    epochs: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    log_interval: int = 10
):
    print(f"[PyTorch Train] Device: {device}")
    vocab_size = 65

    # Load dataset or generate synthetic
    if not os.path.exists(data_path):
        print(f"[PyTorch Train] Warning: {data_path} not found. Generating synthetic dataset.")
        tokens = np.random.randint(0, vocab_size, size=100000, dtype=np.int32)
    else:
        file_size = os.path.getsize(data_path)
        if file_size % 4 == 0:
            tokens = np.fromfile(data_path, dtype=np.int32)
        else:
            tokens = np.fromfile(data_path, dtype=np.uint16).astype(np.int32)

    tokens = torch.tensor(tokens, dtype=torch.long)
    num_batches = (len(tokens) - 1) // (batch_size * seq_len)

    model = PyTorchGPT(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=4 * d_model
    ).to(device)

    optimizer = model.configure_optimizers(lr=lr)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[PyTorch Train] Total Parameters: {total_params} | Batches/epoch: {num_batches}")

    step = 0
    running_loss = 0.0
    batch_tokens = batch_size * seq_len

    for epoch in range(epochs):
        for b in range(num_batches):
            t0 = time.perf_counter()

            offset = b * batch_tokens
            x = tokens[offset : offset + batch_tokens].view(batch_size, seq_len).to(device)
            y = tokens[offset + 1 : offset + 1 + batch_tokens].view(batch_size, seq_len).to(device)

            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if device == "cuda":
                torch.cuda.synchronize()

            dt_ms = (time.perf_counter() - t0) * 1000.0
            running_loss += loss.item()
            step += 1

            if step % log_interval == 0:
                avg_loss = running_loss / log_interval
                tok_per_sec = batch_tokens / (dt_ms / 1000.0) if dt_ms > 0 else 0.0
                print(f"Epoch [{epoch + 1}/{epochs}] Step {step:5d} | "
                      f"Loss: {avg_loss:.4f} | Step: {dt_ms:.2f} ms | "
                      f"Throughput: {tok_per_sec:.1f} tok/s")
                running_loss = 0.0

    print("[PyTorch Train] Training completed.")
    torch.save(model.state_dict(), "pytorch_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch GPT Training Script")
    parser.add_argument("--data", type=str, default="data/input.bin")
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--seq_len", "-t", type=int, default=256)
    parser.add_argument("--d_model", "-d", type=int, default=256)
    parser.add_argument("--num_layers", "-l", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    train_pytorch(
        data_path=args.data,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        lr=args.lr,
        epochs=args.epochs
    )
