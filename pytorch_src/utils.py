import os
import torch
import numpy as np
from .model import PyTorchGPT


def export_weights_to_cuda(model: PyTorchGPT, output_path: str):
    """
    Exports PyTorch model weights to a single contiguous raw float32 binary file
    matching the memory mapping of TransformerModel in CUDA C++.
    """
    tensors = []

    # 1. Embeddings
    tensors.append(model.tok_emb.weight.detach().cpu().float().numpy())  # (V, C)
    tensors.append(model.pos_emb.weight.detach().cpu().float().numpy())  # (T, C)

    # 2. Layers
    for block in model.blocks:
        tensors.append(block.ln_1.weight.detach().cpu().float().numpy())  # (C,)
        tensors.append(block.ln_1.bias.detach().cpu().float().numpy())    # (C,)
        tensors.append(block.attn.qkv_proj.weight.detach().cpu().float().t().numpy())  # (C, 3 * C)
        tensors.append(block.attn.qkv_proj.bias.detach().cpu().float().numpy())        # (3 * C,)
        tensors.append(block.attn.out_proj.weight.detach().cpu().float().t().numpy())  # (C, C)
        tensors.append(block.attn.out_proj.bias.detach().cpu().float().numpy())        # (C,)

        tensors.append(block.ln_2.weight.detach().cpu().float().numpy())  # (C,)
        tensors.append(block.ln_2.bias.detach().cpu().float().numpy())    # (C,)
        tensors.append(block.ffn_1.weight.detach().cpu().float().t().numpy())  # (C, d_ff)
        tensors.append(block.ffn_1.bias.detach().cpu().float().numpy())        # (d_ff,)
        tensors.append(block.ffn_2.weight.detach().cpu().float().t().numpy())  # (d_ff, C)
        tensors.append(block.ffn_2.bias.detach().cpu().float().numpy())        # (C,)

    # 3. Final LN & Head
    tensors.append(model.ln_f.weight.detach().cpu().float().numpy())  # (C,)
    tensors.append(model.ln_f.bias.detach().cpu().float().numpy())    # (C,)
    tensors.append(model.head.weight.detach().cpu().float().t().numpy())  # (C, V)

    flattened = np.concatenate([t.flatten() for t in tensors]).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    flattened.tofile(output_path)
    return len(flattened)


def extract_pytorch_gradients(model: PyTorchGPT) -> np.ndarray:
    """
    Extracts all parameter gradients from PyTorch model in the contiguous
    memory layout matching TransformerModel in CUDA C++.
    """
    grads = []

    # 1. Embeddings
    grads.append(model.tok_emb.weight.grad.detach().cpu().float().numpy())
    grads.append(model.pos_emb.weight.grad.detach().cpu().float().numpy())

    # 2. Layers
    for block in model.blocks:
        grads.append(block.ln_1.weight.grad.detach().cpu().float().numpy())
        grads.append(block.ln_1.bias.grad.detach().cpu().float().numpy())
        grads.append(block.attn.qkv_proj.weight.grad.detach().cpu().float().t().numpy())
        grads.append(block.attn.qkv_proj.bias.grad.detach().cpu().float().numpy())
        grads.append(block.attn.out_proj.weight.grad.detach().cpu().float().t().numpy())
        grads.append(block.attn.out_proj.bias.grad.detach().cpu().float().numpy())

        grads.append(block.ln_2.weight.grad.detach().cpu().float().numpy())
        grads.append(block.ln_2.bias.grad.detach().cpu().float().numpy())
        grads.append(block.ffn_1.weight.grad.detach().cpu().float().t().numpy())
        grads.append(block.ffn_1.bias.grad.detach().cpu().float().numpy())
        grads.append(block.ffn_2.weight.grad.detach().cpu().float().t().numpy())
        grads.append(block.ffn_2.bias.grad.detach().cpu().float().numpy())

    # 3. Final LN & Head
    grads.append(model.ln_f.weight.grad.detach().cpu().float().numpy())
    grads.append(model.ln_f.bias.grad.detach().cpu().float().numpy())
    grads.append(model.head.weight.grad.detach().cpu().float().t().numpy())

    return np.concatenate([g.flatten() for g in grads]).astype(np.float32)


def generate_golden_reference(
    output_dir: str = "tests/parity_data",
    batch_size: int = 4,
    seq_len: int = 16,
    d_model: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    vocab_size: int = 65,
    seed: int = 42,
    device: str = "cpu"
):
    """
    Generates deterministic golden references from PyTorch autograd:
    inputs, targets, model weights, forward logits, loss, and backward gradients.
    """
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = PyTorchGPT(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=4 * d_model
    ).to(device)

    # Initialize weights deterministically
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(mean=0.0, std=0.02)
        for block in model.blocks:
            block.ln_1.weight.fill_(1.0)
            block.ln_1.bias.zero_()
            block.attn.qkv_proj.bias.zero_()
            block.attn.out_proj.bias.zero_()
            block.ln_2.weight.fill_(1.0)
            block.ln_2.bias.zero_()
            block.ffn_1.bias.zero_()
            block.ffn_2.bias.zero_()
        model.ln_f.weight.fill_(1.0)
        model.ln_f.bias.zero_()

    # Generate synthetic input & target tokens
    x = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)

    # Export configuration
    with open(os.path.join(output_dir, "config.txt"), "w") as f:
        f.write(f"{batch_size} {seq_len} {d_model} {num_layers} {num_heads} {vocab_size}\n")

    # Export weights
    weights_path = os.path.join(output_dir, "pytorch_params.bin")
    total_params = export_weights_to_cuda(model, weights_path)

    # Export inputs and targets (int32)
    x.cpu().numpy().astype(np.int32).tofile(os.path.join(output_dir, "inputs.bin"))
    y.cpu().numpy().astype(np.int32).tofile(os.path.join(output_dir, "targets.bin"))

    # Forward & Backward Pass
    logits, loss = model(x, y)
    loss.backward()

    # Save PyTorch golden outputs
    logits.detach().cpu().float().numpy().astype(np.float32).tofile(os.path.join(output_dir, "pytorch_logits.bin"))
    np.array([loss.item()], dtype=np.float32).tofile(os.path.join(output_dir, "pytorch_loss.bin"))

    pt_grads = extract_pytorch_gradients(model)
    pt_grads.tofile(os.path.join(output_dir, "pytorch_grads.bin"))

    print(f"[Reference Generator] Golden reference tensors created successfully in '{output_dir}'.")
    print(f"[Reference Generator] Parameters: {total_params} | Loss: {loss.item():.6f}")
    return model, loss.item()


if __name__ == "__main__":
    generate_golden_reference()
