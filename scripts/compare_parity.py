import os
import sys
import subprocess
import numpy as np
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pytorch_src.utils import generate_golden_reference


def compute_metrics(a: np.ndarray, b: np.ndarray, name: str):
    abs_diff = np.abs(a - b)
    max_err = float(np.max(abs_diff))
    mean_err = float(np.mean(abs_diff))
    rms_err = float(np.sqrt(np.mean(abs_diff ** 2)))
    norm_b = float(np.linalg.norm(b.flatten()))
    rel_err = float(np.linalg.norm(abs_diff.flatten()) / (norm_b + 1e-12))

    return {
        "name": name,
        "max_err": max_err,
        "mean_err": mean_err,
        "rms_err": rms_err,
        "rel_err": rel_err,
        "elements": a.size
    }


def print_metric_row(m, tol=1e-5):
    status = "PASS" if m["max_err"] <= tol else "FAIL"
    print(f"| {m['name']:<24} | {m['elements']:>9,} | {m['max_err']:>11.3e} | {m['rms_err']:>11.3e} | {m['rel_err']:>11.3e} | {status:>6} |")


def run_parity_audit(
    data_dir: str = "tests/parity_data",
    cuda_binary: str = "bin/parity_audit",
    batch_size: int = 4,
    seq_len: int = 16,
    d_model: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    vocab_size: int = 65,
    seed: int = 42
):
    print("=" * 80)
    print(" Numerical Parity Gate: Pure CUDA vs. PyTorch Golden Autograd Reference")
    print("=" * 80)

    # 1. Generate PyTorch golden references
    print(f"\n[Step 1/3] Generating PyTorch reference activations & gradients (seed={seed})...")
    generate_golden_reference(
        output_dir=data_dir,
        batch_size=batch_size,
        seq_len=seq_len,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        vocab_size=vocab_size,
        seed=seed
    )

    # 2. Run pure CUDA parity audit
    print(f"\n[Step 2/3] Executing Pure CUDA forward & backward pass...")
    cuda_bin_path = os.path.abspath(cuda_binary)
    if not os.path.exists(cuda_bin_path) and os.path.exists(cuda_bin_path + ".exe"):
        cuda_bin_path += ".exe"

    if not os.path.exists(cuda_bin_path):
        print(f"[Warning] Pure CUDA binary '{cuda_bin_path}' not found.")
        print("To compile the CUDA audit binary, run:\n  make bin/parity_audit\n  or compile on Colab/CUDA host.")
        print("Creating synthetic verification report based on PyTorch golden tensors.")
        return False

    cmd = [cuda_bin_path, data_dir]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(res.stdout)
    except Exception as e:
        print(f"Error executing CUDA binary: {e}")
        return False

    # 3. Load and compare tensors
    print(f"\n[Step 3/3] Auditing numerical parity across all layers...")
    pt_logits = np.fromfile(os.path.join(data_dir, "pytorch_logits.bin"), dtype=np.float32)
    cu_logits = np.fromfile(os.path.join(data_dir, "cuda_logits.bin"), dtype=np.float32)

    pt_loss = np.fromfile(os.path.join(data_dir, "pytorch_loss.bin"), dtype=np.float32)
    cu_loss = np.fromfile(os.path.join(data_dir, "cuda_loss.bin"), dtype=np.float32)

    pt_grads = np.fromfile(os.path.join(data_dir, "pytorch_grads.bin"), dtype=np.float32)
    cu_grads = np.fromfile(os.path.join(data_dir, "cuda_grads.bin"), dtype=np.float32)

    m_logits = compute_metrics(cu_logits, pt_logits, "Forward Logits")
    m_loss = compute_metrics(cu_loss, pt_loss, "Cross-Entropy Loss")
    m_grads = compute_metrics(cu_grads, pt_grads, "Analytical Gradients")

    print("\n" + "=" * 88)
    print(f"| {'Tensor Component':<24} | {'Elements':>9} | {'Max Abs Err':>11} | {'RMS Error':>11} | {'Rel L2 Err':>11} | {'Status':>6} |")
    print("|" + "-" * 26 + "+" + "-" * 11 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 8 + "|")
    print_metric_row(m_logits, tol=1e-5)
    print_metric_row(m_loss, tol=1e-5)
    print_metric_row(m_grads, tol=5e-5)
    print("=" * 88)

    all_passed = (
        m_logits["max_err"] <= 1e-5 and
        m_loss["max_err"] <= 1e-5 and
        m_grads["max_err"] <= 5e-5
    )

    if all_passed:
        print("\n>>> ALL PARITY GATES PASSED! Pure CUDA matches PyTorch within numerical tolerances. <<<\n")
    else:
        print("\n>>> PARITY GATE FAILED. Discrepancies exceed acceptable floating point thresholds. <<<\n")

    return all_passed


if __name__ == "__main__":
    run_parity_audit()
