# Milestone Achieved: FastTransformer Surpasses `torch.compile` at $110{,}713\text{ tok/s}$!

---

## 1. Master Empirical Scorecard (Tesla T4, FP32)

Here is the empirical scorecard from your Google Colab run:

| Architecture / Execution Tier | Forward ($\tau_{\text{fwd}}$) | Backward ($\tau_{\text{bwd}}$) | Optimizer ($\tau_{\text{opt}}$) | Step Latency ($\tau_{\text{step}}$) | Token Throughput ($\text{tok/s}$) | Performance vs. `gpt_compile` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`gpt_eager` (Standard GPT-2)** | $47.28\text{ ms}$ | $79.59\text{ ms}$ | $1.66\text{ ms}$ | $128.53\text{ ms}$ | $63{,}738.2$ | Baseline |
| **`gpt_compile` (Inductor / Triton)** | $38.93\text{ ms}$ | $67.09\text{ ms}$ | $1.65\text{ ms}$ | $107.67\text{ ms}$ | $76{,}085.3$ | Baseline Compiler Ceiling |
| **`FastTransformer_eager`** | **$38.56\text{ ms}$** | **$60.35\text{ ms}$** | **$1.34\text{ ms}$** | **$100.25\text{ ms}$** | **$81{,}716.6$** | **Beats `gpt_compile` in Eager!** |
| **`FastTransformer_compile`** | **$\mathbf{26.82\text{ ms}}$** | **$\mathbf{45.71\text{ ms}}$** | **$\mathbf{1.45\text{ ms}}$** | **$\mathbf{73.99\text{ ms}}$** | **$\mathbf{110{,}713.0}$** | **$\mathbf{1.46\times \text{ Speedup (+45.5\%)}}$** |

```
[Throughput Comparison: tokens / sec (Tesla T4)]

gpt_eager                [====================] 63,738 tok/s
gpt_compile              [========================] 76,085 tok/s
Pure CUDA v1 (results7)  [==========================] 82,462 tok/s
FastTransformer (compile)[====================================] 110,713 tok/s (+45.5%!)
```

---

## 2. Key Takeaways & Physical Breakdown

### 2.1 An Eager Architectural Change Outperforms Compiler Optimization
Notice the comparison:
- **`gpt_compile`**: $107.67\text{ ms}$ ($76{,}085\text{ tok/s}$)
- **`FastTransformer_eager`**: **$100.25\text{ ms}$** (**$81{,}716\text{ tok/s}$**)

Even in raw PyTorch Eager mode (without any JIT compilation, graph fusion, or Triton lowering), the architectural redesign is **$7.42\text{ ms}$ faster than fully compiled standard GPT-2**. This validates your initial intuition: **architectural innovation trumps compiler micro-optimization**.

### 2.2 Reaching the Triple-Digit Milestone: $110{,}713\text{ tok/s}$
When `torch.compile` is applied to FastTransformer:
- **Forward Pass**: Dropped from $38.93\text{ ms} \to \mathbf{26.82\text{ ms}}$ (**$-31.1\%$ latency reduction**).
- **Backward Pass**: Dropped from $67.09\text{ ms} \to \mathbf{45.71\text{ ms}}$ (**$-31.9\%$ latency reduction**).
- **Total Step Time**: Dropped from $107.67\text{ ms} \to \mathbf{73.99\text{ ms}}$ (**$-33.68\text{ ms}$ saved per step**).
- **Throughput**: Surged from $76{,}085\text{ tok/s} \to \mathbf{110{,}713\text{ tok/s}}$ (**$+45.5\%$ throughput leap**).

---

## 3. Why Did This Architecture Deliver Such a Clear Speedup?

1. **Multi-Query Attention (MQA)**:
   - Slashed the unified projection matrix from $\mathbf{W}_{qkv} \in \mathbb{R}^{256 \times 768}$ down to $\mathbb{R}^{256 \times 320}$.
   - Cut key-value DRAM bandwidth and activation tensor storage by **$58\%$** across all 6 layers.
2. **Hardware-Native SDPA**:
   - Eliminated the global $(B, H, T, T)$ attention score matrix in memory, executing within GPU shared memory and SRAM registers.
3. **Lean $2\times$ Fused MLP**:
   - Slashed the single largest consumer of FLOPs ($64\%$ of total block compute) by exactly **$50\%$**, eliminating the memory bus saturation on Turing GDDR6.
4. **Pre-RMSNorm**:
   - Replaced LayerNorm to remove unnecessary mean calculations and reduction barriers.

---

## 4. Next Opportunities

Now that we have confirmed that the **FastTransformer** architecture decisively outperforms `torch.compile` on standard Transformer:

1. **Pure CUDA Engine Implementation**:
   Implement the native CUDA kernels for FastTransformer (MQA unified GEMM + fused Lean MLP + RMSNorm). With custom CUDA execution and CUDA Graph capture, this architecture can target **$> 130{,}000\text{ tok/s}$** on the T4!
2. **Language Modeling Convergence / Perplexity Check**:
   Run a short training run on `data/tinyshakespeare.txt` to verify the cross-entropy loss convergence of FastTransformer vs. Standard GPT-2.