import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (Shazeer, 2020) with unified Gate-Up projection.
    Computes: SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
    """
    def __init__(self, d_model: int = 256, d_ff: int = 640):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.gate_up_proj = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class MultiScaleRetention(nn.Module):
    """
    Multi-Scale Retention (RetNet, Sun et al., 2023).
    Replaces quadratic causal softmax attention with parallel retention:
        R = (Q * K^T ⊙ D) * V
    where D is a precomputed, static causal decay buffer.
    Features:
      1. Zero softmax row reductions.
      2. Static decay factors: zero autograd power derivatives.
      3. Pure dense Tensor-Core GEMMs.
    """
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        max_seq_len: int = 256,
        scale: float = None
    ):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.scale = scale or (1.0 / math.sqrt(self.d_head))

        # Unified QKV projection (C -> 3C)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        # Gating projection (C -> C)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        # Output projection (C -> C)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Precompute static multi-scale decay buffer D:
        # gamma_h = 1 - 2^(-5 - h) for h in [0, H-1]
        gammas = 1.0 - 2.0 ** (-5.0 - torch.arange(0, num_heads, dtype=torch.float32))
        
        # Construct static D matrix of shape (1, H, max_seq_len, max_seq_len)
        indices = torch.arange(max_seq_len, dtype=torch.float32)
        # diff[i, j] = i - j
        diff = indices.unsqueeze(1) - indices.unsqueeze(0)
        causal_mask = diff >= 0

        # D[h, i, j] = gamma_h^(i - j) if i >= j else 0
        d_matrix = torch.zeros(1, num_heads, max_seq_len, max_seq_len, dtype=torch.float32)
        for h in range(num_heads):
            gamma = gammas[h].item()
            decay = (gamma ** diff.clamp(min=0.0)) * causal_mask.float()
            d_matrix[0, h] = decay

        self.register_buffer("decay_mask", d_matrix, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        H = self.num_heads
        d = self.d_head

        # 1. Project QKV and Gate
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        g = self.g_proj(x)

        # Reshape to (B, H, T, d)
        q = q.view(B, T, H, d).transpose(1, 2)
        k = k.view(B, T, H, d).transpose(1, 2)
        v = v.view(B, T, H, d).transpose(1, 2)

        # 2. Parallel Retention (Pure Tensor Core GEMMs + Static Elementwise Decay)
        # att = (Q * K^T) / sqrt(d) -> (B, H, T, T)
        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # Apply precomputed static decay (zero autograd power gradients!)
        att = att * self.decay_mask[:, :, :T, :T]

        # R = att * V -> (B, H, T, d)
        r = torch.matmul(att, v)

        # 3. Recombine heads: (B, T, C)
        r = r.transpose(1, 2).contiguous().view(B, T, C)

        # 4. Gated Output Projection: (SiLU(g) * r) * W_out
        y = F.silu(g) * r
        return self.out_proj(y)


class RetNetBlock(nn.Module):
    """
    RetNet Block with Pre-RMSNorm, Multi-Scale Retention, and SwiGLU MLP.
    """
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        d_ff: int = 640,
        max_seq_len: int = 256,
        eps: float = 1e-5
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps=eps)
        self.retention = MultiScaleRetention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len
        )
        self.norm2 = RMSNorm(d_model, eps=eps)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.retention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PyTorchGLA(nn.Module):
    """
    Full Decoder-Only Language Model powered by Parallel Multi-Scale Retention (RetNet).
    Eliminates softmax row reductions and dynamic power derivatives.
    """
    def __init__(
        self,
        vocab_size: int = 65,
        max_seq_len: int = 256,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 640,
        chunk_size: int = 64,
        eps: float = 1e-5
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            RetNetBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                eps=eps
            )
            for _ in range(num_layers)
        ])

        self.norm_f = RMSNorm(d_model, eps=eps)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # Token + Positional embeddings
        x = self.tok_emb(idx) + self.pos_emb(pos)

        # Pass through Retention blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def configure_optimizers(self, lr: float = 3e-4, weight_decay: float = 0.01, betas: tuple = (0.9, 0.999)):
        decay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=lr, betas=betas, fused=False)
