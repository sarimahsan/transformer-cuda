import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention matching CUDA implementation.
    Uses unified QKV projection and causal upper-triangular masking.
    """
    def __init__(self, d_model: int = 256, num_heads: int = 8, max_seq_len: int = 256):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model {d_model} must be divisible by num_heads {num_heads}"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.max_seq_len = max_seq_len

        # Unified QKV projection (C -> 3 * C)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        # Output projection (C -> C)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        # Causal mask: 1s in lower triangle, 0s in upper triangle
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # 1. Project to Q, K, V: (B, T, 3 * C)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. Reshape and transpose to (B, H, T, d_head)
        q = q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        # 3. Scaled dot-product attention scores
        scale = 1.0 / math.sqrt(self.d_head)
        att = (q @ k.transpose(-2, -1)) * scale

        # 4. Causal mask
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # 5. Softmax & attention value aggregation
        probs = F.softmax(att, dim=-1)
        y = probs @ v  # (B, H, T, d_head)

        # 6. Recombine heads: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 7. Output projection
        return self.out_proj(y)


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Block with Multi-Head Attention and GELU MLP.
    """
    def __init__(self, d_model: int = 256, num_heads: int = 8, d_ff: int = 1024, max_seq_len: int = 256, eps: float = 1e-5):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model, eps=eps)
        self.attn = CausalSelfAttention(d_model, num_heads, max_seq_len)
        self.ln_2 = nn.LayerNorm(d_model, eps=eps)
        self.ffn_1 = nn.Linear(d_model, d_ff, bias=True)
        self.ffn_2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Self-Attention with residual connection
        x = x + self.attn(self.ln_1(x))
        # Pre-LN Feed-Forward Network with residual connection
        x = x + self.ffn_2(F.gelu(self.ffn_1(self.ln_2(x)), approximate="tanh"))
        return x


class PyTorchGPT(nn.Module):
    """
    Decoder-Only Generative Pre-trained Transformer matching Pure CUDA layout.
    """
    def __init__(
        self,
        vocab_size: int = 65,
        max_seq_len: int = 256,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 1024,
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
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len, eps=eps)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model, eps=eps)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # Token + Position Embeddings
        x = self.tok_emb(idx) + self.pos_emb(pos)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final LayerNorm & Un-embedding Head
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss

    def configure_optimizers(self, lr: float = 3e-4, weight_decay: float = 0.01, betas: tuple = (0.9, 0.999), eps: float = 1e-8):
        return torch.optim.AdamW(self.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
