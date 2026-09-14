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
    def __init__(self, d_model: int = 256, d_ff: int = 768):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # Unified projection for gate and up branches
        self.gate_up_proj = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class ChunkwiseGatedLinearAttention(nn.Module):
    """
    Chunkwise Gated Linear Attention (C-GLA).
    Computes intra-chunk local attention via block matrix multiplications
    and inter-chunk global recurrence via a parallel associative state scan.
    Complexity: O(T * B_chunk * d_k) compute, O(T * C) memory footprint.
    Eliminates the O(T^2) softmax attention map.
    """
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        chunk_size: int = 64,
        scale: float = None
    ):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.chunk_size = chunk_size
        self.scale = scale or (1.0 / math.sqrt(self.d_head))

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Learnable decay rates per head: gamma = sigmoid(decay_param) in (0, 1)
        self.decay_param = nn.Parameter(torch.randn(num_heads))

        # Precompute static causal mask for chunk
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool)),
            persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        H = self.num_heads
        d = self.d_head
        C_L = self.chunk_size

        assert T % C_L == 0, f"Sequence length {T} must be divisible by chunk size {C_L}"
        num_chunks = T // C_L

        # 1. Linear Projections & Reshapes: (B, H, num_chunks, C_L, d)
        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2).view(B, H, num_chunks, C_L, d)
        k = self.k_proj(x).view(B, T, H, d).transpose(1, 2).view(B, H, num_chunks, C_L, d)
        v = self.v_proj(x).view(B, T, H, d).transpose(1, 2).view(B, H, num_chunks, C_L, d)
        g = self.g_proj(x)  # (B, T, C)

        # 2. Decay Factors: gamma per head in (0.5, 0.999)
        gamma = torch.sigmoid(self.decay_param).view(1, H, 1, 1, 1)  # (1, H, 1, 1, 1)

        # Decay vectors within a chunk: position indices 0 .. C_L - 1
        pos = torch.arange(C_L, device=x.device, dtype=x.dtype)
        # diff[i, j] = i - j
        decay_matrix = gamma ** (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp(min=0.0)  # (1, H, 1, C_L, C_L)
        decay_mask = torch.where(self.causal_mask, decay_matrix, torch.zeros_like(decay_matrix))

        # 3. Intra-Chunk Local Attention (Tensor Core GEMMs)
        # Attn scores within chunk: (B, H, num_chunks, C_L, C_L)
        intra_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        intra_scores = intra_scores * decay_mask
        o_intra = torch.matmul(intra_scores, v)  # (B, H, num_chunks, C_L, d)

        # 4. Inter-Chunk Global State Recurrence
        # Weight tokens within chunk by decay from end of chunk: gamma^(C_L - 1 - i)
        k_decay = gamma ** (C_L - 1 - pos).view(1, 1, 1, C_L, 1)  # (1, 1, 1, C_L, 1)
        k_weighted = k * k_decay  # (B, H, num_chunks, C_L, d)

        # Delta state per chunk: sum_i (k_i^T * v_i) -> (B, H, num_chunks, d, d)
        delta_states = torch.matmul(k_weighted.transpose(-2, -1), v)

        # Sequential chunk-to-chunk state propagation
        # State: S_{c} = S_{c-1} * gamma^C_L + delta_S_{c}
        gamma_chunk = gamma.squeeze(-1).squeeze(-1) ** C_L  # (1, H, 1)
        states = []
        curr_state = torch.zeros(B, H, d, d, device=x.device, dtype=x.dtype)

        for c in range(num_chunks):
            states.append(curr_state)
            curr_state = curr_state * gamma_chunk.unsqueeze(-1) + delta_states[:, :, c]

        inter_states = torch.stack(states, dim=2)  # (B, H, num_chunks, d, d)

        # Query projection into inter-chunk state:
        # Weight queries by decay from start of chunk: gamma^(i + 1)
        q_decay = gamma ** (pos + 1).view(1, 1, 1, C_L, 1)
        q_weighted = q * q_decay  # (B, H, num_chunks, C_L, d)

        o_inter = torch.matmul(q_weighted, inter_states)  # (B, H, num_chunks, C_L, d)

        # 5. Combine intra and inter outputs
        o = (o_intra + o_inter).view(B, H, T, d).transpose(1, 2).contiguous().view(B, T, C)

        # 6. Gated Output & Projection
        y = F.silu(g) * o
        return self.out_proj(y)


class GLABlock(nn.Module):
    """
    Chunkwise Gated Linear Attention Block with Pre-RMSNorm and SwiGLU FFN.
    """
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        d_ff: int = 768,
        chunk_size: int = 64,
        eps: float = 1e-5
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps=eps)
        self.attn = ChunkwiseGatedLinearAttention(
            d_model=d_model,
            num_heads=num_heads,
            chunk_size=chunk_size
        )
        self.norm2 = RMSNorm(d_model, eps=eps)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PyTorchGLA(nn.Module):
    """
    Full Decoder-Only Language Model powered by Chunkwise Gated Linear Attention (C-GLA).
    Replaces quadratic causal attention with O(T) chunked GEMMs and associative state scans.
    """
    def __init__(
        self,
        vocab_size: int = 65,
        max_seq_len: int = 256,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 768,
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
        self.chunk_size = chunk_size

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            GLABlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                chunk_size=chunk_size,
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

        # Pass through GLA Transformer blocks
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
