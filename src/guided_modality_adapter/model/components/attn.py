import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .ffn import FeedForward


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block:
      out = LayerNorm( Q + MultiHeadAttention(Q, KV) )
      out = FFN(out)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff=d_ff, dropout=dropout)

    def forward(
        self, q: torch.Tensor, kv: torch.Tensor, kv_mask: Optional[torch.Tensor] = None
    ):
        # q: (B, Qq, D), kv: (B, Qkv, D)
        # kv_mask: (B, Qkv) where True means to **mask** (like key_padding_mask)
        attn_out, _ = self.attn(query=q, key=kv, value=kv, key_padding_mask=kv_mask)
        x = self.ln1(q + attn_out)
        x = self.ff(x)
        return x
