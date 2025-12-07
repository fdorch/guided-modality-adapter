from typing import Optional

import torch
import torch.nn as nn
from .attn import CrossAttentionBlock
from torch import Tensor


class QueryFormerBase(nn.Module):
    """
    Generic Q-former:
      - holds `num_queries` learnable query tokens (Q x D)
      - optionally projects input features to d_model (K/V)
      - returns queries after cross-attending to modality inputs

    Args:
      in_dim: dimensionality of input features (e.g. Whisper encoder dim or spk dim)
      num_queries: number of learnable queries produced for this modality
      d_model: internal projection dimension (model dim)
    """

    def __init__(
        self,
        in_dim: int,
        num_queries: int,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_queries = num_queries
        self.d_model = d_model

        # learnable query tokens (initialized)
        self.queries = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)

        # project input to d_model for keys/values
        self.kv_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
            nn.GELU(),
        )

        # one cross-attention block that turns learnable queries into conditioned queries
        self.cross = CrossAttentionBlock(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )

    def forward(self, x: Tensor, input_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
          inputs: (B, T_in, in_dim)
          input_mask: optional boolean mask of shape (B, T_in) where True means padding (to be masked)
        Returns:
          queries_out: (B, num_queries, d_model)
        """
        B = x.shape[0]
        # project keys/values
        kv = self.kv_proj(x)  # (B, T_in, d_model)

        # expand the learnable queries to batch
        q = self.queries.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, Q, d_model)

        # cross-attention: queries attend to kv
        q = self.cross(q, kv, kv_mask=input_mask)
        return q
