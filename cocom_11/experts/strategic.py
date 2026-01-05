"""
Strategic Expert for COCOM v11.

Sparse global attention using top-k selection.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from ..layers.normalization import RMSNorm
from ..layers.activations import SwiGLU
from ..layers.embeddings import RotaryPositionEmbedding


class StrategicExpert(nn.Module):
    """Strategic Expert: Sparse global attention (top-k)."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_classes: int = 10,
        top_k_ratio: float = 0.25,
        dropout: float = 0.1,
        use_rope: bool = True,
        rope_dims: int = 2,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.top_k_ratio = top_k_ratio
        self.use_rope = use_rope
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.norm = RMSNorm(d_model)
        self.norm_ffn = RMSNorm(d_model)
        
        if use_rope:
            self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len, rope_dims)
        
        self.ffn = SwiGLU(d_model, dropout=dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor, grid_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.use_rope:
            q, k = self.rope(q, k, N, grid_size)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        k_keep = max(1, int(N * self.top_k_ratio))
        topk_vals, topk_indices = torch.topk(scores, k=k_keep, dim=-1)
        
        sparse_attn = torch.zeros_like(scores)
        softmax_vals = torch.softmax(topk_vals, dim=-1).to(sparse_attn.dtype)
        sparse_attn.scatter_(-1, topk_indices, softmax_vals)
        
        out = torch.matmul(sparse_attn, v)
        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        out = self.out_proj(out)
        out = self.dropout(out)
        out = self.norm(x + out)
        
        ffn_out = self.ffn(out)
        out = self.norm_ffn(out + ffn_out)
        
        features = out.mean(dim=1)
        logits = self.classifier(features)
        return logits, features
