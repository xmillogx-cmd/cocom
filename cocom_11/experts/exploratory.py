"""
Exploratory Expert for COCOM v11.

Full global attention for complete sequence coverage.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from ..layers.normalization import RMSNorm
from ..layers.activations import SwiGLU
from ..layers.embeddings import RotaryPositionEmbedding


class ExploratoryExpert(nn.Module):
    """Exploratory Expert: Full global attention."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_classes: int = 10,
        dropout: float = 0.1,
        use_rope: bool = True,
        rope_dims: int = 2,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
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
        
        # Full attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        out = self.out_proj(out)
        out = self.dropout(out)
        out = self.norm(x + out)
        
        ffn_out = self.ffn(out)
        out = self.norm_ffn(out + ffn_out)
        
        features = out.mean(dim=1)
        logits = self.classifier(features)
        return logits, features
