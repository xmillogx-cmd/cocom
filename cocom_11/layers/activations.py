"""
SwiGLU - Swish Gated Linear Unit activation for FFN.

Better quality than GELU - gated linear unit with SiLU.
Used in: LLaMA, PaLM, Gemma

Paper: "GLU Variants Improve Transformer" (Shazeer, 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU activation for FFN."""
    
    def __init__(self, d_model: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or int(d_model * 8 / 3)  # ~2.67x, standard ratio
        
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))
