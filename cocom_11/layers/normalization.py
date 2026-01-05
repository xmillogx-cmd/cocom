"""
RMSNorm - Root Mean Square Layer Normalization.

Faster than LayerNorm - no mean centering, just scale.
Used in: LLaMA, Mistral, Gemma

Paper: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight
