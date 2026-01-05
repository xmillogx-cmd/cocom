"""
Tactical Expert for COCOM v11.

Uses MultiScaleEMA with FFT convolution for O(N log N) complexity.
(Replaces RWKV which was too slow for long sequences)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from ..layers.normalization import RMSNorm


class MultiScaleEMA(nn.Module):
    """
    Multi-Scale EMA using FFT convolution (from MEGA).
    
    O(N log N) complexity via FFT-based convolution.
    Works on all platforms (no CUDA kernels required).
    """
    
    def __init__(self, d_model: int, n_scales: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_scales = n_scales
        
        # Learnable parameters for each scale
        self.delta = nn.Parameter(torch.randn(n_scales, d_model) * 0.2)
        self.alpha = nn.Parameter(torch.randn(n_scales, d_model) * 0.2)
        self.beta = nn.Parameter(torch.randn(n_scales, d_model) * 0.02)
        
        # Output projection
        self.scale_proj = nn.Linear(d_model * n_scales, d_model)
        
        # Gating
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        # Residual weight
        self.omega = nn.Parameter(torch.randn(d_model) * 0.1)
        self.norm = nn.LayerNorm(d_model)
    
    def _compute_kernel(self, length: int, scale_idx: int) -> torch.Tensor:
        """Compute EMA kernel for given length."""
        p = torch.sigmoid(self.delta[scale_idx])
        alpha = torch.sigmoid(self.alpha[scale_idx])
        q = 1.0 - p * alpha
        
        indices = torch.arange(length, device=p.device, dtype=p.dtype)
        log_kernel = indices.unsqueeze(0) * torch.log(q + 1e-8).unsqueeze(1)
        kernel = p.unsqueeze(1) * self.beta[scale_idx].unsqueeze(1) * torch.exp(log_kernel)
        return kernel  # (D, L)
    
    def _fft_conv(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        FFT-based convolution.
        
        Args:
            x: (B, D, L)
            kernel: (D, L)
        Returns:
            (B, D, L)
        """
        B, D, L = x.shape
        fft_len = L
        
        # FFT
        k_f = torch.fft.rfft(kernel.float(), n=2 * fft_len)
        x_f = torch.fft.rfft(x.float(), n=2 * fft_len)
        
        # Convolution in frequency domain
        out = torch.fft.irfft(x_f * k_f.unsqueeze(0), n=2 * fft_len)
        
        return out[:, :, :L].type_as(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) input
        Returns:
            (B, N, D) output
        """
        B, N, D = x.shape
        
        # Residual
        residual = x * self.omega
        
        # (B, N, D) -> (B, D, N) for convolution
        x_conv = x.transpose(1, 2)
        
        scale_outputs = []
        for i in range(self.n_scales):
            kernel = self._compute_kernel(N, i)  # (D, N)
            out = self._fft_conv(x_conv, kernel)  # (B, D, N)
            scale_outputs.append(out.transpose(1, 2))  # (B, N, D)
        
        # Concatenate and project
        multi_scale = torch.cat(scale_outputs, dim=-1)  # (B, N, D*n_scales)
        projected = self.scale_proj(multi_scale)  # (B, N, D)
        
        # Gating with residual
        gate = self.gate(x)
        output = gate * F.silu(projected + residual) + (1 - gate) * x
        
        return self.norm(output)


class TacticalExpert(nn.Module):
    """
    Tactical Expert: EMA-based sequential processing.
    
    Uses MultiScaleEMA with FFT convolution for O(N log N) complexity.
    Optimized for long sequences (ListOps: 2048 tokens).
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_classes: int = 10,
        n_scales: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        
        # EMA with FFT convolution (fast!)
        self.ema = MultiScaleEMA(d_model, n_scales)
        
        # Projection
        self.proj = nn.Linear(d_model, d_model)
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ema_out = self.ema(x)
        proj_out = self.proj(ema_out)
        proj_out = self.dropout(proj_out)
        out = self.norm(ema_out + proj_out)
        
        features = out.mean(dim=1)
        logits = self.classifier(features)
        return logits, features
