"""
Embedding layers for COCOM v11.

- PatchEmbedding: 2D patch extraction for images
- RotaryPositionEmbedding: RoPE for relative position encoding
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


class PatchEmbedding(nn.Module):
    """
    Patch Embedding for 2D images.
    
    Converts (B, H*W) flat sequence back to (B, H, W),
    extracts patches, and projects to d_model.
    """
    
    def __init__(self, img_size: int = 32, patch_size: int = 4, d_model: int = 256, in_channels: int = 1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H*W) flat sequence of pixels [0, 1]
        Returns:
            (B, num_patches, d_model) patch embeddings
        """
        B = x.size(0)
        x = x.view(B, 1, self.img_size, self.img_size)
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return self.norm(x)


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for attention.
    
    Applies rotation to q and k based on position, providing:
    - Relative position encoding (attention depends on distance)
    - Local bias (nearby tokens have stronger attention)
    - Zero additional parameters
    
    Args:
        dim: head dimension (d_model // n_heads)
        max_seq_len: maximum sequence length
        rope_dims: 1 for 1D sequences, 2 for 2D grids (images)
        base: base frequency for rotations
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, rope_dims: int = 1, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.rope_dims = rope_dims
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Build sin/cos cache for given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_cached = emb.cos()[None, None, :, :]
        sin_cached = emb.sin()[None, None, :, :]
        self.register_buffer('cos_cached', cos_cached, persistent=False)
        self.register_buffer('sin_cached', sin_cached, persistent=False)
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of x."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        seq_len: int,
        grid_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors.
        
        Args:
            q: (B, n_heads, N, head_dim)
            k: (B, n_heads, N, head_dim)
            seq_len: actual sequence length
            grid_size: for 2D mode, the grid dimension
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        
        if self.rope_dims == 2 and grid_size is not None:
            return self._apply_rope_2d(q, k, grid_size)
        else:
            return self._apply_rope_1d(q, k, seq_len)
    
    def _apply_rope_1d(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 1D RoPE for sequential data."""
        cos = self.cos_cached[:, :, :seq_len, :].to(q.dtype)
        sin = self.sin_cached[:, :, :seq_len, :].to(q.dtype)
        
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot
    
    def _apply_rope_2d(self, q: torch.Tensor, k: torch.Tensor, grid_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 2D RoPE for grid-structured data (images)."""
        B, H, N, D = q.shape
        half_d = D // 2
        
        rows = torch.arange(grid_size, device=q.device)
        cols = torch.arange(grid_size, device=q.device)
        row_pos = rows.repeat_interleave(grid_size)
        col_pos = cols.repeat(grid_size)
        
        row_freqs = torch.einsum('i,j->ij', row_pos.float(), self.inv_freq[:half_d // 2])
        col_freqs = torch.einsum('i,j->ij', col_pos.float(), self.inv_freq[:half_d // 2])
        
        row_emb = torch.cat([row_freqs, row_freqs], dim=-1)
        col_emb = torch.cat([col_freqs, col_freqs], dim=-1)
        
        cos_row = row_emb.cos()[None, None, :, :]
        sin_row = row_emb.sin()[None, None, :, :]
        cos_col = col_emb.cos()[None, None, :, :]
        sin_col = col_emb.sin()[None, None, :, :]
        
        q_row, q_col = q[..., :half_d], q[..., half_d:]
        k_row, k_col = k[..., :half_d], k[..., half_d:]
        
        q_row_rot = (q_row * cos_row.to(q.dtype)) + (self._rotate_half(q_row) * sin_row.to(q.dtype))
        k_row_rot = (k_row * cos_row.to(k.dtype)) + (self._rotate_half(k_row) * sin_row.to(k.dtype))
        
        q_col_rot = (q_col * cos_col.to(q.dtype)) + (self._rotate_half(q_col) * sin_col.to(q.dtype))
        k_col_rot = (k_col * cos_col.to(k.dtype)) + (self._rotate_half(k_col) * sin_col.to(k.dtype))
        
        q_rot = torch.cat([q_row_rot, q_col_rot], dim=-1)
        k_rot = torch.cat([k_row_rot, k_col_rot], dim=-1)
        return q_rot, k_rot
