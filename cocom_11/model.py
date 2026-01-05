"""
COCOM v11 - Main Model.

Modular architecture with MultiScaleEMA Tactical Expert (FFT convolution).
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, List

from .layers.normalization import RMSNorm
from .layers.embeddings import PatchEmbedding, RotaryPositionEmbedding
from .experts.strategic import StrategicExpert
from .experts.tactical import TacticalExpert
from .experts.exploratory import ExploratoryExpert
from .arbitrator import Arbitrator


class COCOMv11(nn.Module):
    """
    COCOM v11: Modular architecture with MultiScaleEMA Tactical Expert.
    
    Features:
    - Modular file structure (cocom_11/)
    - MultiScaleEMA with FFT convolution O(N log N)
    - RMSNorm and SwiGLU by default
    - RoPE for position encoding (1D/2D)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        num_classes: int = 2,
        img_size: int = 32,
        patch_size: int = 4,
        input_type: str = 'image',
        vocab_size: int = 17,
        seq_len: int = 2048,
        embed_kernel_size: int = 4,
        use_rope: bool = True,
        rope_dims: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.input_type = input_type
        self.use_rope = use_rope
        self.rope_dims = rope_dims
        
        # Input embedding based on type
        if input_type == 'image':
            self.input_embed = PatchEmbedding(img_size, patch_size, d_model)
            self.seq_len = self.input_embed.num_patches
            self.grid_size = img_size // patch_size
        elif input_type == 'image_conv1d':
            self.embed_kernel_size = embed_kernel_size
            self.input_embed = nn.Conv1d(
                in_channels=1,
                out_channels=d_model,
                kernel_size=embed_kernel_size,
                stride=embed_kernel_size,
                padding=0
            )
            self.embed_norm = nn.LayerNorm(d_model)
            self.seq_len = seq_len // embed_kernel_size
            self.grid_size = int(self.seq_len ** 0.5)
        elif input_type == 'tokens':
            self.input_embed = nn.Embedding(vocab_size, d_model)
            self.seq_len = seq_len
            self.grid_size = None
        elif input_type == 'tokens_conv1d':
            # Grouped token embedding: embed -> Conv1d
            self.embed_kernel_size = embed_kernel_size
            self.token_embed = nn.Embedding(vocab_size, d_model)
            self.token_conv = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=embed_kernel_size,
                stride=embed_kernel_size,
                padding=0
            )
            self.embed_norm = nn.LayerNorm(d_model)
            self.seq_len = seq_len // embed_kernel_size
            self.grid_size = None
        elif input_type == 'features':
            # Bypass: input is already (B, seq_len, d_model) from external backbone
            self.input_embed = nn.Identity()
            self.seq_len = seq_len
            self.grid_size = int(seq_len ** 0.5) if seq_len > 0 else None
        else:  # continuous
            self.input_embed = nn.Linear(1, d_model)
            self.seq_len = seq_len
            self.grid_size = int(seq_len ** 0.5)
        
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.02)
        self.norm = RMSNorm(d_model)
        
        # Experts
        self.expert_strategic = StrategicExpert(
            d_model, n_heads, num_classes,
            use_rope=use_rope, rope_dims=rope_dims, max_seq_len=self.seq_len
        )
        self.expert_tactical = TacticalExpert(d_model, n_heads, num_classes)
        self.expert_exploratory = ExploratoryExpert(
            d_model, n_heads, num_classes,
            use_rope=use_rope, rope_dims=rope_dims, max_seq_len=self.seq_len
        )
        
        self.arbitrator = Arbitrator(d_model, num_classes)
    
    def freeze_all_except(self, parts: List[str]):
        """Freeze all parameters except specified parts."""
        if 'all' in parts:
            for param in self.parameters():
                param.requires_grad = True
            return
        
        for param in self.parameters():
            param.requires_grad = False
        
        if 'embed' in parts:
            if self.input_type == 'tokens_conv1d':
                for param in self.token_embed.parameters():
                    param.requires_grad = True
                for param in self.token_conv.parameters():
                    param.requires_grad = True
                for param in self.embed_norm.parameters():
                    param.requires_grad = True
            else:
                for param in self.input_embed.parameters():
                    param.requires_grad = True
            self.pos_embed.requires_grad = True
            for param in self.norm.parameters():
                param.requires_grad = True
        
        if 'strategic' in parts:
            for param in self.expert_strategic.parameters():
                param.requires_grad = True
        
        if 'tactical' in parts:
            for param in self.expert_tactical.parameters():
                param.requires_grad = True
        
        if 'exploratory' in parts:
            for param in self.expert_exploratory.parameters():
                param.requires_grad = True
        
        if 'arbitrator' in parts:
            for param in self.arbitrator.parameters():
                param.requires_grad = True
    
    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Embedding
        if self.input_type == 'continuous':
            x = x.unsqueeze(-1)
            x = self.input_embed(x)
        elif self.input_type == 'image_conv1d':
            x = x.unsqueeze(1)
            x = self.input_embed(x)
            x = x.transpose(1, 2)
            x = self.embed_norm(x)
        elif self.input_type == 'tokens_conv1d':
            # Embed tokens then group with Conv1d
            x = self.token_embed(x)           # (B, seq_len, d_model)
            x = x.transpose(1, 2)              # (B, d_model, seq_len)
            x = self.token_conv(x)             # (B, d_model, seq_len/kernel)
            x = x.transpose(1, 2)              # (B, seq_len/kernel, d_model)
            x = self.embed_norm(x)
        elif self.input_type == 'features':
            # Already embedded features from CNN backbone
            pass  # x is (B, seq_len, d_model)
        else:
            x = self.input_embed(x)
        
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.norm(x)
        
        # Expert predictions
        logits_s, feat_s = self.expert_strategic(x, self.grid_size)
        logits_t, feat_t = self.expert_tactical(x)
        logits_e, feat_e = self.expert_exploratory(x, self.grid_size)
        
        # Input context for arbitrator
        input_context = x.mean(dim=1)
        
        # Arbitrator
        arb_logits, arb_analysis = self.arbitrator(
            input_context, feat_s, feat_t, feat_e, logits_s, logits_t, logits_e
        )
        
        # Majority voting consensus
        pred_s = logits_s.argmax(dim=-1)
        pred_t = logits_t.argmax(dim=-1)
        pred_e = logits_e.argmax(dim=-1)
        
        agree_st = (pred_s == pred_t)
        agree_te = (pred_t == pred_e)
        agree_se = (pred_s == pred_e)
        
        all_agree = agree_st & agree_te
        only_st = agree_st & ~agree_te
        only_te = agree_te & ~agree_st
        only_se = agree_se & ~agree_st
        
        avg_logits = (logits_s + logits_t + logits_e) / 3
        final_logits = arb_logits.clone()
        
        mask = all_agree.unsqueeze(-1).expand(-1, self.num_classes)
        final_logits = torch.where(mask, avg_logits, final_logits)
        
        mask = only_st.unsqueeze(-1).expand(-1, self.num_classes)
        final_logits = torch.where(mask, (logits_s + logits_t) / 2, final_logits)
        
        mask = only_te.unsqueeze(-1).expand(-1, self.num_classes)
        final_logits = torch.where(mask, (logits_t + logits_e) / 2, final_logits)
        
        mask = only_se.unsqueeze(-1).expand(-1, self.num_classes)
        final_logits = torch.where(mask, (logits_s + logits_e) / 2, final_logits)
        
        has_consensus = agree_st | agree_te | agree_se
        
        analysis = {
            'agreement_rate': has_consensus.float().mean().item(),
            'avg_logits': avg_logits,
            'arb_logits': arb_logits,
            'logits_s': logits_s,
            'logits_t': logits_t,
            'logits_e': logits_e,
            'pred_s': pred_s,
            'pred_t': pred_t,
            'pred_e': pred_e,
            'final_logits': final_logits,
            **arb_analysis
        }
        
        return final_logits, analysis
