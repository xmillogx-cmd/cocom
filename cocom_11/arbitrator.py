"""
Arbitrator for COCOM v11.

Fuses expert predictions using learned attention and confidence weighting.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from .layers.normalization import RMSNorm
from .layers.activations import SwiGLU


class Arbitrator(nn.Module):
    """Arbitrator with FFN, own classifier, and input passthrough."""
    
    def __init__(
        self,
        d_model: int,
        num_classes: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        self.ffn = SwiGLU(d_model, dropout=dropout)
        self.norm = RMSNorm(d_model)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        self.mix_weight = nn.Parameter(torch.tensor(0.5))
    
    def _compute_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)
        max_entropy = math.log(self.num_classes)
        return 1.0 - (entropy / max_entropy)
    
    def forward(
        self,
        input_context: torch.Tensor,
        feat_s: torch.Tensor, feat_t: torch.Tensor, feat_e: torch.Tensor,
        logits_s: torch.Tensor, logits_t: torch.Tensor, logits_e: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        B = feat_s.size(0)
        
        all_feats = torch.stack([input_context, feat_s, feat_t, feat_e], dim=1)
        query = self.query.expand(B, -1, -1)
        
        attended, attn_weights = self.attention(query, all_feats, all_feats)
        attended = attended.squeeze(1)
        
        ffn_out = self.ffn(attended)
        arb_features = self.norm(attended + ffn_out)
        
        own_logits = self.classifier(arb_features)
        
        conf_s = self._compute_confidence(logits_s)
        conf_t = self._compute_confidence(logits_t)
        conf_e = self._compute_confidence(logits_e)
        
        expert_attn = attn_weights.squeeze(1)[:, 1:]
        confidence = torch.stack([conf_s, conf_t, conf_e], dim=1)
        
        combined_weights = F.softmax(expert_attn * confidence, dim=-1)
        all_logits = torch.stack([logits_s, logits_t, logits_e], dim=1)
        weighted_logits = (combined_weights.unsqueeze(-1) * all_logits).sum(dim=1)
        
        mix = torch.sigmoid(self.mix_weight)
        arb_logits = mix * weighted_logits + (1 - mix) * own_logits
        
        analysis = {
            'confidence_s': conf_s.mean().item(),
            'confidence_t': conf_t.mean().item(),
            'confidence_e': conf_e.mean().item(),
            'expert_weights': combined_weights.mean(dim=0).tolist(),
            'mix_weight': mix.item(),
        }
        
        return arb_logits, analysis
