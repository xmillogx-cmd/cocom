# COCOM v11: Development History

This document provides a brief overview of COCOM architecture evolution from v4 to v11.

---

## Version Timeline

| Version | Key Innovation | Limitation |
|---------|----------------|------------|
| **v4** | Sequence-based training with temporal learning | Shared QKV weights between experts |
| **v5** | Majority voting + CRF memory | CRF requires sequences, not universal |
| **v6** | Learnable Memory (class_keys) | Memory as separate component |
| **v7** | Memory as arbiter, gradient → winner only | Gradient dilution for non-winners |
| **v8** | ConfidenceArbitrator with entropy-based weighting | Still gradient dilution |
| **v9** | EMA Tactical with FFT (O(N log N)) | EMA needs warmup |
| **v10** | Staged sequential training | Manual stage transitions |
| **v11** | RMSNorm, SwiGLU, RoPE, OneCycleLR | **Current version** |

---

## Key Insights from Development

### 1. Independent Experts Are Critical
Early versions (v3-v4) shared QKV weights — experts were mathematically identical. v5+ uses fully independent weights.

### 2. Memory/Arbitrator Role Evolution
- v5: CRF records only consensus → learns reliable patterns
- v7: Memory arbitrates when experts disagree
- v8+: Confidence-weighted combination + own classifier

### 3. EMA for Tactical Expert
v9 replaced window attention with Multi-Scale EMA:
- O(N log N) via FFT convolution
- Better for sequential patterns
- Requires proper warmup (MEGA-style)

### 4. Staged Training
v10 introduced sequential training:
1. Tactical + Embeddings (warmup critical)
2. Strategic (frozen embeddings)
3. Exploratory (frozen embeddings)
4. Full model (all trainable)

---

## Architecture Comparison

```
v4-v6: Shared QKV → all experts similar
v7+:   Independent experts → true diversity  
v9:    Tactical = EMA (not attention)
v11:   Modern layers (RMSNorm, SwiGLU, RoPE)
```

---

## References

For detailed technical documentation, see:
- [guide.md](guide.md) — Full v11 technical guide
- [cocom_math_foundation.md](cocom_math_foundation.md) — Theoretical foundations
