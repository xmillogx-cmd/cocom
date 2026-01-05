# COCOM v10 Changelog

## v10.0.0 (2024-12)

### New Features
- **Unified Sequential Training**: Single script for complete training pipeline
- **Staged Training**: Train experts separately (Tactical → Strategic → Exploratory → Full)
- **FFT-based EMA**: O(N log N) complexity for Tactical expert
- **Enhanced Arbitrator**: FFN + own classifier + input passthrough
- **Configurable Epochs**: CLI arguments for per-stage epoch count
- **Freeze/Unfreeze API**: `model.freeze_all_except(['tactical', 'embed'])`

### Architecture
- `StrategicExpert`: Top-k sparse attention (25% keys)
- `TacticalExpert`: Multi-scale EMA with linear projection
- `ExploratoryExpert`: Sampled attention with chunked processing
- `Arbitrator`: Cross-attention over [input, S, T, E] + confidence weighting

### Training Improvements
- MEGA-style optimizer for EMA (LR=0.005, betas=(0.9, 0.98))
- Linear warmup + decay scheduler
- Weight decay = 0.01
- Per-step LR updates during training

### Performance
- ListOps (LRA): ~40-42% accuracy
- Parameters: ~2.6M
- Memory-efficient chunked attention for Exploratory

### Breaking Changes from v9
- Tactical now uses simple linear projection instead of FFN
- Warmup is mandatory for Tactical training
- Arbitrator frozen during expert pretraining stages
