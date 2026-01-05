# COCOM-Attention v11
![architecture](https://github.com/user-attachments/assets/4fc32285-98e3-426b-ae4b-2c890107a7dd)

**Cooperative Consensus Multi-Expert Attention** - A multi-expert sequence classification architecture with staged sequential training.

## Overview

COCOM v11 combines three specialized attention experts with a confidence-weighted arbitrator for sequence classification tasks. Each expert uses a different attention mechanism optimized for different pattern types:

| Expert | Mechanism | Best For |
|--------|-----------|----------|
| **Strategic** | Top-k sparse attention | Global patterns, key tokens |
| **Tactical** | Multi-scale EMA (FFT) | Sequential patterns, local dependencies |
| **Exploratory** | Sampled global attention | Wide coverage, structure understanding |
| **Arbitrator** | Cross-attention + FFN | Combining expert predictions |

## Key Features

- **Staged Sequential Training**: Train each expert separately before fine-tuning together
- **Freeze/Unfreeze Support**: Selective parameter training per stage
- **FFT-based EMA**: O(N log N) complexity for efficient sequence processing
- **Confidence-Weighted Arbitration**: Entropy-based expert weighting
- **Dataset Diagnostic**: Expert metrics reveal dataset characteristics

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
python scripts/train_cocom_v11.py \
    --device cuda:0 \
    --batch_size 16 \
    --tactical_epochs 10 \
    --strategic_epochs 5 \
    --exploratory_epochs 5 \
    --full_epochs 20
```

## Architecture

```
Input Sequence
      │
      ▼
┌─────────────────┐
│   Embedding     │ ← Token/Continuous + Positional
│   + LayerNorm   │
└────────┬────────┘
         │
    ┌────┴────┬─────────────┐
    ▼         ▼             ▼
┌───────┐ ┌───────┐   ┌───────────┐
│  S    │ │  T    │   │     E     │
│ Top-k │ │ EMA   │   │ Sampled   │
│ Attn  │ │ (FFT) │   │ Attn      │
└───┬───┘ └───┬───┘   └─────┬─────┘
    │         │             │
    └────┬────┴─────────────┘
         ▼
┌─────────────────┐
│   Arbitrator    │ ← Attention + FFN + Classifier
│   (Consensus)   │
└────────┬────────┘
         ▼
    Final Prediction
```

## Training Stages

### Stage 1: Tactical (EMA)
- **Trainable**: Embeddings + Tactical Expert
- **LR**: 0.005 (MEGA-style)
- **Betas**: (0.9, 0.98)
- **Warmup**: 2 epochs linear warmup + linear decay

> ⚠️ **Important for EMA Training**: The EMA mechanism requires sufficient training samples to reach peak learning rate and stabilize. For a 50K sample dataset, use:
> - 2-3 warmup epochs (~100-150K samples)
> - Minimum 5 training epochs (~250K+ samples total)
> - More epochs recommended for better convergence

### Stage 2: Strategic
- **Trainable**: Strategic Expert only
- **LR**: 1e-4
- **Embeddings**: Frozen (preserves learned representations)

### Stage 3: Exploratory
- **Trainable**: Exploratory Expert only
- **LR**: 1e-4
- **Embeddings**: Frozen

### Stage 4: Full Training
- **Trainable**: All parameters
- **LR**: 5e-5
- **Early stopping**: 5 epochs without improvement

## Model Components

### MultiScaleEMA
FFT-based Exponential Moving Average from MEGA paper:
- O(N log N) complexity via FFT convolution
- Multiple decay scales for multi-resolution processing
- Learnable decay rates and mixing parameters

### StrategicExpert
Sparse global attention using top-k selection:
- Focuses on most important key positions
- Efficient for long sequences with sparse patterns
- Configurable `top_k_ratio` (default: 0.25)

### TacticalExpert
EMA-based sequential processing:
- No attention mechanism (pure EMA)
- Simple linear projection + LayerNorm
- Best for sequential/temporal patterns

### ExploratoryExpert
Sampled global attention:
- Samples `max_keys` positions uniformly
- Chunked processing to avoid OOM
- Good for understanding full sequence structure

### Arbitrator
Confidence-weighted expert combination:
- Cross-attention over [input, S, T, E] features
- FFN for "thinking" about combination
- Own classifier for independent prediction
- Learnable mix between own prediction and expert consensus

> **Note: Arbitrator as Standalone Classifier**
> 
> Tests show that the Arbitrator can classify **without** expert logits mixing:
> 
> | Mode | Accuracy |
> |------|----------|
> | Full system | 40.30% |
> | Arbitrator (mixed) | 40.05% |
> | **Arbitrator (OWN only)** | **38.70%** |
> | Experts average | 39.85% |
> 
> The Arbitrator's own classifier achieves 38.70% using only cross-attention features from input and expert representations — making it a capable classifier on its own, not just an orchestrator.

## Dataset Diagnostic

Run COCOM on your dataset and analyze expert metrics to understand data characteristics:

| Expert Performance | Dataset Characteristic | Recommendation |
|-------------------|------------------------|----------------|
| **S >> T, E** | Key tokens/positions matter most, data may be noisy | Add conv/pooling layer to reduce seq_len. Quality may improve on T or E |
| **T >> S, E** | Sequential/temporal patterns dominant | Consider lighter RNN/RWKV-style model |
| **E >> S, T** | Global structure understanding needed | Consider S4/Mamba for O(N) with infinite memory |
| **All similar** | Mixed pattern types | COCOM is optimal, keep full architecture |
| **Arb >> Experts** | Complex task requiring multi-expert fusion | Ensemble approach is necessary |

### Modular Deployment

Each expert can be used independently as a lightweight classifier:

| Mode | Parameters | Use Case |
|------|------------|----------|
| **Single Expert** | ~800K | Fast inference, edge devices, when one expert dominates |
| **Full Model + Arbitrator** | ~3.4M | Maximum quality, complex tasks |

> **Tip:** If Strategic dominates significantly, your data likely has redundant positions. Adding another convolution layer to compress the input may shift performance to Tactical or Exploratory without quality loss.

## Results on ListOps (LRA)

**Final Results** (trained with sequential staging):

| Stage | Epochs | Best Acc | S | T | E | Arb |
|-------|--------|----------|------|------|------|------|
| Tactical | 5 | 22.2% | 8.6% | 23.0% | 8.1% | 22.2% |
| Strategic | 5 | 38.2% | 38.5% | 23.0% | 8.1% | 38.3% |
| Exploratory | 5 | 38.0% | 38.5% | 23.0% | 37.7% | 37.3% |
| **Full** | 15* | **40.3%** | 34.0% | 23.0% | 36.7% | 40.1% |

*Early stopping after 15/20 epochs

**Key observations:**
- Exploratory (E=36.7%) > Strategic (S=34.0%) > Tactical (T=23.0%)
- Arbitrator combines experts effectively: 40.3% > best single expert
- ListOps requires global attention (S, E) more than sequential EMA (T)

## Command Line Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--task` | str | `listops` | Task: `listops`, `text`, `image`, `pathfinder32`, `pathfinder128` |
| `--data_dir` | str | `./lra_data` | Data directory |
| `--d_model` | int | `256` | Model dimension |
| `--n_heads` | int | `8` | Number of attention heads |
| `--input_type` | str | `None` | Input type (auto-detected from task preset) |
| `--group_size` | int | `None` | Token grouping for 1D tasks |
| `--patch_size` | int | `None` | Patch size for 2D tasks |
| `--use_rope` | flag | `True` | Use Rotary Position Embedding |
| `--no_rope` | flag | `False` | Disable RoPE |
| `--batch_size` | int | `None` | Batch size (auto from task preset) |
| `--staged` | flag | `True` | Use staged training |
| `--full_only` | flag | `False` | Train all parameters from start |
| `--epochs_tactical` | int | `8` | Epochs for tactical stage |
| `--epochs_strategic` | int | `5` | Epochs for strategic stage |
| `--epochs_exploratory` | int | `5` | Epochs for exploratory stage |
| `--epochs_full` | int | `20` | Epochs for full stage |
| `--lr_tactical` | float | `0.003` | Learning rate for tactical stage |
| `--lr_strategic` | float | `0.001` | Learning rate for strategic stage |
| `--lr_exploratory` | float | `0.001` | Learning rate for exploratory stage |
| `--lr_full` | float | `3e-4` | Learning rate for full stage |
| `--early_stop` | int | `5` | Early stopping patience (full stage only) |
| `--device` | str | `cuda` | Device for training |
| `--fp32` | flag | `False` | Disable FP16 (use FP32) |
| `--checkpoint` | str | `None` | Path to checkpoint to resume from |
| `--output_dir` | str | `./checkpoints` | Checkpoint output directory |




## File Structure

```
cocom
├── cocom_v11          # Model architecture
└── scripts            # Train scripts
└── tests
experiments/
├── cocom_v10_unified_training.py  # Training script
└── ...
```

## Citation

If you use COCOM in your research, please cite:

```bibtex
@software{cocom2026,
  title={COCOM-Attention: Cooperative Consensus Multi-Expert Attention},
  author={Ostrozhnykh Yuri Victorovich},
  year={2026}
}
```

## License

MIT License
