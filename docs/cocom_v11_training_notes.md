# COCOM v11 ListOps Training Results

## Summary Table

| Experiment | Group Size | Seq Len | Batch Size | Training Mode | Best Acc | Params | Notes |
|------------|------------|---------|------------|---------------|----------|--------|-------|
| v11 tokens | 1 | 2048 | 16 | Staged (T→S→E→Full) | 40.55% (Arb) | 3.4M | Collapsed on full stage |
| v11 grouped-8 | 8 | 256 | 128 | Staged | ~40% | 3.5M | 10x faster training |
| v11 grouped-32 | 32 | 64 | 1024 | Staged | 38.25% | 5.0M | Early stopping on full |
| v11 full-only | 8 | 256 | 256 | Full only (30 epochs) | 37.65% | 3.5M | Collapsed epoch 6-8, early stop 9 |
| **CIFAR-10 2x2** | 2x2 patch | 256 | 256 | Staged | **68.77%** | 2.96M | Strategic best (63.6%), Arb=69.2% |
| **LRA Text** | 16 | 256 | 128 | Staged | **69.36%** | 4.1M | Tactical best (68.7%), beats Transformer +5% |
| **Pathfinder-32** | patch=4 | 64 | 32 | Staged | **65.33%** | ~3M | Exploratory best (66.9%), Arb=67.4% |

## Detailed Results

### Experiment 1: v11 tokens (group=1)
- **Config:** input_type='tokens', seq_len=2048, batch=16
- **Results:**
  - Strategic: 37.05%
  - Tactical: 38.00%
  - Exploratory: 39.10% ← Best expert
  - Arbitrator: 40.55%
  - Final: 39.85%
- **Observations:**
  - Collapsed on full stage (17.8% later epochs)
  - Class 0 and 9 dominated (80%, 78%)
  - Other classes: 10-20%

### Experiment 2: v11 grouped-8
- **Config:** input_type='tokens_conv1d', kernel=8, seq_len=256, batch=128
- **Results (staged):**
  - Tactical stage best: 36.45%
  - Strategic stage best: 39.30%
  - Exploratory stage best: 39.80%
- **Observations:**
  - Same accuracy as tokens, 10x faster
  - Arb improves as experts improve (frozen but input changes)

### Experiment 3: v11 grouped-32
- **Config:** input_type='tokens_conv1d', kernel=32, seq_len=64, batch=1024
- **Results:**
  - Tactical: 37.55%
  - Strategic: 37.90%
  - Exploratory: 38.25% ← Best
  - Full: 37.25% (degraded)
- **Observations:**
  - Early stopping on full (epoch 6)
  - Very fast training (~40x vs tokens)
  - 2% lower than group=8

## Key Findings

### 1. Tokenization Invariance
- All group sizes achieve ~40% accuracy
- Bottleneck is architecture, not tokenization
- Training speed scales with compression

### 2. Staged Training Benefits
- Each expert finds local optimum
- Arbitrator inherits expert improvements
- No interference during early training

### 3. Full Stage Problems
- Tends to overfit/collapse
- Early stopping helps
- May need lower LR or regularization

## Variables for Correlation Analysis

| Variable | Values Tested |
|----------|---------------|
| group_size | 1, 8, 32 |
| batch_size | 16, 128, 512, 1024 |
| training_mode | staged, full-only |
| seq_len | 64, 256, 2048 |
| lr_tactical | 0.003 |
| lr_full | 3e-4 |
| epochs_per_stage | 5-8 |
| warmup | True/False |

## Hypotheses to Test

1. **H1:** accuracy ~ constant across group_sizes (confirmed ~40%)
2. **H2:** training_time ~ 1/group_size² (attention O(N²))
3. **H3:** staged > full-only for final accuracy
4. **H4:** arbitrator_acc >= max(expert_acc)
5. **H5:** collapse_probability ~ epochs in full stage

## Next Experiments

- [ ] Full-only with group=8, batch=512
- [ ] Transfer: train group=32 → adapt to group=8
- [ ] Multi-layer experts (2-4 layers)
- [ ] Compare with S4/MEGA at same param count
