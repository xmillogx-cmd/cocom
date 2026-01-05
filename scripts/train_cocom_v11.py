"""
COCOM v11: Universal Training Script

Unified trainer for all LRA tasks:
- ListOps, Text, Image (CIFAR-10), Pathfinder-32/128

Usage:
    python train_cocom_v11.py --task listops --batch_size 128
    python train_cocom_v11.py --task text --group_size 16
    python train_cocom_v11.py --task image --patch_size 2
    python train_cocom_v11.py --task pathfinder32
"""

import os
import sys

# Add parent directory to path for cocom_11 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json
import argparse

from cocom_11 import COCOMv11


# ============================================================================
# Dataset Classes
# ============================================================================

class ListOpsDataset(Dataset):
    """ListOps dataset from numpy files."""
    
    def __init__(self, data_dir: str, split: str = 'train'):
        data_path = Path(data_dir) / "listops_kaggle"
        self.sequences = np.load(data_path / f"{split}_sequences.npy", mmap_mode='r')
        self.labels = np.load(data_path / f"{split}_labels.npy", mmap_mode='r')
        self.vocab_size = int(self.sequences.max()) + 1
        self.seq_len = self.sequences.shape[1]
        self.num_classes = int(self.labels.max()) + 1
        print(f"[ListOps {split}] {len(self)} samples, seq_len={self.seq_len}, vocab={self.vocab_size}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx].copy(), dtype=torch.long),
            torch.tensor(int(self.labels[idx]), dtype=torch.long)
        )


class LRATextDataset(Dataset):
    """LRA Text dataset from pickle files."""
    
    def __init__(self, data_dir: str, split: str = 'train'):
        pickle_path = Path(data_dir) / f"lra-text.{split}.pickle"
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        self.sequences = [item['input_ids_0'] for item in data]
        self.labels = [item['label'] for item in data]
        
        max_token = max(max(seq) for seq in self.sequences)
        self.vocab_size = max_token + 1
        self.seq_len = len(self.sequences[0])
        self.num_classes = max(self.labels) + 1
        print(f"[Text {split}] {len(self)} samples, seq_len={self.seq_len}, vocab={self.vocab_size}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


class ImageDataset(Dataset):
    """LRA Image (CIFAR-10 grayscale) dataset."""
    
    def __init__(self, data_dir: str, split: str = 'train'):
        data_path = Path(data_dir) / "image"
        self.sequences = np.load(data_path / f"{split}_sequences.npy")
        self.labels = np.load(data_path / f"{split}_labels.npy")
        self.seq_len = self.sequences.shape[1]
        self.num_classes = int(self.labels.max()) + 1
        self.vocab_size = None  # Continuous input
        print(f"[Image {split}] {len(self)} samples, seq_len={self.seq_len}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(int(self.labels[idx]), dtype=torch.long)
        )


class PathfinderDataset(Dataset):
    """Pathfinder dataset (32x32 or 128x128)."""
    
    def __init__(self, data_dir: str, split: str = 'train', resolution: int = 32):
        if resolution == 32:
            data_path = Path(data_dir) / "pathfinder_kaggle32"
        else:
            data_path = Path(data_dir) / f"pathfinder_kaggle{resolution}"
        
        self.sequences = np.load(data_path / f"{split}_sequences.npy")
        self.labels = np.load(data_path / f"{split}_labels.npy")
        self.seq_len = self.sequences.shape[1]
        self.num_classes = int(self.labels.max()) + 1
        self.vocab_size = None
        print(f"[Pathfinder-{resolution} {split}] {len(self)} samples, seq_len={self.seq_len}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(int(self.labels[idx]), dtype=torch.long)
        )


# ============================================================================
# Task Presets
# ============================================================================

TASK_PRESETS = {
    'listops': {
        'dataset_class': ListOpsDataset,
        'input_type': 'tokens_conv1d',
        'group_size': 8,
        'patch_size': None,
        'rope_dims': 1,
        'batch_size': 128,
        'data_subdir': '',
    },
    'text': {
        'dataset_class': LRATextDataset,
        'input_type': 'tokens_conv1d',
        'group_size': 16,
        'patch_size': None,
        'rope_dims': 1,
        'batch_size': 128,
        'data_subdir': '',
    },
    'image': {
        'dataset_class': ImageDataset,
        'input_type': 'image',
        'group_size': None,
        'patch_size': 2,
        'rope_dims': 2,
        'batch_size': 256,
        'data_subdir': '',
    },
    'pathfinder32': {
        'dataset_class': PathfinderDataset,
        'input_type': 'image',
        'group_size': None,
        'patch_size': 4,
        'rope_dims': 2,
        'batch_size': 64,
        'data_subdir': '',
        'resolution': 32,
    },
    'pathfinder128': {
        'dataset_class': PathfinderDataset,
        'input_type': 'image',
        'group_size': None,
        'patch_size': 8,
        'rope_dims': 2,
        'batch_size': 32,
        'data_subdir': '',
        'resolution': 128,
    },
}


# ============================================================================
# Training Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration with all hyperparameters."""
    
    # Task
    task: str = 'listops'
    data_dir: str = './lra_data'  # Relative to script or absolute path
    
    # Model
    d_model: int = 256
    n_heads: int = 8
    input_type: str = 'tokens_conv1d'
    group_size: int = 8
    patch_size: int = 4
    use_rope: bool = True
    rope_dims: int = 1
    
    # Training
    batch_size: int = 128
    staged: bool = True
    epochs_tactical: int = 8
    epochs_strategic: int = 5
    epochs_exploratory: int = 5
    epochs_full: int = 20
    lr_tactical: float = 0.003
    lr_strategic: float = 0.001
    lr_exploratory: float = 0.001
    lr_full: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    early_stop_patience: int = 5
    
    # System
    device: str = 'cuda'
    fp16: bool = True
    num_workers: int = 0
    checkpoint_dir: str = './checkpoints'  # Relative to script or absolute path
    checkpoint: Optional[str] = None  # Resume from
    
    def apply_preset(self, preset: dict):
        """Apply task preset values."""
        if preset.get('input_type'):
            self.input_type = preset['input_type']
        if preset.get('group_size'):
            self.group_size = preset['group_size']
        if preset.get('patch_size'):
            self.patch_size = preset['patch_size']
        if preset.get('rope_dims'):
            self.rope_dims = preset['rope_dims']
        if preset.get('batch_size'):
            self.batch_size = preset['batch_size']


# ============================================================================
# Unified Trainer
# ============================================================================

class UnifiedTrainer:
    """Universal COCOM v11 trainer for all tasks."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.fp16 = config.fp16 and 'cuda' in str(self.device)
        
        # Get task preset
        preset = TASK_PRESETS.get(config.task)
        if preset is None:
            raise ValueError(f"Unknown task: {config.task}")
        
        # Load datasets
        dataset_class = preset['dataset_class']
        resolution = preset.get('resolution')
        
        if resolution:
            self.train_data = dataset_class(config.data_dir, 'train', resolution)
            self.test_data = dataset_class(config.data_dir, 'test', resolution)
        else:
            self.train_data = dataset_class(config.data_dir, 'train')
            self.test_data = dataset_class(config.data_dir, 'test')
        
        self.train_loader = DataLoader(
            self.train_data, batch_size=config.batch_size, 
            shuffle=True, num_workers=config.num_workers
        )
        self.test_loader = DataLoader(
            self.test_data, batch_size=config.batch_size,
            shuffle=False, num_workers=config.num_workers
        )
        
        # Determine model parameters
        seq_len = self.train_data.seq_len
        num_classes = self.train_data.num_classes
        vocab_size = getattr(self.train_data, 'vocab_size', None) or 256
        
        # Calculate image size for 2D tasks
        img_size = int(seq_len ** 0.5) if config.input_type == 'image' else 32
        
        print(f"\nModel config:")
        print(f"  input_type={config.input_type}, seq_len={seq_len}")
        print(f"  vocab_size={vocab_size}, num_classes={num_classes}")
        
        # Create model
        self.model = COCOMv11(
            d_model=config.d_model,
            n_heads=config.n_heads,
            num_classes=num_classes,
            img_size=img_size,
            patch_size=config.patch_size,
            input_type=config.input_type,
            vocab_size=vocab_size,
            seq_len=seq_len,
            embed_kernel_size=config.group_size,
            use_rope=config.use_rope,
            rope_dims=config.rope_dims,
        ).to(self.device)
        
        # Load checkpoint if specified
        if config.checkpoint:
            self._load_checkpoint(config.checkpoint)
        
        params = sum(p.numel() for p in self.model.parameters())
        print(f"  Parameters: {params:,}")
        print(f"  Effective seq_len: {self.model.seq_len}")
        
        self.scaler = GradScaler(enabled=self.fp16)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.history = []
    
    def _load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = Path(path)
        if checkpoint.exists():
            state_dict = torch.load(checkpoint, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"✅ Loaded checkpoint: {checkpoint}")
        else:
            print(f"⚠️ Checkpoint not found: {checkpoint}")
    
    def _get_stages(self) -> List[dict]:
        """Get training stages based on config."""
        if self.config.staged:
            return [
                {'name': 'tactical', 'epochs': self.config.epochs_tactical, 
                 'trainable': ['embed', 'tactical'], 'lr': self.config.lr_tactical, 'warmup': True},
                {'name': 'strategic', 'epochs': self.config.epochs_strategic,
                 'trainable': ['strategic'], 'lr': self.config.lr_strategic, 'warmup': False},
                {'name': 'exploratory', 'epochs': self.config.epochs_exploratory,
                 'trainable': ['exploratory'], 'lr': self.config.lr_exploratory, 'warmup': False},
                {'name': 'full', 'epochs': self.config.epochs_full,
                 'trainable': ['all'], 'lr': self.config.lr_full, 'warmup': False},
            ]
        else:
            return [
                {'name': 'full', 'epochs': self.config.epochs_full,
                 'trainable': ['all'], 'lr': self.config.lr_full, 'warmup': True},
            ]
    
    def _get_stage_logits(self, stage_name: str, analysis: dict) -> torch.Tensor:
        if stage_name == 'tactical':
            return analysis['logits_t']
        elif stage_name == 'strategic':
            return analysis['logits_s']
        elif stage_name == 'exploratory':
            return analysis['logits_e']
        else:
            return analysis['avg_logits']
    
    def _get_loss(self, stage_name: str, analysis: dict, y: torch.Tensor) -> torch.Tensor:
        if stage_name == 'tactical':
            return F.cross_entropy(analysis['logits_t'], y)
        elif stage_name == 'strategic':
            return F.cross_entropy(analysis['logits_s'], y)
        elif stage_name == 'exploratory':
            return F.cross_entropy(analysis['logits_e'], y)
        else:  # full - v7 style: gradient averaging
            loss_s = F.cross_entropy(analysis['logits_s'], y)
            loss_t = F.cross_entropy(analysis['logits_t'], y)
            loss_e = F.cross_entropy(analysis['logits_e'], y)
            arb_loss = F.cross_entropy(analysis['arb_logits'], y)
            return (loss_s + loss_t + loss_e) / 3 + 0.5 * arb_loss
    
    def train_epoch(self, stage_name: str, optimizer, scheduler=None) -> dict:
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        correct_s = correct_t = correct_e = correct_arb = 0
        
        pbar = tqdm(self.train_loader, desc=f"  {stage_name}", leave=False)
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=self.fp16):
                final_logits, analysis = self.model(x)
                loss = self._get_loss(stage_name, analysis, y)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(optimizer)
            self.scaler.update()
            
            if scheduler is not None:
                scheduler.step()
            
            stage_logits = self._get_stage_logits(stage_name, analysis)
            total_correct += (stage_logits.argmax(-1) == y).sum().item()
            total_samples += y.size(0)
            total_loss += loss.item() * y.size(0)
            
            correct_s += (analysis['pred_s'] == y).sum().item()
            correct_t += (analysis['pred_t'] == y).sum().item()
            correct_e += (analysis['pred_e'] == y).sum().item()
            correct_arb += (analysis['arb_logits'].argmax(-1) == y).sum().item()
            
            pbar.set_postfix(
                loss=loss.item(),
                acc=total_correct/total_samples,
                S=correct_s/total_samples,
                T=correct_t/total_samples,
                E=correct_e/total_samples,
            )
        
        return {
            'loss': total_loss / total_samples,
            'stage_acc': total_correct / total_samples,
            'acc_s': correct_s / total_samples,
            'acc_t': correct_t / total_samples,
            'acc_e': correct_e / total_samples,
            'acc_arb': correct_arb / total_samples,
        }
    
    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        total_correct = 0
        total_samples = 0
        correct_s = correct_t = correct_e = correct_arb = 0
        total_agree = 0
        n_batches = 0
        
        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            with autocast(enabled=self.fp16):
                final_logits, analysis = self.model(x)
            
            total_correct += (final_logits.argmax(-1) == y).sum().item()
            total_samples += y.size(0)
            correct_s += (analysis['pred_s'] == y).sum().item()
            correct_t += (analysis['pred_t'] == y).sum().item()
            correct_e += (analysis['pred_e'] == y).sum().item()
            correct_arb += (analysis['arb_logits'].argmax(-1) == y).sum().item()
            total_agree += analysis['agreement_rate']
            n_batches += 1
        
        return {
            'acc': total_correct / total_samples,
            'acc_s': correct_s / total_samples,
            'acc_t': correct_t / total_samples,
            'acc_e': correct_e / total_samples,
            'acc_arb': correct_arb / total_samples,
            'agree': total_agree / n_batches
        }
    
    def run(self) -> float:
        task = self.config.task
        print("\n" + "="*70)
        print(f"COCOM v11: {task.upper()}")
        print("="*70)
        
        stages = self._get_stages()
        best_acc = 0
        
        for stage in stages:
            stage_name = stage['name']
            epochs = stage['epochs']
            trainable = stage['trainable']
            lr = stage['lr']
            
            print(f"\n{'='*70}")
            print(f"STAGE: {stage_name.upper()} ({epochs} epochs)")
            print("="*70)
            
            self.model.freeze_all_except(trainable)
            trainable_params = self.model.get_trainable_params()
            print(f"Trainable: {trainable_params:,}")
            
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr,
                betas=(0.9, 0.99) if stage_name == 'tactical' else (0.9, 0.999),
                weight_decay=self.config.weight_decay
            )
            
            total_steps = len(self.train_loader) * epochs
            
            if stage.get('warmup', False):
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=lr, total_steps=total_steps,
                    pct_start=0.15, anneal_strategy='cos',
                    div_factor=25, final_div_factor=100
                )
            else:
                scheduler = None
            
            stage_best = 0
            no_improve = 0
            
            for epoch in range(epochs):
                train_metrics = self.train_epoch(stage_name, optimizer, scheduler)
                test_metrics = self.evaluate()
                
                saved = ""
                if test_metrics['acc'] > stage_best:
                    stage_best = test_metrics['acc']
                    no_improve = 0
                    torch.save(self.model.state_dict(), 
                              self.checkpoint_dir / f"cocom_v11_{task}_{stage_name}.pt")
                    saved = " 💾"
                    
                    if test_metrics['acc'] > best_acc:
                        best_acc = test_metrics['acc']
                        torch.save(self.model.state_dict(), 
                                  self.checkpoint_dir / f"cocom_v11_{task}_best.pt")
                else:
                    no_improve += 1
                
                print(f"  Epoch {epoch+1}/{epochs} | Test: {test_metrics['acc']:.4f} | "
                      f"S={test_metrics['acc_s']:.3f} T={test_metrics['acc_t']:.3f} "
                      f"E={test_metrics['acc_e']:.3f} Arb={test_metrics['acc_arb']:.3f}{saved}")
                
                self.history.append({
                    'stage': stage_name, 
                    'epoch': epoch + 1,
                    **test_metrics
                })
                
                if stage_name == 'full' and no_improve >= self.config.early_stop_patience:
                    print(f"  Early stopping")
                    break
            
            print(f"Stage {stage_name} best: {stage_best:.4f}")
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Best Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
        
        # Save final checkpoint and history
        torch.save(self.model.state_dict(), 
                  self.checkpoint_dir / f"cocom_v11_{task}_full.pt")
        
        with open(self.checkpoint_dir / f"cocom_v11_{task}_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return best_acc


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='COCOM v11 Universal Trainer')
    
    # Task
    parser.add_argument('--task', type=str, default='listops',
                        choices=['listops', 'text', 'image', 'pathfinder32', 'pathfinder128'],
                        help='Task to train on')
    parser.add_argument('--data_dir', type=str, default='./lra_data',
                        help='Data directory')
    
    # Model
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--input_type', type=str, default=None,
                        help='Input type (auto-detected from task if not specified)')
    parser.add_argument('--group_size', type=int, default=None,
                        help='Token grouping for 1D tasks')
    parser.add_argument('--patch_size', type=int, default=None,
                        help='Patch size for 2D tasks')
    parser.add_argument('--use_rope', action='store_true', default=True,
                        help='Use Rotary Position Embedding')
    parser.add_argument('--no_rope', action='store_true', help='Disable RoPE')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (auto from task preset if not specified)')
    parser.add_argument('--staged', action='store_true', default=True,
                        help='Use staged training (default)')
    parser.add_argument('--full_only', action='store_true',
                        help='Train all parameters from start')
    parser.add_argument('--epochs_tactical', type=int, default=8)
    parser.add_argument('--epochs_strategic', type=int, default=5)
    parser.add_argument('--epochs_exploratory', type=int, default=5)
    parser.add_argument('--epochs_full', type=int, default=20)
    parser.add_argument('--lr_tactical', type=float, default=0.003)
    parser.add_argument('--lr_strategic', type=float, default=0.001)
    parser.add_argument('--lr_exploratory', type=float, default=0.001)
    parser.add_argument('--lr_full', type=float, default=3e-4)
    parser.add_argument('--early_stop', type=int, default=5,
                        help='Early stopping patience for full stage')
    
    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fp32', action='store_true', help='Disable FP16')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Checkpoint output directory')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Get task preset
    preset = TASK_PRESETS.get(args.task, {})
    
    # Build config, applying preset defaults then CLI overrides
    config = TrainingConfig(
        task=args.task,
        data_dir=args.data_dir,
        d_model=args.d_model,
        n_heads=args.n_heads,
        input_type=args.input_type or preset.get('input_type', 'tokens_conv1d'),
        group_size=args.group_size or preset.get('group_size', 8),
        patch_size=args.patch_size or preset.get('patch_size', 4),
        use_rope=not args.no_rope,
        rope_dims=preset.get('rope_dims', 1),
        batch_size=args.batch_size or preset.get('batch_size', 128),
        staged=not args.full_only,
        epochs_tactical=args.epochs_tactical,
        epochs_strategic=args.epochs_strategic,
        epochs_exploratory=args.epochs_exploratory,
        epochs_full=args.epochs_full,
        lr_tactical=args.lr_tactical,
        lr_strategic=args.lr_strategic,
        lr_exploratory=args.lr_exploratory,
        lr_full=args.lr_full,
        early_stop_patience=args.early_stop,
        device=args.device,
        fp16=not args.fp32,
        checkpoint_dir=args.output_dir,
        checkpoint=args.checkpoint,
    )
    
    print(f"Device: {args.device}")
    print(f"Task: {args.task}")
    print(f"Staged training: {config.staged}")
    
    trainer = UnifiedTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
