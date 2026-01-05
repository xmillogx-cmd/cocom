"""
COCOM v11: ListOps Training with EMA Tactical Expert

ListOps is a SEQUENTIAL task - perfect for Tactical Expert!
(Unlike Pathfinder which is 2D spatial)

Key optimizations for ListOps:
1. input_type='tokens' - vocabulary embedding for discrete tokens
2. RoPE 1D - sequential position encoding (not 2D like images)
3. MultiScaleEMA Tactical - FFT convolution O(N log N) for long sequences
4. Longer warmup - ListOps needs stable early training
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from cocom_11 import COCOMv11


class ListOpsDataset(Dataset):
    """ListOps dataset loader supporting both original and kaggle formats."""
    
    def __init__(self, data_dir: str, split: str = 'train', use_kaggle: bool = True):
        if use_kaggle:
            data_path = Path(data_dir) / "listops_kaggle"
        else:
            data_path = Path(data_dir) / "listops"
        
        self.sequences = np.load(data_path / f"{split}_sequences.npy", mmap_mode='r')
        self.labels = np.load(data_path / f"{split}_labels.npy", mmap_mode='r')
        print(f"Loaded {split}: {len(self.sequences)} samples, seq_len={self.sequences.shape[1]}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx].copy(), dtype=torch.long),
            torch.tensor(int(self.labels[idx]), dtype=torch.long)
        )


class ListOpsTrainerV11:
    """Trainer for COCOM v11 on ListOps - SEQUENTIAL task."""
    
    # Training stages optimized for ListOps
    # Tactical should excel here (sequential data, not spatial)
    STAGES = [
        {'name': 'tactical', 'epochs': 8, 'trainable': ['embed', 'tactical'], 'lr': 0.003, 'betas': (0.9, 0.98), 'warmup': True},
        {'name': 'strategic', 'epochs': 5, 'trainable': ['strategic'], 'lr': 0.001, 'betas': (0.9, 0.999), 'warmup': False},
        {'name': 'exploratory', 'epochs': 5, 'trainable': ['exploratory'], 'lr': 0.001, 'betas': (0.9, 0.999), 'warmup': False},
        {'name': 'full', 'epochs': 20, 'trainable': ['all'], 'lr': 3e-4, 'betas': (0.9, 0.999), 'warmup': False},
    ]
    
    def __init__(
        self,
        data_dir: str = "./lra_data",
        batch_size: int = 16,
        device: str = 'cuda',
        use_fp16: bool = True,
        use_kaggle: bool = True,
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and 'cuda' in device
        
        # Load data
        self.train_data = ListOpsDataset(data_dir, 'train', use_kaggle)
        self.test_data = ListOpsDataset(data_dir, 'test', use_kaggle)
        
        seq_len = self.train_data.sequences.shape[1]
        vocab_size = int(self.train_data.sequences.max()) + 1
        
        print(f"Vocabulary size: {vocab_size}")
        print(f"Sequence length: {seq_len}")
        
        self.train_loader = DataLoader(
            self.train_data, batch_size=batch_size, shuffle=True, num_workers=0
        )
        self.test_loader = DataLoader(
            self.test_data, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        # Create model for SEQUENTIAL data (not image)
        self.model = COCOMv11(
            d_model=256,
            n_heads=8,
            num_classes=10,  # ListOps outputs 0-9
            seq_len=seq_len,
            input_type='tokens',  # Vocabulary embedding
            vocab_size=vocab_size,
            use_rope=True,   # Enable RoPE
            rope_dims=1,     # 1D for sequential data (not 2D like images)!
        ).to(self.device)
        
        params = sum(p.numel() for p in self.model.parameters())
        print(f"COCOMv11 Parameters: {params:,}")
        print(f"Input: tokens (vocab={vocab_size}), RoPE 1D enabled")
        print(f"EMA Tactical (FFT) should excel on sequential ListOps!")
        
        self.scaler = GradScaler(enabled=self.use_fp16)
        self.checkpoint_dir = Path("./checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.history = []
    
    def get_stage_logits(self, stage_name: str, analysis: dict) -> torch.Tensor:
        """Get logits for current training stage."""
        if stage_name == 'tactical':
            return analysis['logits_t']
        elif stage_name == 'strategic':
            return analysis['logits_s']
        elif stage_name == 'exploratory':
            return analysis['logits_e']
        else:  # full
            return analysis['avg_logits']
    
    def get_loss(self, stage_name: str, analysis: dict, y: torch.Tensor) -> torch.Tensor:
        """Compute loss for current training stage."""
        if stage_name == 'tactical':
            return F.cross_entropy(analysis['logits_t'], y)
        elif stage_name == 'strategic':
            return F.cross_entropy(analysis['logits_s'], y)
        elif stage_name == 'exploratory':
            return F.cross_entropy(analysis['logits_e'], y)
        else:  # full - combined loss
            avg_loss = F.cross_entropy(analysis['avg_logits'], y)
            arb_loss = F.cross_entropy(analysis['arb_logits'], y)
            return avg_loss + 0.5 * arb_loss
    
    def train_epoch(self, stage_name: str, optimizer, scheduler=None) -> dict:
        """Train one epoch with optional LR scheduler."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        correct_s = correct_t = correct_e = correct_arb = correct_final = 0
        
        pbar = tqdm(self.train_loader, desc=f"  {stage_name}", leave=False)
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=self.use_fp16):
                final_logits, analysis = self.model(x)
                loss = self.get_loss(stage_name, analysis, y)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
            
            if scheduler is not None:
                scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # Stage-specific accuracy
            stage_logits = self.get_stage_logits(stage_name, analysis)
            total_correct += (stage_logits.argmax(-1) == y).sum().item()
            total_samples += y.size(0)
            total_loss += loss.item() * y.size(0)
            
            # Track all expert accuracies
            correct_s += (analysis['pred_s'] == y).sum().item()
            correct_t += (analysis['pred_t'] == y).sum().item()
            correct_e += (analysis['pred_e'] == y).sum().item()
            correct_arb += (analysis['arb_logits'].argmax(-1) == y).sum().item()
            correct_final += (final_logits.argmax(-1) == y).sum().item()
            
            pbar.set_postfix(
                loss=loss.item(),
                acc=total_correct/total_samples,
                S=correct_s/total_samples,
                T=correct_t/total_samples,
                E=correct_e/total_samples,
                lr=f"{current_lr:.6f}"
            )
        
        return {
            'loss': total_loss / total_samples,
            'stage_acc': total_correct / total_samples,
            'acc_s': correct_s / total_samples,
            'acc_t': correct_t / total_samples,
            'acc_e': correct_e / total_samples,
            'acc_arb': correct_arb / total_samples,
            'acc_final': correct_final / total_samples
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
            
            with autocast(enabled=self.use_fp16):
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
    
    def run(self):
        print("\n" + "="*70)
        print("COCOM v11: LISTOPS TRAINING (RWKV TACTICAL)")
        print("="*70)
        
        print("\nKey points:")
        print("  1. ListOps is SEQUENTIAL - Tactical should excel!")
        print("  2. RoPE 1D for sequential positions")
        print("  3. Tokens embedding for discrete vocabulary")
        print("  4. MultiScaleEMA with FFT convolution O(N log N)")
        
        best_acc = 0
        
        for stage in self.STAGES:
            stage_name = stage['name']
            epochs = stage['epochs']
            trainable = stage['trainable']
            lr = stage['lr']
            
            print(f"\n{'='*70}")
            print(f"STAGE: {stage_name.upper()} ({epochs} epochs)")
            print("="*70)
            
            self.model.freeze_all_except(trainable)
            trainable_params = self.model.get_trainable_params()
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Training: {trainable}")
            
            betas = stage.get('betas', (0.9, 0.999))
            use_scheduler = stage.get('warmup', False) or stage_name == 'tactical'
            
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr,
                betas=betas,
                eps=1e-8,
                weight_decay=0.01
            )
            
            total_steps = len(self.train_loader) * epochs
            
            if use_scheduler:
                max_lr = lr
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=max_lr,
                    total_steps=total_steps,
                    pct_start=0.15,  # 15% warmup for ListOps stability
                    anneal_strategy='cos',
                    div_factor=25,
                    final_div_factor=100
                )
                print(f"Using OneCycleLR: max_lr={max_lr:.6f}, warmup=15%, cosine decay")
            else:
                scheduler = None
                print(f"Using constant LR: {lr}")
            
            stage_best = 0
            epochs_without_improvement = 0
            
            for epoch in range(epochs):
                train_metrics = self.train_epoch(stage_name, optimizer, scheduler)
                test_metrics = self.evaluate()
                
                saved = ""
                if test_metrics['acc'] > stage_best:
                    stage_best = test_metrics['acc']
                    epochs_without_improvement = 0
                    torch.save(self.model.state_dict(), 
                              self.checkpoint_dir / f"cocom_v11_listops_{stage_name}.pt")
                    saved = " 💾"
                    
                    if test_metrics['acc'] > best_acc:
                        best_acc = test_metrics['acc']
                        torch.save(self.model.state_dict(), 
                                  self.checkpoint_dir / "cocom_v11_listops_best.pt")
                else:
                    epochs_without_improvement += 1
                
                print(f"  Epoch {epoch+1}/{epochs} | Test: {test_metrics['acc']:.4f} | "
                      f"S={test_metrics['acc_s']:.3f} T={test_metrics['acc_t']:.3f} "
                      f"E={test_metrics['acc_e']:.3f} Arb={test_metrics['acc_arb']:.3f}{saved}")
                
                self.history.append({
                    'stage': stage_name, 
                    'epoch': epoch + 1,
                    'train_stage_acc': train_metrics['stage_acc'],
                    **test_metrics
                })
                
                # Early stopping for full stage
                if stage_name == 'full' and epochs_without_improvement >= 5:
                    print(f"  Early stopping after {epoch+1} epochs")
                    break
            
            print(f"Stage {stage_name} best: {stage_best:.4f}")
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Best Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
        
        with open(self.checkpoint_dir / "cocom_v11_listops_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return best_acc


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--use_kaggle', action='store_true', default=True,
                        help='Use kaggle ListOps dataset (default: True)')
    # Epochs
    parser.add_argument('--tactical_epochs', type=int, default=8)
    parser.add_argument('--strategic_epochs', type=int, default=5)
    parser.add_argument('--exploratory_epochs', type=int, default=5)
    parser.add_argument('--full_epochs', type=int, default=20)
    # LR settings
    parser.add_argument('--tactical_lr', type=float, default=0.003)
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--full_lr', type=float, default=3e-4)
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Override stages
    ListOpsTrainerV11.STAGES[0]['epochs'] = args.tactical_epochs
    ListOpsTrainerV11.STAGES[1]['epochs'] = args.strategic_epochs
    ListOpsTrainerV11.STAGES[2]['epochs'] = args.exploratory_epochs
    ListOpsTrainerV11.STAGES[3]['epochs'] = args.full_epochs
    
    ListOpsTrainerV11.STAGES[0]['lr'] = args.tactical_lr
    ListOpsTrainerV11.STAGES[1]['lr'] = args.base_lr
    ListOpsTrainerV11.STAGES[2]['lr'] = args.base_lr
    ListOpsTrainerV11.STAGES[3]['lr'] = args.full_lr
    
    print(f"LR: tactical={args.tactical_lr}, base={args.base_lr}, full={args.full_lr}")
    
    trainer = ListOpsTrainerV11(
        batch_size=args.batch_size,
        device=device,
        use_fp16=not args.fp32,
        use_kaggle=args.use_kaggle,
    )
    
    trainer.run()


